#!/usr/bin/env bash
set -euo pipefail

# Weekly pipeline runner
# - Renders stitch.Rmd (auto-dated)
# - Detects latest stitched file and sets cutoff = last_date - 8 weeks
# - Runs retrospective generators (ARIMA, LGBM, SVM)
# - Runs prospective generators (ARIMA, LGBM, SVM)
# - Builds adaptive ensemble from last 4 weeks weights and current prospective

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

# Optional CLI arguments for ensemble control
# --lookback <N>             Number of reference weeks for weighting (default 6)
# --history <N>              Number of retrospective weeks to consider (default 8)
# --include-arima <t/f>      Include ARIMA in ensemble (default true)
# --include-svm <t/f>        Include SVM in ensemble (default true)
# --include-lgbm-blended <t/f>  Include LGBM blended in ensemble (default true)
# --include-lgbm-bounded <t/f>  Include LGBM bounded in ensemble (default true)
# --include-lgbm-bounded-wide-N <t/f>  Include LGBM bounded wide variant N in ensemble (N=1-5)
# --include-lgbm-bounded-wide-4-ne <t/f>  Include LGBM bounded wide 4 non-enhanced in ensemble
# --include-joint-twostage <t/f> Include joint two-stage pooled model in ensembles (default true)
# --joint-warmup-weeks <N>       Joint retrospective warmup weeks before as-of (default auto minimum)
# --include-meta-ensemble <t/f>  Run adaptive meta ensemble (default false)
# --incremental-retrospective <t/f>  Reuse prior retrospective cache and only append new anchors (default false)

ENSEMBLE_LOOKBACK_WEEKS=""
ENSEMBLE_HISTORY_WEEKS=""
ENSEMBLE_INCLUDE_ARIMA=""
ENSEMBLE_INCLUDE_SVM="false"  # SKIPPED this week
ENSEMBLE_INCLUDE_LGBM_BLENDED=""
ENSEMBLE_INCLUDE_LGBM_BOUNDED=""
ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_1=""
ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_2=""
ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_3=""
ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_4=""
ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_4_NE=""
ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_5=""
ENSEMBLE_INCLUDE_JOINT_TWOSTAGE="true"
JOINT_WARMUP_WEEKS=""
ENSEMBLE_INCLUDE_META="false"
INCREMENTAL_RETROSPECTIVE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lookback)
      ENSEMBLE_LOOKBACK_WEEKS="$2"; shift 2;;
    --history)
      ENSEMBLE_HISTORY_WEEKS="$2"; shift 2;;
    --include-arima)
      ENSEMBLE_INCLUDE_ARIMA="$2"; shift 2;;
    --include-svm)
      ENSEMBLE_INCLUDE_SVM="$2"; shift 2;;
    --include-lgbm-blended)
      ENSEMBLE_INCLUDE_LGBM_BLENDED="$2"; shift 2;;
    --include-lgbm-bounded)
      ENSEMBLE_INCLUDE_LGBM_BOUNDED="$2"; shift 2;;
    --include-lgbm-bounded-wide-1)
      ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_1="$2"; shift 2;;
    --include-lgbm-bounded-wide-2)
      ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_2="$2"; shift 2;;
    --include-lgbm-bounded-wide-3)
      ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_3="$2"; shift 2;;
    --include-lgbm-bounded-wide-4)
      ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_4="$2"; shift 2;;
    --include-lgbm-bounded-wide-4-ne)
      ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_4_NE="$2"; shift 2;;
    --include-lgbm-bounded-wide-5)
      ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_5="$2"; shift 2;;
    --include-joint-twostage)
      ENSEMBLE_INCLUDE_JOINT_TWOSTAGE="$2"; shift 2;;
    --joint-warmup-weeks)
      JOINT_WARMUP_WEEKS="$2"; shift 2;;
    --include-meta-ensemble)
      ENSEMBLE_INCLUDE_META="$2"; shift 2;;
    --incremental-retrospective)
      INCREMENTAL_RETROSPECTIVE="$2"; shift 2;;
    *)
      shift;;
  esac
done

is_true() {
  local v="${1:-}"
  v="$(printf '%s' "$v" | tr '[:upper:]' '[:lower:]')"
  [[ "$v" == "1" || "$v" == "true" || "$v" == "t" || "$v" == "yes" || "$v" == "y" ]]
}

extract_r_default_int() {
  local file="$1"
  local var="$2"
  local fallback="$3"
  local value=""
  if [[ -f "$file" ]]; then
    value=$(awk -v var="$var" '
      $0 ~ ("^[[:space:]]*" var "[[:space:]]*<-[[:space:]]*[0-9]+") {
        match($0, /<-[[:space:]]*[0-9]+/)
        if (RSTART > 0) {
          s = substr($0, RSTART, RLENGTH)
          gsub(/<-[[:space:]]*/, "", s)
          print s
          exit
        }
      }
    ' "$file")
  fi
  if [[ -z "${value}" ]]; then
    value="$fallback"
  fi
  printf '%s\n' "$value"
}

next_cutoff_from_reference_file() {
  local reference_file="$1"
  local base_cutoff="$2"
  REFERENCE_FILE="$reference_file" BASE_CUTOFF="$base_cutoff" python - <<'PY'
import os
from datetime import timedelta
import pandas as pd

ref_file = os.environ["REFERENCE_FILE"]
base_cutoff = pd.to_datetime(os.environ["BASE_CUTOFF"]).date()
cutoff = base_cutoff

try:
    df = pd.read_csv(ref_file, usecols=["reference_date"])
    if len(df) > 0:
        max_ref = pd.to_datetime(df["reference_date"], errors="coerce").max()
        if pd.notna(max_ref):
            next_cutoff = (max_ref.date() + timedelta(weeks=1))
            if next_cutoff > cutoff:
                cutoff = next_cutoff
except Exception:
    pass

print(cutoff.isoformat())
PY
}

merge_csv_file() {
  local existing_file="$1"
  local new_file="$2"
  local output_file="$3"

  EXISTING_FILE="$existing_file" NEW_FILE="$new_file" OUTPUT_FILE="$output_file" python - <<'PY'
import os
from pathlib import Path
import pandas as pd

existing = Path(os.environ["EXISTING_FILE"])
incoming = Path(os.environ["NEW_FILE"])
output = Path(os.environ["OUTPUT_FILE"])

if not existing.exists() and not incoming.exists():
    raise SystemExit(0)

if existing.exists():
    try:
        old_df = pd.read_csv(existing)
    except Exception:
        old_df = pd.DataFrame()
else:
    old_df = pd.DataFrame()

if incoming.exists():
    try:
        new_df = pd.read_csv(incoming)
    except Exception:
        new_df = pd.DataFrame()
else:
    new_df = pd.DataFrame()

if old_df.empty and new_df.empty:
    raise SystemExit(0)

if old_df.empty:
    merged = new_df
elif new_df.empty:
    merged = old_df
else:
    all_cols = []
    for c in list(old_df.columns) + list(new_df.columns):
        if c not in all_cols:
            all_cols.append(c)
    old_df = old_df.reindex(columns=all_cols)
    new_df = new_df.reindex(columns=all_cols)
    merged = pd.concat([old_df, new_df], ignore_index=True)

def dedupe_keys(cols):
    candidates = [
        ["reference_date", "horizon", "target", "target_end_date", "location", "output_type", "output_type_id"],
        ["reference_date", "horizon", "target_end_date", "location", "output_type", "output_type_id"],
        ["location", "horizon", "forecast_date", "target_date"],
        ["location", "forecast_date", "target_date"],
    ]
    for key in candidates:
        if all(k in cols for k in key):
            return key
    if {"reference_date", "target_end_date", "location"}.issubset(set(cols)):
        key = [k for k in ["reference_date", "horizon", "target", "target_end_date", "location", "output_type", "output_type_id"] if k in cols]
        if len(key) > 0:
            return key
    if {"forecast_date", "target_date", "location"}.issubset(set(cols)):
        key = [k for k in ["location", "horizon", "forecast_date", "target_date"] if k in cols]
        if len(key) > 0:
            return key
    return None

keys = dedupe_keys(list(merged.columns))
if keys is not None:
    merged = merged.drop_duplicates(subset=keys, keep="last")
else:
    merged = merged.drop_duplicates(keep="last")

sort_cols = [c for c in ["reference_date", "forecast_date", "target_end_date", "target_date", "horizon", "location", "output_type_id"] if c in merged.columns]
if len(sort_cols) > 0:
    merged = merged.sort_values(sort_cols, kind="mergesort")

output.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(output, index=False)
PY
}

merge_csv_tree() {
  local incoming_root="$1"
  local target_root="$2"
  if [[ ! -d "$incoming_root" ]]; then
    return 0
  fi
  while IFS= read -r -d '' incoming_file; do
    local rel_path="${incoming_file#"$incoming_root"/}"
    local target_file="${target_root}/${rel_path}"
    merge_csv_file "$target_file" "$incoming_file" "$target_file"
  done < <(find "$incoming_root" -type f -name '*.csv' -print0)
}

echo "==> Computing as-of date (last Saturday)"
ASOF=$(python - <<'PY'
from datetime import date, timedelta
import sys
today=date.today()
# Week starts Saturday -> compute last Saturday
offset=(today.weekday()+2)%7  # Saturday=0
asof=today - timedelta(days=offset)
print(asof.isoformat())
PY
)
ASTS=$(echo "$ASOF" | tr -d -)
echo "AS OF: $ASOF ($ASTS)"

echo "==> Render stitch.Rmd"
Rscript -e "rmarkdown::render('src/stitch.Rmd', output_dir='src', output_file='stitch.html')"

echo "==> Locate stitched file for ASOF"
STITCHED="data/imputed_sets/imputed_and_stitched_hosp_${ASOF}.csv"
if [[ ! -f "$STITCHED" ]]; then
  echo "Expected stitched file not found for ASOF ($ASOF). Falling back to latest."
  STITCHED=$(ls -1 data/imputed_sets/imputed_and_stitched_hosp_*.csv | sort | tail -n 1)
fi
echo "Stitched: $STITCHED"

echo "==> Compute cutoffs"
# CUTOFF_RECENT: last_date - 8 weeks (for bounded LGBM retrospective)
# CUTOFF_SEASON: Oct 1 before the most recent COMPLETED flu season
#                (so prospective uses last season's residuals)
read LAST_DATA_DATE CUTOFF_RECENT CUTOFF_SEASON <<< $(python - <<PY
import pandas as pd
from datetime import date
df=pd.read_csv('$STITCHED'); df['date']=pd.to_datetime(df['date'])
last=df['date'].max().date()
# Recent cutoff: 8 weeks before last date
cutoff_recent=(pd.Timestamp(last) - pd.Timedelta(weeks=8)).date()

def is_in_flu_season(d):
    # Flu season: Nov 1 - May 1 (inclusive)
    if d.month >= 11:
        return True
    if d.month < 5:
        return True
    if d.month == 5 and d.day <= 1:
        return True
    return False

def get_flu_season_year(d):
    # Season 2025 = Nov 2025 - May 2026
    return d.year if d.month >= 11 else d.year - 1

# Use the most recent COMPLETED season for residuals
current_season_year = get_flu_season_year(last)
if is_in_flu_season(last):
    target_season_year = current_season_year - 1
else:
    target_season_year = current_season_year

cutoff_season = date(target_season_year, 10, 1)
print(f"{last.isoformat()} {cutoff_recent.isoformat()} {cutoff_season.isoformat()}")
PY
)
echo "Data end date: $LAST_DATA_DATE"
echo "Cutoff (recent, for bounded LGBM): $CUTOFF_RECENT"
echo "Cutoff (season, for ARIMA/blended residuals): $CUTOFF_SEASON"

mkdir -p forecasts/retrospective/arima
mkdir -p forecasts/retrospective/lgbm_blended
mkdir -p forecasts/retrospective/lgbm_enhanced_t10_bounded
mkdir -p forecasts/retrospective/lgbm_enhanced_t10_bounded_wide_1
mkdir -p forecasts/retrospective/lgbm_enhanced_t10_bounded_wide_2
mkdir -p forecasts/retrospective/lgbm_enhanced_t10_bounded_wide_3
mkdir -p forecasts/retrospective/lgbm_enhanced_t10_bounded_wide_4
mkdir -p forecasts/retrospective/lgbm_enhanced_t10_bounded_wide_5
mkdir -p forecasts/retrospective/lgbm_t10_bounded_wide_4
mkdir -p forecasts/retrospective/svm_t100
mkdir -p forecasts/retrospective/joint_twostage_pool
mkdir -p forecasts/prospective

if is_true "$INCREMENTAL_RETROSPECTIVE"; then
  echo "==> Retrospective mode: INCREMENTAL (reuse cache + append only new anchors)"

  ARIMA_RETRO_CUTOFF=$(next_cutoff_from_reference_file "forecasts/retrospective/arima/ARIMA_h1_forecasts.csv" "$CUTOFF_SEASON")
  if [[ "$ARIMA_RETRO_CUTOFF" > "$LAST_DATA_DATE" ]]; then
    echo "==> Retrospective ARIMA: no new anchors (next cutoff $ARIMA_RETRO_CUTOFF > data end $LAST_DATA_DATE), reusing cache"
  else
    echo "==> Retrospective ARIMA incremental (cutoff=$ARIMA_RETRO_CUTOFF)"
    TMP_ARIMA_RETRO=$(mktemp -d "/tmp/retro_arima_${ASTS}_XXXXXX")
    python src/generate_retro_arima.py --data-file "$STITCHED" --cut-off "$ARIMA_RETRO_CUTOFF" \
      --output "$TMP_ARIMA_RETRO" --max-horizon 4
    merge_csv_tree "$TMP_ARIMA_RETRO" "forecasts/retrospective/arima"
  fi

  BLENDED_RETRO_CUTOFF=$(next_cutoff_from_reference_file "forecasts/retrospective/lgbm_blended/LGBM-blended_h1_forecasts.csv" "$CUTOFF_SEASON")
  if [[ "$BLENDED_RETRO_CUTOFF" > "$LAST_DATA_DATE" ]]; then
    echo "==> Retrospective Blended LGBM: no new anchors (next cutoff $BLENDED_RETRO_CUTOFF > data end $LAST_DATA_DATE), reusing cache"
  else
    echo "==> Retrospective Blended LGBM incremental (cutoff=$BLENDED_RETRO_CUTOFF)"
    TMP_BLENDED_RETRO=$(mktemp -d "/tmp/retro_blended_${ASTS}_XXXXXX")
    python src/generate_blended_lgbm.py \
      --data-file "$STITCHED" \
      --cut-off "$BLENDED_RETRO_CUTOFF" \
      --t10-models models/lgbm_enhanced_t10 \
      --t100-models models/lgbm_enhanced_t100 \
      --output "$TMP_BLENDED_RETRO" \
      --mode retrospective \
      --horizons 1,2,3,4
    merge_csv_tree "$TMP_BLENDED_RETRO" "forecasts/retrospective/lgbm_blended"
  fi

  BOUNDED_RETRO_CUTOFF=$(next_cutoff_from_reference_file "forecasts/retrospective/lgbm_enhanced_t10_bounded/TwoStage-FrozenMu_h1_forecasts.csv" "$CUTOFF_RECENT")
  if [[ "$BOUNDED_RETRO_CUTOFF" > "$LAST_DATA_DATE" ]]; then
    echo "==> Retrospective LGBM Bounded: no new anchors (next cutoff $BOUNDED_RETRO_CUTOFF > data end $LAST_DATA_DATE), reusing cache"
  else
    echo "==> Retrospective LGBM Bounded incremental (cutoff=$BOUNDED_RETRO_CUTOFF)"
    TMP_BOUNDED_RETRO_BASE=$(mktemp -d "/tmp/retro_lgbm_bounded_${ASTS}_XXXXXX")
    python src/generate_all_retro_lgbm.py --data-file "$STITCHED" --cut-off "$BOUNDED_RETRO_CUTOFF" \
      --models-dir models/lgbm_enhanced_t10_bounded --models-base-dir models/lgbm_enhanced_t10_bounded \
      --output-base "$TMP_BOUNDED_RETRO_BASE"
    merge_csv_tree "$TMP_BOUNDED_RETRO_BASE" "forecasts/retrospective"
  fi

  # Run retrospective for each bounded-wide variant that exists
  for V in 1 2 3 4 5; do
    if [[ -d "models/lgbm_enhanced_t10_bounded_wide_${V}" ]]; then
      MODEL_TAG="lgbm_enhanced_t10_bounded_wide_${V}"
      WIDE_RETRO_CUTOFF=$(next_cutoff_from_reference_file "forecasts/retrospective/${MODEL_TAG}/TwoStage-FrozenMu_h1_forecasts.csv" "$CUTOFF_RECENT")
      if [[ "$WIDE_RETRO_CUTOFF" > "$LAST_DATA_DATE" ]]; then
        echo "==> Retrospective LGBM Bounded Wide ${V}: no new anchors (next cutoff $WIDE_RETRO_CUTOFF > data end $LAST_DATA_DATE), reusing cache"
      else
        echo "==> Retrospective LGBM Bounded Wide ${V} incremental (cutoff=$WIDE_RETRO_CUTOFF)"
        TMP_WIDE_RETRO_BASE=$(mktemp -d "/tmp/retro_lgbm_wide_${V}_${ASTS}_XXXXXX")
        python src/generate_all_retro_lgbm.py --data-file "$STITCHED" --cut-off "$WIDE_RETRO_CUTOFF" \
          --models-dir "models/${MODEL_TAG}" --models-base-dir "models/${MODEL_TAG}" \
          --output-base "$TMP_WIDE_RETRO_BASE"
        merge_csv_tree "$TMP_WIDE_RETRO_BASE" "forecasts/retrospective"
      fi
    fi
  done

  # Run retrospective for non-enhanced bounded-wide-4 model (uses default state lag features)
  if [[ -d "models/lgbm_t10_bounded_wide_4" ]]; then
    NE_WIDE_RETRO_CUTOFF=$(next_cutoff_from_reference_file "forecasts/retrospective/lgbm_t10_bounded_wide_4/TwoStage-FrozenMu_h1_forecasts.csv" "$CUTOFF_RECENT")
    if [[ "$NE_WIDE_RETRO_CUTOFF" > "$LAST_DATA_DATE" ]]; then
      echo "==> Retrospective LGBM Bounded Wide 4 Non-Enhanced: no new anchors (next cutoff $NE_WIDE_RETRO_CUTOFF > data end $LAST_DATA_DATE), reusing cache"
    else
      echo "==> Retrospective LGBM Bounded Wide 4 Non-Enhanced incremental (cutoff=$NE_WIDE_RETRO_CUTOFF)"
      TMP_NE_WIDE_RETRO_BASE=$(mktemp -d "/tmp/retro_lgbm_wide4ne_${ASTS}_XXXXXX")
      python src/generate_all_retro_lgbm.py --data-file "$STITCHED" --cut-off "$NE_WIDE_RETRO_CUTOFF" \
        --models-dir "models/lgbm_t10_bounded_wide_4" --models-base-dir "models/lgbm_t10_bounded_wide_4" \
        --output-base "$TMP_NE_WIDE_RETRO_BASE"
      merge_csv_tree "$TMP_NE_WIDE_RETRO_BASE" "forecasts/retrospective"
    fi
  fi
else
  echo "==> Retrospective mode: FULL REBUILD (default)"

  echo "==> Retrospective ARIMA (using season cutoff for residuals)"
  python src/generate_retro_arima.py --data-file "$STITCHED" --cut-off "$CUTOFF_SEASON" \
    --output forecasts/retrospective/arima --max-horizon 4

  echo "==> Retrospective Blended LGBM (using season cutoff for residuals)"
  python src/generate_blended_lgbm.py \
    --data-file "$STITCHED" \
    --cut-off "$CUTOFF_SEASON" \
    --t10-models models/lgbm_enhanced_t10 \
    --t100-models models/lgbm_enhanced_t100 \
    --output forecasts/retrospective/lgbm_blended \
    --mode retrospective \
    --horizons 1,2,3,4

  echo "==> Retrospective LGBM Bounded (using recent cutoff)"
  python src/generate_all_retro_lgbm.py --data-file "$STITCHED" --cut-off "$CUTOFF_RECENT" \
    --models-dir models/lgbm_enhanced_t10_bounded --models-base-dir models/lgbm_enhanced_t10_bounded \
    --output-base forecasts/retrospective

  # Run retrospective for each bounded-wide variant that exists
  for V in 1 2 3 4 5; do
    if [[ -d "models/lgbm_enhanced_t10_bounded_wide_${V}" ]]; then
      echo "==> Retrospective LGBM Bounded Wide ${V} (using recent cutoff)"
      python src/generate_all_retro_lgbm.py --data-file "$STITCHED" --cut-off "$CUTOFF_RECENT" \
        --models-dir "models/lgbm_enhanced_t10_bounded_wide_${V}" --models-base-dir "models/lgbm_enhanced_t10_bounded_wide_${V}" \
        --output-base forecasts/retrospective
    fi
  done

  # Run retrospective for non-enhanced bounded-wide-4 model (uses default state lag features)
  if [[ -d "models/lgbm_t10_bounded_wide_4" ]]; then
    echo "==> Retrospective LGBM Bounded Wide 4 Non-Enhanced (using recent cutoff)"
    python src/generate_all_retro_lgbm.py --data-file "$STITCHED" --cut-off "$CUTOFF_RECENT" \
      --models-dir "models/lgbm_t10_bounded_wide_4" --models-base-dir "models/lgbm_t10_bounded_wide_4" \
      --output-base forecasts/retrospective
  fi
fi

# SKIPPED this week
# echo "==> Retrospective SVM (h=1..4)"
# for H in 1 2 3 4; do
#   python src/generate_retro_svm.py \
#     --hyperparams models/svm_t100/svm_hyperparameters_h${H}_t100.pkl \
#     --data-file "$STITCHED" \
#     --cut-off "$CUTOFF_SEASON" \
#     --output forecasts/retrospective/svm_t100 \
#     --max-weeks 0 || true
# done

echo "==> Prospective ARIMA (h=1..4)"
python src/generate_prosp_arima.py --data-file "$STITCHED" \
  --residuals-dir forecasts/retrospective/arima \
  --output forecasts/prospective

# SKIPPED this week
# echo "==> Prospective SVM (h=1..4)"
# python src/generate_prosp_svm.py --data-file "$STITCHED" --models models/svm_t100 --output forecasts/prospective

echo "==> Prospective Blended LGBM (0.5*t10 + 0.5*t100 -> plain conformal CIs)"
python src/generate_blended_lgbm.py \
  --data-file "$STITCHED" \
  --t10-models models/lgbm_enhanced_t10 \
  --t100-models models/lgbm_enhanced_t100 \
  --output forecasts/prospective \
  --mode prospective \
  --residuals-dir forecasts/retrospective/lgbm_blended \
  --horizons 1,2,3,4

echo "==> Prospective LGBM Bounded (h=1..4)"
for H in 1 2 3 4; do
  python src/generate_prosp_lgbm.py \
    --hyperparams models/lgbm_enhanced_t10_bounded/two_stage_hyperparameters_h${H}.pkl \
    --data-file "$STITCHED" \
    --horizon ${H} \
    --output forecasts/prospective \
    --model-name TwoStage-FrozenMu-bounded \
    --save-models \
    --models-output-dir models/lgbm_enhanced_t10_bounded || true
done

# Run prospective for each bounded-wide variant that exists
for V in 1 2 3 4 5; do
  if [[ -d "models/lgbm_enhanced_t10_bounded_wide_${V}" ]]; then
    echo "==> Prospective LGBM Bounded Wide ${V} (h=1..4)"
    for H in 1 2 3 4; do
      python src/generate_prosp_lgbm.py \
        --hyperparams "models/lgbm_enhanced_t10_bounded_wide_${V}/two_stage_hyperparameters_h${H}.pkl" \
        --data-file "$STITCHED" \
        --horizon ${H} \
        --output forecasts/prospective \
        --model-name "TwoStage-FrozenMu-bounded-wide-${V}" \
        --save-models \
        --models-output-dir "models/lgbm_enhanced_t10_bounded_wide_${V}" || true
    done
  fi
done

# Run prospective for non-enhanced bounded-wide-4 model (uses default state lag features)
if [[ -d "models/lgbm_t10_bounded_wide_4" ]]; then
  echo "==> Prospective LGBM Bounded Wide 4 Non-Enhanced (h=1..4)"
  for H in 1 2 3 4; do
    python src/generate_prosp_lgbm.py \
      --hyperparams "models/lgbm_t10_bounded_wide_4/two_stage_hyperparameters_h${H}.pkl" \
      --data-file "$STITCHED" \
      --horizon ${H} \
      --output forecasts/prospective \
      --model-name "TwoStage-FrozenMu-bounded-wide-4-ne" \
      --save-models \
      --models-output-dir "models/lgbm_t10_bounded_wide_4" || true
  done
fi

echo "==> Prospective Adaptive Ensemble"
# Align ensemble as-of date with the reference date used by prospective generators
PROSP_ASOF=$(python - <<PY
import pandas as pd
df=pd.read_csv('$STITCHED'); df['date']=pd.to_datetime(df['date'])
print(df['date'].max().date().isoformat())
PY
)
echo "Using prospective as-of date from stitched data: $PROSP_ASOF"
PROSP_ASOF_TS=$(echo "$PROSP_ASOF" | tr -d -)

JOINT_RETRO_FILE=""
JOINT_PROSP_FILE=""
if is_true "$ENSEMBLE_INCLUDE_JOINT_TWOSTAGE"; then
  # Minimum warm-start weeks needed so weighting has enough fully-observed history.
  # For horizon h, a reference date is scorable only when target_end_date <= as_of_date,
  # so horizon 4 requires an extra 3 weeks beyond the history window.
  DEFAULT_AE_HISTORY_WEEKS=$(extract_r_default_int "src/generate_prosp_adaptive_ensemble.R" "HISTORY_WEEKS" "6")
  EFFECTIVE_HISTORY_WEEKS="$DEFAULT_AE_HISTORY_WEEKS"
  if [[ -n "$ENSEMBLE_HISTORY_WEEKS" ]]; then
    EFFECTIVE_HISTORY_WEEKS="$ENSEMBLE_HISTORY_WEEKS"
  fi
  DEFAULT_META_HISTORY_WEEKS=0
  if is_true "$ENSEMBLE_INCLUDE_META"; then
    DEFAULT_META_HISTORY_WEEKS=$(extract_r_default_int "src/generate_prosp_adaptive_ensemble_meta.R" "META_HISTORY_WEEKS" "26")
  fi
  MAX_WEIGHT_HISTORY_WEEKS="$EFFECTIVE_HISTORY_WEEKS"
  if (( DEFAULT_META_HISTORY_WEEKS > MAX_WEIGHT_HISTORY_WEEKS )); then
    MAX_WEIGHT_HISTORY_WEEKS="$DEFAULT_META_HISTORY_WEEKS"
  fi
  REQUIRED_JOINT_WARMUP_WEEKS=$((MAX_WEIGHT_HISTORY_WEEKS + 3))

  if [[ -z "$JOINT_WARMUP_WEEKS" ]]; then
    JOINT_WARMUP_WEEKS="$REQUIRED_JOINT_WARMUP_WEEKS"
  elif (( JOINT_WARMUP_WEEKS < REQUIRED_JOINT_WARMUP_WEEKS )); then
    echo "Requested --joint-warmup-weeks=$JOINT_WARMUP_WEEKS is below required minimum ($REQUIRED_JOINT_WARMUP_WEEKS); using minimum."
    JOINT_WARMUP_WEEKS="$REQUIRED_JOINT_WARMUP_WEEKS"
  fi

  JOINT_WARMUP_START=$(python - <<PY
from datetime import date, timedelta
asof = date.fromisoformat("$PROSP_ASOF")
weeks = int("$JOINT_WARMUP_WEEKS")
print((asof - timedelta(weeks=weeks)).isoformat())
PY
)
  JOINT_RETRO_FILE="forecasts/retrospective/joint_twostage_pool/JointTwoStagePool_backtest_from_${JOINT_WARMUP_START}_to_${PROSP_ASOF}.csv"
  JOINT_PROSP_FILE="forecasts/prospective/JointTwoStagePool_prospective_${PROSP_ASOF_TS}.csv"

  echo "==> Joint Two-Stage Pooled Model (retrospective warm start)"
  echo "Joint warmup weeks: $JOINT_WARMUP_WEEKS (required minimum: $REQUIRED_JOINT_WARMUP_WEEKS; history=$EFFECTIVE_HISTORY_WEEKS, meta_history=$DEFAULT_META_HISTORY_WEEKS, max_horizon=4)"
  echo "Joint warmup start: $JOINT_WARMUP_START"
  echo "Joint retrospective generation is used only for ensemble weighting (not evaluation backtesting)."
  python joint_twostage_pool_test/joint_twostage_pool.py backtest \
    --data-file "$STITCHED" \
    --start-date "$JOINT_WARMUP_START" \
    --max-anchors "$JOINT_WARMUP_WEEKS" \
    --output "$JOINT_RETRO_FILE" \
    --max-horizons 4 \
    --num-bags 50 \
    --bag-frac 0.7 \
    --cov-top-k 5 \
    --cov-lags 1,2,3,4,8,12,52 \
    --target-mode delta_log \
    --include-partial-horizons \
    --sigma-mode wide

  echo "==> Joint Two-Stage Pooled Model (prospective)"
  python joint_twostage_pool_test/joint_twostage_pool.py prospective \
    --data-file "$STITCHED" \
    --output "$JOINT_PROSP_FILE" \
    --max-horizons 4 \
    --num-bags 50 \
    --bag-frac 0.7 \
    --cov-top-k 5 \
    --cov-lags 1,2,3,4,8,12,52 \
    --target-mode delta_log \
    --sigma-mode wide
fi

AE_ARGS=(--asof-date "$PROSP_ASOF")
if [[ -n "$ENSEMBLE_LOOKBACK_WEEKS" ]]; then AE_ARGS+=(--lookback-weeks "$ENSEMBLE_LOOKBACK_WEEKS"); fi
if [[ -n "$ENSEMBLE_HISTORY_WEEKS" ]]; then AE_ARGS+=(--history-weeks "$ENSEMBLE_HISTORY_WEEKS"); fi
if [[ -n "$ENSEMBLE_INCLUDE_ARIMA" ]]; then AE_ARGS+=(--include-arima "$ENSEMBLE_INCLUDE_ARIMA"); fi
if [[ -n "$ENSEMBLE_INCLUDE_SVM" ]]; then AE_ARGS+=(--include-svm "$ENSEMBLE_INCLUDE_SVM"); fi
if [[ -n "$ENSEMBLE_INCLUDE_LGBM_BLENDED" ]]; then AE_ARGS+=(--include-lgbm-blended "$ENSEMBLE_INCLUDE_LGBM_BLENDED"); fi
if [[ -n "$ENSEMBLE_INCLUDE_LGBM_BOUNDED" ]]; then AE_ARGS+=(--include-lgbm-bounded "$ENSEMBLE_INCLUDE_LGBM_BOUNDED"); fi
if [[ -n "$ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_1" ]]; then AE_ARGS+=(--include-lgbm-bounded-wide-1 "$ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_1"); fi
if [[ -n "$ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_2" ]]; then AE_ARGS+=(--include-lgbm-bounded-wide-2 "$ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_2"); fi
if [[ -n "$ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_3" ]]; then AE_ARGS+=(--include-lgbm-bounded-wide-3 "$ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_3"); fi
if [[ -n "$ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_4" ]]; then AE_ARGS+=(--include-lgbm-bounded-wide-4 "$ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_4"); fi
if [[ -n "$ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_4_NE" ]]; then AE_ARGS+=(--include-lgbm-bounded-wide-4-ne "$ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_4_NE"); fi
if [[ -n "$ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_5" ]]; then AE_ARGS+=(--include-lgbm-bounded-wide-5 "$ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_5"); fi
if [[ -n "$ENSEMBLE_INCLUDE_JOINT_TWOSTAGE" ]]; then AE_ARGS+=(--include-joint-twostage "$ENSEMBLE_INCLUDE_JOINT_TWOSTAGE"); fi
if [[ -n "$JOINT_RETRO_FILE" ]]; then AE_ARGS+=(--joint-retro-file "$JOINT_RETRO_FILE"); fi
if [[ -n "$JOINT_PROSP_FILE" ]]; then AE_ARGS+=(--joint-prosp-file "$JOINT_PROSP_FILE"); fi
AE_ARGS+=(--joint-reference-shift-days -7)
Rscript src/generate_prosp_adaptive_ensemble.R "${AE_ARGS[@]}"
Rscript src/generate_prosp_adaptive_ensemble_hedge.R "${AE_ARGS[@]}"
if is_true "$ENSEMBLE_INCLUDE_META"; then
  Rscript src/generate_prosp_adaptive_ensemble_meta.R "${AE_ARGS[@]}"
else
  echo "==> Skipping Adaptive Meta Ensemble (--include-meta-ensemble false)"
fi

echo "==> Done. Outputs under forecasts/{retrospective,prospective}"

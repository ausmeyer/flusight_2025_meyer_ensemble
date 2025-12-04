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

ENSEMBLE_LOOKBACK_WEEKS=""
ENSEMBLE_HISTORY_WEEKS=""
ENSEMBLE_INCLUDE_ARIMA=""
ENSEMBLE_INCLUDE_SVM=""
ENSEMBLE_INCLUDE_LGBM_BLENDED=""
ENSEMBLE_INCLUDE_LGBM_BOUNDED=""

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
    *)
      shift;;
  esac
done

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
# CUTOFF_SEASON: Before last flu season start (for ARIMA/blended LGBM residuals)
#                We need residuals from Nov 2024 - May 2025, so cut-off = Oct 1, 2024
read CUTOFF_RECENT CUTOFF_SEASON <<< $(python - <<PY
import pandas as pd
from datetime import date
df=pd.read_csv('$STITCHED'); df['date']=pd.to_datetime(df['date'])
last=df['date'].max()
# Recent cutoff: 8 weeks before last date
cutoff_recent=(last - pd.Timedelta(weeks=8)).date()
# Season cutoff: Oct 1 of the previous year to capture full Nov-May season
# If we're in Nov 2025, we want Oct 1, 2024 so we get Nov 2024 - May 2025
if last.month >= 11:
    season_year = last.year - 1
else:
    season_year = last.year - 1
cutoff_season = date(season_year, 10, 1)
print(f"{cutoff_recent.isoformat()} {cutoff_season.isoformat()}")
PY
)
echo "Cutoff (recent, for bounded LGBM): $CUTOFF_RECENT"
echo "Cutoff (season, for ARIMA/blended residuals): $CUTOFF_SEASON"

mkdir -p forecasts/retrospective/arima
mkdir -p forecasts/retrospective/lgbm_blended
mkdir -p forecasts/retrospective/lgbm_enhanced_t10_bounded
mkdir -p forecasts/retrospective/svm_t100
mkdir -p forecasts/prospective

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

# SVM disabled - not tracking trend well
# echo "==> Retrospective SVM (h=1..4)"
# for H in 1 2 3 4; do
#   python src/generate_retro_svm.py \
#     --hyperparams models/svm_t100/svm_hyperparameters_h${H}_t100.pkl \
#     --data-file "$STITCHED" \
#     --cut-off "$CUTOFF" \
#     --output forecasts/retrospective/svm_t100 \
#     --max-weeks 0 || true
# done

echo "==> Prospective ARIMA (h=1..4)"
python src/generate_prosp_arima.py --data-file "$STITCHED" \
  --residuals-dir forecasts/retrospective/arima \
  --output forecasts/prospective

# SVM disabled - not tracking trend well
# echo "==> Prospective SVM (h=1..4)"
# python src/generate_prosp_svm.py --data-file "$STITCHED" --models models/svm_t100 --output forecasts/prospective

echo "==> Prospective Blended LGBM (t10+t100+persistence -> conformal CIs)"
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

echo "==> Prospective Adaptive Ensemble"
# Align ensemble as-of date with the reference date used by prospective generators
PROSP_ASOF=$(python - <<PY
import pandas as pd
df=pd.read_csv('$STITCHED'); df['date']=pd.to_datetime(df['date'])
print(df['date'].max().date().isoformat())
PY
)
echo "Using prospective as-of date from stitched data: $PROSP_ASOF"
AE_ARGS=(--asof-date "$PROSP_ASOF")
if [[ -n "$ENSEMBLE_LOOKBACK_WEEKS" ]]; then AE_ARGS+=(--lookback-weeks "$ENSEMBLE_LOOKBACK_WEEKS"); fi
if [[ -n "$ENSEMBLE_HISTORY_WEEKS" ]]; then AE_ARGS+=(--history-weeks "$ENSEMBLE_HISTORY_WEEKS"); fi
if [[ -n "$ENSEMBLE_INCLUDE_ARIMA" ]]; then AE_ARGS+=(--include-arima "$ENSEMBLE_INCLUDE_ARIMA"); fi
if [[ -n "$ENSEMBLE_INCLUDE_SVM" ]]; then AE_ARGS+=(--include-svm "$ENSEMBLE_INCLUDE_SVM"); fi
if [[ -n "$ENSEMBLE_INCLUDE_LGBM_BLENDED" ]]; then AE_ARGS+=(--include-lgbm-blended "$ENSEMBLE_INCLUDE_LGBM_BLENDED"); fi
if [[ -n "$ENSEMBLE_INCLUDE_LGBM_BOUNDED" ]]; then AE_ARGS+=(--include-lgbm-bounded "$ENSEMBLE_INCLUDE_LGBM_BOUNDED"); fi
Rscript src/generate_prosp_adaptive_ensemble.R "${AE_ARGS[@]}"

echo "==> Done. Outputs under forecasts/{retrospective,prospective}"

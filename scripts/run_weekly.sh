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
# --lookback <N>         Number of reference weeks for weighting (default 4)
# --history <N>          Number of retrospective weeks to consider (default 8)
# --include-arima <t/f>  Include ARIMA in ensemble (default true)
# --include-svm <t/f>    Include SVM in ensemble (default true)
# --include-lgbm <t/f>   Include LGBM in ensemble (default true)

ENSEMBLE_LOOKBACK_WEEKS=""
ENSEMBLE_HISTORY_WEEKS=""
ENSEMBLE_INCLUDE_ARIMA=""
ENSEMBLE_INCLUDE_SVM=""
ENSEMBLE_INCLUDE_LGBM=""

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
    --include-lgbm)
      ENSEMBLE_INCLUDE_LGBM="$2"; shift 2;;
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

echo "==> Compute cutoff (last_date - 8 weeks)"
CUTOFF=$(python - <<PY
import pandas as pd, sys
df=pd.read_csv('$STITCHED'); df['date']=pd.to_datetime(df['date'])
last=df['date'].max()
cut=(last - pd.Timedelta(weeks=8)).date()
print(cut.isoformat())
PY
)
echo "Cutoff: $CUTOFF"

mkdir -p forecasts/retrospective/arima
mkdir -p forecasts/retrospective/lgbm_enhanced_t10
mkdir -p forecasts/prospective

echo "==> Retrospective ARIMA"
python src/generate_retro_arima.py --data-file "$STITCHED" --cut-off "$CUTOFF" \
  --output forecasts/retrospective/arima --max-horizon 4

echo "==> Retrospective LGBM (using hyperparams in models/lgbm_enhanced_t10)"
python src/generate_all_retro_lgbm.py --data-file "$STITCHED" --cut-off "$CUTOFF" \
  --models-dir models/lgbm_enhanced_t10 --models-base-dir models/lgbm_enhanced_t10 \
  --output-base forecasts/retrospective

echo "==> Retrospective SVM (h=1..4)"
for H in 1 2 3 4; do
  python src/generate_retro_svm.py \
    --hyperparams models/svm_t100/svm_hyperparameters_h${H}_t100.pkl \
    --data-file "$STITCHED" \
    --cut-off "$CUTOFF" \
    --output forecasts/retrospective/svm_t100 \
    --max-weeks 0 || true
done

echo "==> Prospective ARIMA (h=1..4)"
python src/generate_prosp_arima.py --data-file "$STITCHED" --output forecasts/prospective

echo "==> Prospective SVM (h=1..4)"
python src/generate_prosp_svm.py --data-file "$STITCHED" --models models/svm_t100 --output forecasts/prospective

echo "==> Prospective LGBM (h=1..4; also save models under models/lgbm_enhanced_t10)"
for H in 1 2 3 4; do
  python src/generate_prosp_lgbm.py \
    --hyperparams models/lgbm_enhanced_t10/two_stage_hyperparameters_h${H}.pkl \
    --data-file "$STITCHED" \
    --horizon ${H} \
    --output forecasts/prospective \
    --model-name TwoStage-FrozenMu \
    --save-models \
    --models-output-dir models/lgbm_enhanced_t10 || true
done

echo "==> Prospective Adaptive Ensemble"
AE_ARGS=()
if [[ -n "$ENSEMBLE_LOOKBACK_WEEKS" ]]; then AE_ARGS+=(--lookback-weeks "$ENSEMBLE_LOOKBACK_WEEKS"); fi
if [[ -n "$ENSEMBLE_HISTORY_WEEKS" ]]; then AE_ARGS+=(--history-weeks "$ENSEMBLE_HISTORY_WEEKS"); fi
if [[ -n "$ENSEMBLE_INCLUDE_ARIMA" ]]; then AE_ARGS+=(--include-arima "$ENSEMBLE_INCLUDE_ARIMA"); fi
if [[ -n "$ENSEMBLE_INCLUDE_SVM" ]]; then AE_ARGS+=(--include-svm "$ENSEMBLE_INCLUDE_SVM"); fi
if [[ -n "$ENSEMBLE_INCLUDE_LGBM" ]]; then AE_ARGS+=(--include-lgbm "$ENSEMBLE_INCLUDE_LGBM"); fi
if (( ${#AE_ARGS[@]} )); then
  Rscript src/generate_prosp_adaptive_ensemble.R "${AE_ARGS[@]}"
else
  Rscript src/generate_prosp_adaptive_ensemble.R
fi

echo "==> Done. Outputs under forecasts/{retrospective,prospective}"

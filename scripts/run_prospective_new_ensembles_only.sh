#!/usr/bin/env bash
set -euo pipefail

# Generate the current, hedge, and meta prospective ensembles only.

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

ENSEMBLE_LOOKBACK_WEEKS=""
ENSEMBLE_HISTORY_WEEKS=""
ENSEMBLE_INCLUDE_ARIMA=""
ENSEMBLE_INCLUDE_SVM=""
ENSEMBLE_INCLUDE_LGBM_BLENDED=""
ENSEMBLE_INCLUDE_LGBM_BOUNDED=""
ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_1=""
ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_2=""
ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_3=""
ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_4=""
ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_5=""

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
    --include-lgbm-bounded-wide-5)
      ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_5="$2"; shift 2;;
    *)
      shift;;
  esac
done

STITCHED=$(ls -1 data/imputed_sets/imputed_and_stitched_hosp_*.csv | sort | tail -n 1)
if [[ -z "$STITCHED" ]]; then
  echo "ERROR: No stitched data file found in data/imputed_sets/"
  exit 1
fi
echo "Using stitched file: $STITCHED"

PROSP_ASOF=$(python - <<PY
import pandas as pd
df=pd.read_csv("$STITCHED"); df['date']=pd.to_datetime(df['date'])
print(df['date'].max().date().isoformat())
PY
)
echo "Prospective as-of date: $PROSP_ASOF"

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
if [[ -n "$ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_5" ]]; then AE_ARGS+=(--include-lgbm-bounded-wide-5 "$ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_5"); fi

echo "==> Adaptive Ensemble (Current)"
Rscript src/generate_prosp_adaptive_ensemble.R "${AE_ARGS[@]}"

echo "==> Adaptive Ensemble (Hedge)"
Rscript src/generate_prosp_adaptive_ensemble_hedge.R "${AE_ARGS[@]}"

echo "==> Adaptive Ensemble (Meta)"
Rscript src/generate_prosp_adaptive_ensemble_meta.R "${AE_ARGS[@]}"

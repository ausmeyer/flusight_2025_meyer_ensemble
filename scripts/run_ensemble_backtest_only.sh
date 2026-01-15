#!/usr/bin/env bash
set -euo pipefail

# Retrospective Experiment Runner (Ensemble Backtest Loop Only)
# - Skips base model generation
# - Simulates weekly prospective ensemble generation ("Backtest")
# - Outputs to forecasts/retrospective/ensemble/

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

# Optional CLI arguments for ensemble control
ENSEMBLE_LOOKBACK_WEEKS=""
ENSEMBLE_HISTORY_WEEKS=""
ENSEMBLE_INCLUDE_ARIMA=""
ENSEMBLE_INCLUDE_SVM=""
ENSEMBLE_INCLUDE_LGBM=""
ENSEMBLE_INCLUDE_LGBM_BLENDED=""
ENSEMBLE_INCLUDE_LGBM_BOUNDED=""
ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_1=""
ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_2=""
ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_3=""
ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_4=""
ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_4_NE=""
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
    --include-lgbm)
      ENSEMBLE_INCLUDE_LGBM="$2"; shift 2;;
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
    *)
      shift;;
  esac
done

# Fixed Retrospective Start Date
CUTOFF="2024-10-01"
echo "Retrospective Start Date (Cutoff): $CUTOFF"

# Create output directories if they don't exist
mkdir -p forecasts/retrospective/ensemble
mkdir -p forecasts/prospective

echo "==> Running Ensemble Backtest Loop (Skipping Base Model Generation)"

# Prepare Ensemble Arguments
AE_ARGS=()
if [[ -n "$ENSEMBLE_LOOKBACK_WEEKS" ]]; then AE_ARGS+=(--lookback-weeks "$ENSEMBLE_LOOKBACK_WEEKS"); fi
if [[ -n "$ENSEMBLE_HISTORY_WEEKS" ]]; then AE_ARGS+=(--history-weeks "$ENSEMBLE_HISTORY_WEEKS"); fi
if [[ -n "$ENSEMBLE_INCLUDE_ARIMA" ]]; then AE_ARGS+=(--include-arima "$ENSEMBLE_INCLUDE_ARIMA"); fi
if [[ -n "$ENSEMBLE_INCLUDE_SVM" ]]; then AE_ARGS+=(--include-svm "$ENSEMBLE_INCLUDE_SVM"); fi
if [[ -n "$ENSEMBLE_INCLUDE_LGBM" ]]; then AE_ARGS+=(--include-lgbm "$ENSEMBLE_INCLUDE_LGBM"); fi
if [[ -n "$ENSEMBLE_INCLUDE_LGBM_BLENDED" ]]; then AE_ARGS+=(--include-lgbm-blended "$ENSEMBLE_INCLUDE_LGBM_BLENDED"); fi
if [[ -n "$ENSEMBLE_INCLUDE_LGBM_BOUNDED" ]]; then AE_ARGS+=(--include-lgbm-bounded "$ENSEMBLE_INCLUDE_LGBM_BOUNDED"); fi
if [[ -n "$ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_1" ]]; then AE_ARGS+=(--include-lgbm-bounded-wide-1 "$ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_1"); fi
if [[ -n "$ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_2" ]]; then AE_ARGS+=(--include-lgbm-bounded-wide-2 "$ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_2"); fi
if [[ -n "$ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_3" ]]; then AE_ARGS+=(--include-lgbm-bounded-wide-3 "$ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_3"); fi
if [[ -n "$ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_4" ]]; then AE_ARGS+=(--include-lgbm-bounded-wide-4 "$ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_4"); fi
if [[ -n "$ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_4_NE" ]]; then AE_ARGS+=(--include-lgbm-bounded-wide-4-ne "$ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_4_NE"); fi
if [[ -n "$ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_5" ]]; then AE_ARGS+=(--include-lgbm-bounded-wide-5 "$ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE_5"); fi

# Generate list of As-Of dates (Saturdays) from Cutoff to now
DATES=$(python - <<PY
import pandas as pd
from datetime import datetime
start = pd.to_datetime('$CUTOFF')
# Find first Saturday on or after start
offset = (5 - start.weekday()) % 7
first_sat = start + pd.Timedelta(days=offset)
now = pd.Timestamp.now()
dates = pd.date_range(first_sat, now, freq='W-SAT')
# Print space-separated ISO dates
print(' '.join(d.strftime('%Y-%m-%d') for d in dates))
PY
)

for ASOF in $DATES; do
    TS=$(echo "$ASOF" | tr -d -)
    echo "----------------------------------------------------------------"
    echo "Simulating Prospective Ensemble for As-Of: $ASOF"

    # 1. Slice Retro files to create Temp Prospective files
    python - <<PY
import pandas as pd
import os
import sys

asof = '$ASOF'
ts = '$TS'
h_list = [1, 2, 3, 4]

tasks = []
for h in h_list:
    tasks.append({
        'src': f'forecasts/retrospective/arima/ARIMA_h{h}_forecasts.csv',
        'dst': f'forecasts/prospective/ARIMA_h{h}_prospective_{ts}.csv'
    })
    tasks.append({
        'src': f'forecasts/retrospective/svm_t100/svm_retrospective_h{h}.csv',
        'dst': f'forecasts/prospective/SVM_h{h}_prospective_{ts}.csv'
    })
    tasks.append({
        'src': f'forecasts/retrospective/lgbm_enhanced_t100/TwoStage-FrozenMu_h{h}_forecasts.csv',
        'dst': f'forecasts/prospective/TwoStage-FrozenMu-t100_h{h}_prospective_{ts}.csv'
    })
    tasks.append({
        'src': f'forecasts/retrospective/lgbm_enhanced_t10/TwoStage-FrozenMu_h{h}_forecasts.csv',
        'dst': f'forecasts/prospective/TwoStage-FrozenMu-t10_h{h}_prospective_{ts}.csv'
    })
    tasks.append({
        'src': f'forecasts/retrospective/lgbm_enhanced_t10_blended/TwoStage-FrozenMu_h{h}_forecasts.csv',
        'dst': f'forecasts/prospective/TwoStage-FrozenMu-blended_h{h}_prospective_{ts}.csv'
    })
    tasks.append({
        'src': f'forecasts/retrospective/lgbm_enhanced_t10_bounded/TwoStage-FrozenMu_h{h}_forecasts.csv',
        'dst': f'forecasts/prospective/TwoStage-FrozenMu-bounded_h{h}_prospective_{ts}.csv'
    })
    tasks.append({
        'src': f'forecasts/retrospective/lgbm_enhanced_t10_bounded_wide_1/TwoStage-FrozenMu_h{h}_forecasts.csv',
        'dst': f'forecasts/prospective/TwoStage-FrozenMu-bounded-wide-1_h{h}_prospective_{ts}.csv'
    })
    tasks.append({
        'src': f'forecasts/retrospective/lgbm_enhanced_t10_bounded_wide_2/TwoStage-FrozenMu_h{h}_forecasts.csv',
        'dst': f'forecasts/prospective/TwoStage-FrozenMu-bounded-wide-2_h{h}_prospective_{ts}.csv'
    })
    tasks.append({
        'src': f'forecasts/retrospective/lgbm_enhanced_t10_bounded_wide_3/TwoStage-FrozenMu_h{h}_forecasts.csv',
        'dst': f'forecasts/prospective/TwoStage-FrozenMu-bounded-wide-3_h{h}_prospective_{ts}.csv'
    })
    tasks.append({
        'src': f'forecasts/retrospective/lgbm_enhanced_t10_bounded_wide_4/TwoStage-FrozenMu_h{h}_forecasts.csv',
        'dst': f'forecasts/prospective/TwoStage-FrozenMu-bounded-wide-4_h{h}_prospective_{ts}.csv'
    })
    # Non-enhanced bounded-wide-4 model (uses default state lag features)
    tasks.append({
        'src': f'forecasts/retrospective/lgbm_t10_bounded_wide_4/TwoStage-FrozenMu_h{h}_forecasts.csv',
        'dst': f'forecasts/prospective/TwoStage-FrozenMu-bounded-wide-4-ne_h{h}_prospective_{ts}.csv'
    })
    tasks.append({
        'src': f'forecasts/retrospective/lgbm_enhanced_t10_bounded_wide_5/TwoStage-FrozenMu_h{h}_forecasts.csv',
        'dst': f'forecasts/prospective/TwoStage-FrozenMu-bounded-wide-5_h{h}_prospective_{ts}.csv'
    })

files_created = 0
for task in tasks:
    if os.path.exists(task['src']):
        try:
            df = pd.read_csv(task['src'], dtype={'location': str})
            subset = df[df['reference_date'] == asof]
            if not subset.empty:
                subset.to_csv(task['dst'], index=False)
                files_created += 1
        except Exception as e:
            print(f"Error processing {task['src']}: {e}")

if files_created == 0:
    print("Warning: No prospective files created for this date (maybe no retro forecasts available?).")
PY

    # 2. Run Ensemble Script
    Rscript src/generate_prosp_adaptive_ensemble.R --asof-date "$ASOF" ${AE_ARGS[@]+"${AE_ARGS[@]}"}

    # 3. Move output to safe place
    REF_DATE_TS=$(python -c "from datetime import datetime, timedelta; d = datetime.strptime('$ASOF', '%Y-%m-%d'); print((d + timedelta(days=7)).strftime('%Y%m%d'))")

    SRC_ENS="forecasts/prospective/AdaptiveEnsemble_prospective_${REF_DATE_TS}.csv"
    DST_ENS="forecasts/retrospective/ensemble/AdaptiveEnsemble_retrospective_${REF_DATE_TS}.csv"

    if [[ -f "$SRC_ENS" ]]; then
        mv "$SRC_ENS" "$DST_ENS"
        echo "  -> Saved Ensemble: $DST_ENS"
    else
        echo "  -> Warning: No ensemble output produced for $ASOF"
    fi

    # 4. Cleanup Temp Prospective Files - DISABLED to preserve historical files
    # rm -f forecasts/prospective/*_prospective_${TS}.csv

done

echo "==> Done. Ensemble Backtest Complete."

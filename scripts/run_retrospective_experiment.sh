#!/usr/bin/env bash
set -euo pipefail

# Retrospective Experiment Runner
# - Renders stitch.Rmd (to ensure latest data)
# - Generates retrospective base model forecasts from fixed start date (2024-10-01) to present
# - Simulates weekly prospective ensemble generation ("Backtest")
# - Outputs to forecasts/retrospective/ensemble/

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

# Optional CLI arguments for ensemble control
ENSEMBLE_LOOKBACK_WEEKS=""
ENSEMBLE_HISTORY_WEEKS=""
ENSEMBLE_INCLUDE_ARIMA=""
ENSEMBLE_INCLUDE_SVM=""
ENSEMBLE_INCLUDE_LGBM_BLENDED=""
ENSEMBLE_INCLUDE_LGBM_BOUNDED=""
ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE=""

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
    --include-lgbm-bounded-wide)
      ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE="$2"; shift 2;;
    *)
      shift;;
  esac
done

echo "==> Render stitch.Rmd"
Rscript -e "rmarkdown::render('src/stitch.Rmd', output_dir='src', output_file='stitch.html')"

echo "==> Locate latest stitched file"
STITCHED=$(ls -1 data/imputed_sets/imputed_and_stitched_hosp_*.csv | sort | tail -n 1)
echo "Stitched: $STITCHED"

# Fixed Retrospective Start Date
CUTOFF="2024-07-01"
echo "Retrospective Start Date (Cutoff): $CUTOFF"

# Create output directories
mkdir -p forecasts/retrospective/arima
mkdir -p forecasts/retrospective/lgbm_enhanced_t100
mkdir -p forecasts/retrospective/lgbm_enhanced_t10
mkdir -p forecasts/retrospective/lgbm_enhanced_t10_bounded
mkdir -p forecasts/retrospective/lgbm_enhanced_t10_bounded_wide
mkdir -p forecasts/retrospective/svm_t100
mkdir -p forecasts/retrospective/ensemble
mkdir -p forecasts/prospective # Ensure exists for temp files

echo "==> Generating Base Model Retrospective Forecasts ($CUTOFF to present)"

echo "--> ARIMA"
python src/generate_retro_arima.py --data-file "$STITCHED" --cut-off "$CUTOFF" \
  --output forecasts/retrospective/arima --max-horizon 4

echo "--> LGBM (t100)"
python src/generate_all_retro_lgbm.py --data-file "$STITCHED" --cut-off "$CUTOFF" \
  --models-dir models/lgbm_enhanced_t100 --models-base-dir models/lgbm_enhanced_t100 \
  --output-base forecasts/retrospective

echo "--> LGBM (t10)"
python src/generate_all_retro_lgbm.py --data-file "$STITCHED" --cut-off "$CUTOFF" \
  --models-dir models/lgbm_enhanced_t10 --models-base-dir models/lgbm_enhanced_t10 \
  --output-base forecasts/retrospective

echo "--> LGBM (t10 bounded)"
python src/generate_all_retro_lgbm.py --data-file "$STITCHED" --cut-off "$CUTOFF" \
  --models-dir models/lgbm_enhanced_t10_bounded --models-base-dir models/lgbm_enhanced_t10_bounded \
  --output-base forecasts/retrospective

echo "--> LGBM (t10 bounded wide)"
python src/generate_all_retro_lgbm.py --data-file "$STITCHED" --cut-off "$CUTOFF" \
  --models-dir models/lgbm_enhanced_t10_bounded_wide --models-base-dir models/lgbm_enhanced_t10_bounded_wide \
  --output-base forecasts/retrospective

echo "--> SVM"
for H in 1 2 3 4; do
  python src/generate_retro_svm.py \
    --hyperparams models/svm_t100/svm_hyperparameters_h${H}_t100.pkl \
    --data-file "$STITCHED" \
    --cut-off "$CUTOFF" \
    --output forecasts/retrospective/svm_t100 \
    --max-weeks 0 || true
done

echo "==> Running Ensemble Backtest Loop"

# Prepare Ensemble Arguments
AE_ARGS=()
if [[ -n "$ENSEMBLE_LOOKBACK_WEEKS" ]]; then AE_ARGS+=(--lookback-weeks "$ENSEMBLE_LOOKBACK_WEEKS"); fi
if [[ -n "$ENSEMBLE_HISTORY_WEEKS" ]]; then AE_ARGS+=(--history-weeks "$ENSEMBLE_HISTORY_WEEKS"); fi
if [[ -n "$ENSEMBLE_INCLUDE_ARIMA" ]]; then AE_ARGS+=(--include-arima "$ENSEMBLE_INCLUDE_ARIMA"); fi
if [[ -n "$ENSEMBLE_INCLUDE_SVM" ]]; then AE_ARGS+=(--include-svm "$ENSEMBLE_INCLUDE_SVM"); fi
if [[ -n "$ENSEMBLE_INCLUDE_LGBM_BLENDED" ]]; then AE_ARGS+=(--include-lgbm-blended "$ENSEMBLE_INCLUDE_LGBM_BLENDED"); fi
if [[ -n "$ENSEMBLE_INCLUDE_LGBM_BOUNDED" ]]; then AE_ARGS+=(--include-lgbm-bounded "$ENSEMBLE_INCLUDE_LGBM_BOUNDED"); fi
if [[ -n "$ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE" ]]; then AE_ARGS+=(--include-lgbm-bounded-wide "$ENSEMBLE_INCLUDE_LGBM_BOUNDED_WIDE"); fi

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
    # Map (Source Retro File) -> (Dest Temp Prospective File)
    # Note: SVM retro file naming: svm_retrospective_h{h}.csv
    # LGBM retro file naming: TwoStage-FrozenMu_h{h}_forecasts.csv (inside model folder)
    # ARIMA retro file naming: ARIMA_h{h}_forecasts.csv
    
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
        'src': f'forecasts/retrospective/lgbm_enhanced_t10_bounded/TwoStage-FrozenMu_h{h}_forecasts.csv',
        'dst': f'forecasts/prospective/TwoStage-FrozenMu-bounded_h{h}_prospective_{ts}.csv'
    })
    tasks.append({
        'src': f'forecasts/retrospective/lgbm_enhanced_t10_bounded_wide/TwoStage-FrozenMu_h{h}_forecasts.csv',
        'dst': f'forecasts/prospective/TwoStage-FrozenMu-bounded-wide_h{h}_prospective_{ts}.csv'
    })

files_created = 0
for task in tasks:
    if os.path.exists(task['src']):
        try:
            df = pd.read_csv(task['src'], dtype={'location': str})
            # Filter for reference_date == asof
            # Note: Ensure date string format matches. 
            # The retro files usually have YYYY-MM-DD.
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
    # The script will use the temp files we just created in forecasts/prospective
    Rscript src/generate_prosp_adaptive_ensemble.R --asof-date "$ASOF" ${AE_ARGS[@]+"${AE_ARGS[@]}"}

    # 3. Move output to safe place
    # Output file is typically forecasts/prospective/AdaptiveEnsemble_prospective_{RefDateTS}.csv
    # RefDate = AsOf + 7 days
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

echo "==> Done. Retrospective Experiment Complete."
echo "Ensemble outputs are in forecasts/retrospective/ensemble/"

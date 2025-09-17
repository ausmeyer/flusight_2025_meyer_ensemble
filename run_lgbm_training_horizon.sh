#!/bin/bash

set -euo pipefail

# Usage: ./run_horizon.sh <HORIZON>
# Example (16 vCPU, balanced): OPTUNA_JOBS=2 OMP_NUM_THREADS=4 LGBM_NUM_THREADS=4 ./run_horizon.sh 1

if [ $# -lt 1 ]; then
  echo "Usage: $0 <HORIZON>"
  exit 1
fi

HORIZON="$1"
shift

# Forward any additional args to the Python script (e.g., --no-stage2-pruning)
EXTRA_ARGS=("$@")

# Threading environment defaults; override by exporting before calling
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

# LightGBM threads per process
export LGBM_NUM_THREADS=${LGBM_NUM_THREADS:-16}
export OPTUNA_JOBS=${OPTUNA_JOBS:-1}

# Optional: suppress LightGBM split warnings (default on)
# Note: No output filtering; run directly so Optuna progress renders cleanly

# Default locations array; preserve spaces in multi-word names
DEFAULT_LOCATIONS_ARRAY=(
  Alabama Alaska Arizona Arkansas California Colorado Connecticut Delaware
  "District of Columbia" Florida Georgia Hawaii Idaho Illinois Indiana Iowa Kansas Kentucky
  Louisiana Maine Maryland Massachusetts Michigan Minnesota Mississippi Missouri Montana Nebraska Nevada
  "New Hampshire" "New Jersey" "New Mexico" "New York" "North Carolina" "North Dakota"
  Ohio Oklahoma Oregon Pennsylvania "Puerto Rico" "Rhode Island" "South Carolina" "South Dakota"
  Tennessee Texas US Utah Vermont Virginia Washington "West Virginia" Wisconsin Wyoming
)

# Allow override via LOCATIONS env as a comma-separated list
if [[ -n "${LOCATIONS:-}" ]]; then
  IFS=',' read -r -a LOCATIONS_ARRAY <<< "${LOCATIONS}"
else
  LOCATIONS_ARRAY=("${DEFAULT_LOCATIONS_ARRAY[@]}")
fi

python src/train_two_stage.py \
  --data-file data/imputed_and_stitched_hosp_2025-05-24.csv \
  --cut-off 2024-07-01 \
  --locations "${LOCATIONS_ARRAY[@]}" \
  --horizon "${HORIZON}" \
  --use-enhanced-features \
  --trials-stage1 100 \
  --trials-stage2 100 \
  --n-features 10 \
  --random-seed 1 \
  --num-threads "${LGBM_NUM_THREADS}" \
  --optuna-jobs "${OPTUNA_JOBS}" \
  "${EXTRA_ARGS[@]}"

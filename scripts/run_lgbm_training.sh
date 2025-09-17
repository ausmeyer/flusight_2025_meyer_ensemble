#!/bin/bash

# Ensure we run from the repository root regardless of invocation path
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

set -euo pipefail

# Run two-stage forecasting pipeline on all locations
# with 100 stage1 trials and 100 stage2 trials

# Threading environment to maximize CPU utilization by LightGBM without BLAS oversubscription
# You can override these before calling this script if needed (e.g., OMP_NUM_THREADS=8 for parallel Optuna jobs)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

# Optional: suppress LightGBM split warnings (default on)
# Note: No output filtering; run directly so Optuna progress renders cleanly

python src/train_two_stage.py \
    --data-file data/imputed_and_stitched_hosp_2025-05-24.csv \
    --cut-off 2024-07-01 \
    --locations Alabama Alaska Arizona Arkansas California Colorado Connecticut Delaware "District of Columbia" Florida Georgia Hawaii Idaho Illinois Indiana Iowa Kansas Kentucky Louisiana Maine Maryland Massachusetts Michigan Minnesota Mississippi Missouri Montana Nebraska Nevada "New Hampshire" "New Jersey" "New Mexico" "New York" "North Carolina" "North Dakota" Ohio Oklahoma Oregon Pennsylvania "Puerto Rico" "Rhode Island" "South Carolina" "South Dakota" Tennessee Texas US Utah Vermont Virginia Washington "West Virginia" Wisconsin Wyoming \
    --horizon 1 \
    --use-enhanced-features \
    --trials-stage1 100 \
    --trials-stage2 50 \
    --n-features 10 \
    --random-seed 1 \
    --num-threads ${LGBM_NUM_THREADS:-16} \
    --optuna-jobs ${OPTUNA_JOBS:-1} \
    "$@"
    
python src/train_two_stage.py \
    --data-file data/imputed_and_stitched_hosp_2025-05-24.csv \
    --cut-off 2024-07-01 \
    --locations Alabama Alaska Arizona Arkansas California Colorado Connecticut Delaware "District of Columbia" Florida Georgia Hawaii Idaho Illinois Indiana Iowa Kansas Kentucky Louisiana Maine Maryland Massachusetts Michigan Minnesota Mississippi Missouri Montana Nebraska Nevada "New Hampshire" "New Jersey" "New Mexico" "New York" "North Carolina" "North Dakota" Ohio Oklahoma Oregon Pennsylvania "Puerto Rico" "Rhode Island" "South Carolina" "South Dakota" Tennessee Texas US Utah Vermont Virginia Washington "West Virginia" Wisconsin Wyoming \
    --horizon 2 \
    --use-enhanced-features \
    --trials-stage1 100 \
    --trials-stage2 50 \
    --n-features 10 \
    --random-seed 1 \
    --num-threads ${LGBM_NUM_THREADS:-16} \
    --optuna-jobs ${OPTUNA_JOBS:-1} \
    "$@"
 
python src/train_two_stage.py \
    --data-file data/imputed_and_stitched_hosp_2025-05-24.csv \
    --cut-off 2024-07-01 \
    --locations Alabama Alaska Arizona Arkansas California Colorado Connecticut Delaware "District of Columbia" Florida Georgia Hawaii Idaho Illinois Indiana Iowa Kansas Kentucky Louisiana Maine Maryland Massachusetts Michigan Minnesota Mississippi Missouri Montana Nebraska Nevada "New Hampshire" "New Jersey" "New Mexico" "New York" "North Carolina" "North Dakota" Ohio Oklahoma Oregon Pennsylvania "Puerto Rico" "Rhode Island" "South Carolina" "South Dakota" Tennessee Texas US Utah Vermont Virginia Washington "West Virginia" Wisconsin Wyoming \
    --horizon 3 \
    --use-enhanced-features \
    --trials-stage1 100 \
    --trials-stage2 50 \
    --n-features 10 \
    --random-seed 1 \
    --num-threads ${LGBM_NUM_THREADS:-16} \
    --optuna-jobs ${OPTUNA_JOBS:-1} \
    "$@"
    
python src/train_two_stage.py \
    --data-file data/imputed_and_stitched_hosp_2025-05-24.csv \
    --cut-off 2024-07-01 \
    --locations Alabama Alaska Arizona Arkansas California Colorado Connecticut Delaware "District of Columbia" Florida Georgia Hawaii Idaho Illinois Indiana Iowa Kansas Kentucky Louisiana Maine Maryland Massachusetts Michigan Minnesota Mississippi Missouri Montana Nebraska Nevada "New Hampshire" "New Jersey" "New Mexico" "New York" "North Carolina" "North Dakota" Ohio Oklahoma Oregon Pennsylvania "Puerto Rico" "Rhode Island" "South Carolina" "South Dakota" Tennessee Texas US Utah Vermont Virginia Washington "West Virginia" Wisconsin Wyoming \
    --horizon 4 \
    --use-enhanced-features \
    --trials-stage1 100 \
    --trials-stage2 50 \
    --n-features 10 \
    --random-seed 1 \
    --num-threads ${LGBM_NUM_THREADS:-16} \
    --optuna-jobs ${OPTUNA_JOBS:-1} \
    "$@"

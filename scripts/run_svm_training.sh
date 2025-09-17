#!/bin/bash

# Ensure we run from the repository root regardless of invocation path
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# Run SVM forecasting pipeline on all locations
# for horizons 1-4 with 25 trials and 10 features

# Set data file and cutoff date
DATA_FILE="data/imputed_and_stitched_hosp_2025-05-24.csv"
CUTOFF_DATE="2024-07-01"
TRIALS=100
N_FEATURES=10
RANDOM_SEED=1

# Define all locations - note that multi-word locations need special handling
# We'll pass them directly to the Python script with proper quoting

echo "=================================================="
echo "SVM Pipeline Training - All Locations"
echo "=================================================="
echo "Data file: $DATA_FILE"
echo "Cutoff date: $CUTOFF_DATE"
echo "Trials: $TRIALS"
echo "Features: $N_FEATURES"
echo "Locations: 53 total"
echo "=================================================="

# Train models for each horizon (no enhanced features for initial run)
for HORIZON in 1 2 3 4; do
    echo ""
    echo "=================================================="
    echo "Training Horizon $HORIZON"
    echo "=================================================="

    python src/train_svm.py \
        --data-file "$DATA_FILE" \
        --cut-off "$CUTOFF_DATE" \
        --locations Alabama Alaska Arizona Arkansas California \
                   Colorado Connecticut Delaware "District of Columbia" \
                   Florida Georgia Hawaii Idaho Illinois Indiana \
                   Iowa Kansas Kentucky Louisiana Maine Maryland \
                   Massachusetts Michigan Minnesota Mississippi Missouri \
                   Montana Nebraska Nevada "New Hampshire" "New Jersey" \
                   "New Mexico" "New York" "North Carolina" "North Dakota" Ohio \
                   Oklahoma Oregon Pennsylvania "Puerto Rico" "Rhode Island" \
                   "South Carolina" "South Dakota" Tennessee Texas US Utah \
                   Vermont Virginia Washington "West Virginia" Wisconsin Wyoming \
        --horizon $HORIZON \
        --trials $TRIALS \
        --n-features $N_FEATURES \
        --max-lags 8 \
        --random-seed $RANDOM_SEED \
        --kernels rbf,linear \
        --seasonal-in-search \
        

    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Horizon $HORIZON training completed successfully"
    else
        echo "✗ Horizon $HORIZON training failed"
        exit 1
    fi
done

echo ""
echo "=================================================="
echo "All Training Completed Successfully!"
echo "=================================================="
echo ""
echo "Models saved in: models/"
echo ""
echo "To generate retrospective forecasts, run:"
echo ""
for HORIZON in 1 2 3 4; do
    echo "python src/generate_retrospective_forecasts.py \\"
    echo "    --hyperparams models/svm_hyperparameters_h${HORIZON}_t${TRIALS}.pkl \\"
    echo "    --data-file $DATA_FILE \\"
    echo "    --cut-off $CUTOFF_DATE \\"
    echo "    --output forecasts/retrospective/svm_h${HORIZON}"
    echo ""
done

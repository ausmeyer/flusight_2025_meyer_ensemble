#!/bin/bash

# Generate retrospective forecasts for all trained models
# This script should be run after run_svm_pipeline.sh completes

# Set data file and cutoff date
DATA_FILE="data/imputed_and_stitched_hosp_2025-05-24.csv"
CUTOFF_DATE="2024-10-01"
TRIALS=25
# Optional: Set MAX_WEEKS to limit the number of weeks (uncomment to use)
# MAX_WEEKS=30

echo "=================================================="
echo "Generating Retrospective Forecasts"
echo "=================================================="
echo "Data file: $DATA_FILE"
echo "Cutoff date: $CUTOFF_DATE"
if [ -n "$MAX_WEEKS" ]; then
    echo "Max weeks: $MAX_WEEKS"
else
    echo "Max weeks: Running to end of data file"
fi
echo "=================================================="

# Create retrospective directory if it doesn't exist
mkdir -p forecasts/retrospective

# Generate forecasts for each horizon
for HORIZON in 1 2 3 4; do
    echo ""
    echo "=================================================="
    echo "Generating forecasts for Horizon $HORIZON"
    echo "=================================================="

    # Check for enhanced features model first, fall back to regular if not found
    HYPERPARAMS_FILE="models/svm_hyperparameters_h${HORIZON}_t${TRIALS}_enhanced.pkl"
    if [ ! -f "$HYPERPARAMS_FILE" ]; then
        HYPERPARAMS_FILE="models/svm_hyperparameters_h${HORIZON}_t${TRIALS}.pkl"
    fi
    OUTPUT_FILE="forecasts/retrospective/svm_t${TRIALS}_h${HORIZON}.csv"

    # Check if hyperparameters file exists
    if [ ! -f "$HYPERPARAMS_FILE" ]; then
        echo "✗ Hyperparameters file not found: $HYPERPARAMS_FILE"
        echo "  Please run run_svm_pipeline.sh first"
        continue
    fi

    # Use a temporary directory and then move the file
    TEMP_DIR="forecasts/temp_h${HORIZON}"

    # Generate baseline for each horizon
    BASELINE_FLAG="--include-baseline"

    # Build command with optional max-weeks
    CMD="python src/generate_retrospective_forecasts.py \
        --hyperparams \"$HYPERPARAMS_FILE\" \
        --data-file \"$DATA_FILE\" \
        --cut-off \"$CUTOFF_DATE\" \
        --output \"$TEMP_DIR\""

    # Add max-weeks if specified
    if [ -n "$MAX_WEEKS" ]; then
        CMD="$CMD --max-weeks $MAX_WEEKS"
    fi

    # Add baseline flag if needed
    CMD="$CMD $BASELINE_FLAG"

    # Execute the command
    eval $CMD

    # Check if the command was successful
    if [ $? -eq 0 ]; then
        # Move the generated SVM file to the retrospective folder with proper naming
        if [ -f "$TEMP_DIR/svm_retrospective_h${HORIZON}.csv" ]; then
            mv "$TEMP_DIR/svm_retrospective_h${HORIZON}.csv" "$OUTPUT_FILE"
            echo "✓ Horizon $HORIZON SVM forecasts generated successfully"
            echo "  Output saved to: $OUTPUT_FILE"
        else
            echo "✗ Expected SVM output file not found in $TEMP_DIR"
        fi

        # Move baseline file if it exists (when baseline flag was used)
        if [ -n "$BASELINE_FLAG" ]; then
            BASELINE_FILE="$TEMP_DIR/FluSight-baseline_h${HORIZON}.csv"
            if [ -f "$BASELINE_FILE" ]; then
                mv "$BASELINE_FILE" "forecasts/retrospective/"
                echo "  Baseline horizon $HORIZON saved to: forecasts/retrospective/FluSight-baseline_h${HORIZON}.csv"
            fi
        fi

        # Clean up temp directory and any remaining files
        if [ -d "$TEMP_DIR" ]; then
            rm -rf "$TEMP_DIR"
            echo "  Cleaned up temporary directory"
        fi
    else
        echo "✗ Horizon $HORIZON forecast generation failed"
        # Clean up temp directory even on failure
        if [ -d "$TEMP_DIR" ]; then
            rm -rf "$TEMP_DIR"
        fi
    fi
done

echo ""
echo "=================================================="
echo "All Forecasts Generated!"
echo "=================================================="
echo ""
echo "Forecast files saved in: forecasts/retrospective/"
echo "Files created:"
for HORIZON in 1 2 3 4; do
    OUTPUT_FILE="forecasts/retrospective/svm_t${TRIALS}_h${HORIZON}.csv"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "  - $(basename $OUTPUT_FILE)"
    fi
done
echo ""
echo "To analyze results, run the R Markdown script:"
echo "  Rscript -e \"rmarkdown::render('forecasts/retrospective_svm_comparison.Rmd')\""
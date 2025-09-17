# Influenza Hospitalizations Ensemble Pipeline (2025)

This repository assembles a weekly ensemble pipeline for forecasting influenza hospitalizations using three model families (ARIMA, SVM, LightGBM-LSS) and a prospective adaptive ensemble.

## Weekly Run

- Ensure R and Python dependencies are installed (lightgbmlss, pmdarima, sklearn, lightgbm, tidyverse, etc.).
- Run the full weekly pipeline from the repo root:

```bash
bash scripts/run_weekly.sh
```

What it does:
- Auto-dates and renders `src/stitch.Rmd` (writes dated ILI/ED/stitched CSVs)
- Detects the latest stitched file and computes a cutoff = last_date - 8 weeks
- Generates retrospective forecasts (h=1..4) for ARIMA, LGBM, SVM
- Generates prospective forecasts (h=1..4) for ARIMA, LGBM, SVM
- Builds a prospective adaptive ensemble from the last 4 reference weeks’ weights (using up to the last 8 retrospective weeks)

You can control the ensemble lookback and inclusions via flags:

```bash
# Use 6-week lookback and 8-week history window; include all models
bash scripts/run_weekly.sh \
  --lookback 6 \
  --history 8 \
  --include-arima true \
  --include-svm true \
  --include-lgbm true
```

## Outputs

- Stitched data: `data/imputed_sets/imputed_and_stitched_hosp_<YYYY-MM-DD>.csv`
- Retrospective forecasts (CDC-style, quantiles only):
  - `forecasts/retrospective/arima/ARIMA_h{1..4}_forecasts.csv`
  - `forecasts/retrospective/lgbm_enhanced_t10/TwoStage-FrozenMu_h{1..4}_forecasts.csv`
  - `forecasts/retrospective/svm_retrospective_h{1..4}.csv`
- Prospective forecasts (CDC-style, quantiles only):
  - `forecasts/prospective/ARIMA_h{1..4}_prospective_<YYYYMMDD>.csv`
  - `forecasts/prospective/SVM_h{1..4}_prospective_<YYYYMMDD>.csv`
  - `forecasts/prospective/TwoStage-FrozenMu_h{1..4}_prospective_<YYYYMMDD>.csv`
  - `forecasts/prospective/AdaptiveEnsemble_h{1..4}_prospective_<YYYYMMDD>.csv`

Schema (CDC-style):

```
reference_date, horizon, target, target_end_date, location, output_type, output_type_id, value
```
- `target` is `wk inc flu hosp`
- `horizon` is zero-based for retrospective files (h-1) and `0` for prospective files (horizons are encoded in filenames)
- `output_type` is `quantile`; `output_type_id` contains the CDC quantile levels

## LightGBM Model Saving

Prospective LightGBM training saves models (per location, per horizon) to:
- `models/lgbm_enhanced_t10/point_mu/{location}_h{h}_booster.txt`
- `models/lgbm_enhanced_t10/scale_sigma/{location}_h{h}_lgbmlss_model.pkl`

Retrospective LGBM generation can read from `--models-base-dir models/lgbm_enhanced_t10`.

## Prospective Ensemble Controls

`src/generate_prosp_adaptive_ensemble.R` supports optional flags:

- `--lookback-weeks <N>` (default: 4)
- `--history-weeks <N>` (default: 8)
- `--include-arima <true|false>` (default: true)
- `--include-svm <true|false>` (default: true)
- `--include-lgbm <true|false>` (default: true)
- `--asof-date <YYYY-MM-DD>` (override latest Saturday)

Example (direct call without the weekly runner):

```bash
Rscript src/generate_prosp_adaptive_ensemble.R \
  --lookback-weeks 6 --history-weeks 8 \
  --include-arima true --include-svm true --include-lgbm true
```

## Helpers

- `src/utils/pipeline_utils.py` provides:
  - `find_latest_imputed()` – latest `imputed_and_stitched_hosp_*.csv`
  - `last_and_cutoff_dates(data_file, weeks=8)` – computes last date and cutoff

## Notes

- `src/stitch.Rmd` automatically sets the as-of date to the most recent Saturday and writes dated outputs.
- All model generators accept explicit `--data-file` overrides if needed.
- Retrospective SVM outputs are standardized to the CDC 8-column format.

# Joint KNN3 Two-Stage Pooled-Horizon Bagged Model (Test Module)

This directory is an isolated experiment that does **not** modify your existing pipeline.

It implements:

- Joint fitting across locations
- Pooled horizons (single model across requested horizons with `horizon` as a feature)
- Two-stage training:
  - Stage 1: KNN (`k=3`) for transformed mean target
  - Stage 2: LightGBMLSS with frozen `mu` and bounded `sigma` in log space
- Season-level bagging (Flusion-style): sample a fraction of flu seasons per bag, fit models, aggregate bag forecasts by median
- Covariance-selected donor states: for each location, picks top-k other states by historical covariance and adds donor lag covariates

## Script

- `joint_twostage_pool.py`

## Input

- Any stitched file with `location_name`, `date`, `total_hosp` (for example: `data/imputed_sets/imputed_and_stitched_hosp_2026-02-07.csv`)
- Uses `data/locations.csv` only for location-to-FIPS mapping in outputs

## Outputs

CSV in CDC-style columns:

- `reference_date`
- `target`
- `horizon`
- `target_end_date`
- `location`
- `output_type`
- `output_type_id`
- `value`

## Backtest options

1. Quick backtest (faster sanity check)

```bash
python joint_knn3_pool_test/joint_twostage_pool.py backtest \
  --data-file data/imputed_sets/imputed_and_stitched_hosp_2026-02-07.csv \
  --start-date 2025-10-04 \
  --output joint_knn3_pool_test/outputs/backtest_quick.csv \
  --num-bags 20 \
  --bag-frac 0.7 \
  --anchor-step-weeks 2 \
  --max-horizons 4
```

2. Full backtest (heavier, closer to your intended setup)

```bash
python joint_knn3_pool_test/joint_twostage_pool.py backtest \
  --data-file data/imputed_sets/imputed_and_stitched_hosp_2026-02-07.csv \
  --start-date 2025-10-04 \
  --output joint_knn3_pool_test/outputs/backtest_full.csv \
  --num-bags 100 \
  --bag-frac 0.7 \
  --anchor-step-weeks 1 \
  --max-horizons 4 \
  --sigma-mode wide
```

3. Backtest through end of observed data (includes recent anchors where only horizon 0 may be evaluable yet)

```bash
python joint_knn3_pool_test/joint_twostage_pool.py backtest \
  --data-file data/imputed_sets/imputed_and_stitched_hosp_2026-02-07.csv \
  --start-date 2025-10-04 \
  --output joint_knn3_pool_test/outputs/backtest_to_end.csv \
  --num-bags 100 \
  --bag-frac 0.7 \
  --anchor-step-weeks 1 \
  --max-horizons 4 \
  --include-partial-horizons \
  --sigma-mode wide
```

## Prospective options

1. Quick prospective run

```bash
python joint_knn3_pool_test/joint_twostage_pool.py prospective \
  --data-file data/imputed_sets/imputed_and_stitched_hosp_2026-02-07.csv \
  --output joint_knn3_pool_test/outputs/prospective_quick.csv \
  --num-bags 20 \
  --bag-frac 0.7 \
  --max-horizons 4
```

2. Full prospective run

```bash
python joint_knn3_pool_test/joint_twostage_pool.py prospective \
  --data-file data/imputed_sets/imputed_and_stitched_hosp_2026-02-07.csv \
  --output joint_knn3_pool_test/outputs/prospective_full.csv \
  --num-bags 100 \
  --bag-frac 0.7 \
  --max-horizons 4 \
  --sigma-mode wide
```

## Notes

- This script uses fixed model parameters (no Optuna search) to keep the joint model computationally tractable.
- Donor-state covariates are enabled by default:
  - `--cov-top-k 5`
  - `--cov-lags 1,2,3,4,8,12,52`
  - `--cov-min-overlap 40`
- Target mode:
  - `--target-mode delta_log` (default): fit log-change target `log(1+y_{t+h}) - log(1+y_t)` (Flusion-like)
  - `--target-mode level`: fit log-level target
- For backtests, donor-state selection is fit on data before `--start-date` (to avoid forward leakage).
- If runtime is too high, first reduce:
  - `--num-bags`
  - `--anchor-step-weeks` (for backtests)
  - location scope via `--locations`

## Top-k donor toggle

Use `--cov-top-k 3` or `--cov-top-k 5`:

```bash
python joint_knn3_pool_test/joint_twostage_pool.py backtest \
  --data-file data/imputed_sets/imputed_and_stitched_hosp_2026-02-07.csv \
  --start-date 2025-11-19 \
  --output joint_knn3_pool_test/outputs/backtest_cov3.csv \
  --max-horizons 4 \
  --cov-top-k 3 \
  --cov-lags 1,2,3,4,8,12,52 \
  --target-mode delta_log \
  --include-partial-horizons
```

## Hedge Ensemble (Test Copy)

Use the copied hedge script:

- `joint_knn3_pool_test/ensemble/generate_prosp_adaptive_ensemble_hedge_test.R`

It keeps the existing hedge weighting logic, and adds `JointKNN3Pool` as an optional component using:

- retrospective history file (for weight estimation)
- prospective file (for current blended forecast values)

### Recommended workflow

1. Generate joint backtest history with enough references for weights

```bash
python joint_knn3_pool_test/joint_twostage_pool.py backtest \
  --data-file data/imputed_sets/imputed_and_stitched_hosp_2026-02-07.csv \
  --start-date 2025-11-19 \
  --output joint_knn3_pool_test/outputs/backtest_cov5_delta_log_from_2025-11-19.csv \
  --max-horizons 4 \
  --num-bags 100 \
  --bag-frac 0.7 \
  --cov-top-k 5 \
  --cov-lags 1,2,3,4,8,12,52 \
  --target-mode delta_log \
  --include-partial-horizons \
  --sigma-mode wide
```

2. Generate the joint prospective file

```bash
python joint_knn3_pool_test/joint_twostage_pool.py prospective \
  --data-file data/imputed_sets/imputed_and_stitched_hosp_2026-02-07.csv \
  --output joint_knn3_pool_test/outputs/prospective_cov5_delta_log.csv \
  --max-horizons 4 \
  --num-bags 100 \
  --bag-frac 0.7 \
  --cov-top-k 5 \
  --cov-lags 1,2,3,4,8,12,52 \
  --target-mode delta_log \
  --sigma-mode wide
```

3. Run hedge ensemble in test output directory

```bash
Rscript joint_knn3_pool_test/ensemble/generate_prosp_adaptive_ensemble_hedge_test.R \
  --asof-date 2026-02-07 \
  --include-joint-twostage true \
  --joint-retro-file joint_knn3_pool_test/outputs/backtest_cov5_delta_log_from_2025-11-19.csv \
  --joint-prosp-file joint_knn3_pool_test/outputs/prospective_cov5_delta_log.csv
```

Output is written to:

- `joint_knn3_pool_test/outputs/ensemble_hedge/AdaptiveEnsemble-hedge_prospective_<YYYYMMDD>.csv`

### History requirement for stable weights

- Hedge weights use the most recent `--history-weeks` reference dates (default `6`) before `--asof-date`.
- If the joint model has fewer than that many historical reference dates, the script logs a warning and weights may be noisy.

## Hedge Backtest (Retrospective)

To backtest the hedge ensemble itself with `JointKNN3Pool` included as a component, use:

- `joint_knn3_pool_test/ensemble/backtest_prosp_ensemble_hedge_test.R`

Example:

```bash
Rscript joint_knn3_pool_test/ensemble/backtest_prosp_ensemble_hedge_test.R \
  --warmup-start 2025-09-06 \
  --season-start 2025-11-01 \
  --season-end 2026-02-07 \
  --include-joint-twostage true \
  --joint-retro-file joint_knn3_pool_test/outputs/backtest_cov5_delta_log_from_2025-11-19.csv
```

Notes:

- This uses the same adaptive hedge weighting structure as your production backtest script.
- Use `--warmup-start` earlier than `--season-start` to accumulate model losses/weights before the scored season window.
- Output goes to `joint_knn3_pool_test/outputs/ensemble_hedge_retrospective/`.
- `--joint-reference-shift-days` defaults to `-7` to align the joint model's CDC-style reference dates with legacy component date conventions used in this backtest script.

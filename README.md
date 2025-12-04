# FluSight 2025 Ensemble Forecasts

This repository contains weekly influenza hospitalization forecasts for the United States, submitted to the CDC FluSight forecasting challenge.

## Forecast Visualization

**[View Latest Forecasts](https://ausmeyer.github.io/flusight_2025_meyer_ensemble/)**

The interactive dashboard displays:
- Forecasts for all 50 US states, Puerto Rico, and national aggregates
- Multiple model outputs with uncertainty intervals (50% and 95% CI)
- Ground truth hospitalization data for comparison

## Overview

We generate probabilistic forecasts of weekly incident influenza hospitalizations at 1-4 week horizons for all US states and territories. Our approach combines multiple statistical and machine learning models into an adaptive ensemble.

### Models

- **ARIMA** - Autoregressive integrated moving average time series model
- **SVM** - Support vector machine regression
- **LightGBM** - Gradient boosting with distributional forecasting
- **AdaptiveEnsemble** - Weighted combination of base models, with weights updated based on recent forecast performance

## Forecast Format

Forecasts follow the CDC FluSight hub format:

| Column | Description |
|--------|-------------|
| `reference_date` | Date forecast was generated |
| `target` | `wk inc flu hosp` (weekly incident flu hospitalizations) |
| `horizon` | Weeks ahead (0-3) |
| `target_end_date` | End date of the forecast week |
| `location` | FIPS code or "US" |
| `output_type` | `quantile` |
| `output_type_id` | Quantile level (0.01 to 0.99) |
| `value` | Forecast value |

## Data Sources

- **HHS Protect** - Weekly hospitalization data via healthdata.gov
- **ILINet** - CDC influenza-like illness surveillance data

## Repository Structure

```
├── data/                  # Input data files
├── forecasts/
│   ├── prospective/       # Weekly submission-ready forecasts
│   └── retrospective/     # Historical backtesting results
├── docs/                  # GitHub Pages visualization
└── src/                   # Model and pipeline code
```

## Authors

Santillana Lab, Harvard University

## License

This project is for research purposes. Forecast outputs are publicly available for use with appropriate attribution.

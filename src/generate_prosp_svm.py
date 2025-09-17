#!/usr/bin/env python
"""
Generate prospective SVM forecasts (horizons 1–4) using saved hyperparameters.

For each horizon, trains per-location SVR on all data up to the last date in the
imputed file, then generates probabilistic forecasts using residual-based or
calibrated intervals. Outputs standardized CDC-style quantiles to
forecasts/prospective/SVM_h{h}_prospective_{YYYYMMDD}.csv.

Usage examples:
  - Auto-detect latest imputed file and default hyperparam dir:
      python src/generate_prosp_svm.py --auto-latest --models models/svm_t100

  - Explicit file:
      python src/generate_prosp_svm.py \
          --data-file data/imputed_sets/imputed_and_stitched_hosp_2025-09-13.csv \
          --models models/svm_t100
"""

import os
import sys
import argparse
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
from utils.tabularizer import create_features, create_features_for_prediction
from utils.enhanced_features import create_enhanced_features, create_enhanced_features_for_prediction
from utils.lgbm_timeseries import TimeSeriesDataProcessor
from utils.pipeline_utils import find_latest_imputed


CDC_QUANTILES = np.array([
    0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99
])


STATE_TO_FIPS = {
    'Alabama': '01', 'Alaska': '02', 'Arizona': '04', 'Arkansas': '05',
    'California': '06', 'Colorado': '08', 'Connecticut': '09', 'Delaware': '10',
    'District of Columbia': '11', 'Florida': '12', 'Georgia': '13', 'Hawaii': '15',
    'Idaho': '16', 'Illinois': '17', 'Indiana': '18', 'Iowa': '19',
    'Kansas': '20', 'Kentucky': '21', 'Louisiana': '22', 'Maine': '23',
    'Maryland': '24', 'Massachusetts': '25', 'Michigan': '26', 'Minnesota': '27',
    'Mississippi': '28', 'Missouri': '29', 'Montana': '30', 'Nebraska': '31',
    'Nevada': '32', 'New Hampshire': '33', 'New Jersey': '34', 'New Mexico': '35',
    'New York': '36', 'North Carolina': '37', 'North Dakota': '38', 'Ohio': '39',
    'Oklahoma': '40', 'Oregon': '41', 'Pennsylvania': '42', 'Puerto Rico': '72',
    'Rhode Island': '44', 'South Carolina': '45', 'South Dakota': '46', 'Tennessee': '47',
    'Texas': '48', 'Utah': '49', 'Vermont': '50', 'Virginia': '51',
    'Washington': '53', 'West Virginia': '54', 'Wisconsin': '55', 'Wyoming': '56',
    'US': 'US'
}


def load_and_split(data_file: str) -> Tuple[pd.DataFrame, pd.Timestamp]:
    processor = TimeSeriesDataProcessor()
    df = processor.load_and_pivot_data(data_file, exclude_locations=None)
    last_date = pd.to_datetime(df['date'].max())
    return df, last_date


def apply_log_transform(arr: np.ndarray, enable: bool) -> np.ndarray:
    if not enable:
        return arr
    return np.log1p(np.maximum(arr, 0))


def inverse_log_transform(arr: np.ndarray, enable: bool) -> np.ndarray:
    if not enable:
        return arr
    return np.expm1(arr)


def quantiles_from_residuals(point: float, residuals: List[float] | None, scale: str = 'log1p', fallback_errors: List[float] | None = None) -> np.ndarray:
    residuals = residuals or []
    fallback_errors = fallback_errors or []
    from scipy import stats
    if len(residuals) >= 10:
        rq = np.percentile(residuals, CDC_QUANTILES * 100)
        if scale == 'log1p':
            loc = np.log1p(max(point, 0))
            return np.array([max(0.0, np.expm1(loc + r)) for r in rq])
        elif scale == 'pct':
            return np.array([max(0.0, point * (1.0 + r)) for r in rq])
        else:
            return np.array([max(0.0, point + r) for r in rq])
    if len(fallback_errors) >= 10:
        eq = np.percentile(fallback_errors, CDC_QUANTILES * 100)
        return np.array([max(0.0, point + e) for e in eq])
    std = max(np.std(fallback_errors), 5.0) if len(fallback_errors) > 1 else max(0.25 * abs(point), 5.0)
    return np.array([max(0.0, stats.norm.ppf(q, loc=point, scale=std)) for q in CDC_QUANTILES])


def main():
    ap = argparse.ArgumentParser(description='Generate SVM prospective forecasts (h=1..4)')
    ap.add_argument('--data-file', type=str, default=None, help='Path to imputed_and_stitched_hosp_*.csv')
    ap.add_argument('--auto-latest', action='store_true', help='Use latest imputed file automatically')
    ap.add_argument('--models', type=str, default='models/svm_t100', help='Directory with SVM hyperparameters and models')
    ap.add_argument('--output', type=str, default='forecasts/prospective', help='Output directory')
    args = ap.parse_args()

    data_file = find_latest_imputed() if (args.auto_latest or not args.data_file) else args.data_file
    if not os.path.exists(data_file):
        raise FileNotFoundError(data_file)
    os.makedirs(args.output, exist_ok=True)

    df, last_date = load_and_split(data_file)

    locations = [c for c in df.columns if c != 'date']

    for horizon in [1, 2, 3, 4]:
        # Load hyperparameters per horizon
        hp_enh = os.path.join(args.models, f'svm_hyperparameters_h{horizon}_t100_enhanced.pkl')
        hp_std = os.path.join(args.models, f'svm_hyperparameters_h{horizon}_t100.pkl')
        hp_path = hp_enh if os.path.exists(hp_enh) else hp_std
        if not os.path.exists(hp_path):
            print(f"Skipping h{horizon}: hyperparameters not found: {hp_path}")
            continue
        import pickle
        with open(hp_path, 'rb') as f:
            hyperparams = pickle.load(f)

        records = []
        for loc in locations:
            if loc not in hyperparams:
                continue
            params = hyperparams[loc]
            best = params['best_params']
            selected_states = params['selected_states']
            lags = params.get('lags', None)
            residuals = params.get('residuals', [])
            residual_scale = params.get('residual_scale', 'log1p')
            error_dist = params.get('error_distribution', {})
            errors = error_dist.get(horizon, []) if isinstance(error_dist, dict) else []
            use_log = params.get('use_log_transform', False)

            # Build training features up to last_date - 1 day (avoid leakage)
            end_train = last_date - pd.Timedelta(days=1)
            if lags is None:
                X_train, y_train, _ = create_enhanced_features(df, loc, selected_states, end_date=end_train, horizon=horizon)
            else:
                X_train, y_train, _ = create_features(df, loc, selected_states, lags, end_date=end_train, horizon=horizon)
            if len(X_train) < 25:
                continue

            X_train_t = apply_log_transform(X_train, use_log)
            y_train_t = apply_log_transform(y_train, use_log)
            scaler = StandardScaler().fit(X_train_t)
            model = SVR(**best)
            model.fit(scaler.transform(X_train_t), y_train_t)

            # Build prediction features anchored at last_date
            if lags is None:
                X_pred, _ = create_enhanced_features_for_prediction(df, loc, selected_states, anchor_date=last_date, horizon=horizon)
            else:
                X_pred, _ = create_features_for_prediction(df, loc, selected_states, lags, anchor_date=last_date, horizon=horizon)
            if len(X_pred) == 0:
                continue

            x_last = apply_log_transform(X_pred[-1:], use_log)
            point_t = float(model.predict(scaler.transform(x_last))[0])
            point = float(inverse_log_transform(np.array([point_t]), use_log)[0])

            qvals = quantiles_from_residuals(point, residuals, scale=residual_scale, fallback_errors=errors)
            fips = STATE_TO_FIPS.get(loc, loc)
            target_date = last_date + pd.Timedelta(weeks=horizon)
            for qi, q in enumerate(CDC_QUANTILES):
                records.append({
                    'reference_date': last_date.strftime('%Y-%m-%d'),
                    'horizon': 0,
                    'target': 'wk inc flu hosp',
                    'target_end_date': target_date.strftime('%Y-%m-%d'),
                    'location': fips,
                    'output_type': 'quantile',
                    'output_type_id': q,
                    'value': float(qvals[qi])
                })

        if records:
            out_df = pd.DataFrame(records)
            cols = ['reference_date', 'horizon', 'target', 'target_end_date', 'location', 'output_type', 'output_type_id', 'value']
            out_df = out_df[cols]
            ts = last_date.strftime('%Y%m%d')
            out_path = os.path.join(args.output, f'SVM_h{horizon}_prospective_{ts}.csv')
            out_df.to_csv(out_path, index=False)
            print(f"Saved: {out_path} ({len(out_df)} rows)")


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""
Generate prospective ARIMA forecasts with empirical prediction intervals.

This script trains per-location ARIMA models using data up to the last date in
the imputed-and-stitched file, builds empirical error distributions from a
validation window ending 8 weeks before the last date, then issues prospective
probabilistic forecasts for horizons 1–4 from the last date.

Outputs are standardized to CDC-style quantile format and saved under
forecasts/prospective/ARIMA_h{h}_prospective_{YYYYMMDD}.csv.

Usage examples:
  - Auto-detect latest imputed file and output to default folder
      python src/generate_prosp_arima.py --auto-latest

  - Explicit file
      python src/generate_prosp_arima.py \
        --data-file data/imputed_sets/imputed_and_stitched_hosp_2025-09-13.csv \
        --output forecasts/prospective
"""

import os
import argparse
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pmdarima import auto_arima, ARIMA

# Utils
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
from pipeline_utils import find_latest_imputed

warnings.filterwarnings("ignore")


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


def load_wide(df_path: str) -> pd.DataFrame:
    df = pd.read_csv(df_path)
    df['date'] = pd.to_datetime(df['date'])
    wide = df.pivot(index='date', columns='location_name', values='total_hosp')
    wide = wide.asfreq('W-SAT')
    return wide


def determine_order(series: pd.Series) -> Tuple[int, int, int]:
    series = series.dropna()
    if len(series) < 20:
        return (1, 1, 1)
    try:
        model = auto_arima(series + 1,
                           start_p=0, max_p=8,
                           start_d=0, max_d=2,
                           start_q=0, max_q=8,
                           seasonal=False, stepwise=True, max_order=15,
                           n_fits=100, error_action='ignore', suppress_warnings=True,
                           trace=False)
        return model.order
    except Exception:
        return (1, 1, 1)


def fit_predict(series: pd.Series, order: Tuple[int, int, int], horizon: int) -> np.ndarray:
    series = series.dropna()
    if len(series) == 0:
        return np.full(horizon, np.nan)
    try:
        model = ARIMA(order=order, suppress_warnings=True)
        model.fit(series + 1)
        fc = model.predict(n_periods=horizon)
        return np.maximum(fc - 1, 0)
    except Exception:
        return np.array([series.iloc[-1]] * horizon)


def make_quantiles(point: float, errors: List[float]) -> np.ndarray:
    if len(errors) < 10:
        std = max(abs(point) * 0.25, 5.0)
        from scipy import stats
        return np.array([max(0.0, stats.norm.ppf(q, loc=point, scale=std)) for q in CDC_QUANTILES])
    err_q = np.percentile(errors, CDC_QUANTILES * 100)
    return np.maximum(point + err_q, 0.0)


def format_cdc(reference_date: pd.Timestamp, target_date: pd.Timestamp, fips: str, qvals: np.ndarray) -> List[Dict]:
    rows = []
    for qi, q in enumerate(CDC_QUANTILES):
        rows.append({
            'reference_date': reference_date.strftime('%Y-%m-%d'),
            'horizon': 0,  # prospective files use CDC-style single-target schema; horizon encoded in filename
            'target': 'wk inc flu hosp',
            'target_end_date': target_date.strftime('%Y-%m-%d'),
            'location': fips,
            'output_type': 'quantile',
            'output_type_id': q,
            'value': float(qvals[qi])
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description='Generate prospective ARIMA forecasts (h=1..4)')
    ap.add_argument('--data-file', type=str, default=None, help='Path to imputed_and_stitched_hosp_*.csv')
    ap.add_argument('--auto-latest', action='store_true', help='Use latest imputed file automatically')
    ap.add_argument('--output', type=str, default='forecasts/prospective', help='Output directory')
    args = ap.parse_args()

    if args.auto_latest or not args.data_file:
        data_file = find_latest_imputed()
    else:
        data_file = args.data_file
    if not os.path.exists(data_file):
        raise FileNotFoundError(data_file)

    os.makedirs(args.output, exist_ok=True)

    wide = load_wide(data_file)
    last_date = pd.to_datetime(wide.index.max())
    cutoff_date = last_date - pd.Timedelta(weeks=8)

    # Split for error distributions
    pre_cut = wide[wide.index < cutoff_date]
    val = wide[(wide.index >= cutoff_date) & (wide.index <= last_date)]

    # Determine best orders per location using pre_cut (fall back to all available)
    best_orders = {}
    for loc in wide.columns:
        s = pre_cut[loc].dropna()
        if len(s) < 20:
            s = wide[loc].dropna()
        best_orders[loc] = determine_order(s)

    # Build error distributions from expanding forecasts across validation portion
    error_dist: Dict[str, Dict[int, List[float]]] = {loc: {h: [] for h in range(1, 5)} for loc in wide.columns}
    for loc in wide.columns:
        order = best_orders[loc]
        s_full = wide[loc].dropna()
        s_pre = s_full[s_full.index < cutoff_date]
        s_val = s_full[(s_full.index >= cutoff_date) & (s_full.index <= last_date)]
        if len(s_pre) < 20 or len(s_val) < 6:
            continue
        val_idx = s_val.index
        for i in range(len(val_idx)):
            train_end = val_idx[i] - pd.Timedelta(weeks=1)
            train_series = s_full[s_full.index <= train_end]
            preds = fit_predict(train_series, order, horizon=4)
            # Compare only if actual future exists
            for h in range(1, 5):
                tgt = val_idx[i] + pd.Timedelta(weeks=h)
                if tgt in s_full.index:
                    actual = s_full.loc[tgt]
                    pred = preds[h - 1]
                    if not np.isnan(actual) and not np.isnan(pred):
                        error_dist[loc][h].append(float(actual - pred))

    # Prospective forecasts from last_date
    for h in range(1, 5):
        records = []
        for loc in wide.columns:
            order = best_orders.get(loc, (1, 1, 1))
            preds = fit_predict(wide[loc], order, horizon=4)
            point = float(preds[h - 1]) if len(preds) >= h else np.nan
            errs = error_dist.get(loc, {}).get(h, [])
            qvals = make_quantiles(point, errs)
            fips = STATE_TO_FIPS.get(loc, loc)
            target_date = last_date + pd.Timedelta(weeks=h)
            records.extend(format_cdc(last_date, target_date, fips, qvals))

        if records:
            out_df = pd.DataFrame(records)
            # Ensure column order
            cols = ['reference_date', 'horizon', 'target', 'target_end_date', 'location', 'output_type', 'output_type_id', 'value']
            out_df = out_df[cols]
            ts = last_date.strftime('%Y%m%d')
            out_path = os.path.join(args.output, f'ARIMA_h{h}_prospective_{ts}.csv')
            out_df.to_csv(out_path, index=False)
            print(f"Saved: {out_path} ({len(out_df)} rows)")


if __name__ == '__main__':
    main()

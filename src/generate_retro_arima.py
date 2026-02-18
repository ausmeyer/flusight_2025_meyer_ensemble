#!/usr/bin/env python
"""
Generate retrospective ARIMA forecasts with plain conformal prediction intervals
based on prior out-of-sample residuals.

This script:
1. Fits ARIMA models using the first half of training data to determine lag orders
2. Generates retrospective forecasts from the cut-off date onwards
3. Collects residuals for in-season dates (Nov 1 - May 1)
4. Uses plain empirical conformal residuals for quantiles
5. Outputs forecasts in CDC FluSight format

Usage:
    python src/generate_retro_arima.py \
        --data-file data/imputed_and_stitched_hosp_2025-05-24.csv \
        --cut-off 2024-07-01 \
        --output forecasts/retrospective/arima
"""

import os
import sys
import argparse
import warnings
import pandas as pd
import numpy as np
from pmdarima import auto_arima, ARIMA
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy import stats
import pickle

warnings.filterwarnings("ignore")

# CDC FluSight quantiles
CDC_QUANTILES = np.array([
    0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99
])

# Season definition (inclusive): Nov 1 to May 1
SEASON_START_MONTH = 11  # November
SEASON_START_DAY = 1
SEASON_END_MONTH = 5     # May
SEASON_END_DAY = 1

# Cap on log-space residuals to avoid extreme outliers skewing the distribution
# Set to 1.6 to align with bounded-wide sigma max (0.8 * 1.96 ≈ 1.57)
RESIDUAL_CAP = 1.6

# Minimum residuals for conformal intervals
MIN_RESIDUALS_FOR_CONFORMAL = 4

# Residuals file name pattern
RESIDUALS_FILE_PATTERN = 'arima_residuals_h{horizon}.csv'

# Exogenous feature defaults
DEFAULT_DONOR_TOP_K = 5
DEFAULT_DONOR_MIN_OVERLAP = 30
SEASONAL_PERIOD_WEEKS = 52.1775

# State name to FIPS code mapping
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


def build_seasonal_exog(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build deterministic seasonal exogenous features."""
    iso_week = index.isocalendar().week.to_numpy(dtype=float)
    angle = (2.0 * np.pi * iso_week) / SEASONAL_PERIOD_WEEKS

    exog = pd.DataFrame(index=index)
    exog['season_sin1'] = np.sin(angle)
    exog['season_cos1'] = np.cos(angle)
    exog['season_sin2'] = np.sin(2.0 * angle)
    exog['season_cos2'] = np.cos(2.0 * angle)
    exog['in_flu_season'] = np.array(
        [1.0 if is_in_flu_season(pd.Timestamp(d)) else 0.0 for d in index],
        dtype=float
    )
    return exog


def pairwise_corr_on_initial_window(
    series_a: pd.Series,
    series_b: pd.Series,
    min_overlap: int
) -> float:
    """Correlation in log-delta space on overlapping initial-window data."""
    a = np.log1p(series_a).diff()
    b = np.log1p(series_b).diff()
    aligned = pd.concat([a.rename('a'), b.rename('b')], axis=1).dropna()
    if len(aligned) < min_overlap:
        return np.nan
    return float(aligned['a'].corr(aligned['b']))


def select_donor_states(
    initial_wide: pd.DataFrame,
    top_k: int,
    min_overlap: int
) -> Dict[str, List[str]]:
    """Select top-k correlated donor states per location on initial training data only."""
    donor_map: Dict[str, List[str]] = {}
    locations = list(initial_wide.columns)

    for loc in locations:
        target = initial_wide[loc]
        scores: List[Tuple[str, float]] = []

        for donor in locations:
            if donor == loc:
                continue
            corr = pairwise_corr_on_initial_window(target, initial_wide[donor], min_overlap)
            if np.isnan(corr):
                continue
            scores.append((donor, corr))

        scores.sort(key=lambda x: x[1], reverse=True)
        donor_map[loc] = [name for name, _ in scores[:top_k]]

    return donor_map


def build_location_exog(
    wide_with_future: pd.DataFrame,
    location: str,
    donor_map: Dict[str, List[str]],
    donor_lag_weeks: int
) -> pd.DataFrame:
    """Build location-specific exogenous matrix using seasonal terms + lagged donors."""
    exog = build_seasonal_exog(wide_with_future.index)
    donors = donor_map.get(location, [])

    for i, donor in enumerate(donors, start=1):
        if donor not in wide_with_future.columns:
            continue
        exog[f'donor{i}_lag{donor_lag_weeks}'] = np.log1p(
            wide_with_future[donor].astype(float)
        ).shift(donor_lag_weeks)

    return exog


def prepare_train_inputs(
    y_series: pd.Series,
    exog_df: pd.DataFrame,
    min_rows: int = 20
) -> Tuple[Optional[pd.Series], Optional[np.ndarray]]:
    """Align target + exog and remove rows with NA."""
    merged = pd.concat([y_series.rename('y'), exog_df], axis=1).dropna()
    if len(merged) < min_rows:
        return None, None
    y = merged['y'].astype(float)
    X = merged.drop(columns=['y']).to_numpy(dtype=float)
    return y, X


def is_in_flu_season(date: pd.Timestamp) -> bool:
    """Check if a date is within the flu season (Nov 1 - May 1)."""
    month, day = date.month, date.day
    if month >= SEASON_START_MONTH:
        return True
    if month < SEASON_END_MONTH:
        return True
    if month == SEASON_END_MONTH and day <= SEASON_END_DAY:
        return True
    return False


def get_flu_season_year(date: pd.Timestamp) -> int:
    """Get the flu season year for a date. Season 2024 = Nov 2024 - May 2025."""
    if date.month >= SEASON_START_MONTH:
        return date.year
    else:
        return date.year - 1


def get_target_season_year(date: pd.Timestamp) -> int:
    """Return most recent COMPLETED flu season year for residual lookup."""
    current_season_year = get_flu_season_year(date)
    if is_in_flu_season(date):
        return current_season_year - 1
    return current_season_year


def get_week_of_season(date: pd.Timestamp) -> int:
    """Get the week number within the flu season (0 = first week of Nov)."""
    season_year = get_flu_season_year(date)
    season_start = pd.Timestamp(year=season_year, month=SEASON_START_MONTH, day=SEASON_START_DAY)
    days_since_start = (date - season_start).days
    if days_since_start < 0:
        days_since_start += 365
    return days_since_start // 7


def empirical_percentile(values, quantiles):
    """Compute empirical percentiles."""
    return np.quantile(np.asarray(values), quantiles)


class ARIMAForecaster:
    """Generate ARIMA forecasts with season-based conformal prediction intervals."""

    def __init__(
        self,
        cut_off_date: str,
        max_horizon: int = 4,
        donor_top_k: int = DEFAULT_DONOR_TOP_K,
        donor_min_overlap: int = DEFAULT_DONOR_MIN_OVERLAP,
        donor_lag_weeks: Optional[int] = None
    ):
        self.cut_off_date = pd.to_datetime(cut_off_date)
        self.max_horizon = max_horizon
        self.quantiles = CDC_QUANTILES
        self.donor_top_k = donor_top_k
        self.donor_min_overlap = donor_min_overlap
        self.donor_lag_weeks = donor_lag_weeks if donor_lag_weeks is not None else max_horizon

        # Storage for models and errors
        self.best_orders = {}
        self.donor_map: Dict[str, List[str]] = {}
        self.data = None
        self.prepared_data = None

    def load_data(self, file_path: str) -> None:
        """Load and prepare data for ARIMA modeling."""
        print(f"Loading data from {file_path}")

        raw_data = pd.read_csv(file_path)
        raw_data['date'] = pd.to_datetime(raw_data['date'])

        self.data = raw_data

        # Pivot data to have dates as indices and locations as columns
        self.prepared_data = raw_data.pivot(
            index='date',
            columns='location_name',
            values='total_hosp'
        )

        # Set weekly frequency
        self.prepared_data = self.prepared_data.asfreq('W-SAT')

        print(f"Data shape: {len(self.prepared_data)} weeks x {len(self.prepared_data.columns)} locations")
        print(f"Date range: {self.prepared_data.index.min()} to {self.prepared_data.index.max()}")

    def determine_best_order(self, train: pd.Series, exog_train: pd.DataFrame, location: str) -> Tuple:
        """Use auto_arima to determine the best ARIMA order with exogenous covariates."""
        try:
            y, X = prepare_train_inputs(train, exog_train, min_rows=20)
            if y is None or X is None:
                return (1, 1, 1)
            train_shifted = y + 1
            model = auto_arima(
                train_shifted,
                X=X,
                start_p=0, max_p=8,
                start_d=0, max_d=2,
                start_q=0, max_q=8,
                seasonal=False,
                stepwise=True,
                max_order=15,
                n_fits=100,
                error_action='ignore',
                suppress_warnings=True,
                trace=False
            )
            return model.order
        except Exception as e:
            print(f"  Warning: auto_arima failed for {location}: {str(e)}")
            return (1, 1, 1)

    def fit_and_predict(
        self,
        train: pd.Series,
        exog_train: pd.DataFrame,
        exog_future: pd.DataFrame,
        order: Tuple,
        horizon: int
    ) -> np.ndarray:
        """Fit ARIMAX model and make forecasts."""
        try:
            y, X_train = prepare_train_inputs(train, exog_train, min_rows=20)
            if y is None or X_train is None:
                return np.array([train.iloc[-1]] * horizon)
            if exog_future.isnull().any().any():
                return np.array([train.iloc[-1]] * horizon)

            X_future = exog_future.to_numpy(dtype=float)
            train_shifted = y + 1
            model = ARIMA(order=order, suppress_warnings=True)
            model.fit(train_shifted, X=X_train)
            forecast_shifted = model.predict(n_periods=horizon, X=X_future)
            forecast = forecast_shifted - 1
            return np.maximum(forecast, 0)
        except Exception as e:
            return np.array([train.iloc[-1]] * horizon)

    def generate_forecasts(self, output_dir: str = None) -> Tuple[Dict, List[Dict]]:
        """Generate retrospective forecasts and collect in-season residuals.

        Returns:
            Tuple of (forecasts dict, residuals list)
        """
        print("\n--- Generating Retrospective ARIMA Forecasts ---")

        all_forecasts = {}
        all_residuals = []  # For saving

        # Get validation dates
        dates_sorted = sorted(self.prepared_data.index)
        val_start_idx = next((i for i, d in enumerate(dates_sorted) if d >= self.cut_off_date), len(dates_sorted))
        validation_dates = dates_sorted[val_start_idx:]

        print(f"  Validation dates: {len(validation_dates)} (from {validation_dates[0].date() if validation_dates else 'N/A'})")

        # Select donor states once on initial training window (before cutoff)
        initial_window = self.prepared_data[self.prepared_data.index < self.cut_off_date]
        self.donor_map = select_donor_states(
            initial_window,
            top_k=self.donor_top_k,
            min_overlap=self.donor_min_overlap
        )
        print(
            f"  Donor features: top_k={self.donor_top_k}, "
            f"min_overlap={self.donor_min_overlap}, lag_weeks={self.donor_lag_weeks}"
        )

        # Extend data index so exogenous features are available for forecast horizons
        max_date = self.prepared_data.index.max()
        future_dates = pd.date_range(
            start=max_date + pd.Timedelta(weeks=1),
            periods=self.max_horizon,
            freq='W-SAT'
        )
        wide_with_future = self.prepared_data.reindex(self.prepared_data.index.union(future_dates))

        exog_cache: Dict[str, pd.DataFrame] = {}
        for location in self.prepared_data.columns:
            exog_cache[location] = build_location_exog(
                wide_with_future=wide_with_future,
                location=location,
                donor_map=self.donor_map,
                donor_lag_weeks=self.donor_lag_weeks
            )

        for location in self.prepared_data.columns:
            location_data = self.prepared_data[location].dropna()
            location_exog = exog_cache[location]

            if len(location_data) < 40:
                continue

            # Determine ARIMA order using data before cut-off
            pre_cutoff_data = location_data[location_data.index < self.cut_off_date]
            if len(pre_cutoff_data) < 20:
                continue

            # Use first half for order selection
            split_point = len(pre_cutoff_data) // 2
            order_series = pre_cutoff_data.iloc[:split_point]
            order_exog = location_exog.loc[order_series.index]

            best_order = self.determine_best_order(order_series, order_exog, location)
            self.best_orders[location] = best_order

            print(f"  {location}: order={best_order}")

            # Storage for this location
            location_predictions = []
            location_residuals = {h: [] for h in range(1, self.max_horizon + 1)}

            # Generate forecasts for each validation date
            for val_date in validation_dates:
                if val_date not in location_data.index:
                    continue

                # Train on all data up to val_date
                train_data = location_data[location_data.index < val_date]
                if len(train_data) < 20:
                    continue

                train_exog = location_exog.loc[train_data.index]
                forecast_index = pd.date_range(
                    start=val_date + pd.Timedelta(weeks=1),
                    periods=self.max_horizon,
                    freq='W-SAT'
                )
                future_exog = location_exog.reindex(forecast_index)

                # Generate point forecasts
                point_forecasts = self.fit_and_predict(
                    train_data,
                    train_exog,
                    future_exog,
                    best_order,
                    self.max_horizon
                )

                # Store predictions and compute residuals for each horizon
                for h in range(1, self.max_horizon + 1):
                    target_date = val_date + pd.Timedelta(weeks=h)

                    if target_date not in location_data.index:
                        continue

                    actual = location_data.loc[target_date]
                    predicted = point_forecasts[h - 1]

                    if np.isnan(actual) or np.isnan(predicted):
                        continue

                    # Store prediction
                    location_predictions.append({
                        'forecast_date': val_date,
                        'target_date': target_date,
                        'horizon': h,
                        'actual': actual,
                        'predicted': predicted
                    })

                    # Compute and store residual (only for in-season dates)
                    if is_in_flu_season(val_date):
                        residual_log = np.log1p(actual) - np.log1p(predicted)
                        # Cap residuals to avoid extreme outliers
                        residual_log = np.clip(residual_log, -RESIDUAL_CAP, RESIDUAL_CAP)
                        week_of_season = get_week_of_season(val_date)
                        season_year = get_flu_season_year(val_date)

                        residual_record = {
                            'location': location,
                            'horizon': h,
                            'forecast_date': val_date,
                            'target_date': target_date,
                            'week_of_season': week_of_season,
                            'season_year': season_year,
                            'actual': actual,
                            'predicted': predicted,
                            'residual_log': residual_log
                        }
                        location_residuals[h].append(residual_record)
                        all_residuals.append(residual_record)

            # Generate quantile forecasts from prior observable residuals
            forecast_results = []

            for pred in location_predictions:
                val_date = pred['forecast_date']
                h = pred['horizon']
                predicted = pred['predicted']
                # Get residuals for this horizon that are observable before val_date
                target_season_year = get_target_season_year(val_date)
                available_residuals = [
                    r for r in location_residuals[h]
                    if (r['target_date'] < val_date and r['season_year'] == target_season_year)
                ]

                residual_values = np.array([
                    np.clip(r['residual_log'], -RESIDUAL_CAP, RESIDUAL_CAP)
                    for r in available_residuals
                ])

                # Generate quantiles
                if len(residual_values) >= MIN_RESIDUALS_FOR_CONFORMAL:
                    mu_log = np.log1p(predicted)
                    residual_values_centered = residual_values - np.median(residual_values)
                    residual_quantiles = empirical_percentile(residual_values_centered, CDC_QUANTILES)
                    q_log = mu_log + residual_quantiles
                    quantile_forecasts = np.array([max(0.0, np.expm1(val)) for val in q_log])
                    # Keep q0.5 at the point prediction.
                    quantile_forecasts[np.where(CDC_QUANTILES == 0.5)[0][0]] = max(0.0, predicted)
                else:
                    # Fallback: use empirical spread
                    spread = max(predicted * 0.3, 10.0)
                    quantile_forecasts = np.array([
                        max(0.0, stats.norm.ppf(q, loc=predicted, scale=spread))
                        for q in CDC_QUANTILES
                    ])

                forecast_results.append({
                    'reference_date': val_date,
                    'target_date': pred['target_date'],
                    'horizon': h,
                    'quantile_forecasts': quantile_forecasts,
                    'n_residuals': len(residual_values)
                })

            all_forecasts[location] = forecast_results

        # Save residuals if output_dir provided
        if output_dir and all_residuals:
            self.save_residuals(all_residuals, output_dir)

        return all_forecasts, all_residuals

    def save_residuals(self, residuals: List[Dict], output_dir: str) -> None:
        """Save residuals to CSV for later use in prospective forecasts."""
        os.makedirs(output_dir, exist_ok=True)

        # Save per horizon
        for h in range(1, self.max_horizon + 1):
            h_residuals = [r for r in residuals if r['horizon'] == h]
            if h_residuals:
                df = pd.DataFrame(h_residuals)
                output_file = os.path.join(output_dir, RESIDUALS_FILE_PATTERN.format(horizon=h))
                df.to_csv(output_file, index=False)
                print(f"  Saved {len(h_residuals)} residuals to: {output_file}")

    def format_cdc_flusight(self, forecasts: Dict) -> pd.DataFrame:
        """Convert forecasts to CDC FluSight format."""
        cdc_records = []

        for location, forecast_list in forecasts.items():
            fips_code = STATE_TO_FIPS.get(location, location)

            for forecast in forecast_list:
                reference_date = forecast['reference_date']
                target_date = forecast['target_date']
                horizon = forecast['horizon']
                quantile_forecasts = forecast['quantile_forecasts']

                for i, quantile_level in enumerate(self.quantiles):
                    predicted_value = quantile_forecasts[i]

                    cdc_records.append({
                        'reference_date': reference_date.strftime('%Y-%m-%d'),
                        'horizon': horizon - 1,
                        'target': 'wk inc flu hosp',
                        'target_end_date': target_date.strftime('%Y-%m-%d'),
                        'location': fips_code,
                        'output_type': 'quantile',
                        'output_type_id': quantile_level,
                        'value': max(predicted_value, 0.0)
                    })

        return pd.DataFrame(cdc_records)

    def save_forecasts(self, forecasts: Dict, output_dir: str) -> None:
        """Save forecasts in CDC FluSight format."""
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nSaving forecasts to {output_dir}/")

        for h in range(1, self.max_horizon + 1):
            horizon_forecasts = {}
            for location, forecast_list in forecasts.items():
                horizon_list = [f for f in forecast_list if f['horizon'] == h]
                if len(horizon_list) > 0:
                    horizon_forecasts[location] = horizon_list

            if len(horizon_forecasts) > 0:
                forecast_df = self.format_cdc_flusight(horizon_forecasts)

                cdc_column_order = ['reference_date', 'horizon', 'target', 'target_end_date',
                                   'location', 'output_type', 'output_type_id', 'value']
                forecast_df = forecast_df[cdc_column_order]

                output_file = os.path.join(output_dir, f"ARIMA_h{h}_forecasts.csv")
                forecast_df.to_csv(output_file, index=False)
                print(f"  Saved horizon {h} forecasts: {output_file}")

        # Save model information
        model_info = {
            'best_orders': self.best_orders,
            'donor_map': self.donor_map,
            'donor_top_k': self.donor_top_k,
            'donor_min_overlap': self.donor_min_overlap,
            'donor_lag_weeks': self.donor_lag_weeks,
            'cut_off_date': self.cut_off_date,
            'max_horizon': self.max_horizon
        }

        info_file = os.path.join(output_dir, "model_info.pkl")
        with open(info_file, 'wb') as f:
            pickle.dump(model_info, f)
        print(f"  Saved model information: {info_file}")


def main():
    """Main function for ARIMA retrospective forecasting."""

    parser = argparse.ArgumentParser(description='Generate ARIMA Retrospective Forecasts')

    parser.add_argument('--data-file', type=str, required=True,
                       help='Path to input data file')
    parser.add_argument('--cut-off', type=str, required=True,
                       help='Cut-off date for train/test split (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='forecasts/retrospective/arima',
                       help='Output directory for forecasts')
    parser.add_argument('--max-horizon', type=int, default=4,
                       help='Maximum forecast horizon (default: 4)')
    parser.add_argument('--donor-top-k', type=int, default=DEFAULT_DONOR_TOP_K,
                       help='Number of donor states to include as exogenous features')
    parser.add_argument('--donor-min-overlap', type=int, default=DEFAULT_DONOR_MIN_OVERLAP,
                       help='Minimum overlapping weeks for donor-correlation calculation')
    parser.add_argument('--donor-lag-weeks', type=int, default=None,
                       help='Lag (weeks) for donor exogenous features; default=max-horizon')

    args = parser.parse_args()

    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Data file not found: {args.data_file}")

    print(f"{'='*80}")
    print(f"GENERATING ARIMA RETROSPECTIVE FORECASTS")
    print(f"{'='*80}")
    print(f"Data file: {args.data_file}")
    print(f"Cut-off date: {args.cut_off}")
    print(f"Max horizon: {args.max_horizon}")
    print(
        "Donor exog config: "
        f"top_k={args.donor_top_k}, min_overlap={args.donor_min_overlap}, "
        f"lag_weeks={args.donor_lag_weeks if args.donor_lag_weeks is not None else args.max_horizon}"
    )
    print(f"Output: {args.output}")

    forecaster = ARIMAForecaster(
        cut_off_date=args.cut_off,
        max_horizon=args.max_horizon,
        donor_top_k=args.donor_top_k,
        donor_min_overlap=args.donor_min_overlap,
        donor_lag_weeks=args.donor_lag_weeks
    )

    forecaster.load_data(args.data_file)

    # Generate forecasts and save residuals
    forecasts, residuals = forecaster.generate_forecasts(output_dir=args.output)

    # Save forecasts
    forecaster.save_forecasts(forecasts, args.output)

    print(f"\n{'='*80}")
    print(f"ARIMA FORECASTING COMPLETE")
    print(f"{'='*80}")
    print(f"Output directory: {args.output}/")
    print(f"Generated forecasts for {len(forecasts)} locations")

    total_forecasts = sum(len(f) for f in forecasts.values())
    print(f"Total forecast records: {total_forecasts}")

    in_season_residuals = len([r for r in residuals if is_in_flu_season(r['forecast_date'])])
    print(f"In-season residuals saved: {in_season_residuals}")


if __name__ == "__main__":
    main()

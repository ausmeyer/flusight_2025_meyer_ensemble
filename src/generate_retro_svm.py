#!/usr/bin/env python
"""
Generate retrospective SVM forecasts with prediction intervals based on empirical error distributions.

This script:
1. Loads pre-trained SVM models and error distributions
2. Generates retrospective forecasts on test data
3. Creates probabilistic forecasts using the empirical error distributions
4. Generates FluSight-baseline persistence forecasts for comparison
5. Outputs forecasts in CDC FluSight format

Usage:
    python src/generate_retrospective_forecasts.py \
        --hyperparams models/svm_hyperparameters_h1_t100.pkl \
        --data-file data/imputed_and_stitched_hosp_2025-05-24.csv \
        --cut-off 2024-10-01 \
        --output forecasts/retrospective/svm \
        --include-baseline
"""

import os
import sys
import argparse
import warnings
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Import utilities
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
from utils.tabularizer import create_features, create_features_for_prediction
from utils.enhanced_features import create_enhanced_features, create_enhanced_features_for_prediction
from utils.lgbm_timeseries import TimeSeriesDataProcessor

warnings.filterwarnings("ignore")

# CDC FluSight quantiles
CDC_QUANTILES = np.array([
    0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99
])


def persistence_forecast_quantiles(last_value: float, quantiles: np.ndarray,
                                  historical_values: np.ndarray = None,
                                  use_detrended: bool = True) -> np.ndarray:
    """Generate quantile forecasts for persistence baseline using CDC methodology.

    CDC appears to use detrended residuals for uncertainty estimation rather than
    raw h-step differences. This produces wider, more conservative intervals.
    """

    if historical_values is None or len(historical_values) < 10:
        # Fallback: use 25% of last value as uncertainty
        from scipy import stats
        noise_std = max(0.25 * abs(last_value), 5.0)

        quantile_forecasts = []
        for q in quantiles:
            pred = stats.norm.ppf(q, loc=last_value, scale=noise_std)
            quantile_forecasts.append(max(pred, 0.0))  # Truncate negative values

        return np.array(quantile_forecasts)

    if use_detrended:
        # CDC approach: Use detrended residuals for uncertainty
        # This captures full series variability, not just h-step differences
        from scipy import signal, stats
        detrended = signal.detrend(historical_values)
        residual_std = np.std(detrended)

        # Generate quantiles using normal approximation
        quantile_forecasts = []
        for q in quantiles:
            pred = last_value + stats.norm.ppf(q, loc=0, scale=residual_std)
            quantile_forecasts.append(max(pred, 0.0))  # Force non-negative

        return np.array(quantile_forecasts)
    else:
        # Original approach using h-step differences (kept for comparison)
        # Note: This requires historical_differences to be passed instead of values
        # Keeping for backward compatibility but not used by default
        raise ValueError("Non-detrended approach requires historical_differences")


class SVMForecaster:
    """Generate SVM forecasts with empirical prediction intervals."""

    def __init__(self, hyperparams_file: str, cut_off_date: str):
        self.cut_off_date = pd.to_datetime(cut_off_date)
        self.quantiles = CDC_QUANTILES
        self.processor = TimeSeriesDataProcessor()

        # Load hyperparameters
        print(f"Loading hyperparameters from {hyperparams_file}")
        with open(hyperparams_file, 'rb') as f:
            self.hyperparams = pickle.load(f)

        # Extract settings from first location
        first_loc = list(self.hyperparams.keys())[0]
        self.horizon = self.hyperparams[first_loc].get('horizon', 1)
        self.use_log_transform = self.hyperparams[first_loc].get('use_log_transform', False)
        self.use_enhanced_features = 'lags' not in self.hyperparams[first_loc] or self.hyperparams[first_loc]['lags'] is None

        print(f"Loaded hyperparameters for {len(self.hyperparams)} locations")
        print(f"Horizon: {self.horizon}, Enhanced features: {self.use_enhanced_features}")

    def apply_log_transform(self, X: pd.DataFrame, y: np.ndarray = None) -> Tuple[pd.DataFrame, np.ndarray]:
        """Apply log(1+x) transformation to features and optionally targets."""
        if not self.use_log_transform:
            return X, y

        X_transformed = np.log1p(np.maximum(X, 0))
        y_transformed = None
        if y is not None:
            y_transformed = np.log1p(np.maximum(y, 0))

        return X_transformed, y_transformed

    def inverse_log_transform(self, values: np.ndarray) -> np.ndarray:
        """Apply inverse log transformation."""
        if not self.use_log_transform:
            return values
        return np.expm1(values)

    def load_data(self, data_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and split data into train and test sets."""

        print(f"Loading data from {data_file}")

        # Load all data
        full_df = self.processor.load_and_pivot_data(data_file, exclude_locations=None)

        # Split based on cutoff date
        train_df = full_df[full_df["date"] < self.cut_off_date].copy()
        test_df = full_df[full_df["date"] >= self.cut_off_date].copy()

        print(f"Training data: {len(train_df)} samples up to {train_df['date'].max()}")
        print(f"Test data: {len(test_df)} samples from {test_df['date'].min()}")

        return train_df, test_df

    def create_quantile_forecasts(self, point_forecast: float,
                                  residuals: List[float] = None,
                                  residual_scale: str = 'log1p',
                                  fallback_errors: List[float] = None) -> np.ndarray:
        """Convert point forecast to quantile forecasts using residual quantiles.

        Priority: use residuals on transformed scale gathered from dynamic validation.
        Fallback: use old errors (actual - pred) if provided, else calibrated normal.
        """

        residuals = residuals or []
        fallback_errors = fallback_errors or []

        # Helper to generate from transformed residuals
        def from_residuals(pf: float, res: List[float], scale: str) -> np.ndarray:
            if len(res) < 10:
                # Calibrated fallback on transformed scale using sample std
                # Use robust std if possible
                std = np.std(res) if len(res) > 1 else 0.5
                if std <= 1e-8:
                    std = 0.5
                if scale == 'log1p':
                    loc = np.log1p(max(pf, 0))
                    qvals = [max(0, np.expm1(loc + stats.norm.ppf(q, loc=0, scale=std))) for q in self.quantiles]
                elif scale == 'pct':
                    qvals = [max(0, pf * (1.0 + stats.norm.ppf(q, loc=0, scale=std))) for q in self.quantiles]
                else:
                    qvals = [max(0, stats.norm.ppf(q, loc=pf, scale=std)) for q in self.quantiles]
                return np.array(qvals)

            # Empirical residual quantiles (split conformal)
            res_q = np.percentile(res, self.quantiles * 100)
            if scale == 'log1p':
                loc = np.log1p(max(pf, 0))
                qvals = [max(0, np.expm1(loc + rq)) for rq in res_q]
            elif scale == 'pct':
                qvals = [max(0, pf * (1.0 + rq)) for rq in res_q]
            else:
                qvals = [max(0, pf + rq) for rq in res_q]
            return np.array(qvals)

        # Use new residuals if present
        if len(residuals) > 0:
            return from_residuals(point_forecast, residuals, residual_scale)

        # Fallback: use old error list if available
        if len(fallback_errors) >= 10:
            error_quantiles = np.percentile(fallback_errors, self.quantiles * 100)
            return np.array([max(0, point_forecast + eq) for eq in error_quantiles])

        # Last resort: calibrated normal on level (avoid 25% heuristic)
        std = max(np.std(fallback_errors), 5.0) if len(fallback_errors) > 1 else max(0.25 * abs(point_forecast), 5.0)
        return np.array([max(0, stats.norm.ppf(q, loc=point_forecast, scale=std)) for q in self.quantiles])

    def generate_baseline_forecasts(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                                   output_dir: str, max_weeks: int | None = None, horizon: int = None) -> None:
        """Generate FluSight-baseline persistence forecasts for specified horizon(s)."""

        # If horizon is specified, only generate for that horizon, otherwise all
        horizons_to_generate = [horizon] if horizon else [1, 2, 3, 4]

        if len(horizons_to_generate) == 1:
            print(f"\nGenerating FluSight-baseline persistence forecast for horizon {horizons_to_generate[0]}...")
        else:
            print(f"\nGenerating FluSight-baseline persistence forecasts for all horizons...")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get all locations from data
        all_locations = [col for col in train_df.columns if col != 'date']

        # Generate for specified horizon(s)
        for horizon in horizons_to_generate:
            print(f"\nGenerating baseline for horizon {horizon}...")
            all_forecasts = []

            for location in all_locations:
                # Generate forecasts for each test date (default to all test weeks)
                test_dates_all = sorted(test_df['date'].unique())
                if max_weeks is None or max_weeks <= 0:
                    test_dates = test_dates_all
                else:
                    test_dates = test_dates_all[:max_weeks]

                for i, reference_date in enumerate(test_dates):
                    try:
                        # Expanding window: include all training data plus test data up to and including reference_date
                        # For persistence, we need the value AT reference_date to be our "last observed"
                        if i == 0:
                            historical_data = train_df
                        else:
                            test_subset = test_df[test_df['date'] <= reference_date]
                            historical_data = pd.concat([train_df, test_subset])

                        # Get historical values for this location
                        historical_values = historical_data[location].dropna().values

                        if len(historical_values) == 0:
                            continue

                        # Last observed value (persistence)
                        last_value = historical_values[-1]

                        # Generate quantile forecasts using CDC methodology
                        quantile_forecasts = persistence_forecast_quantiles(
                            last_value, self.quantiles, historical_values
                        )

                        # Calculate target date for this specific horizon
                        target_date = reference_date + pd.Timedelta(weeks=horizon)

                        # Store forecast
                        forecast_entry = {
                            'location': location,
                            'reference_date': reference_date,
                            'target_date': target_date,
                            'horizon': horizon,  # Use the loop horizon, not self.horizon
                            'quantile_forecasts': quantile_forecasts
                        }
                        all_forecasts.append(forecast_entry)

                    except Exception as e:
                        continue

            # Convert to CDC FluSight format
            if len(all_forecasts) > 0:
                cdc_df = self.format_cdc_flusight(all_forecasts)
                # Save forecasts for this horizon
                output_file = os.path.join(output_dir, f"FluSight-baseline_h{horizon}.csv")
                cdc_df.to_csv(output_file, index=False)
                print(f"  Saved baseline horizon {horizon}: {len(all_forecasts)} forecasts")

    def generate_forecasts(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                          output_dir: str, max_weeks: int | None = None, include_baseline: bool = False) -> None:
        """Generate retrospective forecasts for all locations."""

        print(f"\nGenerating retrospective SVM forecasts...")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        all_forecasts = []

        for location in self.hyperparams.keys():
            print(f"\nProcessing {location}...")

            # Get model parameters
            loc_params = self.hyperparams[location]
            best_params = loc_params['best_params']
            selected_states = loc_params['selected_states']
            lags = loc_params.get('lags', None)
            error_dist = loc_params.get('error_distribution', {})
            # New residual-based uncertainty (preferred)
            residuals = loc_params.get('residuals', [])
            residual_scale = loc_params.get('residual_scale', 'log1p')
            point_bias = float(loc_params.get('point_bias', 0.0))
            blend_alpha = float(loc_params.get('blend_alpha', 1.0))
            # Backward compatibility
            errors = error_dist.get(self.horizon, []) if isinstance(error_dist, dict) else []

            # Informative logging based on available uncertainty information
            if len(residuals) >= 5:
                print(f"  Using residual-based intervals ({len(residuals)} samples, scale={residual_scale}); bias={point_bias:.2f}, alpha={blend_alpha:.2f}")
            elif len(errors) >= 10:
                print(f"  Using legacy error distribution ({len(errors)} samples)")
            else:
                print(f"  Insufficient residual/error samples; using calibrated fallback")

            # Generate forecasts for each test date (default to all test weeks)
            test_dates_all = sorted(test_df['date'].unique())
            if max_weeks is None or max_weeks <= 0:
                test_dates = test_dates_all
            else:
                test_dates = test_dates_all[:max_weeks]

            for i, reference_date in enumerate(test_dates):
                try:
                    # Expanding window: include all training data plus sufficient test data
                    # IMPORTANT: For DARTS-based tabularizer, feature creation for anchor_date
                    # requires the target series to extend through anchor_date + horizon.
                    # This does NOT leak information into features, but allows the tabularizer
                    # to construct the row aligned exactly at reference_date.
                    # For enhanced features, we only need data up to reference_date.
                    if self.use_enhanced_features:
                        feature_end_date = reference_date
                    else:
                        feature_end_date = reference_date + pd.Timedelta(weeks=self.horizon)

                    test_subset = test_df[test_df['date'] <= feature_end_date]
                    current_data = pd.concat([train_df, test_subset])

                    # Build training features - use data up to the day before reference_date to avoid leakage
                    if self.use_enhanced_features:
                        X_train, y_train, _ = create_enhanced_features(
                            current_data, location, selected_states,
                            end_date=reference_date - pd.Timedelta(days=1),
                            horizon=self.horizon
                        )
                    else:
                        X_train, y_train, _ = create_features(
                            current_data, location, selected_states, lags,
                            end_date=reference_date - pd.Timedelta(days=1),
                            horizon=self.horizon
                        )

                    if len(X_train) < 25:
                        continue

                    # Apply log transformation
                    X_train, y_train = self.apply_log_transform(X_train, y_train)

                    # Standardize features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)

                    # Train model
                    svm = SVR(**best_params)
                    svm.fit(X_train_scaled, y_train)

                    # Build prediction features - use current_data which includes reference_date
                    if self.use_enhanced_features:
                        X_pred, _ = create_enhanced_features_for_prediction(
                            current_data, location, selected_states,
                            anchor_date=reference_date, horizon=self.horizon
                        )
                    else:
                        X_pred, _ = create_features_for_prediction(
                            current_data, location, selected_states, lags,
                            anchor_date=reference_date, horizon=self.horizon
                        )

                    if len(X_pred) == 0:
                        continue

                    # Apply log transformation to prediction features
                    X_pred, _ = self.apply_log_transform(X_pred, None)
                    X_pred_scaled = scaler.transform(X_pred[-1:])

                    # Make point prediction
                    point_pred = svm.predict(X_pred_scaled)[0]
                    point_pred = self.inverse_log_transform(np.array([point_pred]))[0]

                    # Persistence baseline last value at reference_date
                    last_val_series = current_data.loc[current_data['date'] == reference_date, location]
                    last_value = last_val_series.iloc[0] if len(last_val_series) > 0 else point_pred

                    # Bias-correct and blend with persistence to improve MAE
                    point_pred_adj = point_pred + point_bias
                    point_pred_blend = blend_alpha * point_pred_adj + (1.0 - blend_alpha) * last_value

                    # Generate quantile forecasts using residuals if available
                    quantile_forecasts = self.create_quantile_forecasts(
                        point_pred_blend,
                        residuals=residuals,
                        residual_scale=residual_scale,
                        fallback_errors=errors
                    )

                    # Calculate target date
                    target_date = reference_date + pd.Timedelta(weeks=self.horizon)

                    # Store forecast
                    forecast_entry = {
                        'location': location,
                        'reference_date': reference_date,
                        'target_date': target_date,
                        'horizon': self.horizon,
                        'quantile_forecasts': quantile_forecasts
                    }
                    all_forecasts.append(forecast_entry)

                except Exception as e:
                    print(f"  Warning: Failed to generate forecast for {reference_date}: {str(e)}")
                    continue

            print(f"  Generated {len([f for f in all_forecasts if f['location'] == location])} forecasts")

        # Convert to CDC FluSight format
        print(f"\nConverting to CDC FluSight format...")
        cdc_df = self.format_cdc_flusight(all_forecasts)

        # Save SVM forecasts
        output_file = os.path.join(output_dir, f"svm_retrospective_h{self.horizon}.csv")
        cdc_df.to_csv(output_file, index=False)
        print(f"Saved SVM forecasts to: {output_file}")

        # Print summary statistics
        self.print_summary(cdc_df)

        # Generate baseline if requested (only for the current horizon)
        if include_baseline:
            self.generate_baseline_forecasts(train_df, test_df, output_dir, max_weeks, horizon=self.horizon)

    def format_cdc_flusight(self, forecasts: List[Dict]) -> pd.DataFrame:
        """Convert forecasts to CDC FluSight format."""

        # State name to FIPS code mapping
        state_to_fips = {
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

        cdc_records = []

        for forecast in forecasts:
            location = forecast['location']
            reference_date = forecast['reference_date']
            target_date = forecast['target_date']
            horizon = forecast['horizon']
            quantile_forecasts = forecast['quantile_forecasts']

            # Get FIPS code
            fips_code = state_to_fips.get(location, location)

            # Add quantile forecasts
            for q_idx, q in enumerate(self.quantiles):
                cdc_records.append({
                    'reference_date': reference_date.strftime('%Y-%m-%d'),
                    'target': f'{horizon} wk ahead inc hosp',
                    'horizon': horizon,
                    'target_end_date': target_date.strftime('%Y-%m-%d'),
                    'location': fips_code,
                    'location_name': location,
                    'type': 'quantile',
                    'quantile': q,
                    'value': quantile_forecasts[q_idx]
                })

            # Add point forecast (median)
            median_idx = np.where(self.quantiles == 0.5)[0][0]
            cdc_records.append({
                'reference_date': reference_date.strftime('%Y-%m-%d'),
                'target': f'{horizon} wk ahead inc hosp',
                'horizon': horizon,
                'target_end_date': target_date.strftime('%Y-%m-%d'),
                'location': fips_code,
                'location_name': location,
                'type': 'point',
                'quantile': None,
                'value': quantile_forecasts[median_idx]
            })

        return pd.DataFrame(cdc_records)

    def print_summary(self, df: pd.DataFrame) -> None:
        """Print summary statistics of generated forecasts."""

        print(f"\n{'='*60}")
        print("FORECAST SUMMARY")
        print(f"{'='*60}")

        # Summary by location
        print("\nForecasts by location:")
        location_counts = df.groupby('location_name')['reference_date'].nunique()
        for loc, count in location_counts.items():
            print(f"  {loc}: {count} reference dates")

        # Date range
        ref_dates = pd.to_datetime(df['reference_date']).unique()
        print(f"\nReference date range: {ref_dates.min()} to {ref_dates.max()}")

        # Total forecasts
        n_forecasts = len(df[df['type'] == 'point'])
        print(f"Total point forecasts generated: {n_forecasts}")


def main():
    """Main function for retrospective forecasting."""

    parser = argparse.ArgumentParser(description='Generate SVM Retrospective Forecasts')

    parser.add_argument('--hyperparams', type=str, required=True,
                       help='Path to hyperparameters pickle file')
    parser.add_argument('--data-file', type=str, required=True,
                       help='Path to input data file')
    parser.add_argument('--cut-off', type=str, required=True,
                       help='Cut-off date for training (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='forecasts/retrospective/svm',
                       help='Output directory for forecasts')
    parser.add_argument('--max-weeks', type=int, default=None,
                       help='Maximum number of weeks to forecast (default: all test weeks)')
    parser.add_argument('--include-baseline', action='store_true',
                       help='Also generate FluSight-baseline persistence forecasts')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.hyperparams):
        raise FileNotFoundError(f"Hyperparameters file not found: {args.hyperparams}")
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Data file not found: {args.data_file}")

    # Initialize forecaster
    forecaster = SVMForecaster(args.hyperparams, args.cut_off)

    # Load data
    train_df, test_df = forecaster.load_data(args.data_file)

    # Generate forecasts
    forecaster.generate_forecasts(train_df, test_df, args.output, args.max_weeks, args.include_baseline)

    print(f"\n{'='*60}")
    print("RETROSPECTIVE FORECASTING COMPLETED")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

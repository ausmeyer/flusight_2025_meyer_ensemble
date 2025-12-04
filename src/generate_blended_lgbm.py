#!/usr/bin/env python
"""
Generate blended LGBM forecasts by combining t10 and t100 Stage-1 medians with persistence,
then applying conformal prediction intervals.

This script implements the proper blending workflow:
1. Load t10 and t100 hyperparameters
2. Train Stage-1 models for both to get raw medians
3. Blend medians: 0.4*t10 + 0.4*t100 + 0.2*persistence
4. Generate conformal residual quantiles using LAST SEASON's in-season errors
   - Residuals collected from Nov 1 - May 1 of previous season
   - Weighted by calendar week proximity (same week = highest weight)
5. Output as LGBM-blended (the only unbounded LGBM output for ensemble)

Usage:
    # Retrospective
    python src/generate_blended_lgbm.py \
        --data-file data/imputed_and_stitched_hosp_YYYY-MM-DD.csv \
        --cut-off 2024-07-01 \
        --t10-models models/lgbm_enhanced_t10 \
        --t100-models models/lgbm_enhanced_t100 \
        --output forecasts/retrospective/lgbm_blended \
        --mode retrospective

    # Prospective
    python src/generate_blended_lgbm.py \
        --data-file data/imputed_and_stitched_hosp_YYYY-MM-DD.csv \
        --t10-models models/lgbm_enhanced_t10 \
        --t100-models models/lgbm_enhanced_t100 \
        --output forecasts/prospective \
        --mode prospective
"""

import os
import sys
import pickle
import argparse
import warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy.stats import norm

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

# Blending weights
W_T10 = 0.4
W_T100 = 0.4
W_PERS = 0.2

# Conformal settings - using last season's in-season residuals
MIN_RESIDUALS_FOR_CONFORMAL = 4
# Season definition (inclusive): Nov 1 to May 1
SEASON_START_MONTH = 11  # November
SEASON_START_DAY = 1
SEASON_END_MONTH = 5     # May
SEASON_END_DAY = 1
# Week proximity weighting: weight = exp(-WEEK_DECAY * |week_diff|)
# WEEK_DECAY = 0.1 means ~60% weight at 5 weeks apart, ~37% at 10 weeks
WEEK_DECAY = 0.1
# Cap on log-space residuals to avoid extreme outliers skewing the distribution
RESIDUAL_CAP = 2.5
# Residuals file name pattern
RESIDUALS_FILE_PATTERN = 'blended_residuals_h{horizon}.csv'

# FIPS mappings
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


def weighted_percentile(values, weights, quantiles):
    """Compute weighted percentiles using linear interpolation."""
    values = np.array(values)
    weights = np.array(weights)
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    cumsum = np.cumsum(weights)
    cdf = cumsum / cumsum[-1]
    return np.interp(quantiles, cdf, values)


def is_in_flu_season(date: pd.Timestamp) -> bool:
    """Check if a date is within the flu season (Nov 1 - May 1)."""
    month, day = date.month, date.day
    # In season if: Nov 1 onwards OR before May 1
    if month >= SEASON_START_MONTH:
        return True
    if month < SEASON_END_MONTH:
        return True
    if month == SEASON_END_MONTH and day <= SEASON_END_DAY:
        return True
    return False


def get_flu_season_year(date: pd.Timestamp) -> int:
    """Get the flu season year for a date. Season 2023 = Nov 2023 - May 2024."""
    if date.month >= SEASON_START_MONTH:
        return date.year
    else:
        return date.year - 1


def get_week_of_season(date: pd.Timestamp) -> int:
    """Get the week number within the flu season (0 = first week of Nov).

    Returns a week index that allows comparison across seasons.
    Week 0 = Nov 1-7, Week 1 = Nov 8-14, etc.
    Weeks after Dec 31 continue: Jan 1 is approximately week 9.
    """
    season_year = get_flu_season_year(date)
    season_start = pd.Timestamp(year=season_year, month=SEASON_START_MONTH, day=SEASON_START_DAY)
    days_since_start = (date - season_start).days
    if days_since_start < 0:
        # Date is before Nov 1 of its season year, so it's in previous season
        # This shouldn't happen if is_in_flu_season is checked first
        days_since_start += 365
    return days_since_start // 7


def week_proximity_weight(current_week: int, residual_week: int) -> float:
    """Calculate weight based on week proximity within season.

    Uses exponential decay based on week distance.
    Same week = weight 1.0, farther weeks get lower weights.
    """
    week_diff = abs(current_week - residual_week)
    return np.exp(-WEEK_DECAY * week_diff)


def get_last_season_dates(current_date: pd.Timestamp, horizon: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Get the date range for last season's in-season period.

    Returns (season_start, season_end) for the previous flu season.
    """
    current_season = get_flu_season_year(current_date)
    last_season = current_season - 1

    season_start = pd.Timestamp(year=last_season, month=SEASON_START_MONTH, day=SEASON_START_DAY)
    season_end = pd.Timestamp(year=last_season + 1, month=SEASON_END_MONTH, day=SEASON_END_DAY)

    return season_start, season_end


class BlendedLGBMGenerator:
    """Generate blended LGBM forecasts with conformal prediction intervals."""

    def __init__(self, cut_off_date: Optional[str] = None):
        self.cut_off_date = pd.to_datetime(cut_off_date) if cut_off_date else None
        self.data = None
        self.quantiles = CDC_QUANTILES
        self.processor = TimeSeriesDataProcessor()

    def load_data(self, data_file: str) -> None:
        """Load and prepare data (pivoted to wide format with locations as columns)."""
        print(f"\nLoading data from: {data_file}")
        self.data = self.processor.load_and_pivot_data(data_file, exclude_locations=None)
        print(f"  Loaded {len(self.data)} rows")
        print(f"  Date range: {self.data['date'].min().date()} to {self.data['date'].max().date()}")

    def load_hyperparams(self, models_dir: str, horizon: int) -> Dict:
        """Load hyperparameters for a given horizon."""
        hp_file = os.path.join(models_dir, f'two_stage_hyperparameters_h{horizon}.pkl')
        if not os.path.exists(hp_file):
            raise FileNotFoundError(f"Hyperparameters not found: {hp_file}")
        with open(hp_file, 'rb') as f:
            return pickle.load(f)

    def train_stage1_model(self, location: str, hyperparams: Dict, horizon: int,
                           train_end_date: pd.Timestamp) -> Tuple[lgb.Booster, bool, np.ndarray]:
        """Train Stage-1 model for a location and return the booster.

        Returns:
            Tuple of (booster, use_log_transform, last_X_train for feature reference)
        """
        params = hyperparams.get(location, {})
        if not params:
            return None, False, None

        stage1_params = params.get('stage1', {})
        use_log = params.get('use_log_transform', False)
        use_enhanced = params.get('use_enhanced_features', True)
        lags = params.get('lags', [1, 2, 3, 4])
        selected_states = params.get('selected_states', [location])

        # Create features (enhanced_features handles end_date filtering internally)
        if use_enhanced:
            X, y, _ = create_enhanced_features(self.data, location, selected_states,
                                               end_date=train_end_date, horizon=horizon)
        else:
            # For non-enhanced, filter data first
            train_data = self.data[self.data['date'] <= train_end_date].copy()
            X, y, _ = create_features(train_data, location, lags, horizon)

        if len(X) == 0 or len(y) == 0:
            return None, use_log, None

        # Transform if needed
        if use_log:
            X_train = np.log1p(np.maximum(X, 0))
            y_train = np.log1p(np.maximum(y, 0))
        else:
            X_train = X
            y_train = y

        # Train Stage-1
        dtrain = lgb.Dataset(X_train, label=y_train, params={'verbose': -1})
        p = stage1_params['best_params'].copy()
        p['verbose'] = -1
        p['verbosity'] = -1
        booster = lgb.train(p, dtrain, num_boost_round=stage1_params['num_boost_round'], callbacks=[])

        return booster, use_log, X_train

    def get_stage1_prediction(self, booster: lgb.Booster, location: str, hyperparams: Dict,
                              horizon: int, anchor_date: pd.Timestamp, use_log: bool) -> float:
        """Get Stage-1 prediction (raw median) for a specific anchor date."""
        params = hyperparams.get(location, {})
        use_enhanced = params.get('use_enhanced_features', True)
        lags = params.get('lags', [1, 2, 3, 4])
        selected_states = params.get('selected_states', [location])

        if use_enhanced:
            X_pred, _ = create_enhanced_features_for_prediction(
                self.data, location, selected_states, anchor_date=anchor_date, horizon=horizon
            )
        else:
            X_pred, _ = create_features_for_prediction(
                self.data, location, selected_states, lags, anchor_date=anchor_date, horizon=horizon
            )

        if len(X_pred) == 0:
            return None

        if use_log:
            X_pred = np.log1p(np.maximum(X_pred, 0))

        mu_pred_raw = booster.predict(X_pred[-1:])[0]

        if use_log:
            mu_pred = np.expm1(mu_pred_raw)
        else:
            mu_pred = mu_pred_raw

        return max(0.0, mu_pred)

    def generate_retrospective_forecasts(self, t10_hyperparams: Dict, t100_hyperparams: Dict,
                                         horizon: int, output_dir: str = None) -> Dict:
        """Generate retrospective forecasts with blended median.

        This method:
        1. Generates blended forecasts for the validation period
        2. Computes and stores residuals for in-season dates (Nov-May)
        3. Uses stored residuals with week-proximity weighting for conformal intervals

        Args:
            t10_hyperparams: Hyperparameters for t10 models
            t100_hyperparams: Hyperparameters for t100 models
            horizon: Forecast horizon (1-4 weeks)
            output_dir: Directory to save residuals (optional, for caching)

        Returns:
            Dict of forecasts by location
        """
        print(f"\n--- Generating Retrospective Blended Forecasts (Horizon {horizon}) ---")

        all_forecasts = {}
        all_residuals = []  # Collect residuals for saving
        locations = list(set(t10_hyperparams.keys()) & set(t100_hyperparams.keys()))

        # Get validation dates - convert to pandas Timestamps for consistent comparison
        dates_sorted = [pd.Timestamp(d) for d in sorted(self.data['date'].unique())]
        val_start_idx = next((i for i, d in enumerate(dates_sorted) if d >= self.cut_off_date), len(dates_sorted))
        validation_dates = dates_sorted[val_start_idx:]

        print(f"  Locations: {len(locations)}")
        print(f"  Validation dates: {len(validation_dates)} (from {validation_dates[0].date() if validation_dates else 'N/A'})")

        # PASS 1: Generate all blended predictions and collect residuals
        print(f"\n  Pass 1: Generating blended predictions and collecting residuals...")
        location_residuals = {loc: [] for loc in locations}  # residuals per location
        location_predictions = {loc: [] for loc in locations}  # predictions per location

        for loc_idx, location in enumerate(locations):
            print(f"\n  [{loc_idx+1}/{len(locations)}] {location}")

            for val_date in validation_dates:
                try:
                    # Train/retrain models up to val_date (expanding window)
                    train_end = val_date - pd.Timedelta(days=1)

                    # Train Stage-1 for t10
                    t10_booster, t10_use_log, _ = self.train_stage1_model(
                        location, t10_hyperparams, horizon, train_end
                    )
                    # Train Stage-1 for t100
                    t100_booster, t100_use_log, _ = self.train_stage1_model(
                        location, t100_hyperparams, horizon, train_end
                    )

                    if t10_booster is None or t100_booster is None:
                        continue

                    # Get raw medians from both models
                    mu_t10 = self.get_stage1_prediction(
                        t10_booster, location, t10_hyperparams, horizon, val_date, t10_use_log
                    )
                    mu_t100 = self.get_stage1_prediction(
                        t100_booster, location, t100_hyperparams, horizon, val_date, t100_use_log
                    )

                    if mu_t10 is None or mu_t100 is None:
                        continue

                    # Get persistence value
                    last_value_row = self.data.loc[self.data['date'] == val_date, location]
                    last_value = last_value_row.iloc[0] if len(last_value_row) > 0 else (mu_t10 + mu_t100) / 2

                    # Compute blended median
                    blended_mu = W_T10 * mu_t10 + W_T100 * mu_t100 + W_PERS * last_value
                    blended_mu = max(0.0, blended_mu)

                    # Get target date and actual value
                    target_date = val_date + pd.Timedelta(weeks=horizon)
                    actual_row = self.data.loc[self.data['date'] == target_date, location]
                    if len(actual_row) == 0:
                        continue
                    actual_value = actual_row.iloc[0]

                    # Store prediction
                    location_predictions[location].append({
                        'forecast_date': val_date,
                        'target_date': target_date,
                        'actual_value': actual_value,
                        'blended_mu': blended_mu,
                        'mu_t10': mu_t10,
                        'mu_t100': mu_t100,
                        'last_value': last_value
                    })

                    # Compute and store residual (only for in-season dates)
                    if is_in_flu_season(val_date):
                        residual_log = np.log1p(actual_value) - np.log1p(blended_mu)
                        # Cap residuals to avoid extreme outliers
                        residual_log = np.clip(residual_log, -RESIDUAL_CAP, RESIDUAL_CAP)
                        week_of_season = get_week_of_season(val_date)
                        season_year = get_flu_season_year(val_date)

                        residual_record = {
                            'location': location,
                            'forecast_date': val_date,
                            'target_date': target_date,
                            'week_of_season': week_of_season,
                            'season_year': season_year,
                            'actual': actual_value,
                            'predicted': blended_mu,
                            'residual_log': residual_log
                        }
                        location_residuals[location].append(residual_record)
                        all_residuals.append(residual_record)

                except Exception as e:
                    print(f"      [ERROR] {location} @ {val_date.date()}: {str(e)}")
                    continue

            n_in_season = len(location_residuals[location])
            print(f"    {len(location_predictions[location])} predictions, {n_in_season} in-season residuals")

        # Save residuals if output_dir provided
        if output_dir and all_residuals:
            self.save_residuals(all_residuals, horizon, output_dir)

        # PASS 2: Generate quantile forecasts using stored residuals with week-proximity weighting
        print(f"\n  Pass 2: Generating conformal quantiles with week-proximity weighting...")

        for loc_idx, location in enumerate(locations):
            forecast_results = []
            loc_preds = location_predictions[location]
            loc_resids = location_residuals[location]

            for pred in loc_preds:
                val_date = pred['forecast_date']
                blended_mu = pred['blended_mu']
                current_week = get_week_of_season(val_date)

                # Get residuals from the SAME SEASON (for in-season dates) or LAST SEASON
                # For retrospective, use residuals from earlier in the same season
                # that have targets observable before val_date
                available_residuals = [
                    r for r in loc_resids
                    if r['target_date'] < val_date  # must be observable
                ]

                # Apply week-proximity weighting
                weighted_residuals = []
                for r in available_residuals:
                    weight = week_proximity_weight(current_week, r['week_of_season'])
                    # Cap residuals to avoid extreme outliers
                    res_log = np.clip(r['residual_log'], -RESIDUAL_CAP, RESIDUAL_CAP)
                    weighted_residuals.append((res_log, weight))

                # Generate quantiles
                if len(weighted_residuals) >= MIN_RESIDUALS_FOR_CONFORMAL:
                    mu_log = np.log1p(blended_mu)
                    res_vals = [r for r, w in weighted_residuals]
                    res_wts = [w for r, w in weighted_residuals]

                    residual_quantiles = weighted_percentile(res_vals, res_wts, CDC_QUANTILES)
                    q_log = mu_log + residual_quantiles
                    quantile_forecasts = np.array([max(0.0, np.expm1(val)) for val in q_log])
                else:
                    # Fallback: use empirical spread from blended median
                    spread = max(blended_mu * 0.3, 10.0)
                    quantile_forecasts = np.array([
                        max(0.0, blended_mu + norm.ppf(q) * spread) for q in CDC_QUANTILES
                    ])

                forecast_results.append({
                    'forecast_date': val_date,
                    'target_date': pred['target_date'],
                    'actual_value': pred['actual_value'],
                    'quantile_forecasts': quantile_forecasts,
                    'blended_mu': blended_mu,
                    'mu_t10': pred['mu_t10'],
                    'mu_t100': pred['mu_t100'],
                    'last_value': pred['last_value'],
                    'n_residuals': len(weighted_residuals)
                })

            all_forecasts[location] = forecast_results

        return all_forecasts

    def save_residuals(self, residuals: List[Dict], horizon: int, output_dir: str) -> None:
        """Save residuals to CSV for later use in prospective forecasts."""
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(residuals)
        output_file = os.path.join(output_dir, RESIDUALS_FILE_PATTERN.format(horizon=horizon))
        df.to_csv(output_file, index=False)
        print(f"\n  Saved {len(residuals)} residuals to: {output_file}")

    def load_residuals(self, horizon: int, residuals_dir: str) -> pd.DataFrame:
        """Load residuals from CSV."""
        residuals_file = os.path.join(residuals_dir, RESIDUALS_FILE_PATTERN.format(horizon=horizon))
        if not os.path.exists(residuals_file):
            raise FileNotFoundError(f"Residuals file not found: {residuals_file}")
        df = pd.read_csv(residuals_file, parse_dates=['forecast_date', 'target_date'])
        print(f"  Loaded {len(df)} residuals from: {residuals_file}")
        return df

    def generate_prospective_forecasts(self, t10_hyperparams: Dict, t100_hyperparams: Dict,
                                        horizon: int, residuals_dir: str = None) -> Dict:
        """Generate prospective forecasts with blended median + conformal CIs.

        Uses pre-computed residuals from retrospective forecasts for conformal intervals.

        Args:
            t10_hyperparams: Hyperparameters for t10 models
            t100_hyperparams: Hyperparameters for t100 models
            horizon: Forecast horizon (1-4 weeks)
            residuals_dir: Directory containing saved residuals from retrospective run

        Returns:
            Dict of forecasts by location
        """
        print(f"\n--- Generating Prospective Blended Forecasts (Horizon {horizon}) ---")

        all_forecasts = {}
        locations = list(set(t10_hyperparams.keys()) & set(t100_hyperparams.keys()))
        last_date = pd.Timestamp(self.data['date'].max())

        print(f"  Locations: {len(locations)}")
        print(f"  Forecast from: {last_date.date()}")

        # Load pre-computed residuals if available
        residuals_df = None
        if residuals_dir:
            try:
                residuals_df = self.load_residuals(horizon, residuals_dir)
            except FileNotFoundError as e:
                print(f"  Warning: {e}")
                print(f"  Will use fallback spread for conformal intervals")

        current_week = get_week_of_season(last_date)

        for loc_idx, location in enumerate(locations):
            try:
                # Train Stage-1 for t10 on all data
                t10_booster, t10_use_log, _ = self.train_stage1_model(
                    location, t10_hyperparams, horizon, last_date
                )
                # Train Stage-1 for t100 on all data
                t100_booster, t100_use_log, _ = self.train_stage1_model(
                    location, t100_hyperparams, horizon, last_date
                )

                if t10_booster is None or t100_booster is None:
                    print(f"  [{loc_idx+1}/{len(locations)}] {location}: Skipped (no model)")
                    continue

                # Get raw medians
                mu_t10 = self.get_stage1_prediction(
                    t10_booster, location, t10_hyperparams, horizon, last_date, t10_use_log
                )
                mu_t100 = self.get_stage1_prediction(
                    t100_booster, location, t100_hyperparams, horizon, last_date, t100_use_log
                )

                if mu_t10 is None or mu_t100 is None:
                    print(f"  [{loc_idx+1}/{len(locations)}] {location}: Skipped (no prediction)")
                    continue

                # Get persistence
                last_value_row = self.data.loc[self.data['date'] == last_date, location]
                last_value = last_value_row.iloc[0] if len(last_value_row) > 0 else (mu_t10 + mu_t100) / 2

                # Blend
                blended_mu = W_T10 * mu_t10 + W_T100 * mu_t100 + W_PERS * last_value
                blended_mu = max(0.0, blended_mu)

                target_date = last_date + pd.Timedelta(weeks=horizon)

                # Get residuals for this location from saved data
                weighted_residuals = []
                if residuals_df is not None:
                    loc_residuals = residuals_df[residuals_df['location'] == location]

                    # Apply week-proximity weighting
                    for _, row in loc_residuals.iterrows():
                        weight = week_proximity_weight(current_week, row['week_of_season'])
                        # Cap residuals to avoid extreme outliers
                        res_log = np.clip(row['residual_log'], -RESIDUAL_CAP, RESIDUAL_CAP)
                        weighted_residuals.append((res_log, weight))

                # Generate quantiles
                if len(weighted_residuals) >= MIN_RESIDUALS_FOR_CONFORMAL:
                    mu_log = np.log1p(blended_mu)

                    res_vals = [r for r, w in weighted_residuals]
                    res_wts = [w for r, w in weighted_residuals]

                    residual_quantiles = weighted_percentile(res_vals, res_wts, CDC_QUANTILES)
                    q_log = mu_log + residual_quantiles
                    quantile_forecasts = np.array([max(0.0, np.expm1(val)) for val in q_log])
                    print(f"  [{loc_idx+1}/{len(locations)}] {location}: blended_mu={blended_mu:.1f}, {len(weighted_residuals)} residuals (week {current_week})")
                else:
                    # Fallback: use empirical spread from blended median
                    spread = max(blended_mu * 0.3, 10.0)
                    quantile_forecasts = np.array([
                        max(0.0, blended_mu + norm.ppf(q) * spread) for q in CDC_QUANTILES
                    ])
                    print(f"  [{loc_idx+1}/{len(locations)}] {location}: blended_mu={blended_mu:.1f}, fallback (only {len(weighted_residuals)} residuals)")

                all_forecasts[location] = {
                    'forecast_date': last_date,
                    'target_date': target_date,
                    'quantile_forecasts': quantile_forecasts,
                    'blended_mu': blended_mu,
                    'mu_t10': mu_t10,
                    'mu_t100': mu_t100,
                    'last_value': last_value,
                    'n_residuals': len(weighted_residuals)
                }

            except Exception as e:
                print(f"  [{loc_idx+1}/{len(locations)}] {location}: ERROR - {str(e)}")
                continue

        return all_forecasts

    def save_retrospective_forecasts(self, forecasts: Dict, horizon: int, output_dir: str) -> None:
        """Save retrospective forecasts in CDC format."""
        os.makedirs(output_dir, exist_ok=True)

        cdc_records = []
        for location, forecast_list in forecasts.items():
            fips_code = STATE_TO_FIPS.get(location, location)

            for forecast in forecast_list:
                for i, q in enumerate(CDC_QUANTILES):
                    cdc_records.append({
                        'reference_date': forecast['forecast_date'].strftime('%Y-%m-%d'),
                        'horizon': horizon - 1,
                        'target': 'wk inc flu hosp',
                        'target_end_date': forecast['target_date'].strftime('%Y-%m-%d'),
                        'location': fips_code,
                        'output_type': 'quantile',
                        'output_type_id': q,
                        'value': max(0.0, forecast['quantile_forecasts'][i])
                    })

        df = pd.DataFrame(cdc_records)
        col_order = ['reference_date', 'horizon', 'target', 'target_end_date',
                     'location', 'output_type', 'output_type_id', 'value']
        df = df[col_order]

        output_file = os.path.join(output_dir, f'LGBM-blended_h{horizon}_forecasts.csv')
        df.to_csv(output_file, index=False)
        print(f"\n  Saved: {output_file}")

    def save_prospective_forecasts(self, forecasts: Dict, horizon: int, output_dir: str) -> None:
        """Save prospective forecasts in CDC format."""
        os.makedirs(output_dir, exist_ok=True)

        if not forecasts:
            print("  No forecasts to save")
            return

        # Get timestamp from first forecast
        first_forecast = next(iter(forecasts.values()))
        timestamp = first_forecast['forecast_date'].strftime('%Y%m%d')

        cdc_records = []
        for location, forecast in forecasts.items():
            fips_code = STATE_TO_FIPS.get(location, location)

            for i, q in enumerate(CDC_QUANTILES):
                cdc_records.append({
                    'reference_date': forecast['forecast_date'].strftime('%Y-%m-%d'),
                    'horizon': horizon - 1,
                    'target': 'wk inc flu hosp',
                    'target_end_date': forecast['target_date'].strftime('%Y-%m-%d'),
                    'location': fips_code,
                    'output_type': 'quantile',
                    'output_type_id': q,
                    'value': max(0.0, forecast['quantile_forecasts'][i])
                })

        df = pd.DataFrame(cdc_records)
        col_order = ['reference_date', 'horizon', 'target', 'target_end_date',
                     'location', 'output_type', 'output_type_id', 'value']
        df = df[col_order]

        output_file = os.path.join(output_dir, f'LGBM-blended_h{horizon}_prospective_{timestamp}.csv')
        df.to_csv(output_file, index=False)
        print(f"\n  Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate Blended LGBM Forecasts')

    parser.add_argument('--data-file', type=str, required=True,
                        help='Path to input data file')
    parser.add_argument('--t10-models', type=str, required=True,
                        help='Directory containing t10 model hyperparameters')
    parser.add_argument('--t100-models', type=str, required=True,
                        help='Directory containing t100 model hyperparameters')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for forecasts')
    parser.add_argument('--mode', type=str, choices=['retrospective', 'prospective'], required=True,
                        help='Mode: retrospective or prospective')
    parser.add_argument('--cut-off', type=str, default=None,
                        help='Cut-off date for retrospective mode (YYYY-MM-DD)')
    parser.add_argument('--horizons', type=str, default='1,2,3,4',
                        help='Comma-separated list of horizons (default: 1,2,3,4)')
    parser.add_argument('--residuals-dir', type=str, default=None,
                        help='Directory containing residuals (for prospective mode, defaults to --output)')

    args = parser.parse_args()

    if args.mode == 'retrospective' and not args.cut_off:
        raise ValueError("--cut-off is required for retrospective mode")

    # Default residuals-dir to output dir if not specified
    residuals_dir = args.residuals_dir if args.residuals_dir else args.output

    print(f"{'='*80}")
    print(f"BLENDED LGBM FORECAST GENERATION")
    print(f"{'='*80}")
    print(f"Mode: {args.mode}")
    print(f"Data: {args.data_file}")
    print(f"T10 models: {args.t10_models}")
    print(f"T100 models: {args.t100_models}")
    print(f"Output: {args.output}")
    if args.mode == 'prospective':
        print(f"Residuals dir: {residuals_dir}")

    horizons = [int(h.strip()) for h in args.horizons.split(',')]

    generator = BlendedLGBMGenerator(cut_off_date=args.cut_off)
    generator.load_data(args.data_file)

    for horizon in horizons:
        print(f"\n{'='*60}")
        print(f"HORIZON {horizon}")
        print(f"{'='*60}")

        # Load hyperparameters
        try:
            t10_hp = generator.load_hyperparams(args.t10_models, horizon)
            t100_hp = generator.load_hyperparams(args.t100_models, horizon)
        except FileNotFoundError as e:
            print(f"  Skipping horizon {horizon}: {e}")
            continue

        if args.mode == 'retrospective':
            # Generate forecasts and save residuals
            forecasts = generator.generate_retrospective_forecasts(t10_hp, t100_hp, horizon, output_dir=args.output)
            generator.save_retrospective_forecasts(forecasts, horizon, args.output)
        else:
            # Load residuals from retrospective run and generate forecasts
            forecasts = generator.generate_prospective_forecasts(t10_hp, t100_hp, horizon, residuals_dir=residuals_dir)
            generator.save_prospective_forecasts(forecasts, horizon, args.output)

    print(f"\n{'='*80}")
    print(f"BLENDED LGBM GENERATION COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

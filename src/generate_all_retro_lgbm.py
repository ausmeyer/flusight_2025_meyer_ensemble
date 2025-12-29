#!/usr/bin/env python
"""
Generate retrospective out-of-sample forecasts for models.

This script processes trained models in a models directory and generates
probabilistic forecasts in CDC FluSight format for each model configuration.

Key Features:
- Processes all model folders in the provided models directory
- Extracts horizon from hyperparameter filenames
- Generates forecasts for each model/horizon combination
- Outputs to forecasts/retrospective/{model_folder}/
- Creates both model and baseline forecasts
- No summary statistics (unlike the single model script)

Usage:
    python src/generate_all_retrospective_forecasts.py \
        --data-file data/imputed_and_stitched_hosp_2025-05-24.csv \
        --cut-off 2023-07-01
"""

import os
import sys
import glob
import pickle
import argparse
import warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from contextlib import redirect_stdout
from io import StringIO
from collections import defaultdict

# Import utilities
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
from utils.tabularizer import create_features, create_features_for_prediction
from utils.enhanced_features import create_enhanced_features, create_enhanced_features_for_prediction

# Import from utils directory  
from utils.lgbm_timeseries import TimeSeriesDataProcessor

# Import LightGBMLSS
try:
    import lightgbmlss
    from lightgbmlss.model import LightGBMLSS
    from lightgbmlss.distributions.Gaussian import Gaussian
    from utils.distributions import GaussianFrozenLoc, GaussianFrozenLocBounded, GaussianFrozenLocBoundedWide
except ImportError:
    raise ImportError("lightgbmlss is required. Install with: pip install lightgbmlss")

warnings.filterwarnings("ignore")

# CDC FluSight quantiles
CDC_QUANTILES = np.array([
    0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99
])


def weighted_percentile(values, weights, quantiles):
    """Compute weighted percentiles using linear interpolation.

    Args:
        values: Array of values
        weights: Array of weights (will be normalized)
        quantiles: Array of quantile levels (0-1)

    Returns:
        Array of interpolated values at the requested quantile levels
    """
    values = np.array(values)
    weights = np.array(weights)
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    # Normalize weights to get CDF
    cumsum = np.cumsum(weights)
    cdf = cumsum / cumsum[-1]
    # Interpolate quantiles
    return np.interp(quantiles, cdf, values)


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
        raise ValueError("Non-detrended approach requires historical_differences")


class BatchRetrospectiveForecastGenerator:
    """
    Generate retrospective probabilistic forecasts for all saved models.
    """
    
    def __init__(self, cut_off_date: str, quantiles: np.ndarray = None):
        self.cut_off_date = pd.to_datetime(cut_off_date)
        self.quantiles = quantiles if quantiles is not None else CDC_QUANTILES
        self.processor = TimeSeriesDataProcessor()
        self.data = None
        
    def load_data(self, data_file: str) -> None:
        """Load and prepare data for forecasting."""
        
        print(f"Loading data from {data_file}")
        
        # Load all data
        self.data = self.processor.load_and_pivot_data(data_file, exclude_locations=None)
        
        # Get training data (all data before cutoff)
        train_data = self.data[self.data['date'] < self.cut_off_date]
        
        # Get validation period info
        validation_data = self.data[self.data['date'] >= self.cut_off_date]
        val_weeks = len(validation_data)
        
        print(f"Data loaded: {len(self.data)} total samples")
        print(f"Training data: up to {(train_data['date'].max() if len(train_data) > 0 else 'N/A')}")
        print(f"Cut-off date: {self.cut_off_date.strftime('%Y-%m-%d')}")
        print(f"Validation period: {val_weeks} weeks from {self.cut_off_date.strftime('%Y-%m-%d')} onwards")
        
    def find_model_configurations(self, models_dir: str = "models") -> List[Dict]:
        """Find all model configurations in the models directory."""
        
        print(f"\nSearching for model configurations in {models_dir}/")
        
        configurations = []
        
        # Always search for hyperparameters at the root of models_dir
        root_folder_name = os.path.basename(models_dir.rstrip('/'))
        root_hyperparams = glob.glob(os.path.join(models_dir, "two_stage_hyperparameters_h*.pkl"))
        for hyperparam_file in sorted(root_hyperparams):
            filename = os.path.basename(hyperparam_file)
            if 'parameters_h' not in filename:
                print(f"  Warning: Could not find 'parameters_h' pattern in {filename}, skipping")
                continue
            horizon_str = filename.split('parameters_h')[1].split('.pkl')[0]
            try:
                horizon = int(horizon_str)
            except ValueError:
                print(f"  Warning: Could not parse horizon from {filename}, skipping")
                continue
            configurations.append({
                'folder_name': root_folder_name,
                'hyperparam_file': hyperparam_file,
                'horizon': horizon
            })
            print(f"  Found: {root_folder_name}/h{horizon}")

        # Also search immediate subdirectories for hyperparameters (if any)
        model_folders = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
        for folder_name in sorted(model_folders):
            folder_path = os.path.join(models_dir, folder_name)
            hyperparam_files = glob.glob(os.path.join(folder_path, "two_stage_hyperparameters_h*.pkl"))
            for hyperparam_file in sorted(hyperparam_files):
                filename = os.path.basename(hyperparam_file)
                if 'parameters_h' not in filename:
                    print(f"  Warning: Could not find 'parameters_h' pattern in {filename}, skipping")
                    continue
                horizon_str = filename.split('parameters_h')[1].split('.pkl')[0]
                try:
                    horizon = int(horizon_str)
                except ValueError:
                    print(f"  Warning: Could not parse horizon from {filename}, skipping")
                    continue
                configurations.append({
                    'folder_name': folder_name,
                    'hyperparam_file': hyperparam_file,
                    'horizon': horizon
                })
                print(f"  Found: {folder_name}/h{horizon}")
        
        print(f"Total configurations found: {len(configurations)}")
        return configurations
        
    def process_single_configuration(self, config: Dict, output_base_dir: str, models_base_dir: str) -> None:
        """Process a single model configuration and generate forecasts."""
        
        folder_name = config['folder_name']
        hyperparam_file = config['hyperparam_file']
        horizon = config['horizon']
        
        print(f"\n{'='*60}")
        print(f"Processing: {folder_name} (horizon={horizon})")
        print(f"{'='*60}")
        
        # Create output directory
        output_dir = os.path.join(output_base_dir, folder_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Load hyperparameters
        print(f"Loading hyperparameters from {hyperparam_file}")
        with open(hyperparam_file, 'rb') as f:
            hyperparams = pickle.load(f)
        
        locations = list(hyperparams.keys())
        print(f"Loaded hyperparameters for {len(locations)} locations")
        
        # Extract log transform and bounded sigma settings
        use_log_transform = {}
        for location in locations:
            use_log_transform[location] = hyperparams[location].get('use_log_transform', False)

        # Detect bounded sigma mode from hyperparameters
        first_location = locations[0]
        is_bounded = hyperparams[first_location].get('bounded_sigma', False)
        is_bounded_wide = hyperparams[first_location].get('bounded_sigma_wide', False)
        if is_bounded_wide:
            print(f"Model type: BOUNDED SIGMA WIDE (log-space quantiles, sigma in [0.1, 0.8])")
        elif is_bounded:
            print(f"Model type: BOUNDED SIGMA (log-space quantiles, sigma in [0.15, 0.45])")
        else:
            print(f"Model type: STANDARD (conformal residual quantiles)")

        # Load models
        stage1_models = {}
        stage2_models = {}

        print("Loading trained models...")
        for location in locations:
            # Load Stage 1 model
            booster_file = os.path.join(models_base_dir, "point_mu", f"{location}_booster.txt")
            # Try horizon-aware first
            booster_file_h = os.path.join(models_base_dir, "point_mu", f"{location}_h{horizon}_booster.txt")
            if os.path.exists(booster_file_h):
                booster_file = booster_file_h
            if not os.path.exists(booster_file):
                print(f"  Warning: Stage 1 model not found: {booster_file}, skipping {location}")
                continue
            
            stage1_models[location] = lgb.Booster(model_file=booster_file)
            
            # Load Stage 2 model
            model_file = os.path.join(models_base_dir, "scale_sigma", f"{location}_lgbmlss_model.pkl")
            model_file_h = os.path.join(models_base_dir, "scale_sigma", f"{location}_h{horizon}_lgbmlss_model.pkl")
            if os.path.exists(model_file_h):
                model_file = model_file_h
            if not os.path.exists(model_file):
                print(f"  Warning: Stage 2 model not found: {model_file}, skipping {location}")
                continue
                
            with open(model_file, 'rb') as f:
                stage2_models[location] = pickle.load(f)
        
        print(f"Loaded models for {len(stage1_models)} locations")
        
        # If models are not available on disk, train from hyperparameters using training data
        if len(stage1_models) == 0 or len(stage2_models) == 0:
            print("No pre-trained models found on disk. Training from hyperparameters...")
            train_data = self.data[self.data['date'] < self.cut_off_date]
            train_end_date = train_data['date'].max() if len(train_data) > 0 else self.cut_off_date
            for location in locations:
                try:
                    params = hyperparams[location]
                    stage1_params = params['stage1']
                    stage2_params = params['stage2']
                    lags = stage1_params['lags']
                    selected_states = stage1_params['selected_states']

                    # Build training features up to last training date
                    if lags is None:
                        X_train, y_train, _ = create_enhanced_features(
                            self.data, location, selected_states,
                            end_date=train_end_date, horizon=horizon
                        )
                    else:
                        X_train, y_train, _ = create_features(
                            self.data, location, selected_states, lags,
                            end_date=train_end_date, horizon=horizon
                        )
                    if len(X_train) < 50:
                        continue
                    # Log transform if enabled
                    use_log = use_log_transform.get(location, False)
                    if use_log:
                        X_train_transformed = np.log1p(np.maximum(X_train, 0))
                        y_train_transformed = np.log1p(np.maximum(y_train, 0))
                    else:
                        X_train_transformed = X_train
                        y_train_transformed = y_train
                    # Train Stage 1
                    dtrain1 = lgb.Dataset(X_train_transformed, label=y_train_transformed, params={'verbose': -1})
                    p1 = stage1_params['best_params'].copy(); p1['verbose'] = -1; p1['verbosity'] = -1
                    stage1_model = lgb.train(p1, dtrain1, num_boost_round=stage1_params['num_boost_round'], callbacks=[])
                    # Train Stage 2 with frozen mu
                    mu_predictions = stage1_model.predict(X_train_transformed)
                    init_score = np.column_stack([mu_predictions, np.zeros_like(mu_predictions)]).ravel(order='F')
                    dtrain2 = lgb.Dataset(X_train_transformed, label=y_train_transformed, init_score=init_score, params={'verbose': -1})
                    if is_bounded_wide:
                        dist = GaussianFrozenLocBoundedWide()
                    elif is_bounded:
                        dist = GaussianFrozenLocBounded()
                    else:
                        dist = GaussianFrozenLoc()
                    stage2_model = LightGBMLSS(dist)
                    # Pre-set start_values to prevent LightGBMLSS from overwriting our init_score
                    mu_mean = init_score[::2].mean()
                    stage2_model.start_values = np.array([mu_mean, 0.0], dtype=np.float32)
                    p2 = stage2_params['best_params'].copy(); p2['verbose'] = -1; p2['verbosity'] = -1
                    stage2_model.train(p2, dtrain2, num_boost_round=stage2_params['num_boost_round'])
                    # Store
                    stage1_models[location] = stage1_model
                    stage2_models[location] = stage2_model
                except Exception as e:
                    print(f"  Warning: Training failed for {location}: {e}")
        
        # Generate model forecasts
        model_forecasts = self.generate_model_forecasts(
            hyperparams, stage1_models, stage2_models, use_log_transform, horizon,
            is_bounded=is_bounded, is_bounded_wide=is_bounded_wide
        )
        
        # Generate persistence forecasts
        persistence_forecasts = self.generate_persistence_forecasts(
            list(hyperparams.keys()), horizon
        )
        
        # Save forecasts (without summary)
        self.save_forecasts(model_forecasts, persistence_forecasts, horizon, output_dir)

    def generate_model_forecasts(self, hyperparams: Dict, stage1_models: Dict,
                                 stage2_models: Dict, use_log_transform: Dict,
                                 horizon: int, is_bounded: bool = False,
                                 is_bounded_wide: bool = False) -> Dict:
        """Generate two-stage model forecasts using expanding window validation.

        If is_bounded or is_bounded_wide: Use Stage 2 sigma directly in log space (skip conformal).
        Otherwise: Use conformal residual quantiles (legacy behavior).
        """

        if is_bounded or is_bounded_wide:
            print(f"\nGenerating two-stage model forecasts (BOUNDED: log-space sigma)...")
        else:
            print(f"\nGenerating two-stage model forecasts (STANDARD: conformal residual quantiles)...")

        all_forecasts = {}

        # Conformal residual quantiles settings (for non-bounded models only)
        RESIDUAL_LOOKBACK = 6  # Keep last 6 weeks of residuals (was 12)
        MIN_RESIDUALS_FOR_CONFORMAL = 6  # Minimum residuals needed (keep in sync with lookback)
        EWMA_ALPHA = 0.5  # Decay factor for residual weights (0 < alpha < 1, smaller = faster decay)

        # Blending weights for unbounded models (t10, t100, persistence)
        # These are static weights that sum to 1.0
        W_T10 = 0.4    # Weight for t10 model median
        W_T100 = 0.4   # Weight for t100 model median
        W_PERS = 0.2   # Weight for persistence (last observed value)

        # Try to load pre-computed medians from other model variants for blending
        # This allows us to blend t10 + t100 + persistence even when running one model at a time
        other_model_medians = {}  # Dict of (location, target_date) -> {'t10': val, 't100': val}

        if not is_bounded:
            # Attempt to load existing retrospective forecasts from both t10 and t100
            for model_variant in ['lgbm_enhanced_t10', 'lgbm_enhanced_t100']:
                forecast_file = os.path.join(
                    os.path.dirname(os.path.dirname(hyperparams[list(hyperparams.keys())[0]].get('_source_file', ''))),
                    'forecasts', 'retrospective', model_variant,
                    f'TwoStage-FrozenMu_h{horizon}_forecasts.csv'
                )
                # Try standard path structure
                for base_path in ['forecasts/retrospective', '../forecasts/retrospective']:
                    candidate = os.path.join(base_path, model_variant, f'TwoStage-FrozenMu_h{horizon}_forecasts.csv')
                    if os.path.exists(candidate):
                        forecast_file = candidate
                        break

                if os.path.exists(forecast_file):
                    try:
                        df = pd.read_csv(forecast_file)
                        # Extract median (quantile 0.5)
                        median_df = df[df['output_type_id'] == 0.5].copy()
                        median_df['target_end_date'] = pd.to_datetime(median_df['target_end_date'])

                        # Build lookup: FIPS -> state name
                        fips_to_state = {
                            '01': 'Alabama', '02': 'Alaska', '04': 'Arizona', '05': 'Arkansas',
                            '06': 'California', '08': 'Colorado', '09': 'Connecticut', '10': 'Delaware',
                            '11': 'District of Columbia', '12': 'Florida', '13': 'Georgia', '15': 'Hawaii',
                            '16': 'Idaho', '17': 'Illinois', '18': 'Indiana', '19': 'Iowa',
                            '20': 'Kansas', '21': 'Kentucky', '22': 'Louisiana', '23': 'Maine',
                            '24': 'Maryland', '25': 'Massachusetts', '26': 'Michigan', '27': 'Minnesota',
                            '28': 'Mississippi', '29': 'Missouri', '30': 'Montana', '31': 'Nebraska',
                            '32': 'Nevada', '33': 'New Hampshire', '34': 'New Jersey', '35': 'New Mexico',
                            '36': 'New York', '37': 'North Carolina', '38': 'North Dakota', '39': 'Ohio',
                            '40': 'Oklahoma', '41': 'Oregon', '42': 'Pennsylvania', '72': 'Puerto Rico',
                            '44': 'Rhode Island', '45': 'South Carolina', '46': 'South Dakota', '47': 'Tennessee',
                            '48': 'Texas', '49': 'Utah', '50': 'Vermont', '51': 'Virginia',
                            '53': 'Washington', '54': 'West Virginia', '55': 'Wisconsin', '56': 'Wyoming',
                            'US': 'US'
                        }

                        variant_key = 't10' if 't10' in model_variant else 't100'
                        for _, row in median_df.iterrows():
                            loc_fips = str(row['location']).zfill(2) if row['location'] != 'US' else 'US'
                            loc_name = fips_to_state.get(loc_fips, loc_fips)
                            target_date = row['target_end_date']
                            key = (loc_name, target_date)
                            if key not in other_model_medians:
                                other_model_medians[key] = {}
                            other_model_medians[key][variant_key] = row['value']

                        print(f"    Loaded {len(median_df)} medians from {model_variant} for blending")
                    except Exception as e:
                        print(f"    Warning: Could not load {forecast_file} for blending: {e}")

        for location in hyperparams.keys():
            if location not in stage1_models or location not in stage2_models:
                continue

            print(f"  Generating forecasts for {location}")

            # Get model parameters
            params = hyperparams[location]
            stage1_params = params['stage1']
            stage2_params = params['stage2']
            lags = stage1_params['lags']
            selected_states = stage1_params['selected_states']

            # Check if using enhanced features
            if lags is None:
                use_enhanced = True
            else:
                use_enhanced = False
                if len(lags) == 0:
                    raise ValueError(f"{location}: no lags specified")
                if min(lags) < 1:
                    raise ValueError(f"{location}: lag_0 detected in {lags}")

            # Get training data last date
            train_data = self.data[self.data['date'] < self.cut_off_date]
            first_anchor_date = train_data['date'].max() if len(train_data) > 0 else self.cut_off_date

            # Get validation dates
            val_data = self.data[self.data['date'] >= first_anchor_date].copy()
            val_dates = sorted(val_data['date'].unique())
            all_dates = sorted(self.data['date'].unique())

            # Stop early enough to have target dates
            val_dates_to_use = val_dates[:-horizon] if len(val_dates) > horizon else []

            forecast_results = []
            total_dates = len(val_dates_to_use)

            # Rolling residual store for this location (log1p space)
            # Each entry is (residual_log, weight) for EWMA weighting
            residuals_log = []  # List of (residual, weight) tuples

            # Optional: Season bucket stores for extra stability
            # Key: is_high_season boolean (True = Oct-Mar, False = Apr-Sep)
            residual_stores_by_season = defaultdict(list)

            # Expanding window validation
            for i, val_date in enumerate(val_dates_to_use):
                try:
                    # Expanding window: include all previous validation points
                    if i == 0:
                        train_end_date = first_anchor_date
                    else:
                        train_end_date = val_date - pd.Timedelta(days=1)

                    # Create training features
                    if use_enhanced:
                        X_train, y_train, _ = create_enhanced_features(
                            self.data, location, selected_states,
                            end_date=train_end_date, horizon=horizon
                        )
                    else:
                        X_train, y_train, _ = create_features(
                            self.data, location, selected_states, lags,
                            end_date=train_end_date, horizon=horizon
                        )

                    if len(X_train) < 50:
                        continue

                    # Apply log transformation if enabled
                    if use_log_transform.get(location, False):
                        X_train_transformed = np.log1p(np.maximum(X_train, 0))
                        y_train_transformed = np.log1p(np.maximum(y_train, 0))
                    else:
                        X_train_transformed = X_train
                        y_train_transformed = y_train

                    # Re-train models on expanding window
                    with redirect_stdout(StringIO()):
                        # Re-train Stage 1 model
                        dtrain1 = lgb.Dataset(X_train_transformed, label=y_train_transformed)
                        temp_stage1 = lgb.train(
                            stage1_params['best_params'],
                            dtrain1,
                            num_boost_round=stage1_params['num_boost_round'],
                            callbacks=[]
                        )

                        # Get μ predictions for training data (needed for Stage 2 init_score)
                        mu_predictions = temp_stage1.predict(X_train_transformed)

                        # Re-train Stage 2 model with frozen μ (only used as fallback)
                        init_score = np.column_stack([
                            mu_predictions,
                            np.zeros_like(mu_predictions)
                        ]).ravel(order='F')

                        dtrain2 = lgb.Dataset(X_train_transformed, label=y_train_transformed, init_score=init_score)
                        if is_bounded_wide:
                            dist = GaussianFrozenLocBoundedWide()
                        elif is_bounded:
                            dist = GaussianFrozenLocBounded()
                        else:
                            dist = GaussianFrozenLoc()
                        temp_stage2 = LightGBMLSS(dist)
                        # Pre-set start_values to prevent LightGBMLSS from overwriting our init_score
                        mu_mean = init_score[::2].mean()
                        temp_stage2.start_values = np.array([mu_mean, 0.0], dtype=np.float32)
                        temp_stage2.train(
                            stage2_params['best_params'],
                            dtrain2,
                            num_boost_round=stage2_params['num_boost_round']
                        )

                    # Create prediction features for current val_date
                    if use_enhanced:
                        X_pred, _ = create_enhanced_features_for_prediction(
                            self.data, location, selected_states,
                            anchor_date=val_date, horizon=horizon
                        )
                    else:
                        X_pred, _ = create_features_for_prediction(
                            self.data, location, selected_states, lags,
                            anchor_date=val_date, horizon=horizon
                        )

                    if len(X_pred) == 0:
                        continue

                    # Apply log transformation to prediction features
                    if use_log_transform.get(location, False):
                        X_pred_transformed = np.log1p(np.maximum(X_pred, 0))
                    else:
                        X_pred_transformed = X_pred

                    # Compute target date aligned with forecast horizon
                    target_date = val_date + pd.Timedelta(weeks=horizon)
                    if target_date not in all_dates:
                        continue
                    actual_value = self.data.loc[self.data['date'] == target_date, location]
                    if len(actual_value) == 0:
                        continue
                    actual_value = actual_value.iloc[0]

                    # Get predictions
                    X_pred_last = X_pred_transformed[-1:]

                    # Get μ prediction (in transformed space if log transform used)
                    mu_pred_raw = temp_stage1.predict(X_pred_last)[0]

                    # Convert mu to linear space for reporting and residual calculation
                    if use_log_transform.get(location, False):
                        mu_pred = np.expm1(mu_pred_raw)
                    else:
                        mu_pred = mu_pred_raw
                    mu_pred_raw_linear = max(0.0, mu_pred)  # Store original model prediction
                    mu_pred_corrected = mu_pred_raw_linear

                    # --- BLENDED MEDIAN (for unbounded models only) ---
                    # Get persistence value (last observed value at reference date)
                    last_value_row = self.data.loc[self.data['date'] == val_date, location]
                    last_value = last_value_row.iloc[0] if len(last_value_row) > 0 else mu_pred_corrected

                    # For unbounded models, compute blended median from t10, t100, and persistence
                    blended_mu = mu_pred_corrected  # Default to current model's prediction

                    if not is_bounded:
                        # Look up pre-computed medians from other model variants
                        blend_key = (location, target_date)
                        other_medians = other_model_medians.get(blend_key, {})

                        mu_t10 = other_medians.get('t10')
                        mu_t100 = other_medians.get('t100')

                        # Compute blended median based on available components
                        if mu_t10 is not None and mu_t100 is not None:
                            # Full blend: t10 + t100 + persistence
                            blended_mu = W_T10 * mu_t10 + W_T100 * mu_t100 + W_PERS * last_value
                        elif mu_t10 is not None:
                            # t10 + persistence (no t100)
                            w_t10_adj = W_T10 / (W_T10 + W_PERS)
                            w_pers_adj = W_PERS / (W_T10 + W_PERS)
                            blended_mu = w_t10_adj * mu_t10 + w_pers_adj * last_value
                        elif mu_t100 is not None:
                            # t100 + persistence (no t10)
                            w_t100_adj = W_T100 / (W_T100 + W_PERS)
                            w_pers_adj = W_PERS / (W_T100 + W_PERS)
                            blended_mu = w_t100_adj * mu_t100 + w_pers_adj * last_value
                        else:
                            # Fallback: current model + persistence (50/50)
                            blended_mu = 0.5 * mu_pred_corrected + 0.5 * last_value

                        blended_mu = max(0.0, blended_mu)

                        # Use blended_mu as the corrected prediction for conformal residuals
                        mu_pred_corrected = blended_mu

                    # --- QUANTILE GENERATION ---
                    from scipy.stats import norm

                    if is_bounded or is_bounded_wide:
                        # BOUNDED MODE: Use Stage 2 sigma directly in log space
                        # mu_pred_raw is already in log space (because use_log_transform=True for bounded)
                        mu_log = mu_pred_raw

                        # Get sigma from Stage 2 (in log space)
                        dist_params = temp_stage2.predict(X_pred_last, pred_type="parameters")
                        if hasattr(dist_params, 'values'):
                            dist_params = dist_params.values
                        if dist_params.ndim > 1:
                            sigma_log = dist_params[0, 1]
                        else:
                            sigma_log = dist_params[1]
                        sigma_log = max(sigma_log, 1e-6)

                        # Generate quantiles in log space, then transform back
                        quantile_forecasts = []
                        for q in self.quantiles:
                            q_log = norm.ppf(q, loc=mu_log, scale=sigma_log)
                            q_linear = np.expm1(q_log)
                            quantile_forecasts.append(max(0.0, q_linear))

                        # No residual tracking for bounded models

                    else:
                        # STANDARD MODE: Use conformal residual quantiles with EWMA weighting

                        # Determine current season for optional season bucketing
                        is_high_season = (val_date.month >= 10 or val_date.month <= 3)

                        # Try season-specific residuals first, then fall back to combined
                        season_residuals = residual_stores_by_season.get(is_high_season, [])

                        if len(residuals_log) >= MIN_RESIDUALS_FOR_CONFORMAL:
                            # Use EWMA-weighted conformal residual quantiles
                            mu_log = np.log1p(max(mu_pred_corrected, 0.0))

                            # Extract values and weights
                            res_vals = [r for r, w in residuals_log]
                            res_wts = [w for r, w in residuals_log]

                            # If we have enough season-specific residuals, prefer those
                            if len(season_residuals) >= MIN_RESIDUALS_FOR_CONFORMAL:
                                res_vals = [r for r, w in season_residuals]
                                res_wts = [w for r, w in season_residuals]

                            # Compute weighted quantiles
                            residual_quantiles = weighted_percentile(res_vals, res_wts, self.quantiles)
                            q_log = mu_log + residual_quantiles
                            quantile_forecasts = [max(0.0, np.expm1(val)) for val in q_log]
                        else:
                            # Fallback to Stage-2 sigma with conservative cap
                            dist_params = temp_stage2.predict(X_pred_last, pred_type="parameters")
                            if hasattr(dist_params, 'values'):
                                dist_params = dist_params.values

                            if dist_params.ndim > 1:
                                sigma_pred = dist_params[0, 1]
                            else:
                                sigma_pred = dist_params[1]

                            sigma_pred = max(sigma_pred, 1e-6)

                            # Cap sigma conservatively until we have enough residuals
                            sigma_cap = max(1.0 * abs(mu_pred_corrected), 50.0)
                            sigma_pred = min(sigma_pred, sigma_cap)

                            quantile_forecasts = []
                            for q in self.quantiles:
                                pred = norm.ppf(q, loc=mu_pred_corrected, scale=sigma_pred)
                                quantile_forecasts.append(max(pred, 0.0))

                        # Record residual for future forecasts (only for standard mode)
                        residual_log = np.log1p(max(actual_value, 0.0)) - np.log1p(max(mu_pred_corrected, 0.0))

                        # Apply EWMA decay to existing weights before adding new residual
                        residuals_log = [(res, wt * EWMA_ALPHA) for res, wt in residuals_log]
                        residuals_log.append((residual_log, 1.0))

                        # Also add to season-specific store with EWMA decay
                        residual_stores_by_season[is_high_season] = [
                            (res, wt * EWMA_ALPHA) for res, wt in residual_stores_by_season[is_high_season]
                        ]
                        residual_stores_by_season[is_high_season].append((residual_log, 1.0))

                        # Prune to RESIDUAL_LOOKBACK entries
                        if len(residuals_log) > RESIDUAL_LOOKBACK:
                            residuals_log = residuals_log[-RESIDUAL_LOOKBACK:]
                        if len(residual_stores_by_season[is_high_season]) > RESIDUAL_LOOKBACK:
                            residual_stores_by_season[is_high_season] = residual_stores_by_season[is_high_season][-RESIDUAL_LOOKBACK:]

                    forecast_results.append({
                        'forecast_date': val_date,
                        'target_date': target_date,
                        'actual_value': actual_value,
                        'quantile_forecasts': np.array(quantile_forecasts),
                        'mu_pred': mu_pred_corrected,  # This is blended_mu for unbounded, raw for bounded
                        'mu_pred_raw': mu_pred_raw_linear,  # Original model prediction before blending
                        'last_value': last_value if not is_bounded else None,  # Persistence value
                        'n_residuals': len(residuals_log) if not is_bounded else 0
                    })

                except Exception as e:
                    print(f"      [ERROR] {location} @ {val_date.date()}: {str(e)}")
                    continue

            all_forecasts[location] = forecast_results
            if is_bounded or is_bounded_wide:
                print(f"    Generated {len(forecast_results)} forecasts (bounded log-space sigma)")
            else:
                print(f"    Generated {len(forecast_results)} forecasts (conformal after {MIN_RESIDUALS_FOR_CONFORMAL} residuals)")

        return all_forecasts
        
    def generate_persistence_forecasts(self, locations: List[str], horizon: int) -> Dict:
        """Generate persistence baseline forecasts."""
        
        print(f"\nGenerating persistence baseline forecasts...")
        
        all_persistence_forecasts = {}
        
        for location in locations:
            print(f"  Generating persistence forecasts for {location}")
            
            # Get training data last date
            train_data = self.data[self.data['date'] < self.cut_off_date]
            first_anchor_date = train_data['date'].max() if len(train_data) > 0 else self.cut_off_date
            
            # Get validation dates
            val_data = self.data[self.data['date'] >= first_anchor_date].copy()
            val_dates = sorted(val_data['date'].unique())
            all_dates = sorted(self.data['date'].unique())
            
            # Stop early enough to have target dates
            val_dates_to_use = val_dates[:-horizon] if len(val_dates) > horizon else []
            
            persistence_results = []
            
            for i, val_date in enumerate(val_dates_to_use):
                # Get target date
                target_date = val_date + pd.Timedelta(weeks=horizon)
                
                if target_date not in all_dates:
                    continue
                
                # Get actual value
                actual_value = self.data.loc[self.data['date'] == target_date, location]
                if len(actual_value) == 0:
                    continue
                actual_value = actual_value.iloc[0]
                
                # Find last observed value (expanding window)
                if i == 0:
                    historical_data = self.data[self.data['date'] <= first_anchor_date]
                else:
                    expand_end = val_date
                    historical_data = self.data[self.data['date'] <= expand_end]
                
                # Get historical values
                if len(historical_data) > 0:
                    historical_values = historical_data[location].values
                    last_value = historical_data[location].iloc[-1]
                    
                    # Generate persistence quantile forecast
                    quantile_forecasts = persistence_forecast_quantiles(
                        last_value, self.quantiles, historical_values
                    )
                    
                    persistence_results.append({
                        'forecast_date': val_date,
                        'target_date': target_date,
                        'actual_value': actual_value,
                        'quantile_forecasts': quantile_forecasts
                    })
                    
            all_persistence_forecasts[location] = persistence_results
            print(f"    Generated {len(persistence_results)} forecasts")
            
        return all_persistence_forecasts
        
    def format_cdc_flusight(self, forecasts: Dict, model_name: str, horizon: int) -> pd.DataFrame:
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
        
        for location, forecast_list in forecasts.items():
            # Get FIPS code for location
            fips_code = state_to_fips.get(location, location)
            
            for forecast in forecast_list:
                forecast_date = forecast['forecast_date']
                target_date = forecast.get('target_date', forecast_date + pd.Timedelta(weeks=horizon))
                quantile_forecasts = forecast['quantile_forecasts']
                
                # Create record for each quantile
                for i, quantile_level in enumerate(self.quantiles):
                    predicted_value = quantile_forecasts[i]
                    
                    # Make CDC fields consistent
                    target_end_date = target_date
                    reference_date = target_end_date - pd.Timedelta(weeks=horizon)
                    
                    cdc_records.append({
                        'reference_date': reference_date.strftime('%Y-%m-%d'),
                        'horizon': horizon - 1,  # CDC uses 0-based horizon
                        'target': 'wk inc flu hosp',
                        'target_end_date': target_end_date.strftime('%Y-%m-%d'),
                        'location': fips_code,
                        'output_type': 'quantile',
                        'output_type_id': quantile_level,
                        'value': max(predicted_value, 0.0)
                    })
        
        return pd.DataFrame(cdc_records)
        
    def save_forecasts(self, model_forecasts: Dict, persistence_forecasts: Dict, 
                      horizon: int, output_dir: str) -> None:
        """Save forecasts in CDC FluSight format (without summary)."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving forecasts to {output_dir}/")
        
        # Format and save model forecasts (if any)
        model_df = self.format_cdc_flusight(model_forecasts, "TwoStage-FrozenMu", horizon)
        if len(model_df) > 0:
            cdc_column_order = ['reference_date', 'horizon', 'target', 'target_end_date', 
                               'location', 'output_type', 'output_type_id', 'value']
            # Ensure absent columns exist
            for col in cdc_column_order:
                if col not in model_df.columns:
                    model_df[col] = []
            model_df = model_df[cdc_column_order]
            model_file = os.path.join(output_dir, f"TwoStage-FrozenMu_h{horizon}_forecasts.csv")
            model_df.to_csv(model_file, index=False)
            print(f"  Model forecasts saved: {model_file}")
        else:
            print("  No model forecasts generated; skipping model file.")
        
        # Format and save persistence forecasts
        persistence_df = self.format_cdc_flusight(persistence_forecasts, "Flusight-baseline", horizon)
        if len(persistence_df) > 0:
            cdc_column_order = ['reference_date', 'horizon', 'target', 'target_end_date', 
                               'location', 'output_type', 'output_type_id', 'value']
            for col in cdc_column_order:
                if col not in persistence_df.columns:
                    persistence_df[col] = []
            persistence_df = persistence_df[cdc_column_order]
            # Save a single baseline file per horizon at the retrospective root
            retrospective_root = os.path.dirname(output_dir)
            persistence_file = os.path.join(retrospective_root, f"Flusight-baseline_h{horizon}_forecasts.csv")
            persistence_df.to_csv(persistence_file, index=False)
            print(f"  Baseline forecasts saved: {persistence_file}")
        else:
            print("  No baseline forecasts generated; skipping baseline file.")

        # Save summary statistics (NEW)
        summary_stats = []
        for location in model_forecasts.keys():
            # Defensive check for empty forecasts
            if len(model_forecasts[location]) == 0:
                continue
            
            model_preds = [f['quantile_forecasts'][11] for f in model_forecasts[location]]  # median
            actuals = [f['actual_value'] for f in model_forecasts[location]]
            biases = [f.get('bias_applied', 0.0) for f in model_forecasts[location]]
            
            summary_stats.append({
                'location': location,
                'n_forecasts': len(actuals),
                'actual_mean': np.mean(actuals),
                'actual_std': np.std(actuals),
                'model_median_mean': np.mean(model_preds),
                'mean_bias_applied': np.mean(biases),
                'date_range': f"{model_forecasts[location][0]['forecast_date'].strftime('%Y-%m-%d')} to {model_forecasts[location][-1]['forecast_date'].strftime('%Y-%m-%d')}"
            })
        
        if summary_stats:
            summary_df = pd.DataFrame(summary_stats)
            summary_file = os.path.join(output_dir, f"forecast_summary_h{horizon}.csv")
            summary_df.to_csv(summary_file, index=False)
            print(f"  Summary statistics saved to: {summary_file}")


def main():
    """Main function for batch retrospective forecasting."""
    
    parser = argparse.ArgumentParser(description='Generate Retrospective Forecasts for All Saved Models')
    
    # Required arguments
    parser.add_argument('--data-file', type=str, required=True,
                       help='Path to input data file')
    parser.add_argument('--cut-off', type=str, required=True,
                       help='Cut-off date for train/validation split (YYYY-MM-DD)')
    
    # Optional arguments
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directory containing model hyperparameters (default: models)')
    parser.add_argument('--output-base', type=str, default='forecasts/retrospective',
                       help='Base output directory for forecasts (default: forecasts/retrospective)')
    parser.add_argument('--models-base-dir', type=str, default='models',
                       help='Base directory for saved models (default: models)')
    parser.add_argument('--horizons', type=str, default=None,
                       help='Optional comma-separated list of horizons to process (e.g., "1,2")')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Data file not found: {args.data_file}")
    
    if not os.path.exists(args.models_dir):
        raise FileNotFoundError(f"Models directory not found: {args.models_dir}")
    
    print(f"{'='*80}")
    print(f"BATCH RETROSPECTIVE FORECAST GENERATION")
    print(f"{'='*80}")
    print(f"Data file: {args.data_file}")
    print(f"Cut-off date: {args.cut_off}")
    print(f"Models directory: {args.models_dir}")
    print(f"Output base: {args.output_base}")
    
    # Initialize generator
    generator = BatchRetrospectiveForecastGenerator(cut_off_date=args.cut_off)
    
    # Load data once
    generator.load_data(args.data_file)
    
    # Find all model configurations
    configurations = generator.find_model_configurations(args.models_dir)
    
    if len(configurations) == 0:
        print("\nNo model configurations found!")
        return
    
    # Optionally filter configurations by user-requested horizons
    if args.horizons:
        wanted = set(int(h.strip()) for h in args.horizons.split(',') if h.strip())
        configurations = [c for c in configurations if c['horizon'] in wanted]

    # Process each configuration
    for i, config in enumerate(configurations, 1):
        print(f"\n[{i}/{len(configurations)}] ", end="")
        try:
            generator.process_single_configuration(config, args.output_base, args.models_base_dir)
        except Exception as e:
            print(f"  ERROR processing {config['folder_name']}/h{config['horizon']}: {str(e)}")
            continue
    
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Processed {len(configurations)} model configurations")
    print(f"Output directory: {args.output_base}/")


if __name__ == "__main__":
    main()

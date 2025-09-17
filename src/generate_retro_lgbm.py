#!/usr/bin/env python
"""
Generate retrospective out-of-sample forecasts using trained two-stage models.

This script loads trained Stage 1 (μ) and Stage 2 (σ) models and generates
probabilistic forecasts in CDC FluSight format for evaluation and visualization.

Key Features:
- Uses trained two-stage models (μ + σ)
- Expanding window validation 
- CDC FluSight quantile format
- Persistence baseline comparison
- Rolling forecast generation
- Compatible with R visualization scripts

Usage:
    python src/generate_retrospective_forecasts.py \\
        --hyperparams models/two_stage_hyperparameters_h1.pkl \\
        --data-file data/imputed_and_stitched_hosp_2025-05-24.csv \\
        --cut-off 2023-07-01 \\
        --output forecasts/retrospective
"""

import os
import sys
import json
import pickle
import argparse
import warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

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
    from utils.distributions import GaussianFrozenLoc
except ImportError:
    raise ImportError("lightgbmlss is required. Install with: pip install lightgbmlss")

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


class RetrospectiveForecastGenerator:
    """
    Generate retrospective probabilistic forecasts using trained two-stage models.
    """
    
    def __init__(self, cut_off_date: str, horizon: int = 1, quantiles: np.ndarray = None):
        self.cut_off_date = pd.to_datetime(cut_off_date)
        self.horizon = horizon
        self.quantiles = quantiles if quantiles is not None else CDC_QUANTILES
        self.processor = TimeSeriesDataProcessor()
        
        # Storage for loaded models and data
        self.hyperparams = {}
        self.stage1_models = {}
        self.stage2_models = {}
        self.data = None
        self.use_log_transform = {}
        
    def load_hyperparameters(self, hyperparams_file: str) -> None:
        """Load trained model hyperparameters."""
        
        print(f"Loading hyperparameters from {hyperparams_file}")
        
        with open(hyperparams_file, 'rb') as f:
            self.hyperparams = pickle.load(f)
            
        locations = list(self.hyperparams.keys())
        print(f"Loaded hyperparameters for {len(locations)} locations: {', '.join(locations)}")
        
        # Extract log transform settings
        for location in locations:
            self.use_log_transform[location] = self.hyperparams[location].get('use_log_transform', False)
            if self.use_log_transform[location]:
                print(f"  {location}: Using log transformation")
        
    def load_trained_models(self) -> None:
        """Load trained Stage 1 and Stage 2 models."""
        
        print("Loading trained models...")
        
        for location in self.hyperparams.keys():
            # Load Stage 1 model (LightGBM booster) with horizon-aware path first
            booster_file = f"models/point_mu/{location}_h{self.horizon}_booster.txt"
            if not os.path.exists(booster_file):
                # Fallback to legacy filename (no horizon)
                booster_file = f"models/point_mu/{location}_booster.txt"
            if not os.path.exists(booster_file):
                raise FileNotFoundError(f"Stage 1 model not found for {location} (tried horizon-aware and legacy paths)")
            self.stage1_models[location] = lgb.Booster(model_file=booster_file)
            
            # Load Stage 2 model (LightGBMLSS) with horizon-aware path first
            model_file = f"models/scale_sigma/{location}_h{self.horizon}_lgbmlss_model.pkl"
            if not os.path.exists(model_file):
                # Fallback to legacy filename
                model_file = f"models/scale_sigma/{location}_lgbmlss_model.pkl"
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"Stage 2 model not found for {location} (tried horizon-aware and legacy paths)")
            with open(model_file, 'rb') as f:
                self.stage2_models[location] = pickle.load(f)
                
            print(f"  Loaded models for {location}")
            
        print(f"All models loaded successfully")
        
    def load_data(self, data_file: str) -> None:
        """Load and prepare data for forecasting."""
        
        print(f"Loading data from {data_file}")
        
        # Load all data
        self.data = self.processor.load_and_pivot_data(data_file, exclude_locations=None)
        
        # Get training data (all data before cutoff)
        train_data = self.data[self.data['date'] < self.cut_off_date]
        
        # Get validation period info
        # Validation set starts at the cutoff date
        validation_data = self.data[self.data['date'] >= self.cut_off_date]
        val_weeks = len(validation_data)
        
        print(f"Data loaded: {len(self.data)} total samples")
        print(f"Training data: up to {(train_data['date'].max() if len(train_data) > 0 else 'N/A')}")
        print(f"Cut-off date: {self.cut_off_date.strftime('%Y-%m-%d')}")
        print(f"Validation period: {val_weeks} weeks from {self.cut_off_date.strftime('%Y-%m-%d')} onwards")
        
    def generate_model_forecasts(self) -> Dict:
        """Generate two-stage model forecasts using expanding window validation."""
        
        print(f"\\nGenerating two-stage model forecasts...")
        
        all_forecasts = {}
        
        for location in self.hyperparams.keys():
            print(f"  Generating forecasts for {location}")
            
            # Get model parameters
            params = self.hyperparams[location]
            stage1_params = params['stage1']
            stage2_params = params['stage2']
            lags = stage1_params['lags']
            selected_states = stage1_params['selected_states']
            
            # Guard: Ensure we have lags and no lag_0 (data leakage)
            # Handle case where enhanced features were used (lags = None)
            if lags is None:
                # Enhanced features mode - no explicit lags needed
                print(f"    Using enhanced features mode (no explicit lags)")
                use_enhanced = True
            else:
                use_enhanced = False
                if len(lags) == 0:
                    raise ValueError(
                        f"{location}: no lags specified. "
                        "Ensure hyperparameters were trained properly."
                    )
                if min(lags) < 1:
                    raise ValueError(
                        f"{location}: lag_0 detected in {lags}. "
                        "Lag 0 would cause data leakage."
                    )
            
            # Get models
            stage1_model = self.stage1_models[location]
            stage2_model = self.stage2_models[location]
            
            # Get training data last date as the first anchor date
            train_data = self.data[self.data['date'] < self.cut_off_date]
            first_anchor_date = train_data['date'].max() if len(train_data) > 0 else self.cut_off_date
            
            # Get validation dates starting from the first anchor date
            # But make sure we only use dates that actually exist in the data
            val_data = self.data[self.data['date'] >= first_anchor_date].copy()
            val_dates = sorted(val_data['date'].unique())
            all_dates = sorted(self.data['date'].unique())
            
            # Stop early enough to have target dates
            val_dates_to_use = val_dates[:-self.horizon] if len(val_dates) > self.horizon else []
                
            forecast_results = []
            total_dates = len(val_dates_to_use)
            progress_interval = max(10, total_dates // 10) if total_dates > 0 else 1
            
            # Expanding window validation
            for i, val_date in enumerate(val_dates_to_use):
                if total_dates > 0 and (i % progress_interval == 0 or i == total_dates - 1):
                    print(f"    Progress: {i+1}/{total_dates} forecasts ({100*(i+1)/total_dates:.1f}%)")
                
                try:
                    # Expanding window: include all previous validation points in training
                    # For the first forecast, train_end_date should be the last training date
                    if i == 0:
                        train_end_date = first_anchor_date
                    else:
                        train_end_date = val_date - pd.Timedelta(days=1)
                    
                    # Create training features up to train_end_date
                    if use_enhanced:
                        X_train, y_train, _ = create_enhanced_features(
                            self.data, location, selected_states,
                            end_date=train_end_date, horizon=self.horizon
                        )
                    else:
                        X_train, y_train, _ = create_features(
                            self.data, location, selected_states, lags,
                            end_date=train_end_date, horizon=self.horizon
                        )
                    
                    if len(X_train) < 50:
                        continue
                    
                    # Apply log transformation if enabled
                    if self.use_log_transform.get(location, False):
                        X_train_transformed = np.log1p(np.maximum(X_train, 0))
                        y_train_transformed = np.log1p(np.maximum(y_train, 0))
                    else:
                        X_train_transformed = X_train
                        y_train_transformed = y_train
                    
                    # Create prediction features using prediction-mode creators
                    if use_enhanced:
                        X_pred, _ = create_enhanced_features_for_prediction(
                            self.data, location, selected_states,
                            anchor_date=val_date, horizon=self.horizon
                        )
                    else:
                        X_pred, _ = create_features_for_prediction(
                            self.data, location, selected_states, lags,
                            anchor_date=val_date, horizon=self.horizon
                        )
                    
                    if len(X_pred) == 0:
                        continue
                    
                    # Apply log transformation to prediction features if enabled
                    if self.use_log_transform.get(location, False):
                        X_pred_transformed = np.log1p(np.maximum(X_pred, 0))
                    else:
                        X_pred_transformed = X_pred
                    
                    # Compute target date (anchor + horizon)
                    # Note: times from tabularizer are for training samples, not predictions
                    target_date = val_date + pd.Timedelta(weeks=self.horizon)
                    
                    # Validation logging for anchor/target alignment
                    if i == 0 or i == len(val_dates_to_use) - 1:
                        print(f"      [VALIDATION] Anchor={val_date.strftime('%Y-%m-%d')}, Target={target_date.strftime('%Y-%m-%d')} (h={self.horizon})")
                    
                    if target_date not in all_dates:
                        continue
                    
                    # Get actual value at target date
                    actual_value = self.data.loc[self.data['date'] == target_date, location]
                    if len(actual_value) == 0:
                        continue
                    actual_value = actual_value.iloc[0]
                    
                    # Re-train models on expanding window
                    # Re-train Stage 1 model (use transformed data)
                    dtrain1 = lgb.Dataset(X_train_transformed, label=y_train_transformed, params={'verbose': -1})
                    p1 = stage1_params['best_params'].copy()
                    p1['verbose'] = -1
                    p1['verbosity'] = -1
                    temp_stage1 = lgb.train(
                        p1, 
                        dtrain1, 
                        num_boost_round=stage1_params['num_boost_round'],
                        callbacks=[]
                    )

                    # Get μ predictions for training data (in log space if transformed)
                    mu_predictions = temp_stage1.predict(X_train_transformed)

                    # Re-train Stage 2 model with frozen μ
                    # NEW: Flatten init_score (Fortran order)
                    init_score = np.column_stack([
                        mu_predictions,
                        np.zeros_like(mu_predictions)
                    ]).ravel(order='F')

                    dtrain2 = lgb.Dataset(X_train_transformed, label=y_train_transformed, init_score=init_score, params={'verbose': -1})
                    temp_stage2 = LightGBMLSS(GaussianFrozenLoc())
                    p2 = stage2_params['best_params'].copy()
                    p2['verbose'] = -1
                    p2['verbosity'] = -1
                    temp_stage2.train(
                        p2, 
                        dtrain2, 
                        num_boost_round=stage2_params['num_boost_round']
                    )
                    
                    # Use the already created X_pred from above (transformed)
                    X_pred_last = X_pred_transformed[-1:]
                    
                    # Get μ prediction from Stage 1
                    mu_pred = temp_stage1.predict(X_pred_last)[0]
                    
                    # Inverse transform μ if using log
                    if self.use_log_transform.get(location, False):
                        mu_pred = np.expm1(mu_pred)
                    
                    # Get σ prediction from Stage 2
                    dist_params = temp_stage2.predict(X_pred_last, pred_type="parameters")
                    if hasattr(dist_params, 'values'):
                        dist_params = dist_params.values
                    
                    if dist_params.ndim > 1:
                        sigma_pred = dist_params[0, 1]
                    else:
                        sigma_pred = dist_params[1]
                    
                    sigma_pred = max(sigma_pred, 1e-6)  # Ensure positive
                    
                    # If using log transform, scale sigma appropriately
                    if self.use_log_transform.get(location, False):
                        # Convert sigma from log space to original space
                        # This is approximate - assumes log-normal distribution
                        sigma_pred = sigma_pred * mu_pred
                    
                    # Generate quantile forecasts
                    from scipy.stats import norm
                    quantile_forecasts = []
                    for q in self.quantiles:
                        pred = norm.ppf(q, loc=mu_pred, scale=sigma_pred)
                        quantile_forecasts.append(max(pred, 0.0))  # Truncate negatives
                    
                    forecast_results.append({
                        'forecast_date': val_date,  # Store ANCHOR date as forecast_date
                        'target_date': target_date,  # Target = anchor + horizon
                        'actual_value': actual_value,
                        'quantile_forecasts': np.array(quantile_forecasts)
                    })
                        
                except Exception as e:
                    continue
                    
            all_forecasts[location] = forecast_results
            
        return all_forecasts
        
    def generate_persistence_forecasts(self) -> Dict:
        """Generate persistence baseline forecasts."""
        
        print(f"\\nGenerating persistence baseline forecasts...")
        
        all_persistence_forecasts = {}
        
        for location in self.hyperparams.keys():
            print(f"  Generating persistence forecasts for {location}")
            
            
            # Get training data last date as the first anchor date
            train_data = self.data[self.data['date'] < self.cut_off_date]
            first_anchor_date = train_data['date'].max() if len(train_data) > 0 else self.cut_off_date
            
            # Get validation dates starting from the first anchor date
            # But make sure we only use dates that actually exist in the data
            val_data = self.data[self.data['date'] >= first_anchor_date].copy()
            val_dates = sorted(val_data['date'].unique())
            all_dates = sorted(self.data['date'].unique())
            
            # Stop early enough to have target dates (same as model forecasts)
            val_dates_to_use = val_dates[:-self.horizon] if len(val_dates) > self.horizon else []
            
            persistence_results = []
            total_dates = len(val_dates_to_use)
            progress_interval = max(10, total_dates // 5) if total_dates > 0 else 1
            
            for i, val_date in enumerate(val_dates_to_use):
                if total_dates > 0 and (i % progress_interval == 0 or i == total_dates - 1):
                    print(f"    Progress: {i+1}/{total_dates} forecasts ({100*(i+1)/total_dates:.1f}%)")
                
                # Get target date consistent with model forecasts
                # For persistence, target is val_date + horizon weeks
                target_date = val_date + pd.Timedelta(weeks=self.horizon)
                
                # Check if target date exists in the data
                if target_date not in all_dates:
                    continue
                
                # Get actual value at target date
                actual_value = self.data.loc[self.data['date'] == target_date, location]
                if len(actual_value) == 0:
                    continue
                actual_value = actual_value.iloc[0]
                
                # Find last observed value (expanding window)
                if i == 0:
                    historical_data = self.data[self.data['date'] <= first_anchor_date]
                else:
                    # Use val_date directly to include the current anchor date value for persistence
                    expand_end = val_date
                    historical_data = self.data[self.data['date'] <= expand_end]
                
                # CDC baseline uses ALL historical data, not just flu season
                # This matches the official CDC FluSight-baseline methodology
                
                # Get all historical values for the location
                if len(historical_data) > 0:
                    historical_values = historical_data[location].values
                    last_value = historical_data[location].iloc[-1]
                    
                    # Generate persistence quantile forecast using CDC methodology
                    # CDC appears to use detrended residuals for wider, more conservative intervals
                    quantile_forecasts = persistence_forecast_quantiles(
                        last_value, self.quantiles, historical_values
                    )
                    
                    persistence_results.append({
                        'forecast_date': val_date,  # Store ANCHOR date as forecast_date
                        'target_date': target_date,  # Store the actual target date
                        'actual_value': actual_value,
                        'quantile_forecasts': quantile_forecasts
                    })
                    
            all_persistence_forecasts[location] = persistence_results
            
        return all_persistence_forecasts
        
    def format_cdc_flusight(self, forecasts: Dict, model_name: str) -> pd.DataFrame:
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
            fips_code = state_to_fips.get(location, location)  # Default to location name if not found
            
            for forecast in forecast_list:
                forecast_date = forecast['forecast_date']
                # Prefer the stored target_date; fall back for old records
                target_date = forecast.get('target_date', forecast_date + pd.Timedelta(weeks=self.horizon))
                actual_value = forecast['actual_value']
                quantile_forecasts = forecast['quantile_forecasts']
                
                # Create record for each quantile
                for i, quantile_level in enumerate(self.quantiles):
                    predicted_value = quantile_forecasts[i]
                    
                    # Make CDC fields consistent with the model's actual target
                    target_end_date = target_date
                    reference_date = target_end_date - pd.Timedelta(weeks=self.horizon)
                    
                    cdc_records.append({
                        'reference_date': reference_date.strftime('%Y-%m-%d'),
                        'horizon': self.horizon - 1,  # CDC uses 0-based horizon
                        'target': 'wk inc flu hosp',
                        'target_end_date': target_end_date.strftime('%Y-%m-%d'),
                        'location': fips_code,
                        'output_type': 'quantile',
                        'output_type_id': quantile_level,
                        'value': max(predicted_value, 0.0)
                    })
        
        return pd.DataFrame(cdc_records)
        
    def save_forecasts(self, model_forecasts: Dict, persistence_forecasts: Dict, output_dir: str) -> Tuple[str, str, str]:
        """Save forecasts in CDC FluSight format."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\\nSaving forecasts to {output_dir}/")
        
        # Format and save model forecasts
        model_df = self.format_cdc_flusight(model_forecasts, "TwoStage-FrozenMu")
        # Ensure column order matches CDC format
        cdc_column_order = ['reference_date', 'horizon', 'target', 'target_end_date', 
                           'location', 'output_type', 'output_type_id', 'value']
        model_df = model_df[cdc_column_order]
        model_file = os.path.join(output_dir, f"TwoStage-FrozenMu_h{self.horizon}_forecasts.csv")
        model_df.to_csv(model_file, index=False)
        print(f"  Model forecasts saved to: {model_file}")
        
        # Format and save persistence forecasts
        persistence_df = self.format_cdc_flusight(persistence_forecasts, "Flusight-baseline")
        persistence_df = persistence_df[cdc_column_order]
        persistence_file = os.path.join(output_dir, f"Flusight-baseline_h{self.horizon}_forecasts.csv")
        persistence_df.to_csv(persistence_file, index=False)
        print(f"  Persistence forecasts saved to: {persistence_file}")
        
        # Save summary statistics
        summary_stats = []
        for location in model_forecasts.keys():
            # Defensive check for empty forecasts
            if len(model_forecasts[location]) == 0:
                print(f"  Warning: No model forecasts generated for {location}")
                continue
            
            if len(persistence_forecasts[location]) == 0:
                print(f"  Warning: No persistence forecasts generated for {location}")
                continue
                
            model_preds = [f['quantile_forecasts'][11] for f in model_forecasts[location]]  # median
            persistence_preds = [f['quantile_forecasts'][11] for f in persistence_forecasts[location]]
            actuals = [f['actual_value'] for f in model_forecasts[location]]
            
            summary_stats.append({
                'location': location,
                'n_forecasts': len(actuals),
                'actual_mean': np.mean(actuals),
                'actual_std': np.std(actuals),
                'model_median_mean': np.mean(model_preds),
                'persistence_median_mean': np.mean(persistence_preds),
                'date_range': f"{model_forecasts[location][0]['forecast_date'].strftime('%Y-%m-%d')} to {model_forecasts[location][-1]['forecast_date'].strftime('%Y-%m-%d')}"
            })
        
        summary_df = pd.DataFrame(summary_stats)
        summary_file = os.path.join(output_dir, f"forecast_summary_h{self.horizon}.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"  Summary statistics saved to: {summary_file}")
        
        return model_file, persistence_file, summary_file


def main():
    """Main function for retrospective forecasting."""
    
    parser = argparse.ArgumentParser(description='Generate Retrospective Two-Stage Forecasts')
    
    # Required arguments
    parser.add_argument('--hyperparams', type=str, required=True,
                       help='Path to saved hyperparameters file')
    parser.add_argument('--data-file', type=str, required=True,
                       help='Path to input data file')
    parser.add_argument('--cut-off', type=str, required=True,
                       help='Cut-off date for train/validation split (YYYY-MM-DD)')
    
    # Optional arguments
    parser.add_argument('--output', type=str, default='forecasts/retrospective',
                       help='Output directory for forecasts (default: forecasts/retrospective)')
    parser.add_argument('--horizon', type=int, default=1,
                       help='Forecast horizon (default: 1)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.hyperparams):
        raise FileNotFoundError(f"Hyperparameters file not found: {args.hyperparams}")
    
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Data file not found: {args.data_file}")
    
    print(f"{'='*80}")
    print(f"GENERATING RETROSPECTIVE TWO-STAGE FORECASTS")
    print(f"{'='*80}")
    print(f"Hyperparameters: {args.hyperparams}")
    print(f"Data file: {args.data_file}")
    print(f"Cut-off date: {args.cut_off}")
    print(f"Horizon: {args.horizon}")
    print(f"Output: {args.output}")
    
    # Initialize generator
    generator = RetrospectiveForecastGenerator(
        cut_off_date=args.cut_off,
        horizon=args.horizon
    )
    
    # Load hyperparameters and models
    generator.load_hyperparameters(args.hyperparams)
    generator.load_trained_models()
    
    # Load data
    generator.load_data(args.data_file)
    
    # Generate forecasts
    model_forecasts = generator.generate_model_forecasts()
    persistence_forecasts = generator.generate_persistence_forecasts()
    
    # Save results
    model_file, persistence_file, summary_file = generator.save_forecasts(
        model_forecasts, persistence_forecasts, args.output
    )
    
    print(f"\\n{'='*80}")
    print(f"RETROSPECTIVE FORECASTING COMPLETE")
    print(f"{'='*80}")
    print(f"Model forecasts: {model_file}")
    print(f"Persistence forecasts: {persistence_file}")
    print(f"Summary statistics: {summary_file}")
    print(f"\\nFiles are ready for evaluation and visualization.")
    
    # Show sample output
    print(f"\\nSample of model forecast format:")
    model_df = pd.read_csv(model_file)
    print(model_df.head(10))


if __name__ == "__main__":
    main()

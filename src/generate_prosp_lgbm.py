#!/usr/bin/env python
"""
Generate prospective forecasts using trained two-stage models.

This script loads trained Stage 1 (μ) and Stage 2 (σ) models and generates
probabilistic forecasts from the end of available data for a specified horizon.

Key Features:
- Uses trained two-stage models (μ + σ)
- Generates forecast from the latest available data point
- CDC FluSight quantile format output
- Single forecast per location at specified horizon

Usage:
    python src/generate_prospective_forecasts.py \
        --hyperparams models/two_stage_hyperparameters_h1.pkl \
        --data-file data/imputed_and_stitched_hosp_2025-05-24.csv \
        --horizon 1 \
        --output forecasts/prospective
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


class ProspectiveForecastGenerator:
    """
    Generate prospective probabilistic forecasts using trained two-stage models.
    """
    
    def __init__(self, horizon: int = 1, quantiles: np.ndarray = None):
        self.horizon = horizon
        self.quantiles = quantiles if quantiles is not None else CDC_QUANTILES
        self.processor = TimeSeriesDataProcessor()
        
        # Storage for hyperparameters and data
        self.hyperparams = {}
        self.data = None
        self.last_date = None
        self.use_log_transform = {}
        
    def load_hyperparameters(self, hyperparams_file: str) -> None:
        """Load trained model hyperparameters."""
        
        print(f"Loading hyperparameters from {hyperparams_file}")
        
        with open(hyperparams_file, 'rb') as f:
            self.hyperparams = pickle.load(f)
            
        locations = list(self.hyperparams.keys())
        print(f"Loaded hyperparameters for {len(locations)} locations: {', '.join(locations)}")
        
    def verify_hyperparameters(self) -> None:
        """Verify that all required hyperparameters are present."""
        
        print("Verifying hyperparameters...")
        
        for location in self.hyperparams.keys():
            params = self.hyperparams[location]
            
            # Check for required keys
            if 'stage1' not in params or 'stage2' not in params:
                raise ValueError(f"Missing stage parameters for {location}")
            
            stage1_params = params['stage1']
            stage2_params = params['stage2']
            
            # Check Stage 1 parameters
            required_stage1 = ['best_params', 'num_boost_round', 'selected_states', 'lags']
            for key in required_stage1:
                if key not in stage1_params:
                    raise ValueError(f"Missing Stage 1 parameter '{key}' for {location}")
            
            # Check Stage 2 parameters
            required_stage2 = ['best_params', 'num_boost_round']
            for key in required_stage2:
                if key not in stage2_params:
                    raise ValueError(f"Missing Stage 2 parameter '{key}' for {location}")
            
            # Extract log transform setting
            self.use_log_transform[location] = params.get('use_log_transform', False)
            if self.use_log_transform[location]:
                print(f"  {location}: Using log transformation")
            else:
                print(f"  Verified parameters for {location}")
            
        print(f"All hyperparameters verified successfully")
        
    def load_data(self, data_file: str) -> None:
        """Load and prepare data for forecasting."""
        
        print(f"Loading data from {data_file}")
        
        # Load all data
        self.data = self.processor.load_and_pivot_data(data_file, exclude_locations=None)
        
        # Get the last date in the data
        self.last_date = self.data['date'].max()
        
        print(f"Data loaded: {len(self.data)} total samples")
        print(f"Last date in data: {self.last_date.strftime('%Y-%m-%d')}")
        print(f"Forecasting {self.horizon} week(s) ahead from this date")
        
    def generate_forecasts(self) -> Dict:
        """Generate prospective forecasts from the end of available data."""
        
        print(f"\nGenerating prospective forecasts...")
        
        all_forecasts = {}
        
        for location in self.hyperparams.keys():
            print(f"  Generating forecast for {location}")
            
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
            
            try:
                # Train on all available data (no cutoff, use everything)
                if use_enhanced:
                    X_train, y_train, _ = create_enhanced_features(
                        self.data, location, selected_states,
                        end_date=None, horizon=self.horizon  # Use all data
                    )
                else:
                    X_train, y_train, _ = create_features(
                        self.data, location, selected_states, lags,
                        end_date=None, horizon=self.horizon  # Use all data
                    )
                
                if len(X_train) < 50:
                    print(f"    Warning: Insufficient training data for {location}")
                    continue
                
                print(f"    Training on {len(X_train)} samples")
                
                # Apply log transformation if enabled
                if self.use_log_transform.get(location, False):
                    X_train_transformed = np.log1p(np.maximum(X_train, 0))
                    y_train_transformed = np.log1p(np.maximum(y_train, 0))
                else:
                    X_train_transformed = X_train
                    y_train_transformed = y_train
                
                # Train Stage 1 model on all available data (use transformed data)
                dtrain1 = lgb.Dataset(X_train_transformed, label=y_train_transformed, params={'verbose': -1})
                p1 = stage1_params['best_params'].copy()
                p1['verbose'] = -1
                p1['verbosity'] = -1
                final_stage1 = lgb.train(
                    p1, 
                    dtrain1, 
                    num_boost_round=stage1_params['num_boost_round'],
                    callbacks=[]
                )

                # Get μ predictions for training data (in log space if transformed)
                mu_predictions = final_stage1.predict(X_train_transformed)

                # Train Stage 2 model with frozen μ
                init_score = np.column_stack([
                    mu_predictions,
                    np.zeros_like(mu_predictions)
                ]).ravel(order='F')

                dtrain2 = lgb.Dataset(X_train_transformed, label=y_train_transformed, init_score=init_score, params={'verbose': -1})
                final_stage2 = LightGBMLSS(GaussianFrozenLoc())
                p2 = stage2_params['best_params'].copy()
                p2['verbose'] = -1
                p2['verbosity'] = -1
                final_stage2.train(
                    p2, 
                    dtrain2, 
                    num_boost_round=stage2_params['num_boost_round']
                )
                
                # Create features for prediction from the last available date
                # Use the prediction-mode creators like in retrospective
                if use_enhanced:
                    X_pred, _ = create_enhanced_features_for_prediction(
                        self.data, location, selected_states,
                        anchor_date=self.last_date, horizon=self.horizon
                    )
                else:
                    X_pred, _ = create_features_for_prediction(
                        self.data, location, selected_states, lags,
                        anchor_date=self.last_date, horizon=self.horizon
                    )
                
                if len(X_pred) == 0:
                    print(f"    Warning: Cannot create prediction features for {location}")
                    continue
                
                # Apply log transformation to prediction features if enabled
                if self.use_log_transform.get(location, False):
                    X_pred_transformed = np.log1p(np.maximum(X_pred, 0))
                else:
                    X_pred_transformed = X_pred
                
                # Use the last row for prediction (should be just one row anyway)
                X_pred_last = X_pred_transformed[-1:]
                
                # Get μ prediction from Stage 1
                mu_pred = final_stage1.predict(X_pred_last)[0]
                
                # Inverse transform μ if using log
                if self.use_log_transform.get(location, False):
                    mu_pred = np.expm1(mu_pred)
                
                # Get σ prediction from Stage 2
                dist_params = final_stage2.predict(X_pred_last, pred_type="parameters")
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
                
                # Calculate target date
                target_date = self.last_date + pd.Timedelta(weeks=self.horizon)
                
                # Validation logging for anchor/target alignment
                print(f"    [VALIDATION] Anchor={self.last_date.strftime('%Y-%m-%d')}, Target={target_date.strftime('%Y-%m-%d')} (h={self.horizon})")
                
                all_forecasts[location] = {
                    'forecast_date': self.last_date,
                    'target_date': target_date,
                    'mu': mu_pred,
                    'sigma': sigma_pred,
                    'quantile_forecasts': np.array(quantile_forecasts)
                }
                
                print(f"    Forecast generated: μ={mu_pred:.2f}, σ={sigma_pred:.2f}")
                    
            except Exception as e:
                print(f"    Error generating forecast for {location}: {str(e)}")
                continue
                
        return all_forecasts

    def save_trained_models(self, forecasts: Dict, hyperparams: Dict, final_stage1_models: Dict, final_stage2_models: Dict,
                             models_output_dir: str, horizon: int) -> None:
        """Persist trained Stage1/Stage2 models to disk under the provided base directory.
        Layout: {models_output_dir}/point_mu/{location}_h{h}_booster.txt and
                {models_output_dir}/scale_sigma/{location}_h{h}_lgbmlss_model.pkl
        """
        os.makedirs(os.path.join(models_output_dir, 'point_mu'), exist_ok=True)
        os.makedirs(os.path.join(models_output_dir, 'scale_sigma'), exist_ok=True)
        import pickle
        for location in forecasts.keys():
            if location not in final_stage1_models or location not in final_stage2_models:
                continue
            booster_path = os.path.join(models_output_dir, 'point_mu', f'{location}_h{horizon}_booster.txt')
            final_stage1_models[location].save_model(booster_path)

            sigma_path = os.path.join(models_output_dir, 'scale_sigma', f'{location}_h{horizon}_lgbmlss_model.pkl')
            with open(sigma_path, 'wb') as f:
                pickle.dump(final_stage2_models[location], f)
            
            # Optional: note presence
            # print(f"Saved models for {location}: {booster_path}, {sigma_path}")
        
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
        
        for location, forecast in forecasts.items():
            # Get FIPS code for location
            fips_code = state_to_fips.get(location, location)
            
            forecast_date = forecast['forecast_date']
            target_date = forecast['target_date']
            quantile_forecasts = forecast['quantile_forecasts']
            
            # Create record for each quantile
            for i, quantile_level in enumerate(self.quantiles):
                predicted_value = quantile_forecasts[i]
                
                # CDC fields
                reference_date = forecast_date
                
                cdc_records.append({
                    'reference_date': reference_date.strftime('%Y-%m-%d'),
                    'horizon': self.horizon - 1,  # CDC uses 0-based horizon
                    'target': 'wk inc flu hosp',
                    'target_end_date': target_date.strftime('%Y-%m-%d'),
                    'location': fips_code,
                    'output_type': 'quantile',
                    'output_type_id': quantile_level,
                    'value': max(predicted_value, 0.0)
                })
        
        return pd.DataFrame(cdc_records)
        
    def save_forecasts(self, forecasts: Dict, output_dir: str, model_name: str = "TwoStage-FrozenMu") -> str:
        """Save forecasts in CDC FluSight format."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving forecasts to {output_dir}/")
        
        # Format forecasts
        forecast_df = self.format_cdc_flusight(forecasts, model_name)
        
        # Ensure column order matches CDC format
        cdc_column_order = ['reference_date', 'horizon', 'target', 'target_end_date', 
                           'location', 'output_type', 'output_type_id', 'value']
        forecast_df = forecast_df[cdc_column_order]
        
        # Generate filename with timestamp
        timestamp = self.last_date.strftime('%Y%m%d')
        forecast_file = os.path.join(output_dir, f"{model_name}_h{self.horizon}_prospective_{timestamp}.csv")
        forecast_df.to_csv(forecast_file, index=False)
        print(f"  Forecasts saved to: {forecast_file}")
        
        # Save summary
        summary_data = []
        for location, forecast in forecasts.items():
            summary_data.append({
                'location': location,
                'forecast_date': forecast['forecast_date'].strftime('%Y-%m-%d'),
                'target_date': forecast['target_date'].strftime('%Y-%m-%d'),
                'horizon': self.horizon,
                'mu': forecast['mu'],
                'sigma': forecast['sigma'],
                'median_forecast': forecast['quantile_forecasts'][11],  # 0.5 quantile
                'lower_95': forecast['quantile_forecasts'][1],  # 0.025 quantile
                'upper_95': forecast['quantile_forecasts'][21]  # 0.975 quantile
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(output_dir, f"prospective_summary_h{self.horizon}_{timestamp}.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"  Summary saved to: {summary_file}")
        
        return forecast_file


def main():
    """Main function for prospective forecasting."""
    
    parser = argparse.ArgumentParser(description='Generate Prospective Two-Stage Forecasts')
    
    # Required arguments
    parser.add_argument('--hyperparams', type=str, required=True,
                       help='Path to saved hyperparameters file (e.g., models/two_stage_hyperparameters_h1.pkl)')
    parser.add_argument('--data-file', type=str, required=True,
                       help='Path to input data file')
    parser.add_argument('--horizon', type=int, required=True,
                       help='Forecast horizon in weeks')
    
    # Optional arguments
    parser.add_argument('--output', type=str, default='forecasts/prospective',
                       help='Output directory for forecasts (default: forecasts/prospective)')
    parser.add_argument('--model-name', type=str, default='TwoStage-FrozenMu',
                       help='Model name for output files (default: TwoStage-FrozenMu)')
    parser.add_argument('--save-models', action='store_true',
                       help='Save trained Stage1/Stage2 models to disk')
    parser.add_argument('--models-output-dir', type=str, default='models/lgbm_enhanced_t10',
                       help='Base directory to save models when --save-models is set')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.hyperparams):
        raise FileNotFoundError(f"Hyperparameters file not found: {args.hyperparams}")
    
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Data file not found: {args.data_file}")
    
    if args.horizon < 1:
        raise ValueError("Horizon must be at least 1 week")
    
    print(f"{'='*80}")
    print(f"GENERATING PROSPECTIVE TWO-STAGE FORECASTS")
    print(f"{'='*80}")
    print(f"Hyperparameters: {args.hyperparams}")
    print(f"Data file: {args.data_file}")
    print(f"Horizon: {args.horizon} week(s)")
    print(f"Output: {args.output}")
    print(f"Model name: {args.model_name}")
    
    # Initialize generator
    generator = ProspectiveForecastGenerator(horizon=args.horizon)
    
    # Load hyperparameters
    generator.load_hyperparameters(args.hyperparams)
    generator.verify_hyperparameters()
    
    # Load data
    generator.load_data(args.data_file)
    
    # Generate forecasts
    # Generate forecasts and capture trained models to optionally persist
    # We re-run core logic to harvest the trained model objects without large refactor:
    # Monkey-patch structures to track trained models per location
    generator._final_stage1_models = {}
    generator._final_stage2_models = {}

    # Wrap the original generate_forecasts to record models
    _orig_generate = generator.generate_forecasts
    def _wrapped_generate():
        forecasts_local = {}
        # repeat core of generate_forecasts with small duplication to expose models
        # Load data is already called
        print(f"\nGenerating prospective forecasts...")
        all_forecasts_local = {}
        for location in generator.hyperparams.keys():
            print(f"  Generating forecast for {location}")
            try:
                params = generator.hyperparams[location]
                stage1_params = params['stage1']
                stage2_params = params['stage2']
                lags = stage1_params['lags']
                selected_states = stage1_params['selected_states']
                use_enhanced = (lags is None)
                if use_enhanced:
                    X_train, y_train, _ = create_enhanced_features(generator.data, location, selected_states, end_date=None, horizon=generator.horizon)
                else:
                    X_train, y_train, _ = create_features(generator.data, location, selected_states, lags, end_date=None, horizon=generator.horizon)
                if len(X_train) < 50:
                    continue
                if generator.use_log_transform.get(location, False):
                    X_train_transformed = np.log1p(np.maximum(X_train, 0))
                    y_train_transformed = np.log1p(np.maximum(y_train, 0))
                else:
                    X_train_transformed = X_train
                    y_train_transformed = y_train
                dtrain1 = lgb.Dataset(X_train_transformed, label=y_train_transformed, params={'verbose': -1})
                p1 = stage1_params['best_params'].copy(); p1['verbose'] = -1; p1['verbosity'] = -1
                final_stage1 = lgb.train(p1, dtrain1, num_boost_round=stage1_params['num_boost_round'], callbacks=[])
                generator._final_stage1_models[location] = final_stage1
                mu_predictions = final_stage1.predict(X_train_transformed)
                init_score = np.column_stack([mu_predictions, np.zeros_like(mu_predictions)]).ravel(order='F')
                dtrain2 = lgb.Dataset(X_train_transformed, label=y_train_transformed, init_score=init_score, params={'verbose': -1})
                final_stage2 = LightGBMLSS(GaussianFrozenLoc())
                p2 = stage2_params['best_params'].copy(); p2['verbose'] = -1; p2['verbosity'] = -1
                final_stage2.train(p2, dtrain2, num_boost_round=stage2_params['num_boost_round'])
                generator._final_stage2_models[location] = final_stage2

                if use_enhanced:
                    X_pred, _ = create_enhanced_features_for_prediction(generator.data, location, selected_states, anchor_date=generator.last_date, horizon=generator.horizon)
                else:
                    X_pred, _ = create_features_for_prediction(generator.data, location, selected_states, lags, anchor_date=generator.last_date, horizon=generator.horizon)
                if len(X_pred) == 0:
                    continue
                if generator.use_log_transform.get(location, False):
                    X_pred_transformed = np.log1p(np.maximum(X_pred, 0))
                else:
                    X_pred_transformed = X_pred
                mu_pred = final_stage1.predict(X_pred_transformed[-1:])[0]
                if generator.use_log_transform.get(location, False):
                    mu_pred = np.expm1(mu_pred)
                dist_params = final_stage2.predict(X_pred_transformed[-1:], pred_type="parameters")
                if hasattr(dist_params, 'values'):
                    dist_params = dist_params.values
                sigma_pred = dist_params[0, 1] if dist_params.ndim > 1 else dist_params[1]
                sigma_pred = max(float(sigma_pred), 1e-6)
                if generator.use_log_transform.get(location, False):
                    sigma_pred = sigma_pred * mu_pred
                from scipy.stats import norm
                qvals = np.array([max(0.0, norm.ppf(q, loc=mu_pred, scale=sigma_pred)) for q in CDC_QUANTILES])
                target_date = generator.last_date + pd.Timedelta(weeks=generator.horizon)
                all_forecasts_local[location] = {
                    'forecast_date': generator.last_date,
                    'target_date': target_date,
                    'mu': mu_pred,
                    'sigma': sigma_pred,
                    'quantile_forecasts': qvals
                }
            except Exception as e:
                continue
        return all_forecasts_local

    forecasts = _wrapped_generate()
    
    if len(forecasts) == 0:
        print("\nError: No forecasts were generated. Check your data and models.")
        return
    
    # Save results
    forecast_file = generator.save_forecasts(forecasts, args.output, args.model_name)

    # Optionally save trained models for reuse
    if args.save_models and len(forecasts) > 0:
        generator.save_trained_models(forecasts, generator.hyperparams, generator._final_stage1_models, generator._final_stage2_models,
                                      args.models_output_dir, generator.horizon)
    
    print(f"\n{'='*80}")
    print(f"PROSPECTIVE FORECASTING COMPLETE")
    print(f"{'='*80}")
    print(f"Generated forecasts for {len(forecasts)} locations")
    print(f"Forecast file: {forecast_file}")
    print(f"\nFiles are ready for submission or further analysis.")
    
    # Show sample output
    print(f"\nSample of forecast format:")
    forecast_df = pd.read_csv(forecast_file)
    print(forecast_df.head(10))


if __name__ == "__main__":
    main()

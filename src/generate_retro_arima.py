#!/usr/bin/env python
"""
Generate retrospective ARIMA forecasts with prediction intervals based on empirical error distributions.

This script:
1. Fits ARIMA models using the first half of training data to determine lag orders
2. Generates retrospective forecasts on the second half of training data
3. Creates empirical error distributions from validation forecasts
4. Generates probabilistic forecasts on test data using the error distributions
5. Outputs forecasts in CDC FluSight format

Usage:
    python src/generate_arima_retrospective_forecasts.py \
        --data-file data/imputed_and_stitched_hosp_2025-05-24.csv \
        --cut-off 2024-10-01 \
        --output forecasts/retrospective/saved_models/arima
"""

import os
import sys
import argparse
import warnings
import pandas as pd
import numpy as np
from pmdarima import auto_arima, ARIMA
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from scipy import stats
import pickle

warnings.filterwarnings("ignore")

# CDC FluSight quantiles
CDC_QUANTILES = np.array([
    0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99
])


class ARIMAForecaster:
    """Generate ARIMA forecasts with empirical prediction intervals."""
    
    def __init__(self, cut_off_date: str, max_horizon: int = 4):
        self.cut_off_date = pd.to_datetime(cut_off_date)
        self.max_horizon = max_horizon
        self.quantiles = CDC_QUANTILES
        
        # Storage for models and errors
        self.best_orders = {}
        self.error_distributions = {}
        self.data = None
        self.prepared_data = None
        
    def load_data(self, file_path: str) -> None:
        """Load and prepare data for ARIMA modeling."""
        
        print(f"Loading data from {file_path}")
        
        # Load data
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
        
        # For each location, find where actual data starts (non-NaN values)
        # and report data availability
        print("\nData availability by location:")
        location_start_dates = {}
        for location in self.prepared_data.columns:
            location_data = self.prepared_data[location].dropna()
            if len(location_data) > 0:
                start_date = location_data.index.min()
                location_start_dates[location] = start_date
                weeks_available = len(location_data)
                if weeks_available >= 100:  # Only show if reasonable amount of data
                    print(f"  {location}: {start_date.strftime('%Y-%m-%d')} ({weeks_available} weeks)")
        
        print(f"\nOverall data shape: {len(self.prepared_data)} weeks x {len(self.prepared_data.columns)} locations")
        print(f"Date range: {self.prepared_data.index.min()} to {self.prepared_data.index.max()}")
        
    def split_data_for_location(self, location: str) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Split data for a specific location into train, validation, and test sets."""
        
        # Get location data and drop NaN values
        location_data = self.prepared_data[location].dropna()
        
        if len(location_data) == 0:
            return pd.Series(), pd.Series(), pd.Series()
        
        # Split based on cutoff date
        train_valid_data = location_data[location_data.index < self.cut_off_date]
        test_data = location_data[location_data.index >= self.cut_off_date]
        
        if len(train_valid_data) < 40:  # Need minimum data for train+valid
            return pd.Series(), pd.Series(), test_data
        
        # Split training data in half for order selection and validation
        split_point = len(train_valid_data) // 2
        train_order = train_valid_data.iloc[:split_point]
        train_valid = train_valid_data.iloc[split_point:]
        
        return train_order, train_valid, test_data
    
    def determine_best_order(self, train: pd.Series, location: str) -> Tuple:
        """Use auto_arima to determine the best ARIMA order."""
        
        try:
            # Add 1 to handle non-positive values for Box-Cox transformation
            train_shifted = train + 1
            
            model = auto_arima(
                train_shifted,
                start_p=0, max_p=8,
                start_d=0, max_d=2,
                start_q=0, max_q=8,
                seasonal=False,  # Disable seasonal for simplicity
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
            return (1, 1, 1)  # Default order
    
    def fit_and_predict(self, train: pd.Series, order: Tuple, horizon: int) -> np.ndarray:
        """Fit ARIMA model and make forecasts."""
        
        try:
            # Add 1 to handle non-positive values
            train_shifted = train + 1
            
            # Fit model with Box-Cox transformation
            model = ARIMA(order=order, suppress_warnings=True)
            model.fit(train_shifted)
            
            # Generate forecast
            forecast_shifted = model.predict(n_periods=horizon)
            
            # Subtract 1 to revert to original scale
            forecast = forecast_shifted - 1
            
            return forecast
            
        except Exception as e:
            print(f"  Warning: Model fitting failed: {str(e)}")
            # Return persistence forecast as fallback
            return np.array([train.iloc[-1]] * horizon)
    
    def generate_validation_forecasts(self) -> Dict:
        """Generate forecasts on validation set to build error distributions."""
        
        print("\nGenerating validation forecasts...")
        
        validation_errors = {}
        
        for location in self.prepared_data.columns:
            # Split data for this specific location
            order_series, valid_series, _ = self.split_data_for_location(location)
            
            # Check if we have enough data for this location
            min_required = 20  # Minimum weeks needed for ARIMA
            if len(order_series) < min_required:
                print(f"  Skipping {location}: insufficient order selection data ({len(order_series)} < {min_required} weeks)")
                continue
                
            if len(valid_series) < self.max_horizon + 5:  # Need some validation samples
                print(f"  Skipping {location}: insufficient validation data ({len(valid_series)} weeks)")
                continue
            
            print(f"  Processing {location} ({len(order_series)} order weeks, {len(valid_series)} validation weeks)...")
                
            best_order = self.determine_best_order(order_series, location)
            self.best_orders[location] = best_order
            print(f"    Best ARIMA order: {best_order}")
            
            # Initialize error storage for each horizon
            location_errors = {h: [] for h in range(1, self.max_horizon + 1)}
            
            # Expanding window validation
            for i in range(len(valid_series) - self.max_horizon):
                # Combine order selection data with part of validation data
                if i == 0:
                    train_data = order_series
                else:
                    train_data = pd.concat([order_series, valid_series.iloc[:i]])
                
                if len(train_data) < min_required:
                    continue
                
                # Generate forecast
                forecast = self.fit_and_predict(train_data, best_order, self.max_horizon)
                
                # Calculate errors for each horizon
                for h in range(1, min(self.max_horizon + 1, len(valid_series) - i)):
                    actual = valid_series.iloc[i + h - 1]
                    predicted = forecast[h - 1]
                    
                    if not np.isnan(actual) and not np.isnan(predicted):
                        error = actual - predicted
                        location_errors[h].append(error)
            
            validation_errors[location] = location_errors
            
        self.error_distributions = validation_errors
        
        # Print error statistics
        print("\nValidation error statistics:")
        for location in list(validation_errors.keys())[:3]:  # Show first 3 locations
            print(f"  {location}:")
            for h in range(1, self.max_horizon + 1):
                errors = validation_errors[location][h]
                if len(errors) > 0:
                    print(f"    Horizon {h}: mean={np.mean(errors):.2f}, std={np.std(errors):.2f}, n={len(errors)}")
        
        return validation_errors
    
    def create_quantile_forecasts(self, point_forecast: float, errors: List[float], 
                                 quantiles: np.ndarray) -> np.ndarray:
        """Convert point forecast to quantile forecasts using empirical errors."""
        
        if len(errors) < 10:
            # Fallback: use normal distribution with conservative std
            std = max(abs(point_forecast) * 0.25, 5.0)
            quantile_forecasts = [
                max(0, stats.norm.ppf(q, loc=point_forecast, scale=std))
                for q in quantiles
            ]
        else:
            # Use empirical error distribution
            error_quantiles = np.percentile(errors, quantiles * 100)
            quantile_forecasts = [
                max(0, point_forecast + eq)  # Ensure non-negative
                for eq in error_quantiles
            ]
        
        return np.array(quantile_forecasts)
    
    def generate_test_forecasts(self) -> Dict:
        """Generate probabilistic forecasts on test data."""
        
        print("\nGenerating test forecasts...")
        
        all_forecasts = {}
        
        for location in self.best_orders.keys():
            print(f"  Processing {location}...")
            
            # Split data for this specific location
            order_series, valid_series, test_series = self.split_data_for_location(location)
            
            if len(test_series) == 0:
                print(f"    Skipping {location}: no test data")
                continue
            
            best_order = self.best_orders[location]
            location_errors = self.error_distributions[location]
            
            # Get full training data (order + validation)
            full_train = pd.concat([order_series, valid_series])
            
            location_forecasts = []
            
            # Generate forecasts for each test date
            for i in range(min(len(test_series), 30)):  # Limit to 30 weeks for speed
                # Expanding window: include all previous data
                if i == 0:
                    train_data = full_train
                else:
                    train_data = pd.concat([full_train, test_series.iloc[:i]])
                
                forecast_date = test_series.index[i]
                
                # Generate point forecast
                point_forecasts = self.fit_and_predict(train_data, best_order, self.max_horizon)
                
                # Convert to quantile forecasts for each horizon
                for h in range(1, self.max_horizon + 1):
                    point_forecast = point_forecasts[h - 1]
                    errors = location_errors[h]
                    
                    # Generate quantile forecasts
                    quantile_forecasts = self.create_quantile_forecasts(
                        point_forecast, errors, self.quantiles
                    )
                    
                    # Calculate target date
                    target_date = forecast_date + pd.Timedelta(weeks=h)
                    
                    # Store forecast
                    location_forecasts.append({
                        'reference_date': forecast_date,
                        'target_date': target_date,
                        'horizon': h,
                        'quantile_forecasts': quantile_forecasts
                    })
            
            all_forecasts[location] = location_forecasts
            
        return all_forecasts
    
    def format_cdc_flusight(self, forecasts: Dict) -> pd.DataFrame:
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
            # Get FIPS code
            fips_code = state_to_fips.get(location, location)
            
            for forecast in forecast_list:
                reference_date = forecast['reference_date']
                target_date = forecast['target_date']
                horizon = forecast['horizon']
                quantile_forecasts = forecast['quantile_forecasts']
                
                # Create record for each quantile
                for i, quantile_level in enumerate(self.quantiles):
                    predicted_value = quantile_forecasts[i]
                    
                    cdc_records.append({
                        'reference_date': reference_date.strftime('%Y-%m-%d'),
                        'horizon': horizon - 1,  # CDC uses 0-based horizon
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
        
        # Save forecasts for each horizon
        for h in range(1, self.max_horizon + 1):
            # Filter forecasts for this horizon
            horizon_forecasts = {}
            for location, forecast_list in forecasts.items():
                horizon_list = [f for f in forecast_list if f['horizon'] == h]
                if len(horizon_list) > 0:
                    horizon_forecasts[location] = horizon_list
            
            if len(horizon_forecasts) > 0:
                # Format as CDC FluSight
                forecast_df = self.format_cdc_flusight(horizon_forecasts)
                
                # Ensure column order
                cdc_column_order = ['reference_date', 'horizon', 'target', 'target_end_date',
                                   'location', 'output_type', 'output_type_id', 'value']
                forecast_df = forecast_df[cdc_column_order]
                
                # Save to file
                output_file = os.path.join(output_dir, f"ARIMA_h{h}_forecasts.csv")
                forecast_df.to_csv(output_file, index=False)
                print(f"  Saved horizon {h} forecasts: {output_file}")
        
        # Save model information
        model_info = {
            'best_orders': self.best_orders,
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
    
    # Required arguments
    parser.add_argument('--data-file', type=str, required=True,
                       help='Path to input data file')
    parser.add_argument('--cut-off', type=str, required=True,
                       help='Cut-off date for train/test split (YYYY-MM-DD)')
    
    # Optional arguments
    parser.add_argument('--output', type=str, default='forecasts/retrospective/saved_models/arima',
                       help='Output directory for forecasts')
    parser.add_argument('--max-horizon', type=int, default=4,
                       help='Maximum forecast horizon (default: 4)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Data file not found: {args.data_file}")
    
    print(f"{'='*80}")
    print(f"GENERATING ARIMA RETROSPECTIVE FORECASTS")
    print(f"{'='*80}")
    print(f"Data file: {args.data_file}")
    print(f"Cut-off date: {args.cut_off}")
    print(f"Max horizon: {args.max_horizon}")
    print(f"Output: {args.output}")
    
    # Initialize forecaster
    forecaster = ARIMAForecaster(
        cut_off_date=args.cut_off,
        max_horizon=args.max_horizon
    )
    
    # Load data
    forecaster.load_data(args.data_file)
    
    # Generate validation forecasts and build error distributions
    # This now handles splitting internally for each location
    forecaster.generate_validation_forecasts()
    
    # Generate test forecasts with prediction intervals
    test_forecasts = forecaster.generate_test_forecasts()
    
    # Save results
    forecaster.save_forecasts(test_forecasts, args.output)
    
    print(f"\n{'='*80}")
    print(f"ARIMA FORECASTING COMPLETE")
    print(f"{'='*80}")
    print(f"Output directory: {args.output}/")
    print(f"Generated forecasts for {len(test_forecasts)} locations")
    
    # Show sample statistics
    total_forecasts = sum(len(f) for f in test_forecasts.values())
    print(f"Total forecast records: {total_forecasts}")


if __name__ == "__main__":
    main()
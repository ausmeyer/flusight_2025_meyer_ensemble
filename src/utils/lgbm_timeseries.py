import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


class TimeSeriesDataProcessor:
    """
    Handles time series data preprocessing without DARTS dependency.
    Provides functionality for creating lag/lead features, time-based splits, 
    and preparing data for LightGBM training.
    """
    
    def __init__(self, freq='W-SAT'):
        self.freq = freq
        
    def load_and_pivot_data(self, file_path: str, exclude_locations: List[str] = None) -> pd.DataFrame:
        """Load CSV data and pivot to wide format with locations as columns."""
        df = pd.read_csv(file_path, on_bad_lines='skip')
        df['date'] = pd.to_datetime(df['date'])
        
        # Drop unnecessary columns
        df = df.drop(['ili', 'pred_hosp', 'true_hosp', 'population'], axis=1, errors='ignore')
        
        # Pivot to wide format
        pivoted_df = df.pivot_table(index='date', columns='location_name', values='total_hosp').reset_index()
        pivoted_df.columns.name = None
        
        # Exclude specified locations
        if exclude_locations:
            pivoted_df = pivoted_df.drop(exclude_locations, axis=1, errors='ignore')
        
        # Drop rows with any missing values
        pivoted_df = pivoted_df.dropna()
        
        return pivoted_df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str, 
                           covariate_cols: List[str], lags: List[int]) -> pd.DataFrame:
        """Create lagged features for target and covariates."""
        result_df = df.copy()
        
        # Create lagged target features
        for lag in lags:
            if lag > 0:
                result_df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        # Create lagged covariate features
        for col in covariate_cols:
            for lag in lags:
                if lag > 0:
                    result_df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return result_df
    
    def create_future_covariates(self, df: pd.DataFrame, periods_ahead: int = 4) -> pd.DataFrame:
        """Create future covariates (week numbers) for forecasting."""
        result_df = df.copy()
        
        # Create week numbers
        result_df['week_num'] = pd.to_datetime(result_df['date']).dt.isocalendar().week
        
        # Extend for future periods if needed
        if periods_ahead > 0:
            last_date = result_df['date'].max()
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(weeks=1),
                periods=periods_ahead,
                freq=self.freq
            )
            
            future_weeks = []
            for date in future_dates:
                week_num = date.isocalendar().week
                future_weeks.append({'date': date, 'week_num': week_num})
            
            future_df = pd.DataFrame(future_weeks)
            result_df = pd.concat([result_df, future_df], ignore_index=True)
        
        return result_df
    
    def time_based_split(self, df: pd.DataFrame, train_end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data based on time for time series validation."""
        train_end = pd.to_datetime(train_end_date)
        
        train_df = df[df['date'] <= train_end].copy()
        val_df = df[df['date'] > train_end].copy()
        
        return train_df, val_df
    
    def prepare_features_and_target(self, df: pd.DataFrame, target_col: str, 
                                   feature_cols: List[str], 
                                   lags: List[int] = None, horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare feature matrix and target variable - FIXED to match DARTS exactly."""
        if lags is None:
            lags = [1, 2, 3, 4]
        
        # CRITICAL FIX: Create ALL features first on the full dataset, then slice
        # This matches DARTS which creates lags after slicing but we need to do it before
        # to avoid losing data points due to the order of operations
        
        # Start with the input dataframe
        df_work = df.copy()
        
        # Add week number as a covariate (like DARTS treats it)
        df_work['week_num'] = pd.to_datetime(df_work['date']).dt.isocalendar().week
        df_work['week_num'] = df_work['week_num'].clip(upper=52)
        
        # Create lagged features for target and all covariates
        df_with_lags = self.create_lag_features(df_work, target_col, feature_cols, lags)
        
        # Create lagged week features just like other covariates
        for lag in lags:
            if lag > 0:
                df_with_lags[f'week_num_lag_{lag}'] = df_with_lags['week_num'].shift(lag)
        
        # DO NOT add week_num_lag_0 - this creates data leakage!
        # At prediction time, we don't know the "current" week_num for the target date
        
        # Select ALL lag features including week lags
        lag_cols = [col for col in df_with_lags.columns if '_lag_' in col]
        feature_columns = lag_cols
        
        # CRITICAL: Drop NaN rows AFTER all feature creation to preserve more data
        # Find the maximum lag to know how many rows to drop from the beginning
        max_lag = max(lags) if lags else 0
        if max_lag > 0:
            # Drop only the first max_lag rows (where lagged features are NaN)
            df_clean = df_with_lags.iloc[max_lag:].copy()
        else:
            df_clean = df_with_lags.copy()
        
        # Final check for any remaining NaN values
        df_clean = df_clean.dropna()
        
        X = df_clean[feature_columns]
        y = df_clean[target_col]
        dates = df_clean['date']
        
        return X, y, dates


class TimeSeriesValidator:
    """
    Implements time series cross-validation for hyperparameter tuning.
    Ensures no data leakage by respecting temporal order.
    """
    
    def __init__(self, min_train_size: int = 100, horizon: int = 1, step_size: int = 1):
        self.min_train_size = min_train_size
        self.horizon = horizon
        self.step_size = step_size
    
    def time_series_split(self, dates: pd.Series, n_splits: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate time series splits for cross-validation.
        Returns list of (train_indices, test_indices) tuples.
        Fixed to match DARTS behavior more closely.
        """
        # Reset index to ensure proper indexing
        dates_reset = dates.reset_index(drop=True)
        dates_sorted_idx = dates_reset.sort_values().index
        n_samples = len(dates_reset)
        
        if n_samples <= self.min_train_size + self.horizon:
            raise ValueError(f"Not enough data points. Need at least {self.min_train_size + self.horizon}")
        
        splits = []
        
        # Calculate step size for splits - distribute evenly if n_splits specified
        if n_splits:
            available_range = n_samples - self.min_train_size - self.horizon
            step_size = max(1, available_range // n_splits)
        else:
            step_size = self.step_size
        
        # Start from min_train_size and create splits
        current_idx = self.min_train_size
        max_train_end = n_samples - self.horizon
        
        while current_idx <= max_train_end:
            train_end_idx = current_idx
            test_start_idx = current_idx
            test_end_idx = current_idx + self.horizon
            
            # Ensure we don't exceed data bounds
            if test_end_idx > n_samples:
                break
            
            # Get indices for this split using sorted order
            train_indices = dates_sorted_idx[:train_end_idx].values
            test_indices = dates_sorted_idx[test_start_idx:test_end_idx].values
            
            # Only add if we have exactly the right horizon length
            if len(test_indices) == self.horizon and len(train_indices) >= self.min_train_size:
                splits.append((train_indices, test_indices))
            
            current_idx += step_size
            
            # Limit number of splits if specified
            if n_splits and len(splits) >= n_splits:
                break
        
        return splits


class LightGBMTimeSeriesModel:
    """
    LightGBM wrapper for time series forecasting without DARTS dependency.
    Handles multi-horizon forecasting and feature engineering.
    """
    
    def __init__(self, horizon: int = 1, lags: List[int] = None, **lgb_params):
        self.horizon = horizon
        self.lags = lags if lags is not None else list(range(1, horizon + 4))
        self.lgb_params = lgb_params
        self.models = {}  # Store models for each horizon
        self.feature_importance_ = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, dates: pd.Series):
        """Fit the model using LGBMRegressor like DARTS does."""
        self.feature_names = X.columns.tolist()
        
        # Ensure all inputs have the same length
        min_len = min(len(X), len(y), len(dates))
        X_train = X.iloc[:min_len].reset_index(drop=True)
        y_train = y.iloc[:min_len].reset_index(drop=True) 
        dates_train = dates.iloc[:min_len].reset_index(drop=True)
        
        # Verify alignment
        if len(X_train) != len(y_train):
            raise ValueError(f"Feature-target mismatch: X_train={len(X_train)}, y_train={len(y_train)}")
        
        if len(X_train) == 0:
            raise ValueError("No valid training samples")
        
        # Store training data info for debugging
        self.train_info_ = {
            'original_samples': min_len,
            'train_samples': len(X_train),
            'horizon': self.horizon,
            'date_range': f"{dates_train.iloc[0]} to {dates_train.iloc[-1]}" if len(dates_train) > 0 else "empty"
        }
        
        # DARTS likely uses the native LightGBM interface with num_iterations instead of n_estimators
        # Let's match DARTS exactly by using the native interface
        import lightgbm as lgb
        
        lgbm_params = {
            'objective': 'regression',
            'metric': 'mae',
            'verbosity': -1
        }
        
        # Convert sklearn params to native LightGBM params
        native_params = self.lgb_params.copy()
        if 'n_estimators' in native_params:
            num_iterations = native_params.pop('n_estimators')
        else:
            num_iterations = 100
            
        if 'random_state' in native_params:
            native_params.pop('random_state')
            
        lgbm_params.update(native_params)
        
        # Use native LightGBM interface like DARTS does
        train_data = lgb.Dataset(X_train, label=y_train, params={'verbose': -1})
        self.model = lgb.train(
            lgbm_params,
            train_data,
            num_boost_round=num_iterations,
            callbacks=[lgb.log_evaluation(0)]
        )
        
        # Store feature importance (native LightGBM uses different method)
        self.feature_importance_ = self.model.feature_importance(importance_type='gain')
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not hasattr(self, 'model'):
            raise ValueError("Model has not been fitted yet")
        
        return self.model.predict(X.values)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if self.feature_importance_ is None:
            raise ValueError("Model has not been fitted yet")
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importance_
        }).sort_values('importance', ascending=False)


class FeatureSelector:
    """
    Feature selection for time series data using LightGBM feature importance.
    """
    
    def __init__(self, n_features: int = 10):
        self.n_features = n_features
        self.selected_features_ = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureSelector':
        """Fit feature selector using LightGBM feature importance."""
        # Use a portion of data for feature selection to avoid overfitting
        n_samples = len(X)
        end_idx = min(n_samples * 2 // 3, n_samples - 1)
        
        X_fs = X.iloc[:end_idx]
        y_fs = y.iloc[:end_idx]
        
        # Train simple LightGBM model for feature importance
        train_data = lgb.Dataset(X_fs, label=y_fs, params={'verbose': -1})
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'verbosity': -1
        }
        
        model = lgb.train(
            params,
            train_data,
            callbacks=[lgb.log_evaluation(0)]
        )
        
        # Get feature importance and select top features
        importance = model.feature_importance(importance_type='gain')
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        self.selected_features_ = importance_df['feature'].head(self.n_features).tolist()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using selected features."""
        if self.selected_features_ is None:
            raise ValueError("FeatureSelector has not been fitted yet")
        
        return X[self.selected_features_]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

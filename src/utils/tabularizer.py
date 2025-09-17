"""
DARTS tabularization utilities for feature engineering.

This module provides the create_features function that uses DARTS 
tabularization to create lagged features for time series forecasting.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Tuple
from darts import TimeSeries
from darts.utils.data.tabularization import create_lagged_training_data

# Add path to import from parent directory
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

class DARTSTabularizer:
    """
    Helper class to use DARTS tabularization for creating lagged features.
    Converts pandas data to DARTS TimeSeries and uses DARTS' create_lagged_training_data.
    """
    
    def pandas_to_darts_timeseries(self, df: pd.DataFrame, target_col: str, 
                                  covariate_cols: List[str]) -> Tuple[TimeSeries, TimeSeries, TimeSeries]:
        """Convert pandas DataFrame to DARTS TimeSeries format."""
        
        # Create target series
        target_df = df[['date', target_col]].copy()
        target_df = target_df.set_index('date')
        target_series = TimeSeries.from_dataframe(target_df, time_col=None, value_cols=[target_col])
        
        # Create past covariates series (state data only, NO week_num)
        cov_df = df[['date'] + covariate_cols].copy()
        cov_df = cov_df.set_index('date')
        past_covariates = TimeSeries.from_dataframe(cov_df, time_col=None)
        
        # Create future covariates series (week_num only, like DARTS)
        week_df = df[['date']].copy()
        week_df['week_num'] = pd.to_datetime(week_df['date']).dt.isocalendar().week
        week_df['week_num'] = week_df['week_num'].clip(upper=52)
        week_df = week_df.set_index('date')[['week_num']]
        future_covariates = TimeSeries.from_dataframe(week_df, time_col=None)
        
        return target_series, past_covariates, future_covariates
    
    def create_training_data(self, df: pd.DataFrame, target_col: str, 
                           covariate_cols: List[str], lags: List[int], 
                           end_date: pd.Timestamp = None, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
        """Create training data using DARTS tabularization."""
        
        # Defensive check: ensure no lag_0 (which would be data leakage)
        if lags and min(lags) < 1:
            raise ValueError(f"min(lag)={min(lags)} < 1. Lag 0 would cause data leakage. Use lags >= 1.")
        
        # Filter to end_date
        if end_date is not None:
            df_filtered = df[df['date'] <= end_date].copy()
        else:
            df_filtered = df.copy()
        
        if len(df_filtered) < (max(lags) if lags else 0) + horizon:
            return np.array([]), np.array([]), pd.Index([])
        
        target_series, past_covariates, future_covariates = self.pandas_to_darts_timeseries(
            df_filtered, target_col, covariate_cols
        )
        
        # Convert lags to DARTS format (negative values)
        darts_lags = [-lag for lag in lags]
        
        # week_num known-in-advance for steps 0..h-1
        lags_future_covs = list(range(0, horizon))
        
        try:
            result = create_lagged_training_data(
                target_series=target_series,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
                lags=darts_lags,
                lags_past_covariates=darts_lags,
                lags_future_covariates=lags_future_covs,
                output_chunk_length=1,
                output_chunk_shift=horizon - 1,  # key alignment for h-step ahead
                uses_static_covariates=False,
                concatenate=True
            )
            
            if len(result) >= 3:
                X, y, times = result[0], result[1], result[2]
                if len(X) > 0:
                    times_index = times[0] if isinstance(times, list) else times
                    X_flat = X.reshape(X.shape[0], -1) if X.ndim == 3 else X
                    y_squeezed = y.squeeze()
                    return X_flat, y_squeezed, times_index
                else:
                    return np.array([]), np.array([]), pd.Index([])
            else:
                return np.array([]), np.array([]), pd.Index([])
                
        except Exception:
            return np.array([]), np.array([]), pd.Index([])
            
    def create_features_for_prediction(
        self, df: pd.DataFrame, target_col: str, covariate_cols: List[str], 
        lags: List[int], anchor_date: pd.Timestamp, horizon: int = 1
    ) -> Tuple[np.ndarray, pd.Index]:
        """
        Create features for prediction at a specific anchor date.
        
        Args:
            df: DataFrame with date column and target/covariate columns
            target_col: Name of target column
            covariate_cols: List of covariate column names
            lags: List of lag values to use
            anchor_date: Date to anchor the prediction features on
            horizon: Forecast horizon
            
        Returns:
            Tuple of (X_features, time_index)
        """
        # Extend end_date so that the row with anchor=anchor_date exists in the DARTS output
        # (training-mode create_training_data needs y present)
        end_date = anchor_date + pd.Timedelta(weeks=horizon)
        X_all, y_all, times = self.create_training_data(
            df, target_col, covariate_cols, lags, end_date=end_date, horizon=horizon
        )
        
        if len(X_all) == 0:
            return np.empty((0, 0)), pd.Index([])
            
        # Find the row corresponding to anchor_date
        idx = np.where(times == anchor_date)[0]
        if len(idx) == 0:
            # Fallback: if time index granularity mismatches, take the last row <= anchor_date
            idx = np.where(times <= anchor_date)[0]
            if len(idx) == 0:
                return np.empty((0, X_all.shape[1] if X_all.size else 0)), pd.Index([])
            idx = [idx[-1]]
            
        X_one = X_all[idx[0]:idx[0]+1]
        return X_one, pd.Index([anchor_date])


def create_features(df: pd.DataFrame, target_col: str, covariate_cols: List[str], 
                   lags: List[int], end_date: pd.Timestamp = None, 
                   horizon: int = 1) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
    """
    Create lagged features using DARTS tabularization.
    
    This is the main function called by the training scripts.
    
    Args:
        df: DataFrame with date column and target/covariate columns
        target_col: Name of target column
        covariate_cols: List of covariate column names
        lags: List of lag values to use
        end_date: Optional end date to filter data
        horizon: Forecast horizon
        
    Returns:
        Tuple of (X_features, y_target, time_index)
    """
    tabularizer = DARTSTabularizer()
    return tabularizer.create_training_data(df, target_col, covariate_cols, lags, end_date, horizon)


def create_features_for_prediction(
    df: pd.DataFrame,
    target_col: str,
    covariate_cols: List[str],
    lags: List[int],
    anchor_date: pd.Timestamp,
    horizon: int = 1
) -> Tuple[np.ndarray, pd.Index]:
    """
    Create features for prediction at a specific anchor date.
    
    Args:
        df: DataFrame with date column and target/covariate columns
        target_col: Name of target column
        covariate_cols: List of covariate column names
        lags: List of lag values to use
        anchor_date: Date to anchor the prediction features on
        horizon: Forecast horizon
        
    Returns:
        Tuple of (X_features, time_index)
    """
    tabularizer = DARTSTabularizer()
    return tabularizer.create_features_for_prediction(
        df, target_col, covariate_cols, lags, anchor_date, horizon
    )
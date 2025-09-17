"""
Enhanced feature engineering for time series forecasting.

This module provides additional features beyond basic lags:
- Dense lag ladder (1-12, plus seasonal 52)
- Rolling statistics (mean, std, min, max)
- Differences and momentum features
- Seasonal encoding (sin/cos)
- Level and trend features
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from scipy import stats


def _create_enhanced_features_internal(df: pd.DataFrame, target_col: str, 
                                      covariate_cols: List[str], 
                                      use_dense_lags: bool = True,
                                      use_rolling_stats: bool = True,
                                      use_seasonal: bool = True,
                                      use_differences: bool = True) -> pd.DataFrame:
    """
    Internal helper function to create enhanced features without creating target or dropping rows.
    
    Args:
        df: DataFrame with date column and target/covariate columns
        target_col: Name of target column
        covariate_cols: List of covariate column names
        use_dense_lags: Include dense lag features
        use_rolling_stats: Include rolling statistics
        use_seasonal: Include seasonal features
        use_differences: Include difference features
        
    Returns:
        DataFrame with enhanced features
    """
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Initialize feature list
    feature_dfs = []
    
    # Get target values for alignment
    target_series = df[target_col].values
    dates = pd.to_datetime(df['date'])
    
    # 1. Dense lag features (key for LightGBM performance)
    if use_dense_lags:
        # Dense recent lags: 1-12 weeks
        dense_lags = list(range(1, 13))
        # Add seasonal lags
        seasonal_lags = [52]
        all_lags = dense_lags + seasonal_lags
        
        lag_features = {}
        for lag in all_lags:
            # Target lags
            lag_features[f'{target_col}_lag_{lag}'] = pd.Series(target_series).shift(lag)
            
            # Covariate lags (top 3 most important)
            for cov in covariate_cols[:3]:
                cov_series = df[cov].values
                lag_features[f'{cov}_lag_{lag}'] = pd.Series(cov_series).shift(lag)
        
        lag_df = pd.DataFrame(lag_features, index=df.index)
        feature_dfs.append(lag_df)
    
    # 2. Rolling statistics (causal - no future information)
    if use_rolling_stats:
        rolling_features = {}
        windows = [2, 4, 8, 12]  # Different window sizes
        
        for window in windows:
            # Rolling mean
            rolling_features[f'{target_col}_roll_mean_{window}'] = (
                pd.Series(target_series).rolling(window=window, min_periods=1).mean()
            )
            
            # Rolling std
            rolling_features[f'{target_col}_roll_std_{window}'] = (
                pd.Series(target_series).rolling(window=window, min_periods=2).std()
            )
            
            # Rolling min/max
            rolling_features[f'{target_col}_roll_min_{window}'] = (
                pd.Series(target_series).rolling(window=window, min_periods=1).min()
            )
            rolling_features[f'{target_col}_roll_max_{window}'] = (
                pd.Series(target_series).rolling(window=window, min_periods=1).max()
            )
        
        # Exponentially weighted moving average
        for alpha in [0.1, 0.3, 0.5]:
            rolling_features[f'{target_col}_ewm_alpha_{alpha}'] = (
                pd.Series(target_series).ewm(alpha=alpha, min_periods=1).mean()
            )
        
        rolling_df = pd.DataFrame(rolling_features, index=df.index)
        feature_dfs.append(rolling_df)
    
    # 3. Difference features (momentum indicators)
    if use_differences:
        diff_features = {}
        diff_lags = [1, 2, 4, 8, 52]
        
        for lag in diff_lags:
            # First difference
            diff_features[f'{target_col}_diff_{lag}'] = (
                pd.Series(target_series).diff(lag)
            )
            
            # Percentage change
            with np.errstate(divide='ignore', invalid='ignore'):
                pct_change = pd.Series(target_series).pct_change(lag)
                pct_change = pct_change.replace([np.inf, -np.inf], np.nan)
                diff_features[f'{target_col}_pct_change_{lag}'] = pct_change
        
        # Momentum (current vs N weeks ago)
        for lag in [4, 8, 12]:
            shifted = pd.Series(target_series).shift(lag)
            with np.errstate(divide='ignore', invalid='ignore'):
                momentum = target_series / (shifted + 1e-8) - 1
                momentum = pd.Series(momentum).replace([np.inf, -np.inf], np.nan)
                diff_features[f'{target_col}_momentum_{lag}'] = momentum
        
        diff_df = pd.DataFrame(diff_features, index=df.index)
        feature_dfs.append(diff_df)
    
    # 4. Seasonal features (sin/cos encoding)
    if use_seasonal:
        seasonal_features = {}
        
        # Week of year (continuous encoding)
        week_of_year = dates.dt.isocalendar().week.values
        # Handle week 53 properly (don't clip to 52)
        day_of_year = dates.dt.dayofyear.values
        year_progress = day_of_year / 365.25  # Normalized to [0, 1]
        
        # Sine/cosine encoding for cyclical nature
        seasonal_features['week_sin'] = np.sin(2 * np.pi * year_progress)
        seasonal_features['week_cos'] = np.cos(2 * np.pi * year_progress)
        
        # Quarter encoding
        seasonal_features['quarter_sin'] = np.sin(2 * np.pi * year_progress * 4)
        seasonal_features['quarter_cos'] = np.cos(2 * np.pi * year_progress * 4)
        
        # Month encoding
        seasonal_features['month_sin'] = np.sin(2 * np.pi * year_progress * 12)
        seasonal_features['month_cos'] = np.cos(2 * np.pi * year_progress * 12)
        
        # Binary indicators for flu season
        month = dates.dt.month.values
        seasonal_features['is_flu_season'] = ((month >= 10) | (month <= 3)).astype(int)
        seasonal_features['is_peak_flu'] = ((month == 12) | (month == 1) | (month == 2)).astype(int)
        
        seasonal_df = pd.DataFrame(seasonal_features, index=df.index)
        feature_dfs.append(seasonal_df)
    
    # 5. Level and trend features
    trend_features = {}
    
    # Cumulative statistics
    expanding_mean = pd.Series(target_series).expanding(min_periods=1).mean()
    trend_features[f'{target_col}_cumulative_mean'] = expanding_mean
    
    # Local trend using Theil-Sen estimator (robust linear trend)
    for window in [8, 12]:
        trend_vals = []
        for i in range(len(target_series)):
            if i < window:
                trend_vals.append(np.nan)
            else:
                y_window = target_series[i-window+1:i+1]
                x_window = np.arange(window)
                if len(y_window) == window and not np.all(np.isnan(y_window)):
                    try:
                        slope, _ = stats.theilslopes(y_window, x_window)
                        trend_vals.append(slope)
                    except:
                        trend_vals.append(np.nan)
                else:
                    trend_vals.append(np.nan)
        trend_features[f'{target_col}_trend_{window}w'] = trend_vals
    
    trend_df = pd.DataFrame(trend_features, index=df.index)
    feature_dfs.append(trend_df)
    
    # Combine all features
    X_df = pd.concat(feature_dfs, axis=1)
    
    # Add basic features (current values of covariates)
    for cov in covariate_cols:
        X_df[f'{cov}_current'] = df[cov].values
    
    # Fill remaining NaNs with forward fill
    X_df = X_df.ffill()
    
    # If still NaNs, fill with 0
    X_df = X_df.fillna(0)
    
    return X_df


def create_enhanced_features(df: pd.DataFrame, target_col: str, 
                            covariate_cols: List[str], 
                            end_date: pd.Timestamp = None,
                            horizon: int = 1,
                            use_dense_lags: bool = True,
                            use_rolling_stats: bool = True,
                            use_seasonal: bool = True,
                            use_differences: bool = True) -> Tuple[pd.DataFrame, np.ndarray, pd.Index]:
    """
    Create enhanced features for time series forecasting.
    
    Args:
        df: DataFrame with date column and target/covariate columns
        target_col: Name of target column
        covariate_cols: List of covariate column names
        end_date: Optional end date to filter data
        horizon: Forecast horizon
        use_dense_lags: Include dense lag features
        use_rolling_stats: Include rolling statistics
        use_seasonal: Include seasonal features
        use_differences: Include difference features
        
    Returns:
        Tuple of (X_features_df, y_target, time_index)
    """
    
    # Filter to end_date if provided
    if end_date is not None:
        df = df[df['date'] <= end_date].copy()
    else:
        df = df.copy()
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Get target values for alignment
    target_series = df[target_col].values
    dates = pd.to_datetime(df['date'])
    
    # Create features using internal helper
    X_df = _create_enhanced_features_internal(
        df, target_col, covariate_cols,
        use_dense_lags, use_rolling_stats, use_seasonal, use_differences
    )
    
    # Create target with horizon shift
    y = pd.Series(target_series).shift(-horizon).values
    
    # Remove rows with NaN in target (from shift) or too many NaN features
    valid_mask = ~np.isnan(y)
    
    # Also ensure we have enough non-NaN features (at least 50% of features)
    feature_nan_pct = X_df.isna().mean(axis=1)
    valid_mask = valid_mask & (feature_nan_pct < 0.5)
    
    # Apply mask
    X_df = X_df[valid_mask].reset_index(drop=True)
    y = y[valid_mask]
    # Note: time_index represents anchor dates, not target dates
    # The target date is anchor + horizon weeks
    time_index = dates[valid_mask]
    
    return X_df, y, time_index


def create_enhanced_features_for_prediction(
    df: pd.DataFrame,
    target_col: str,
    covariate_cols: List[str],
    anchor_date: pd.Timestamp,
    horizon: int = 1,
    use_dense_lags: bool = True,
    use_rolling_stats: bool = True,
    use_seasonal: bool = True,
    use_differences: bool = True,
) -> Tuple[np.ndarray, pd.Index]:
    """
    Create enhanced features for prediction at a specific anchor date.
    
    Args:
        df: DataFrame with date column and target/covariate columns
        target_col: Name of target column
        covariate_cols: List of covariate column names
        anchor_date: Date to anchor the prediction features on
        horizon: Forecast horizon
        use_dense_lags: Include dense lag features
        use_rolling_stats: Include rolling statistics
        use_seasonal: Include seasonal features
        use_differences: Include difference features
        
    Returns:
        Tuple of (X_features_array, time_index)
    """
    # Filter data to anchor_date (no need to extend beyond this for prediction)
    df = df[df["date"] <= anchor_date].copy().sort_values("date").reset_index(drop=True)
    
    # Create features using internal helper (no target creation or row dropping)
    X_df = _create_enhanced_features_internal(
        df, target_col, covariate_cols,
        use_dense_lags, use_rolling_stats, use_seasonal, use_differences
    )
    
    # Select the row with the anchor_date
    anchor_mask = df['date'] == anchor_date
    if anchor_mask.sum() == 0:
        # Fallback: if exact date not found, take the last row <= anchor_date
        anchor_mask = df['date'] <= anchor_date
        if anchor_mask.sum() == 0:
            return np.empty((0, X_df.shape[1] if X_df.size else 0)), pd.Index([])
        # Take the last row that is <= anchor_date
        idx = anchor_mask[::-1].idxmax()  # Find last True index
        X_one = X_df.iloc[idx:idx+1].values
        time_index = pd.Index([df.loc[idx, 'date']])
    else:
        # Take the row with exact anchor_date match
        X_one = X_df[anchor_mask].values
        time_index = pd.Index([anchor_date])
    
    return X_one, time_index


def add_external_features(X_df: pd.DataFrame, external_data: Optional[pd.DataFrame] = None,
                         external_cols: Optional[List[str]] = None,
                         allow_lag0: bool = False) -> pd.DataFrame:
    """
    Add external features like cases, ED visits, test positivity if available.
    
    Args:
        X_df: Existing feature DataFrame
        external_data: DataFrame with external data
        external_cols: Columns to use from external data
        allow_lag0: If True, include lag_0 (contemporaneous) features. 
                   WARNING: Only enable if data is truly available in real-time at anchor.
        
    Returns:
        Enhanced feature DataFrame
    """
    if external_data is not None and external_cols is not None:
        for col in external_cols:
            if col in external_data.columns:
                # Add with appropriate lags (exclude lag_0 by default to prevent leakage)
                lags = [0, 1, 2, 3, 4] if allow_lag0 else [1, 2, 3, 4]
                for lag in lags:
                    X_df[f'{col}_lag_{lag}'] = external_data[col].shift(lag)
    
    return X_df


def select_top_features(X_df: pd.DataFrame, y: np.ndarray, 
                       n_features: int = 50, method: str = 'lgbm') -> List[str]:
    """
    Select top N features using feature importance.
    
    Args:
        X_df: Feature DataFrame
        y: Target values
        n_features: Number of features to select
        method: Method to use ('lgbm' or 'mutual_info')
        
    Returns:
        List of selected feature names
    """
    import lightgbm as lgb
    from sklearn.feature_selection import mutual_info_regression
    
    if method == 'lgbm':
        # Use LightGBM feature importance
        train_data = lgb.Dataset(X_df, label=y, params={'verbose': -1})
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'verbosity': -1,
            'num_leaves': 31,
            'learning_rate': 0.05
        }
        model = lgb.train(params, train_data, num_boost_round=100)
        
        importance = pd.DataFrame({
            'feature': X_df.columns,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        return importance['feature'].head(n_features).tolist()
    
    elif method == 'mutual_info':
        # Use mutual information
        mi_scores = mutual_info_regression(X_df, y, random_state=42)
        importance = pd.DataFrame({
            'feature': X_df.columns,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)
        
        return importance['feature'].head(n_features).tolist()
    
    else:
        raise ValueError(f"Unknown method: {method}")

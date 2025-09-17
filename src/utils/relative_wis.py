"""
Relative WIS implementation following CDC FluSight methodology.

This module implements relative WIS as described in the CDC FluSight evaluation:
- Uses log-transformed counts to minimize impact of count magnitude
- Calculates geometric mean WIS for both model and baseline
- Returns ratio of model to baseline geometric mean WIS
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Tuple
from .wis_function_python import wis as cdc_wis


def calculate_geometric_mean_wis(
    y_true: np.ndarray,
    quantile_preds: np.ndarray,
    quantiles: np.ndarray,
    use_log_transform: bool = True
) -> float:
    """
    Calculate geometric mean WIS for a set of forecasts.
    
    According to CDC methodology:
    - Forecasts are evaluated using log-transformed counts
    - Geometric mean allows for more direct comparison across jurisdictions
    
    Args:
        y_true: True observed values
        quantile_preds: Quantile predictions, shape (n_samples, n_quantiles)
        quantiles: Quantile levels
        use_log_transform: Whether to apply log transformation (CDC standard)
        
    Returns:
        Geometric mean of WIS scores
    """
    # Apply log transformation if specified (CDC standard)
    if use_log_transform:
        # Add small constant to avoid log(0)
        epsilon = 1e-6
        y_true_transformed = np.log(y_true + epsilon)
        quantile_preds_transformed = np.log(quantile_preds + epsilon)
    else:
        y_true_transformed = y_true
        quantile_preds_transformed = quantile_preds
    
    # Calculate WIS for each observation
    wis_scores = cdc_wis(y_true_transformed, quantile_preds_transformed, quantiles)
    
    # Remove any invalid scores
    valid_scores = wis_scores[np.isfinite(wis_scores) & (wis_scores > 0)]
    
    if len(valid_scores) == 0:
        return np.nan
    
    # Calculate geometric mean
    # Geometric mean = exp(mean(log(scores)))
    geometric_mean = np.exp(np.mean(np.log(valid_scores)))
    
    return geometric_mean


def calculate_relative_wis(
    y_true: np.ndarray,
    model_quantile_preds: np.ndarray,
    baseline_quantile_preds: np.ndarray,
    quantiles: np.ndarray,
    use_log_transform: bool = True
) -> float:
    """
    Calculate relative WIS following CDC FluSight methodology.
    
    From CDC documentation:
    "Relative WIS was calculated using the geometric mean WIS of each model 
    forecast compared to the geometric mean WIS of the corresponding FluSight 
    baseline model forecast."
    
    Args:
        y_true: True observed values
        model_quantile_preds: Model's quantile predictions
        baseline_quantile_preds: Baseline's quantile predictions
        quantiles: Quantile levels
        use_log_transform: Whether to use log transformation (CDC standard)
        
    Returns:
        Relative WIS (values < 1 indicate better than baseline)
    """
    # Calculate geometric mean WIS for model
    model_geo_mean_wis = calculate_geometric_mean_wis(
        y_true, model_quantile_preds, quantiles, use_log_transform
    )
    
    # Calculate geometric mean WIS for baseline
    baseline_geo_mean_wis = calculate_geometric_mean_wis(
        y_true, baseline_quantile_preds, quantiles, use_log_transform
    )
    
    # Handle edge cases
    if np.isnan(baseline_geo_mean_wis) or baseline_geo_mean_wis == 0:
        return np.nan
    
    # Calculate relative WIS
    relative_wis = model_geo_mean_wis / baseline_geo_mean_wis
    
    return relative_wis


def generate_baseline_quantiles(
    last_value: float,
    quantiles: np.ndarray,
    historical_values: np.ndarray = None,
    horizon: int = 1
) -> np.ndarray:
    """
    Generate baseline quantile forecasts following CDC FluSight methodology.
    
    From CDC documentation:
    "The FluSight baseline model forecasted a median incidence equal to the 
    last week with uncertainty based on observation noise."
    
    Args:
        last_value: Last observed value
        quantiles: Quantile levels to generate
        historical_values: Historical values for estimating uncertainty
        horizon: Forecast horizon
        
    Returns:
        Array of quantile predictions
    """
    from scipy import stats
    
    if historical_values is not None and len(historical_values) > horizon:
        # Calculate h-step differences for uncertainty estimation
        differences = historical_values[horizon:] - historical_values[:-horizon]
        
        # Remove outliers using IQR method
        q1, q3 = np.percentile(differences, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_diffs = differences[(differences >= lower_bound) & (differences <= upper_bound)]
        
        if len(filtered_diffs) > 0:
            # Use empirical distribution
            forecast_distribution = last_value + filtered_diffs
            forecast_distribution = np.maximum(forecast_distribution, 0)  # Ensure non-negative
            
            quantile_forecasts = []
            for q in quantiles:
                quantile_forecasts.append(np.quantile(forecast_distribution, q))
            
            return np.array(quantile_forecasts)
    
    # Fallback: use normal distribution with estimated noise
    noise_std = max(0.25 * abs(last_value), 5.0)
    
    quantile_forecasts = []
    for q in quantiles:
        pred = stats.norm.ppf(q, loc=last_value, scale=noise_std)
        quantile_forecasts.append(max(pred, 0.0))
    
    return np.array(quantile_forecasts)


def evaluate_forecasts_relative_wis(
    forecasts_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    baseline_df: pd.DataFrame = None,
    quantile_columns: List[str] = None,
    group_by: List[str] = ['location', 'target_end_date']
) -> pd.DataFrame:
    """
    Evaluate forecasts using relative WIS for a dataset.
    
    Args:
        forecasts_df: DataFrame with model forecasts
        truth_df: DataFrame with observed values
        baseline_df: DataFrame with baseline forecasts (optional)
        quantile_columns: List of quantile column names
        group_by: Columns to group by for evaluation
        
    Returns:
        DataFrame with relative WIS scores
    """
    # If no baseline provided, generate it
    if baseline_df is None:
        # This would need to be implemented based on your data structure
        raise NotImplementedError("Automatic baseline generation not implemented")
    
    # Extract quantile levels from column names if not provided
    if quantile_columns is None:
        quantile_columns = [col for col in forecasts_df.columns if col.startswith('q_')]
    
    quantiles = np.array([float(col.split('_')[1]) for col in quantile_columns])
    
    results = []
    
    for group_vals, group_data in forecasts_df.groupby(group_by):
        # Get corresponding truth and baseline
        truth_mask = truth_df[group_by[0]] == group_vals[0]
        if len(group_by) > 1:
            for i, col in enumerate(group_by[1:], 1):
                truth_mask &= truth_df[col] == group_vals[i]
        
        truth_value = truth_df[truth_mask]['value'].values
        if len(truth_value) == 0:
            continue
        
        # Get baseline predictions
        baseline_mask = baseline_df[group_by[0]] == group_vals[0]
        if len(group_by) > 1:
            for i, col in enumerate(group_by[1:], 1):
                baseline_mask &= baseline_df[col] == group_vals[i]
        
        baseline_preds = baseline_df[baseline_mask][quantile_columns].values
        if len(baseline_preds) == 0:
            continue
        
        # Get model predictions
        model_preds = group_data[quantile_columns].values
        
        # Calculate relative WIS
        rel_wis = calculate_relative_wis(
            truth_value,
            model_preds,
            baseline_preds,
            quantiles,
            use_log_transform=True
        )
        
        result = {col: group_vals[i] for i, col in enumerate(group_by)}
        result['relative_wis'] = rel_wis
        results.append(result)
    
    return pd.DataFrame(results)


# CDC standard quantiles for FluSight
CDC_QUANTILES = np.array([
    0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99
])
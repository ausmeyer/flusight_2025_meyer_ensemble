"""
Scoring utilities for model evaluation.

This module provides evaluation metrics including MAE, WIS, and relative WIS
for the two-stage frozen-μ pipeline.
"""

import numpy as np
import sys
import os

# Import WIS function from local utils directory
from .wis_function_python import wis as cdc_wis
from .relative_wis import calculate_relative_wis, calculate_geometric_mean_wis


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAE score
    """
    return np.mean(np.abs(y_true - y_pred))


def wis(y_true: np.ndarray, quantile_preds: np.ndarray, quantiles: np.ndarray) -> float:
    """
    Calculate Weighted Interval Score using CDC FluSight methodology.
    
    Args:
        y_true: True values
        quantile_preds: Quantile predictions, shape (n_samples, n_quantiles)
        quantiles: Quantile levels
        
    Returns:
        Mean WIS score
    """
    wis_scores = cdc_wis(y_true, quantile_preds, quantiles)
    return np.mean(wis_scores)


def relative_wis(y_true: np.ndarray, model_quantile_preds: np.ndarray, 
                 baseline_quantile_preds: np.ndarray, quantiles: np.ndarray,
                 use_log_transform: bool = True) -> float:
    """
    Calculate relative WIS following CDC FluSight methodology.
    
    Args:
        y_true: True values
        model_quantile_preds: Model's quantile predictions
        baseline_quantile_preds: Baseline's quantile predictions  
        quantiles: Quantile levels
        use_log_transform: Whether to use log transformation (CDC standard)
        
    Returns:
        Relative WIS score (< 1 means better than baseline)
    """
    return calculate_relative_wis(y_true, model_quantile_preds, 
                                 baseline_quantile_preds, quantiles, 
                                 use_log_transform)


def geometric_mean_wis(y_true: np.ndarray, quantile_preds: np.ndarray, 
                       quantiles: np.ndarray, use_log_transform: bool = True) -> float:
    """
    Calculate geometric mean WIS following CDC methodology.
    
    Args:
        y_true: True values
        quantile_preds: Quantile predictions
        quantiles: Quantile levels
        use_log_transform: Whether to use log transformation (CDC standard)
        
    Returns:
        Geometric mean of WIS scores
    """
    return calculate_geometric_mean_wis(y_true, quantile_preds, quantiles, 
                                       use_log_transform)


def combined_mae_wis_loss(y_true: np.ndarray, quantile_preds: np.ndarray, 
                         quantiles: np.ndarray, mae_weight: float = 1.0, 
                         wis_weight: float = 0.0) -> float:
    """
    Combined loss function balancing MAE and WIS.
    
    Args:
        y_true: True values
        quantile_preds: Quantile predictions, shape (n_samples, n_quantiles)
        quantiles: Quantile levels
        mae_weight: Weight for MAE component (default 1.0 for pure MAE optimization)
        wis_weight: Weight for WIS component (default 0.0 for pure MAE optimization)
        
    Returns:
        Combined loss score
    """
    # Calculate WIS component
    wis_score = wis(y_true, quantile_preds, quantiles)
    
    # Calculate MAE component using median (0.5 quantile)
    median_idx = np.argmin(np.abs(quantiles - 0.5))
    predicted_medians = quantile_preds[:, median_idx]
    mae_score = mae(y_true, predicted_medians)
    
    # Normalize components to similar scales
    normalized_wis = wis_score / (np.mean(y_true) + 1.0) if np.mean(y_true) > 0 else wis_score
    normalized_mae = mae_score / (np.mean(y_true) + 1.0) if np.mean(y_true) > 0 else mae_score
    
    # Combined loss
    combined_loss = mae_weight * normalized_mae + wis_weight * normalized_wis
    
    return combined_loss
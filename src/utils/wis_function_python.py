import numpy as np

def wis(observed, predicted, quantile_level, separate_results=False, weigh=True):
    """
    Calculate the Weighted Interval Score (WIS) for quantile-based forecasts.

    Parameters:
    observed (array-like): Observed values
    predicted (array-like): Predicted quantiles, shape (n_observations, n_quantiles)
    quantile_level (array-like): Quantile levels corresponding to predicted quantiles
    separate_results (bool): If True, return separate components of WIS
    weigh (bool): If True, apply weights to the interval scores

    Returns:
    If separate_results is False:
        array-like: WIS values for each observation
    If separate_results is True:
        dict: Contains 'wis', 'dispersion', 'underprediction', and 'overprediction'
    """
    observed = np.asarray(observed)
    predicted = np.asarray(predicted)
    quantile_level = np.asarray(quantile_level)

    n_obs, n_quantiles = predicted.shape
    assert len(observed) == n_obs, "Number of observations must match number of predictions"
    assert len(quantile_level) == n_quantiles, "Number of quantile levels must match number of predicted quantiles"

    # Sort quantiles and predictions
    sort_idx = np.argsort(quantile_level)
    quantile_level = quantile_level[sort_idx]
    predicted = predicted[:, sort_idx]

    # Use exact CDC logic to avoid divide-by-zero
    # Find median index
    median_idx = np.argmin(np.abs(quantile_level - 0.5))
    
    # Calculate alphas for symmetric intervals around median
    alphas = 2 * np.abs(quantile_level[quantile_level < 0.5] - 0.5)
    
    # Weights: median weight (0.5) + interval weights (alpha/2)
    if weigh:
        weights = np.concatenate(([0.5], alphas/2))
    else:
        weights = np.ones(len(alphas) + 1)
    
    # Calculate scores
    scores = []
    
    # Median component (absolute error)
    median_score = np.abs(observed - predicted[:, median_idx])
    scores.append(median_score)
    
    # Interval components
    for alpha in alphas:
        lower_q = 0.5 - alpha/2
        upper_q = 0.5 + alpha/2
        
        # Find closest quantile indices
        lower_idx = np.argmin(np.abs(quantile_level - lower_q))
        upper_idx = np.argmin(np.abs(quantile_level - upper_q))
        
        lower_pred = predicted[:, lower_idx]
        upper_pred = predicted[:, upper_idx]
        
        # Interval score components
        interval_width = upper_pred - lower_pred
        penalty_lower = (2/alpha) * np.maximum(0, lower_pred - observed)
        penalty_upper = (2/alpha) * np.maximum(0, observed - upper_pred)
        
        interval_score = interval_width + penalty_lower + penalty_upper
        scores.append(interval_score)
    
    # Combine scores with weights
    scores_array = np.vstack(scores)  # Shape: (n_components, n_observations)
    # NEW: CDC weighted-sum scaling
    wis_values = np.sum(scores_array * weights[:, None], axis=0)

    if separate_results:
        # Extract components from scores_array for separate results
        if len(scores) > 1:
            # Build separate components from the calculated scores
            dispersion_scores = []
            underprediction_scores = []
            overprediction_scores = []
            
            # Extract components from interval scores (skip median score at index 0)
            for i, alpha in enumerate(alphas):
                lower_q = 0.5 - alpha/2
                upper_q = 0.5 + alpha/2
                lower_idx = np.argmin(np.abs(quantile_level - lower_q))
                upper_idx = np.argmin(np.abs(quantile_level - upper_q))
                
                lower_pred = predicted[:, lower_idx]
                upper_pred = predicted[:, upper_idx]
                
                dispersion_scores.append(upper_pred - lower_pred)
                underprediction_scores.append((2/alpha) * np.maximum(0, lower_pred - observed))
                overprediction_scores.append((2/alpha) * np.maximum(0, observed - upper_pred))
            
            avg_dispersion = np.mean(dispersion_scores, axis=0) if dispersion_scores else np.zeros_like(observed)
            avg_underprediction = np.mean(underprediction_scores, axis=0) if underprediction_scores else np.zeros_like(observed)
            avg_overprediction = np.mean(overprediction_scores, axis=0) if overprediction_scores else np.zeros_like(observed)
        else:
            # Only median score available
            avg_dispersion = np.zeros_like(observed)
            avg_underprediction = np.zeros_like(observed)
            avg_overprediction = np.zeros_like(observed)
        
        return {
            'wis': wis_values,
            'dispersion': avg_dispersion,
            'underprediction': avg_underprediction,
            'overprediction': avg_overprediction
        }
    else:
        return wis_values

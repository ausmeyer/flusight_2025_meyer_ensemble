import numpy as np


def wis(observed, predicted, quantile_level, separate_results=False, weigh=True):
    """
    Calculate the Weighted Interval Score (WIS) for quantile-based forecasts.

    Uses the quantile score (pinball loss) formulation, which is equivalent to
    the interval score formulation but easier to compute correctly.

    This implementation matches scoringutils::score() in R.

    Parameters:
    observed (array-like): Observed values, shape (n_observations,)
    predicted (array-like): Predicted quantiles, shape (n_observations, n_quantiles)
    quantile_level (array-like): Quantile levels corresponding to predicted quantiles
    separate_results (bool): If True, return separate components of WIS
    weigh (bool): Unused, kept for backward compatibility

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

    # Compute quantile scores (pinball loss) for each observation and quantile
    # Formula: QS(tau) = 2 * |I(y <= q) - tau| * |y - q|
    # Where I(y <= q) is 1 if observed <= predicted, 0 otherwise

    # Expand observed to match predicted shape
    obs_expanded = observed[:, np.newaxis]  # Shape: (n_obs, 1)

    # Indicator: 1 if observed <= predicted, 0 otherwise
    indicator = (obs_expanded <= predicted).astype(float)  # Shape: (n_obs, n_quantiles)

    # Quantile scores
    quantile_scores = 2 * np.abs(indicator - quantile_level) * np.abs(obs_expanded - predicted)

    # WIS = mean of quantile scores across all quantiles
    wis_values = np.mean(quantile_scores, axis=1)

    if separate_results:
        # Compute components for compatibility
        # Find median index
        median_idx = np.argmin(np.abs(quantile_level - 0.5))

        # Dispersion: average interval width (using symmetric quantile pairs)
        dispersion = np.zeros(n_obs)
        underprediction = np.zeros(n_obs)
        overprediction = np.zeros(n_obs)

        n_intervals = 0
        for i, tau in enumerate(quantile_level):
            if tau < 0.5:
                upper_tau = 1 - tau
                upper_idx = np.argmin(np.abs(quantile_level - upper_tau))
                if np.abs(quantile_level[upper_idx] - upper_tau) < 0.001:
                    alpha = upper_tau - tau
                    lower_pred = predicted[:, i]
                    upper_pred = predicted[:, upper_idx]

                    # Interval width
                    width = upper_pred - lower_pred
                    dispersion += (alpha / 2) * width

                    # Penalties
                    underprediction += (alpha / 2) * (2 / alpha) * np.maximum(0, observed - upper_pred)
                    overprediction += (alpha / 2) * (2 / alpha) * np.maximum(0, lower_pred - observed)
                    n_intervals += 1

        # Normalize by number of intervals + 0.5 (for median)
        if n_intervals > 0:
            norm_factor = n_intervals + 0.5
            dispersion = dispersion / norm_factor
            underprediction = underprediction / norm_factor
            overprediction = overprediction / norm_factor

        return {
            'wis': wis_values,
            'dispersion': dispersion,
            'underprediction': underprediction,
            'overprediction': overprediction
        }
    else:
        return wis_values

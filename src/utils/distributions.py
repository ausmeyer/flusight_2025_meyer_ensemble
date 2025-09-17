"""
Custom distributions for the two-stage frozen-μ pipeline.

This module contains the GaussianFrozenLoc distribution which freezes
the location parameter (μ) and only learns the scale parameter (σ).
"""

import numpy as np
from lightgbmlss.distributions.Gaussian import Gaussian
from lightgbmlss.distributions.distribution_utils import DistributionClass


class GaussianFrozenLoc(DistributionClass):
    """
    Custom Gaussian distribution that freezes the location (μ) parameter completely
    and only learns the scale (σ) parameter. The μ parameter is set via init_score
    and its gradients are zeroed out, forcing all boosting effort into σ learning.
    
    This implements the "frozen-μ" approach where:
    1. A standard LightGBM model provides μ predictions
    2. This LightGBMLSS model only learns σ with μ frozen
    """
    _class_printed = False
    
    def __init__(self, stabilization="MAD"):
        # Import required functions and classes
        from lightgbmlss.distributions.Gaussian import identity_fn, exp_fn, Gaussian_Torch
        
        # Set the parameters specific to the distribution
        distribution = Gaussian_Torch
        param_dict = {"loc": identity_fn, "scale": exp_fn}
        
        # Call parent constructor with required parameters
        super().__init__(
            distribution=distribution,
            univariate=True,
            discrete=False,
            n_dist_param=len(param_dict),
            stabilization=stabilization,
            param_dict=param_dict,
            distribution_arg_names=list(param_dict.keys()),
            loss_fn="nll"
        )
        
        # Use the original Gaussian for sampling, quantile calculation etc.
        self.dist_class = Gaussian()
        
    def compute_gradients_and_hessians(self, loss, predt, weights=None):
        """
        Freeze μ gradients completely and only allow σ learning.
        """
        # Print diagnostic message only once
        if not GaussianFrozenLoc._class_printed:
            print("*** Using GaussianFrozenLoc distribution (frozen μ, learning σ only) ***")
            GaussianFrozenLoc._class_printed = True
        
        # Get standard Gaussian gradients first
        grad, hess = self.dist_class.compute_gradients_and_hessians(loss, predt, weights)
        
        # Freeze μ parameters completely
        if grad.ndim == 1 and self.n_dist_param == 2:
            n_samples = len(grad) // 2
            
            # Zero out gradients for μ (first half)
            grad[:n_samples] = 0.0
            
            # Set tiny positive hessians for μ to keep booster happy
            hess[:n_samples] = 1e-12
        
        return grad, hess

    def quantile(self, quantiles: list, pred_dist: np.ndarray, **kwargs) -> np.ndarray:
        """
        Calculates the quantiles of the distribution using scipy.stats.
        """
        from scipy import stats
        import numpy as np
        
        # Convert to numpy array if it's a DataFrame
        if hasattr(pred_dist, 'values'):
            pred_dist = pred_dist.values
            
        # pred_dist should contain [loc, scale] parameters for each sample
        if pred_dist.ndim == 1:
            # Single sample case
            loc, scale = pred_dist[0], pred_dist[1]
            quantile_preds = stats.norm.ppf(quantiles, loc=loc, scale=scale)
        else:
            # Multiple samples case
            loc = pred_dist[:, 0]
            scale = pred_dist[:, 1]
            # Ensure scale is positive
            scale = np.maximum(scale, 1e-6)
            # Calculate quantiles for each sample
            quantile_preds = np.array([stats.norm.ppf(quantiles, loc=loc[i], scale=scale[i]) 
                                     for i in range(len(loc))])
        
        return quantile_preds
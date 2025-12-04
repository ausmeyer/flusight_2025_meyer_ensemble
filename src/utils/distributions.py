"""
Custom distributions for the two-stage frozen-μ pipeline.

This module contains the GaussianFrozenLoc distribution which freezes
the location parameter (μ) and only learns the scale parameter (σ).

Also contains GaussianFrozenLocBounded which bounds sigma to [0.1, 0.6]
for use with bounded-sigma mode.
"""

import numpy as np
from lightgbmlss.distributions.Gaussian import Gaussian
from lightgbmlss.distributions.distribution_utils import DistributionClass


def bounded_sigmoid_fn(x, min_val=0.1, max_val=0.6):
    """
    Bounded response function that maps any real value to [min_val, max_val].
    Uses sigmoid to bound the range: min_val + (max_val - min_val) * sigmoid(x)

    For log-space forecasting:
    - sigma=0.3 means ~65% of values within factor of 1.35x
    - sigma=0.5 means ~65% of values within factor of 1.65x
    - sigma=0.6 means ~65% of values within factor of 1.82x

    Works with both NumPy arrays and PyTorch tensors.
    """
    # Check if input is a PyTorch tensor
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x_clamped = torch.clamp(x, -20, 20)
            sigmoid = 1.0 / (1.0 + torch.exp(-x_clamped))
            return min_val + (max_val - min_val) * sigmoid
    except ImportError:
        pass

    # NumPy path
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
    return min_val + (max_val - min_val) * sigmoid


class BoundedSigmoidFn:
    """
    Picklable callable class for bounded sigmoid response function.

    This replaces the closure-based approach which cannot be pickled.
    Used as the scale response function in GaussianFrozenLocBounded.
    """
    def __init__(self, min_val=0.15, max_val=0.45):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, x):
        return bounded_sigmoid_fn(x, self.min_val, self.max_val)

    def __reduce__(self):
        """Enable pickling by returning constructor and args."""
        return (BoundedSigmoidFn, (self.min_val, self.max_val))


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


class GaussianFrozenLocBounded(DistributionClass):
    """
    Bounded version of GaussianFrozenLoc that constrains sigma to [0.15, 0.45].

    Uses a sigmoid response function instead of exp for the scale parameter,
    which naturally bounds sigma to a reasonable range for log-space forecasting.

    This ensures the model learns meaningful uncertainty within the bounded range
    rather than predicting unbounded sigma values that need to be clipped.

    IMPORTANT: This class uses the parent's compute_gradients_and_hessians to ensure
    autograd correctly handles the bounded sigmoid response function.
    """
    _class_printed = False

    def __init__(self, stabilization="MAD", sigma_min=0.15, sigma_max=0.45):
        from lightgbmlss.distributions.Gaussian import identity_fn, Gaussian_Torch

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # Use picklable callable class instead of closure
        # Tighter range [0.15, 0.45] allows model to learn meaningful variation
        # sigma=0.3 (midpoint) means ~65% within factor of 1.35x
        bounded_scale_fn = BoundedSigmoidFn(sigma_min, sigma_max)

        distribution = Gaussian_Torch
        param_dict = {"loc": identity_fn, "scale": bounded_scale_fn}

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

    def compute_gradients_and_hessians(self, loss, predt, weights=None):
        """
        Freeze μ gradients completely and only allow σ learning.

        Uses parent class gradient computation to ensure autograd correctly
        handles the bounded sigmoid response function for scale.
        """
        if not GaussianFrozenLocBounded._class_printed:
            print(f"*** Using GaussianFrozenLocBounded (frozen μ, σ ∈ [{self.sigma_min}, {self.sigma_max}]) ***")
            GaussianFrozenLocBounded._class_printed = True

        # Use parent class gradient computation (which uses self.param_dict with bounded sigmoid)
        grad, hess = super().compute_gradients_and_hessians(loss, predt, weights)

        if grad.ndim == 1 and self.n_dist_param == 2:
            n_samples = len(grad) // 2
            grad[:n_samples] = 0.0
            hess[:n_samples] = 1e-12

        return grad, hess

    def quantile(self, quantiles: list, pred_dist: np.ndarray, **kwargs) -> np.ndarray:
        """
        Calculates the quantiles of the distribution using scipy.stats.
        """
        from scipy import stats

        if hasattr(pred_dist, 'values'):
            pred_dist = pred_dist.values

        if pred_dist.ndim == 1:
            loc, scale = pred_dist[0], pred_dist[1]
            quantile_preds = stats.norm.ppf(quantiles, loc=loc, scale=scale)
        else:
            loc = pred_dist[:, 0]
            scale = pred_dist[:, 1]
            scale = np.maximum(scale, 1e-6)
            quantile_preds = np.array([stats.norm.ppf(quantiles, loc=loc[i], scale=scale[i])
                                     for i in range(len(loc))])

        return quantile_preds


class GaussianFrozenLocBoundedWide(DistributionClass):
    """
    Wide-bounded version of GaussianFrozenLoc that constrains sigma to [0.1, 0.8].

    This variant has a wider sigma range than GaussianFrozenLocBounded to allow
    the model to learn larger uncertainty when the data supports it.

    For log-space forecasting:
    - sigma=0.3 means ~65% of values within factor of 1.35x
    - sigma=0.5 means ~65% of values within factor of 1.65x
    - sigma=0.8 means ~65% of values within factor of 2.23x

    IMPORTANT: This class uses the parent's compute_gradients_and_hessians to ensure
    autograd correctly handles the bounded sigmoid response function.
    """
    _class_printed = False

    def __init__(self, stabilization="MAD", sigma_min=0.1, sigma_max=0.8):
        from lightgbmlss.distributions.Gaussian import identity_fn, Gaussian_Torch

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        bounded_scale_fn = BoundedSigmoidFn(sigma_min, sigma_max)

        distribution = Gaussian_Torch
        param_dict = {"loc": identity_fn, "scale": bounded_scale_fn}

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

    def compute_gradients_and_hessians(self, loss, predt, weights=None):
        """
        Freeze μ gradients completely and only allow σ learning.

        Uses parent class gradient computation to ensure autograd correctly
        handles the bounded sigmoid response function for scale.
        """
        if not GaussianFrozenLocBoundedWide._class_printed:
            print(f"*** Using GaussianFrozenLocBoundedWide (frozen μ, σ ∈ [{self.sigma_min}, {self.sigma_max}]) ***")
            GaussianFrozenLocBoundedWide._class_printed = True

        # Use parent class gradient computation (which uses self.param_dict with bounded sigmoid)
        grad, hess = super().compute_gradients_and_hessians(loss, predt, weights)

        if grad.ndim == 1 and self.n_dist_param == 2:
            n_samples = len(grad) // 2
            grad[:n_samples] = 0.0
            hess[:n_samples] = 1e-12

        return grad, hess

    def quantile(self, quantiles: list, pred_dist: np.ndarray, **kwargs) -> np.ndarray:
        """
        Calculates the quantiles of the distribution using scipy.stats.
        """
        from scipy import stats

        if hasattr(pred_dist, 'values'):
            pred_dist = pred_dist.values

        if pred_dist.ndim == 1:
            loc, scale = pred_dist[0], pred_dist[1]
            quantile_preds = stats.norm.ppf(quantiles, loc=loc, scale=scale)
        else:
            loc = pred_dist[:, 0]
            scale = pred_dist[:, 1]
            scale = np.maximum(scale, 1e-6)
            quantile_preds = np.array([stats.norm.ppf(quantiles, loc=loc[i], scale=scale[i])
                                     for i in range(len(loc))])

        return quantile_preds
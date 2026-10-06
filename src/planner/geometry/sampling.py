from __future__ import annotations

import math
import numpy as np
from typing import Dict, Any

from .bspline import BSplineCurve, evaluate, evaluate_derivative

def compute_sample_count(v_max: float, d_safe: float, t_max: float) -> int:
    """
    Compute minimum number of samples to ensure spacing between samples < d_safe.
    
    Args:
        v_max: Maximum velocity magnitude.
        d_safe: Maximum safe distance between consecutive samples.
        t_max: Total duration of the trajectory.
        
    Returns:
        int: Number of samples, with a 20% safety margin applied.
    """
    base_count = math.ceil(v_max * t_max / d_safe) + 1
    safe_count = max(base_count, 10)
    
    # 20% safety margin
    final_count = math.ceil(safe_count * 1.2)
    return final_count


def uniform_parameter_samples(n_samples: int) -> np.ndarray:
    """
    Return uniformly spaced parameter values in [0, 1].
    
    Args:
        n_samples: Number of samples.
        
    Returns:
        np.ndarray: Parametric values, shape (n_samples,).
    """
    return np.linspace(0.0, 1.0, n_samples)


def sample_trajectory(curve: BSplineCurve, n_samples: int) -> Dict[str, np.ndarray]:
    """
    Sample position, and parametric derivatives at uniform parametric points.
    
    Args:
        curve: B-spline curve to sample.
        n_samples: Number of samples to take.
        
    Returns:
        dict: Dictionary containing sampled parametric data.
    """
    u = uniform_parameter_samples(n_samples)
    
    positions = evaluate(curve, u)
    param_vel = evaluate_derivative(curve, u, 1)
    param_acc = evaluate_derivative(curve, u, 2)
    
    # Evaluate snap (order 4) if degree allows, else zeros
    param_snap = evaluate_derivative(curve, u, 4) if curve.degree >= 4 else np.zeros_like(positions)
    
    return {
        'u': u,
        'positions': positions,
        'param_vel': param_vel,
        'param_acc': param_acc,
        'param_snap': param_snap
    }


def parametric_to_physical(samples: Dict[str, np.ndarray], t_max: float) -> Dict[str, np.ndarray]:
    """
    Convert parametric derivatives to physical derivatives.
    
    Args:
        samples: Dictionary of parametric samples.
        t_max: Total time duration of the physical trajectory.
        
    Returns:
        dict: A new dictionary containing both parametric and physical data.
    """
    if t_max <= 0:
        raise ValueError("t_max must be positive")
        
    physical_samples = dict(samples)
    
    u = samples['u']
    
    physical_samples['t'] = u * t_max
    
    # dp/dt = dp/du * (1/T_max)
    physical_samples['velocity'] = samples['param_vel'] / t_max
    
    # d2p/dt2 = d2p/du2 * (1/T_max^2)
    physical_samples['acceleration'] = samples['param_acc'] / (t_max ** 2)
    
    # d4p/dt4 = d4p/du4 * (1/T_max^4)
    physical_samples['snap'] = samples['param_snap'] / (t_max ** 4)
    
    return physical_samples

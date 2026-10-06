from __future__ import annotations

import numpy as np

from planner.environment.obstacles import compute_min_distance


def compute_path_length(positions: np.ndarray) -> float:
    """
    Compute the path length as the sum of Euclidean distances between consecutive sample points.
    
    Args:
        positions: Array of shape (n_samples, dim) containing the position at each sample.
        
    Returns:
        float: The numerical approximation of the arc length.
    """
    diffs = np.diff(positions, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    return float(np.sum(dists))


def compute_clearance_risk(
    positions: np.ndarray,
    obstacles: list,
    d_safe: float,
    d_comfort: float | None = None,
    times: np.ndarray | None = None
) -> float:
    """
    Compute the clearance risk, penalizing samples that are closer than d_safe to obstacles.
    
    Args:
        positions: Array of shape (n_samples, dim).
        obstacles: List of obstacles in the environment.
        d_safe: The hard safe distance threshold.
        d_comfort: The comfort threshold where risk decays to zero. Defaults
            to d_safe for backwards-compatible hard-threshold behavior.
        times: Optional array of times for dynamic obstacles, shape (n_samples,).
        
    Returns:
        float: The clearance risk.
    """
    n_samples = positions.shape[0]
    if n_samples == 0:
        return 0.0
    if not obstacles:
        return 0.0
    if d_comfort is None:
        d_comfort = d_safe
    if d_comfort < d_safe:
        raise ValueError("d_comfort must be greater than or equal to d_safe.")

    risk_sum = 0.0
    for i in range(n_samples):
        t = 0.0 if times is None else times[i]
        d_i = compute_min_distance(positions[i], obstacles, t)
        if d_i <= d_safe:
            risk_sum += 1.0
        elif d_i < d_comfort:
            if d_comfort == d_safe:
                continue
            risk_sum += ((d_comfort - d_i) / (d_comfort - d_safe)) ** 2
            
    return float(risk_sum / n_samples)


def compute_snap_objective(snap_samples: np.ndarray, t_max: float, n_samples: int) -> float:
    """
    Compute the integrated squared snap over the trajectory using the trapezoidal rule.
    
    Args:
        snap_samples: Array of shape (n_samples, dim) with physical snap vectors.
        t_max: Total duration of the trajectory.
        n_samples: Total number of samples.
        
    Returns:
        float: The integrated squared snap.
    """
    if n_samples <= 1:
        return 0.0
        
    snap_sq = np.sum(snap_samples**2, axis=1)
    dt = t_max / (n_samples - 1)
    
    # NumPy versions differ on the trapezoidal integration helper name.
    _trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    return float(_trapz(snap_sq, dx=dt))


def compute_objectives(
    positions: np.ndarray,
    snap_samples: np.ndarray,
    obstacles: list,
    d_safe: float,
    t_max: float,
    d_comfort: float | None = None,
    times: np.ndarray | None = None
) -> tuple[np.ndarray, float]:
    """
    Compute all three objectives and return the raw physical snap.
    
    Args:
        positions: Array of shape (n_samples, dim).
        snap_samples: Array of shape (n_samples, dim).
        obstacles: List of obstacles.
        d_safe: Safe distance threshold.
        d_comfort: Comfort distance threshold for clearance risk.
        t_max: Total time.
        times: Optional array of sample times.
        
    Returns:
        tuple[np.ndarray, float]: 
            - Array of shape (3,) containing [path_length, clearance_risk, snap_objective], where snap_objective is log-scaled.
            - The raw physical integrated squared snap.
    """
    n_samples = positions.shape[0]
    
    path_length = compute_path_length(positions)
    clearance_risk = compute_clearance_risk(positions, obstacles, d_safe, d_comfort, times)
    raw_snap = compute_snap_objective(snap_samples, t_max, n_samples)
    
    # Log-scaled snap for the optimizer
    snap_obj = float(np.log10(1.0 + raw_snap))
    
    return np.array([path_length, clearance_risk, snap_obj], dtype=float), raw_snap

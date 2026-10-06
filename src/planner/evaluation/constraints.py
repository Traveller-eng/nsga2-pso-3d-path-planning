from __future__ import annotations

import numpy as np

from planner.environment.obstacles import compute_min_distance


def compute_collision_violation(positions: np.ndarray, obstacles: list, d_safe: float, times: np.ndarray | None = None) -> float:
    """
    Compute the normalized collision violation.
    
    Args:
        positions: Path positions, shape (n_samples, dim)
        obstacles: List of obstacles.
        d_safe: Safe clearance distance.
        times: Sample times.
        
    Returns:
        float: Normalized violation.
    """
    if not obstacles:
        return 0.0
        
    max_violation = 0.0
    
    for i in range(positions.shape[0]):
        t = 0.0 if times is None else times[i]
        d_i = compute_min_distance(positions[i], obstacles, t)
        if d_i < d_safe:
            max_violation = max(max_violation, (d_safe - d_i) / d_safe)
            
    return float(max_violation)


def compute_velocity_violation(velocity: np.ndarray, v_max: float) -> float:
    """
    Compute the normalized velocity constraint violation.
    """
    speeds = np.linalg.norm(velocity, axis=1)
    violations = np.maximum(0.0, (speeds - v_max) / v_max)
    return float(np.max(violations))


def compute_acceleration_violation(acceleration: np.ndarray, a_max: float) -> float:
    """
    Compute the normalized acceleration constraint violation.
    """
    accels = np.linalg.norm(acceleration, axis=1)
    violations = np.maximum(0.0, (accels - a_max) / a_max)
    return float(np.max(violations))


def compute_bounds_violation(positions: np.ndarray, lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> float:
    """
    Compute the normalized bounds constraint violation.
    """
    bounds_range = np.max(upper_bounds - lower_bounds)
    if bounds_range == 0.0:
        bounds_range = 1.0
        
    lower_viol = np.maximum(0.0, lower_bounds - positions) / bounds_range
    upper_viol = np.maximum(0.0, positions - upper_bounds) / bounds_range
    return float(np.max(lower_viol + upper_viol))


def compute_total_violation(
    positions: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    obstacles: list,
    d_safe: float,
    v_max: float,
    a_max: float,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    times: np.ndarray | None = None
) -> tuple[float, dict[str, float]]:
    """
    Compute total normalized constraint violation and specific violations.
    """
    col_viol = compute_collision_violation(positions, obstacles, d_safe, times)
    vel_viol = compute_velocity_violation(velocity, v_max)
    acc_viol = compute_acceleration_violation(acceleration, a_max)
    bounds_viol = compute_bounds_violation(positions, lower_bounds, upper_bounds)
    
    total = col_viol + vel_viol + acc_viol + bounds_viol
    
    details = {
        'collision': col_viol,
        'velocity': vel_viol,
        'acceleration': acc_viol,
        'bounds': bounds_viol
    }
    
    return float(total), details


def is_feasible(total_violation: float) -> bool:
    """
    Check if a solution is feasible given its total violation score.
    """
    return total_violation <= 1e-12


def deb_constrained_dominates(obj_a: np.ndarray, viol_a: float, obj_b: np.ndarray, viol_b: float) -> bool:
    """
    Deb's constrained domination principle.
    
    Returns True if solution A dominates solution B based on constraints and objectives.
    """
    feas_a = is_feasible(viol_a)
    feas_b = is_feasible(viol_b)
    
    if feas_a and not feas_b:
        return True
    elif not feas_a and feas_b:
        return False
    elif not feas_a and not feas_b:
        return viol_a < viol_b
    else:
        better_in_all = np.all(obj_a <= obj_b)
        strictly_better_in_one = np.any(obj_a < obj_b)
        return bool(better_in_all and strictly_better_in_one)

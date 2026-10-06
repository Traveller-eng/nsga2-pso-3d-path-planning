from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline
from dataclasses import dataclass
from typing import Union

@dataclass
class BSplineCurve:
    """
    Represents a B-spline curve.
    
    Attributes:
        control_points: Control points array of shape (n_cp, dim).
        knot_vector: Knot vector array.
        degree: Degree of the B-spline (default: 5).
    """
    control_points: np.ndarray
    knot_vector: np.ndarray
    degree: int = 5


def make_clamped_knot_vector(n_control_points: int, degree: int = 5) -> np.ndarray:
    """
    Creates a clamped (open) knot vector for a B-spline in domain [0, 1].
    
    The clamped knot vector ensures the curve passes through the first and last
    control points. The first (degree + 1) knots are 0.0 and the last
    (degree + 1) knots are 1.0. Interior knots are uniformly distributed.
    
    Args:
        n_control_points: Number of control points.
        degree: Degree of the B-spline curve.
        
    Returns:
        np.ndarray: Knot vector of shape (n_control_points + degree + 1,).
    """
    if n_control_points < degree + 1:
        raise ValueError(f"Number of control points ({n_control_points}) must be >= degree + 1 ({degree + 1})")
        
    m = n_control_points + degree + 1
    # Number of interior knots
    num_interior = m - 2 * (degree + 1)
    
    knots = np.zeros(m)
    knots[-(degree + 1):] = 1.0
    
    if num_interior > 0:
        # Uniformly space the interior knots between 0 and 1
        interior_knots = np.linspace(0, 1, num_interior + 2)[1:-1]
        knots[degree + 1 : degree + 1 + num_interior] = interior_knots
        
    return knots


def evaluate(curve: BSplineCurve, t: Union[float, np.ndarray]) -> np.ndarray:
    """
    Evaluate the B-spline curve at parametric values t.
    
    Args:
        curve: The B-spline curve object.
        t: Parametric evaluation points in [0, 1]. Can be scalar or 1D array.
        
    Returns:
        np.ndarray: Evaluated points. Shape is (len(t), dim) if t is array,
                    or (dim,) if t is scalar.
    """
    t_clamped = np.clip(t, 0.0, 1.0)
    spline = BSpline(curve.knot_vector, curve.control_points, curve.degree, extrapolate=False)
    return spline(t_clamped)


def evaluate_derivative(curve: BSplineCurve, t: Union[float, np.ndarray], order: int) -> np.ndarray:
    """
    Evaluate the k-th derivative of the B-spline at parameter values t.
    
    Args:
        curve: The B-spline curve object.
        t: Parametric evaluation points in [0, 1]. Can be scalar or 1D array.
        order: The derivative order (e.g., 1 for velocity, 2 for acceleration).
        
    Returns:
        np.ndarray: Evaluated derivative points. Shape is (len(t), dim) if t is array,
                    or (dim,) if t is scalar.
    """
    if order == 0:
        return evaluate(curve, t)
        
    t_clamped = np.clip(t, 0.0, 1.0)
    spline = BSpline(curve.knot_vector, curve.control_points, curve.degree, extrapolate=False)
    derivative_spline = spline.derivative(order)
    return derivative_spline(t_clamped)


def create_trajectory_bspline(control_points: np.ndarray, degree: int = 5) -> BSplineCurve:
    """
    Create a clamped B-spline curve from control points.
    
    Args:
        control_points: Control points array of shape (n_cp, dim).
        degree: Degree of the B-spline curve.
        
    Returns:
        BSplineCurve: The constructed B-spline curve object.
    """
    control_points = np.asarray(control_points)
    n_cp = control_points.shape[0]
    
    if n_cp < degree + 1:
        raise ValueError(f"Number of control points ({n_cp}) must be >= degree + 1 ({degree + 1})")
        
    knot_vector = make_clamped_knot_vector(n_cp, degree)
    return BSplineCurve(control_points, knot_vector, degree)

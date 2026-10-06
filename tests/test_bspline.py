import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from planner.geometry.bspline import (
    BSplineCurve,
    make_clamped_knot_vector,
    evaluate,
    evaluate_derivative,
    create_trajectory_bspline
)

def test_knot_vector_length():
    n_cp = 8
    degree = 5
    knots = make_clamped_knot_vector(n_cp, degree)
    assert len(knots) == n_cp + degree + 1

def test_knot_vector_clamped():
    n_cp = 8
    degree = 5
    knots = make_clamped_knot_vector(n_cp, degree)
    np.testing.assert_allclose(knots[:degree+1], 0.0)
    np.testing.assert_allclose(knots[-(degree+1):], 1.0)

def test_knot_vector_non_decreasing():
    n_cp = 8
    degree = 5
    knots = make_clamped_knot_vector(n_cp, degree)
    diffs = np.diff(knots)
    assert np.all(diffs >= -1e-12)

def test_knot_vector_too_few_points():
    n_cp = 5
    degree = 5
    with pytest.raises(ValueError):
        make_clamped_knot_vector(n_cp, degree)

def test_evaluate_start_equals_first_cp():
    cps = np.random.rand(8, 3)
    curve = create_trajectory_bspline(cps, degree=5)
    p = evaluate(curve, 0.0)
    np.testing.assert_allclose(p, cps[0])

def test_evaluate_end_equals_last_cp():
    cps = np.random.rand(8, 3)
    curve = create_trajectory_bspline(cps, degree=5)
    p = evaluate(curve, 1.0)
    np.testing.assert_allclose(p, cps[-1])

def test_evaluate_start_2d():
    cps = np.random.rand(8, 2)
    curve = create_trajectory_bspline(cps, degree=5)
    p = evaluate(curve, 0.0)
    np.testing.assert_allclose(p, cps[0])

def test_evaluate_dimensions_3d():
    cps = np.random.rand(8, 3)
    curve = create_trajectory_bspline(cps, degree=5)
    t = np.linspace(0, 1, 10)
    p = evaluate(curve, t)
    assert p.shape == (10, 3)

def test_evaluate_dimensions_2d():
    cps = np.random.rand(8, 2)
    curve = create_trajectory_bspline(cps, degree=5)
    t = np.linspace(0, 1, 10)
    p = evaluate(curve, t)
    assert p.shape == (10, 2)

def test_evaluate_scalar_t():
    cps = np.random.rand(8, 3)
    curve = create_trajectory_bspline(cps, degree=5)
    p = evaluate(curve, 0.5)
    assert p.shape == (3,)

def test_derivative_dimensions():
    cps = np.random.rand(8, 3)
    curve = create_trajectory_bspline(cps, degree=5)
    t = np.linspace(0, 1, 10)
    d = evaluate_derivative(curve, t, order=1)
    assert d.shape == (10, 3)

def test_derivative_order_0_equals_evaluate():
    cps = np.random.rand(8, 3)
    curve = create_trajectory_bspline(cps, degree=5)
    t = np.linspace(0, 1, 10)
    p0 = evaluate(curve, t)
    d0 = evaluate_derivative(curve, t, order=0)
    np.testing.assert_allclose(d0, p0)

def test_snap_evaluable():
    cps = np.random.rand(8, 3)
    curve = create_trajectory_bspline(cps, degree=5)
    t = np.linspace(0, 1, 10)
    snap = evaluate_derivative(curve, t, order=4)
    assert snap is not None

def test_snap_dimensions():
    cps = np.random.rand(8, 3)
    curve = create_trajectory_bspline(cps, degree=5)
    t = np.linspace(0, 1, 10)
    snap = evaluate_derivative(curve, t, order=4)
    assert snap.shape == (10, 3)

def test_straight_line_bspline():
    """Collinear CPs produce a geometrically straight line (non-uniform param is OK)."""
    t_cps = np.linspace(0, 1, 8)
    cps = np.zeros((8, 3))
    cps[:, 0] = t_cps * 10.0
    cps[:, 1] = t_cps * 5.0
    cps[:, 2] = t_cps * 2.0
    
    curve = create_trajectory_bspline(cps, degree=5)
    t_eval = np.linspace(0, 1, 50)
    p = evaluate(curve, t_eval)
    
    # Verify endpoints
    np.testing.assert_allclose(p[0], cps[0], atol=1e-10)
    np.testing.assert_allclose(p[-1], cps[-1], atol=1e-10)
    
    # Verify collinearity: cross product of (p_i - p_0) x (p_end - p_0) should be ~0
    direction = cps[-1] - cps[0]
    for i in range(1, len(p) - 1):
        cross = np.cross(p[i] - cps[0], direction)
        np.testing.assert_allclose(cross, 0.0, atol=1e-10)

def test_create_trajectory_bspline_validation():
    cps = np.random.rand(5, 3)
    with pytest.raises(ValueError):
        create_trajectory_bspline(cps, degree=5)

def test_endpoint_derivative_behavior():
    """
    Explicitly tests and documents the endpoint derivative behavior of the current B-spline implementation.
    A degree-5 clamped B-spline with strictly distinct control points (like our straight-line init) 
    does NOT have zero velocity or zero acceleration at the endpoints.
    v(0) is proportional to (p1 - p0).
    """
    n_cp = 8
    t_max = 5.0
    
    # Collinear, evenly spaced CPs from (0,0,0) to (10,10,10)
    start = np.array([0.0, 0.0, 0.0])
    goal = np.array([10.0, 10.0, 10.0])
    fractions = np.linspace(0, 1, n_cp).reshape(-1, 1)
    cps = start + fractions * (goal - start)
    
    curve = create_trajectory_bspline(cps, degree=5)
    
    # Evaluate at u=0 and u=1
    u_eval = np.array([0.0, 1.0])
    p = evaluate(curve, u_eval)
    v_param = evaluate_derivative(curve, u_eval, order=1)
    a_param = evaluate_derivative(curve, u_eval, order=2)
    
    # Parametric to physical conversion factors
    dt_du = t_max
    v_phys = v_param / dt_du
    a_phys = a_param / (dt_du**2)
    
    # p(0) == start, p(1) == goal
    np.testing.assert_allclose(p[0], start)
    np.testing.assert_allclose(p[1], goal)
    
    # The derivatives are definitively NON-ZERO because p1 != p0 and p_{n-2} != p_{n-1}
    assert np.linalg.norm(v_phys[0]) > 1.0
    assert np.linalg.norm(v_phys[1]) > 1.0
    assert np.linalg.norm(a_phys[0]) > 1.0
    assert np.linalg.norm(a_phys[1]) > 1.0

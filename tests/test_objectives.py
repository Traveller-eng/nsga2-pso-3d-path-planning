import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from planner.evaluation.objectives import (
    compute_path_length,
    compute_clearance_risk,
    compute_snap_objective,
    compute_objectives
)
from planner.environment.obstacles import SphereObstacle

def test_path_length_straight_line():
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0]
    ])
    length = compute_path_length(positions)
    expected = 2.0 * np.sqrt(3.0)
    np.testing.assert_allclose(length, expected)

def test_path_length_positive():
    positions = np.random.rand(10, 3)
    length = compute_path_length(positions)
    assert length > 0.0

def test_path_length_zero():
    positions = np.array([[1.0, 2.0, 3.0]])
    length = compute_path_length(positions)
    assert length == 0.0

def test_clearance_risk_zero_no_obstacles():
    positions = np.random.rand(10, 3)
    obstacles = []
    risk = compute_clearance_risk(positions, obstacles, d_safe=1.0)
    assert risk == 0.0

def test_clearance_risk_zero_far_away():
    positions = np.array([[0.0, 0.0, 0.0]])
    obstacles = [SphereObstacle(center=np.array([10.0, 10.0, 10.0]), radius=1.0)]
    risk = compute_clearance_risk(positions, obstacles, d_safe=1.0)
    assert risk == 0.0

def test_clearance_risk_positive_near_obstacle():
    positions = np.array([[0.0, 0.0, 0.0]])
    obstacles = [SphereObstacle(center=np.array([0.5, 0.0, 0.0]), radius=0.1)]
    risk = compute_clearance_risk(positions, obstacles, d_safe=1.0)
    assert risk > 0.0

def test_clearance_risk_quadratic_comfort_zone():
    positions = np.array([[1.5, 0.0, 0.0]])
    obstacles = [SphereObstacle(center=np.array([0.0, 0.0, 0.0]), radius=0.0)]
    risk = compute_clearance_risk(positions, obstacles, d_safe=1.0, d_comfort=2.0)
    assert risk == pytest.approx(0.25)

def test_clearance_risk_one_inside_safe_distance():
    positions = np.array([[0.5, 0.0, 0.0]])
    obstacles = [SphereObstacle(center=np.array([0.0, 0.0, 0.0]), radius=0.0)]
    risk = compute_clearance_risk(positions, obstacles, d_safe=1.0, d_comfort=2.0)
    assert risk == 1.0

def test_snap_objective_finite():
    snap_samples = np.random.rand(10, 3)
    snap = compute_snap_objective(snap_samples, t_max=5.0, n_samples=10)
    assert np.isfinite(snap)

def test_snap_objective_zero_for_constant():
    snap_samples = np.zeros((10, 3))
    snap = compute_snap_objective(snap_samples, t_max=5.0, n_samples=10)
    assert snap == 0.0

def test_compute_objectives_shape():
    positions = np.random.rand(10, 3)
    snap_samples = np.random.rand(10, 3)
    obstacles = []
    obj, raw = compute_objectives(positions, snap_samples, obstacles, d_safe=1.0, t_max=5.0)
    assert obj.shape == (3,)

def test_compute_objectives_all_finite():
    positions = np.random.rand(10, 3)
    snap_samples = np.random.rand(10, 3)
    obstacles = [SphereObstacle(center=np.array([10.0, 10.0, 10.0]), radius=1.0)]
    obj, raw = compute_objectives(positions, snap_samples, obstacles, d_safe=1.0, t_max=5.0)
    assert np.all(np.isfinite(obj))
    assert np.isfinite(raw)

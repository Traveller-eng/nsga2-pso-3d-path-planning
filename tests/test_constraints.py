import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from planner.evaluation.constraints import (
    compute_collision_violation,
    compute_velocity_violation,
    compute_acceleration_violation,
    compute_bounds_violation,
    compute_total_violation,
    is_feasible,
    deb_constrained_dominates
)
from planner.environment.obstacles import SphereObstacle

def test_no_collision_violation():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    obstacles = [SphereObstacle(center=np.array([10.0, 10.0, 10.0]), radius=1.0)]
    d_safe = 0.5
    viol = compute_collision_violation(positions, obstacles, d_safe)
    assert viol == 0.0

def test_collision_violation_positive():
    positions = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
    obstacles = [SphereObstacle(center=np.array([5.0, 5.0, 5.0]), radius=1.0)]
    d_safe = 0.5
    viol = compute_collision_violation(positions, obstacles, d_safe)
    assert viol > 0.0

def test_velocity_no_violation():
    velocity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    v_max = 2.0
    viol = compute_velocity_violation(velocity, v_max)
    assert viol == 0.0

def test_velocity_violation_positive():
    velocity = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    v_max = 2.0
    viol = compute_velocity_violation(velocity, v_max)
    assert viol > 0.0

def test_velocity_violation_uses_max_normalized_violation():
    velocity = np.array([[3.0, 0.0, 0.0], [0.0, 6.0, 0.0]])
    viol = compute_velocity_violation(velocity, v_max=2.0)
    assert viol == pytest.approx(2.0)

def test_acceleration_no_violation():
    acceleration = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    a_max = 2.0
    viol = compute_acceleration_violation(acceleration, a_max)
    assert viol == 0.0

def test_acceleration_violation_positive():
    acceleration = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    a_max = 2.0
    viol = compute_acceleration_violation(acceleration, a_max)
    assert viol > 0.0

def test_acceleration_violation_uses_max_normalized_violation():
    acceleration = np.array([[3.0, 0.0, 0.0], [0.0, 8.0, 0.0]])
    viol = compute_acceleration_violation(acceleration, a_max=2.0)
    assert viol == pytest.approx(3.0)

def test_bounds_no_violation():
    positions = np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([10.0, 10.0, 10.0])
    viol = compute_bounds_violation(positions, lower, upper)
    assert viol == 0.0

def test_bounds_violation_positive():
    positions = np.array([[-1.0, 5.0, 5.0], [11.0, 6.0, 6.0]])
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([10.0, 10.0, 10.0])
    viol = compute_bounds_violation(positions, lower, upper)
    assert viol > 0.0

def test_collision_violation_uses_max_normalized_violation():
    positions = np.array([[0.75, 0.0, 0.0], [0.25, 0.0, 0.0]])
    obstacles = [SphereObstacle(center=np.array([0.0, 0.0, 0.0]), radius=0.0)]
    viol = compute_collision_violation(positions, obstacles, d_safe=1.0)
    assert viol == pytest.approx(0.75)

def test_total_violation_feasible():
    positions = np.array([[5.0, 5.0, 5.0]])
    velocity = np.array([[1.0, 0.0, 0.0]])
    acceleration = np.array([[1.0, 0.0, 0.0]])
    obstacles = [SphereObstacle(center=np.array([10.0, 10.0, 10.0]), radius=1.0)]
    d_safe = 0.5
    v_max = 2.0
    a_max = 2.0
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([10.0, 10.0, 10.0])
    
    total, details = compute_total_violation(positions, velocity, acceleration, obstacles, d_safe, v_max, a_max, lower, upper)
    assert total == 0.0
    assert all(v == 0.0 for v in details.values())

def test_total_violation_infeasible():
    positions = np.array([[10.0, 10.0, 10.0]])
    velocity = np.array([[3.0, 0.0, 0.0]])
    acceleration = np.array([[3.0, 0.0, 0.0]])
    obstacles = [SphereObstacle(center=np.array([10.0, 10.0, 10.0]), radius=1.0)]
    d_safe = 0.5
    v_max = 2.0
    a_max = 2.0
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([10.0, 10.0, 10.0])
    
    total, details = compute_total_violation(positions, velocity, acceleration, obstacles, d_safe, v_max, a_max, lower, upper)
    assert total > 0.0

def test_is_feasible_true():
    assert is_feasible(0.0) is True

def test_is_feasible_false():
    assert is_feasible(0.1) is False

def test_deb_feasible_beats_infeasible():
    obj_a = np.array([1.0, 1.0])
    viol_a = 0.0
    obj_b = np.array([0.5, 0.5]) # Better objective
    viol_b = 0.1 # But infeasible
    assert deb_constrained_dominates(obj_a, viol_a, obj_b, viol_b) is True
    assert deb_constrained_dominates(obj_b, viol_b, obj_a, viol_a) is False

def test_deb_infeasible_lower_violation_wins():
    obj_a = np.array([2.0, 2.0])
    viol_a = 0.1
    obj_b = np.array([1.0, 1.0])
    viol_b = 0.2
    assert deb_constrained_dominates(obj_a, viol_a, obj_b, viol_b) is True
    assert deb_constrained_dominates(obj_b, viol_b, obj_a, viol_a) is False

def test_deb_pareto_dominance():
    obj_a = np.array([1.0, 1.0])
    viol_a = 0.0
    obj_b = np.array([2.0, 2.0])
    viol_b = 0.0
    assert deb_constrained_dominates(obj_a, viol_a, obj_b, viol_b) is True
    assert deb_constrained_dominates(obj_b, viol_b, obj_a, viol_a) is False

def test_deb_no_dominance():
    obj_a = np.array([1.0, 2.0])
    viol_a = 0.0
    obj_b = np.array([2.0, 1.0])
    viol_b = 0.0
    assert deb_constrained_dominates(obj_a, viol_a, obj_b, viol_b) is False
    assert deb_constrained_dominates(obj_b, viol_b, obj_a, viol_a) is False

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from planner.environment.world import make_3d_static_world
from planner.representation.trajectory import TrajectoryConfig, create_straight_line_trajectory
from planner.evaluation.evaluator import TrajectoryEvaluator, EvaluationResult
from planner.environment.obstacles import SphereObstacle

@pytest.fixture
def clean_world():
    return make_3d_static_world(
        bounds_max=(10.0, 10.0, 10.0),
        start=(0.0, 0.0, 0.0),
        goal=(10.0, 10.0, 10.0),
        obstacles=[],
        v_max=10.0,
        a_max=10.0,
        d_safe=0.5
    )

@pytest.fixture
def obstacle_world():
    world = make_3d_static_world(
        bounds_max=(10.0, 10.0, 10.0),
        start=(0.0, 0.0, 0.0),
        goal=(10.0, 10.0, 10.0),
        obstacles=[SphereObstacle(center=np.array([5.0, 5.0, 5.0]), radius=2.0)],
        v_max=10.0,
        a_max=10.0,
        d_safe=0.5
    )
    return world

def test_evaluator_returns_result(clean_world):
    evaluator = TrajectoryEvaluator(clean_world)
    config = TrajectoryConfig(8, 3, clean_world.start, clean_world.goal, clean_world.t_max)
    traj = create_straight_line_trajectory(config)
    res = evaluator.evaluate(traj)
    assert isinstance(res, EvaluationResult)

def test_evaluator_objectives_finite(clean_world):
    evaluator = TrajectoryEvaluator(clean_world)
    config = TrajectoryConfig(8, 3, clean_world.start, clean_world.goal, clean_world.t_max)
    traj = create_straight_line_trajectory(config)
    res = evaluator.evaluate(traj)
    assert np.all(np.isfinite(res.objectives))

def test_evaluator_n_samples_adequate(clean_world):
    evaluator = TrajectoryEvaluator(clean_world)
    dt = clean_world.t_max / (evaluator.n_samples - 1)
    assert clean_world.v_max * dt < clean_world.d_safe

def test_evaluator_straight_line_feasibility(clean_world):
    # Bump limits without changing t_max
    clean_world.v_max = 1000.0
    clean_world.a_max = 10000.0
    evaluator = TrajectoryEvaluator(clean_world)
    config = TrajectoryConfig(8, 3, clean_world.start, clean_world.goal, clean_world.t_max)
    traj = create_straight_line_trajectory(config)
    res = evaluator.evaluate(traj)
    assert res.is_feasible

def test_evaluator_collision_detection(obstacle_world):
    evaluator = TrajectoryEvaluator(obstacle_world)
    config = TrajectoryConfig(8, 3, obstacle_world.start, obstacle_world.goal, obstacle_world.t_max)
    traj = create_straight_line_trajectory(config)
    res = evaluator.evaluate(traj)
    assert not res.is_feasible
    assert res.violation_details['collision'] > 0.0

def test_evaluator_velocity_violation_detection(clean_world):
    clean_world.v_max = 0.1 # Very low v_max
    evaluator = TrajectoryEvaluator(clean_world)
    config = TrajectoryConfig(8, 3, clean_world.start, clean_world.goal, clean_world.t_max)
    traj = create_straight_line_trajectory(config)
    res = evaluator.evaluate(traj)
    assert not res.is_feasible
    assert res.violation_details['velocity'] > 0.0

def test_evaluator_infeasible_positive_violation(obstacle_world):
    evaluator = TrajectoryEvaluator(obstacle_world)
    config = TrajectoryConfig(8, 3, obstacle_world.start, obstacle_world.goal, obstacle_world.t_max)
    traj = create_straight_line_trajectory(config)
    res = evaluator.evaluate(traj)
    assert not res.is_feasible
    assert res.total_violation > 0.0

def test_evaluator_feasible_zero_violation(clean_world):
    """Obstacle-free straight line with generous limits must be feasible."""
    # Bump limits without changing t_max
    clean_world.v_max = 1000.0
    clean_world.a_max = 10000.0
    evaluator = TrajectoryEvaluator(clean_world)
    config = TrajectoryConfig(8, 3, clean_world.start, clean_world.goal, clean_world.t_max)
    traj = create_straight_line_trajectory(config)
    res = evaluator.evaluate(traj)
    assert res.is_feasible, f"Expected feasible but got violations: {res.violation_details}"
    assert res.total_violation == 0.0

def test_evaluate_from_genes(clean_world):
    evaluator = TrajectoryEvaluator(clean_world)
    config = TrajectoryConfig(8, 3, clean_world.start, clean_world.goal, clean_world.t_max)
    traj = create_straight_line_trajectory(config)
    res = evaluator.evaluate_from_genes(traj.genes, config)
    assert isinstance(res, EvaluationResult)

def test_evaluator_diagnostics(clean_world):
    evaluator = TrajectoryEvaluator(clean_world)
    config = TrajectoryConfig(8, 3, clean_world.start, clean_world.goal, clean_world.t_max)
    traj = create_straight_line_trajectory(config)
    res = evaluator.evaluate(traj)
    assert res.min_clearance == float('inf')
    assert np.isfinite(res.max_speed)
    assert np.isfinite(res.max_accel)

import sys
import os
import pytest
import numpy as np
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from planner.representation.trajectory import (
    TrajectoryConfig,
    Trajectory,
    create_straight_line_trajectory,
    create_perturbed_trajectory,
    genes_to_trajectory,
    repair_trajectory
)
from planner.geometry.bspline import BSplineCurve

@pytest.fixture
def base_config():
    return TrajectoryConfig(
        n_control_points=8,
        dim=3,
        start=np.array([0.0, 0.0, 0.0]),
        goal=np.array([10.0, 10.0, 10.0]),
        t_max=5.0,
        degree=5,
        optimize_time=False
    )

def test_config_n_genes(base_config):
    # n_free_points = 8 - 2 = 6
    # n_position_genes = 6 * 3 = 18
    # optimize_time = False -> n_time_genes = 0
    assert base_config.n_genes == 18

def test_config_n_genes_with_time():
    config = TrajectoryConfig(
        n_control_points=8,
        dim=3,
        start=np.array([0.0, 0.0, 0.0]),
        goal=np.array([10.0, 10.0, 10.0]),
        t_max=5.0,
        degree=5,
        optimize_time=True
    )
    # n_time_genes = 8 - 5 = 3
    assert config.n_genes == 21

def test_config_validation():
    with pytest.raises(ValueError):
        TrajectoryConfig(
            n_control_points=5,
            dim=3,
            start=np.array([0.0, 0.0, 0.0]),
            goal=np.array([10.0, 10.0, 10.0]),
            t_max=5.0,
            degree=5
        )

def test_straight_line_start_goal(base_config):
    traj = create_straight_line_trajectory(base_config)
    cps = traj.control_points
    np.testing.assert_allclose(cps[0], base_config.start)
    np.testing.assert_allclose(cps[-1], base_config.goal)

def test_straight_line_interpolation(base_config):
    traj = create_straight_line_trajectory(base_config)
    cps = traj.control_points
    # Interior points should lie on the straight line from start to goal
    for i in range(1, 7):
        alpha = i / 7.0
        expected = base_config.start + alpha * (base_config.goal - base_config.start)
        np.testing.assert_allclose(cps[i], expected)

def test_control_points_shape(base_config):
    traj = create_straight_line_trajectory(base_config)
    assert traj.control_points.shape == (8, 3)

def test_free_control_points_shape(base_config):
    traj = create_straight_line_trajectory(base_config)
    assert traj.free_control_points.shape == (6, 3)

def test_genes_to_trajectory(base_config):
    traj1 = create_straight_line_trajectory(base_config)
    genes = traj1.genes.copy()
    traj2 = genes_to_trajectory(genes, base_config)
    np.testing.assert_allclose(traj1.genes, traj2.genes)
    np.testing.assert_allclose(traj1.control_points, traj2.control_points)

def test_perturbed_differs_from_straight(base_config):
    rng = np.random.default_rng(42)
    straight = create_straight_line_trajectory(base_config)
    perturbed = create_perturbed_trajectory(base_config, rng, scale=0.5)
    assert not np.allclose(straight.genes, perturbed.genes)

def test_perturbed_preserves_start_goal(base_config):
    rng = np.random.default_rng(42)
    perturbed = create_perturbed_trajectory(base_config, rng, scale=0.5)
    cps = perturbed.control_points
    np.testing.assert_allclose(cps[0], base_config.start)
    np.testing.assert_allclose(cps[-1], base_config.goal)

def test_to_bspline(base_config):
    traj = create_straight_line_trajectory(base_config)
    curve = traj.to_bspline()
    assert isinstance(curve, BSplineCurve)

def test_p0_equals_start(base_config):
    traj = create_straight_line_trajectory(base_config)
    curve = traj.to_bspline()
    from planner.geometry.bspline import evaluate
    p0 = evaluate(curve, 0.0)
    np.testing.assert_allclose(p0, base_config.start)

def test_p1_equals_goal(base_config):
    traj = create_straight_line_trajectory(base_config)
    curve = traj.to_bspline()
    from planner.geometry.bspline import evaluate
    p1 = evaluate(curve, 1.0)
    np.testing.assert_allclose(p1, base_config.goal)

def test_time_variables_none(base_config):
    traj = create_straight_line_trajectory(base_config)
    assert traj.time_variables is None

def test_time_normalization():
    config = TrajectoryConfig(
        n_control_points=8,
        dim=3,
        start=np.array([0.0, 0.0, 0.0]),
        goal=np.array([10.0, 10.0, 10.0]),
        t_max=5.0,
        degree=5,
        optimize_time=True
    )
    traj = create_straight_line_trajectory(config)
    norm = traj.normalized_time_segments()
    assert norm is not None
    np.testing.assert_allclose(np.sum(norm), 1.0)

def test_repair_clamps_to_bounds(base_config):
    rng = np.random.default_rng(42)
    perturbed = create_perturbed_trajectory(base_config, rng, scale=10.0)
    
    world_config = SimpleNamespace(
        lower_bounds=np.array([0.0, 0.0, 0.0]),
        upper_bounds=np.array([10.0, 10.0, 10.0])
    )
    
    repaired = repair_trajectory(perturbed, world_config)
    cps = repaired.free_control_points
    assert np.all(cps >= world_config.lower_bounds)
    assert np.all(cps <= world_config.upper_bounds)

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from planner.environment.world import make_3d_static_world
from planner.evaluation.evaluator import EvaluationResult
from planner.optimization.mopso import (
    ExternalArchive,
    MOPSOOptimizer,
    ObjectiveNormalization,
    Particle,
    build_neighborhoods,
    evaluation_better_for_weight,
    generate_weight_vectors,
    tchebycheff_scalarization,
)
from planner.representation.trajectory import Trajectory, TrajectoryConfig


def make_result(feasible: bool, violation: float, objectives: list[float]) -> EvaluationResult:
    return EvaluationResult(
        objectives=np.array(objectives, dtype=float),
        total_violation=violation,
        violation_details={},
        is_feasible=feasible,
        n_samples=10,
        min_clearance=float("inf"),
        max_speed=0.0,
        max_accel=0.0,
        path_length=float(objectives[0]),
        raw_snap=0.0,
    )


def make_dummy_trajectory() -> Trajectory:
    config = TrajectoryConfig(8, 3, np.zeros(3), np.ones(3), 5.0)
    return Trajectory(config=config, genes=np.zeros(config.n_genes))


def test_weight_vectors_valid_and_diverse():
    weights = generate_weight_vectors(10, 3)
    assert weights.shape == (10, 3)
    assert np.all(weights >= 0.0)
    np.testing.assert_allclose(np.sum(weights, axis=1), 1.0)
    assert len(np.unique(weights, axis=0)) == 10
    assert any(np.allclose(weight, [0.0, 0.0, 1.0]) for weight in weights)


def test_normalization_finite_with_zero_ranges_and_fixed_state():
    objs = np.array([[1.0, 2.0, 3.0], [1.0, 5.0, 3.0]])
    norm = ObjectiveNormalization.from_objectives(objs)
    out = norm.normalize(np.array([1.0, 3.5, 3.0]))
    assert np.all(np.isfinite(out))
    saved_ideal = norm.ideal.copy()
    norm.normalize(np.array([100.0, 100.0, 100.0]))
    np.testing.assert_allclose(norm.ideal, saved_ideal)


def test_tchebycheff_weighted_max():
    norm = ObjectiveNormalization.from_objectives(np.array([[1.0, 2.0, 3.0], [3.0, 6.0, 7.0]]))
    value = tchebycheff_scalarization(np.array([2.0, 6.0, 5.0]), np.array([0.5, 0.25, 0.25]), norm)
    assert value == pytest.approx(0.25)


def test_pbest_feasible_beats_infeasible():
    norm = ObjectiveNormalization.from_objectives(np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))
    candidate = make_result(True, 0.0, [2.0, 2.0, 2.0])
    incumbent = make_result(False, 0.1, [1.0, 1.0, 1.0])
    assert evaluation_better_for_weight(candidate, incumbent, np.array([1 / 3] * 3), norm)


def test_pbest_lower_violation_wins_for_infeasible():
    norm = ObjectiveNormalization.from_objectives(np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))
    candidate = make_result(False, 0.1, [2.0, 2.0, 2.0])
    incumbent = make_result(False, 0.2, [1.0, 1.0, 1.0])
    assert evaluation_better_for_weight(candidate, incumbent, np.array([1 / 3] * 3), norm)


def test_pbest_scalarization_decides_feasible_candidates():
    norm = ObjectiveNormalization.from_objectives(np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]]))
    candidate = make_result(True, 0.0, [1.0, 8.0, 8.0])
    incumbent = make_result(True, 0.0, [2.0, 1.0, 1.0])
    assert evaluation_better_for_weight(candidate, incumbent, np.array([1.0, 0.0, 0.0]), norm)


def test_neighborhood_uses_nearest_weights_without_global_best():
    weights = np.array([[0.0, 0.0, 1.0], [0.0, 0.2, 0.8], [1.0, 0.0, 0.0]])
    neighborhoods = build_neighborhoods(weights, neighborhood_size=2)
    assert neighborhoods[0] == [0, 1]
    assert 2 not in neighborhoods[0]


def test_pso_update_clamps_velocity_and_position():
    world = make_3d_static_world(
        bounds_max=(10.0, 10.0, 10.0),
        start=(0.0, 0.0, 0.0),
        goal=(10.0, 10.0, 10.0),
        obstacles=[],
    )
    world.v_max = 100.0
    world.a_max = 1000.0
    config = TrajectoryConfig(8, 3, world.start, world.goal, world.t_max)
    opt = MOPSOOptimizer(population_size=2, max_evaluations=2, velocity_clamp=0.05, seed=1)
    particles = opt.optimize(world, config)
    opt.max_evaluations = 3
    lower, upper = opt._get_bounds(config, world)
    evaluator = __import__("planner.evaluation.evaluator", fromlist=["TrajectoryEvaluator"]).TrajectoryEvaluator(world)
    opt._step_particle(particles[0], particles[1], world, config, evaluator, lower, upper)
    limit = opt._velocity_bounds(lower, upper)
    assert np.all(np.abs(particles[0].velocity) <= limit + 1e-12)
    assert np.all(particles[0].position >= lower)
    assert np.all(particles[0].position <= upper)


def test_archive_removes_dominated_and_enforces_cap():
    archive = ExternalArchive(max_size=2)
    traj = make_dummy_trajectory()
    archive.update(traj, make_result(True, 0.0, [2.0, 2.0, 2.0]), np.zeros(traj.config.n_genes))
    archive.update(traj, make_result(True, 0.0, [1.0, 1.0, 1.0]), np.zeros(traj.config.n_genes))
    assert len(archive.entries) == 1
    np.testing.assert_allclose(archive.entries[0].evaluation.objectives, [1.0, 1.0, 1.0])
    archive.update(traj, make_result(True, 0.0, [0.0, 3.0, 3.0]), np.zeros(traj.config.n_genes))
    archive.update(traj, make_result(True, 0.0, [3.0, 0.0, 3.0]), np.zeros(traj.config.n_genes))
    assert len(archive.entries) == 2


def test_mopso_budget_smaller_than_population():
    world = make_3d_static_world(obstacles=[])
    world.v_max = 100.0
    world.a_max = 1000.0
    config = TrajectoryConfig(8, 3, world.start, world.goal, world.t_max)
    opt = MOPSOOptimizer(population_size=20, max_evaluations=7, seed=42)
    particles = opt.optimize(world, config)
    assert opt.evals == 7
    assert len(particles) == 7


def test_mopso_partial_final_iteration_exact_budget():
    world = make_3d_static_world(obstacles=[])
    world.v_max = 100.0
    world.a_max = 1000.0
    config = TrajectoryConfig(8, 3, world.start, world.goal, world.t_max)
    opt = MOPSOOptimizer(population_size=10, max_evaluations=23, seed=42)
    particles = opt.optimize(world, config)
    assert opt.evals == 23
    assert len(particles) == 10


def test_mopso_reproducible_with_same_seed():
    world = make_3d_static_world(obstacles=[])
    world.v_max = 100.0
    world.a_max = 1000.0
    config = TrajectoryConfig(8, 3, world.start, world.goal, world.t_max)
    opt1 = MOPSOOptimizer(population_size=10, max_evaluations=35, seed=123)
    opt2 = MOPSOOptimizer(population_size=10, max_evaluations=35, seed=123)
    pop1 = opt1.optimize(world, config)
    pop2 = opt2.optimize(world, config)
    np.testing.assert_allclose(
        np.array([p.evaluation.objectives for p in pop1]),
        np.array([p.evaluation.objectives for p in pop2]),
    )


def test_mopso_integration_simple_world():
    world = make_3d_static_world(
        bounds_max=(10.0, 10.0, 10.0),
        start=(0.0, 0.0, 0.0),
        goal=(10.0, 10.0, 10.0),
        obstacles=[],
    )
    world.v_max = 100.0
    world.a_max = 1000.0
    config = TrajectoryConfig(8, 3, world.start, world.goal, world.t_max)
    opt = MOPSOOptimizer(population_size=20, max_evaluations=80, archive_size=30, seed=42)
    particles = opt.optimize(world, config)
    assert opt.evals == 80
    assert len(particles) == 20
    assert len(opt.archive.entries) > 0
    assert all(entry.evaluation.is_feasible for entry in opt.archive.entries)
    for particle in particles:
        assert np.all(np.isfinite(particle.evaluation.objectives))
        np.testing.assert_allclose(particle.trajectory.control_points[0], world.start)
        np.testing.assert_allclose(particle.trajectory.control_points[-1], world.goal)

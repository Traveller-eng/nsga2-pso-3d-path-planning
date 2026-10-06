import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from planner.environment.world import make_3d_static_world
from planner.evaluation.evaluator import EvaluationResult
from planner.optimization.hybrid import HybridNSGA2PSOOptimizer
from planner.optimization.nsga2 import Individual
from planner.representation.trajectory import Trajectory, TrajectoryConfig, create_straight_line_trajectory


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


def test_hybrid_selects_feasible_diverse_elites():
    opt = HybridNSGA2PSOOptimizer(population_size=10, max_evaluations=100, seed=1)
    feasible = [
        Individual(make_dummy_trajectory(), make_result(True, 0.0, [1.0, 3.0, 3.0])),
        Individual(make_dummy_trajectory(), make_result(True, 0.0, [2.0, 2.0, 2.0])),
        Individual(make_dummy_trajectory(), make_result(True, 0.0, [3.0, 1.0, 1.0])),
    ]
    infeasible = [
        Individual(make_dummy_trajectory(), make_result(False, 0.8, [8.0, 8.0, 8.0])),
        Individual(make_dummy_trajectory(), make_result(False, 0.7, [7.0, 7.0, 7.0])),
    ]
    population = feasible + infeasible
    for ind in population:
        ind.rank = 0
        ind.crowding_distance = 1.0

    elites = opt._select_elites(population, elite_count=3)
    assert len(elites) == 3
    assert all(ind.evaluation.is_feasible for ind in elites)
    assert all(id(ind) in {id(item) for item in population} for ind in elites)


def test_hybrid_pso_initializes_from_elites_and_keeps_endpoints_fixed():
    world = make_3d_static_world(
        bounds_max=(10.0, 10.0, 10.0),
        start=(0.0, 0.0, 0.0),
        goal=(10.0, 10.0, 10.0),
        obstacles=[],
    )
    world.v_max = 100.0
    world.a_max = 1000.0
    config = TrajectoryConfig(8, 3, world.start, world.goal, world.t_max)
    elite = Individual(
        trajectory=create_straight_line_trajectory(config),
        evaluation=make_result(True, 0.0, [1.0, 2.0, 3.0]),
    )
    opt = HybridNSGA2PSOOptimizer(population_size=2, max_evaluations=20, seed=3)
    particles = opt._initialize_pso_particles([elite], world, config, 2)
    assert len(particles) == 1
    np.testing.assert_allclose(particles[0].position, elite.trajectory.genes)
    np.testing.assert_allclose(particles[0].trajectory.control_points[0], world.start)
    np.testing.assert_allclose(particles[0].trajectory.control_points[-1], world.goal)


def test_hybrid_writeback_probability_is_reproducible_and_respects_p_values():
    opt = HybridNSGA2PSOOptimizer(population_size=3, max_evaluations=50, seed=4, writeback_probability=1.0)
    population = [
        Individual(make_dummy_trajectory(), make_result(True, 0.0, [1.0, 1.0, 1.0])),
        Individual(make_dummy_trajectory(), make_result(True, 0.0, [2.0, 2.0, 2.0])),
        Individual(make_dummy_trajectory(), make_result(True, 0.0, [3.0, 3.0, 3.0])),
    ]
    for ind in population:
        ind.rank = 0
        ind.crowding_distance = 1.0
    elite_indices = [0, 1]
    refined = [
        type("ParticleStub", (), {"position": np.array([10.0, 10.0, 10.0]), "trajectory": population[0].trajectory, "evaluation": make_result(True, 0.0, [0.5, 0.5, 0.5])})(),
        type("ParticleStub", (), {"position": np.array([11.0, 11.0, 11.0]), "trajectory": population[1].trajectory, "evaluation": make_result(True, 0.0, [0.4, 0.4, 0.4])})(),
    ]
    opt._apply_lamarckian_writeback(population, elite_indices, refined, writeback_probability=1.0)
    assert population[0].evaluation.objectives[0] == 0.5
    assert population[1].evaluation.objectives[0] == 0.4

    opt2 = HybridNSGA2PSOOptimizer(population_size=3, max_evaluations=50, seed=4, writeback_probability=0.0)
    pop2 = [
        Individual(make_dummy_trajectory(), make_result(True, 0.0, [1.0, 1.0, 1.0])),
        Individual(make_dummy_trajectory(), make_result(True, 0.0, [2.0, 2.0, 2.0])),
        Individual(make_dummy_trajectory(), make_result(True, 0.0, [3.0, 3.0, 3.0])),
    ]
    opt2._apply_lamarckian_writeback(pop2, elite_indices, refined, writeback_probability=0.0)
    np.testing.assert_allclose(pop2[0].evaluation.objectives, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(pop2[1].evaluation.objectives, [2.0, 2.0, 2.0])


def test_hybrid_budget_is_shared_and_integration_runs():
    world = make_3d_static_world(
        bounds_max=(10.0, 10.0, 10.0),
        start=(0.0, 0.0, 0.0),
        goal=(10.0, 10.0, 10.0),
        obstacles=[],
    )
    world.v_max = 100.0
    world.a_max = 1000.0
    config = TrajectoryConfig(8, 3, world.start, world.goal, world.t_max)
    opt = HybridNSGA2PSOOptimizer(
        population_size=10,
        max_evaluations=120,
        elite_fraction=0.2,
        min_elites=3,
        pso_iterations=4,
        seed=42,
    )
    population = opt.optimize(world, config)
    assert len(population) == 10
    assert opt.evals <= 120
    assert opt.evals == opt.nsga2_evaluations + opt.pso_evaluations
    assert opt.history
    feasible = [ind for ind in population if ind.evaluation.is_feasible]
    assert len(feasible) > 0
    assert all(np.isfinite(ind.evaluation.objectives).all() for ind in population)
    assert all(np.allclose(ind.trajectory.control_points[0], world.start) for ind in population)
    assert all(np.allclose(ind.trajectory.control_points[-1], world.goal) for ind in population)

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from planner.optimization.nsga2 import (
    Individual, dominates, fast_non_dominated_sort, calculate_crowding_distance,
    binary_tournament, sbx_crossover, polynomial_mutation, NSGA2Optimizer
)
from planner.representation.trajectory import TrajectoryConfig, Trajectory
from planner.evaluation.evaluator import EvaluationResult
from planner.environment.world import make_3d_static_world

def make_dummy_individual(feasible: bool, violation: float, obj: list[float]) -> Individual:
    res = EvaluationResult(
        objectives=np.array(obj),
        total_violation=violation,
        violation_details={},
        is_feasible=feasible,
        n_samples=10,
        min_clearance=0.0,
        max_speed=0.0,
        max_accel=0.0,
        path_length=obj[0],
        raw_snap=1.0
    )
    # Trajectory is a dummy for these tests
    config = TrajectoryConfig(8, 3, np.zeros(3), np.zeros(3), 5.0)
    genes = np.zeros(config.n_genes)
    t = Trajectory(config=config, genes=genes)
    return Individual(trajectory=t, evaluation=res)

def test_dominates_feasible_beats_infeasible():
    i1 = make_dummy_individual(True, 0.0, [10.0, 10.0, 10.0])
    i2 = make_dummy_individual(False, 1.0, [1.0, 1.0, 1.0])
    assert dominates(i1, i2)
    assert not dominates(i2, i1)

def test_dominates_infeasible_lower_violation_wins():
    i1 = make_dummy_individual(False, 1.0, [10.0, 10.0, 10.0])
    i2 = make_dummy_individual(False, 2.0, [1.0, 1.0, 1.0])
    assert dominates(i1, i2)
    assert not dominates(i2, i1)

def test_dominates_pareto():
    i1 = make_dummy_individual(True, 0.0, [1.0, 1.0, 1.0])
    i2 = make_dummy_individual(True, 0.0, [2.0, 2.0, 2.0])
    i3 = make_dummy_individual(True, 0.0, [1.0, 2.0, 1.0])
    
    assert dominates(i1, i2)
    assert dominates(i1, i3)
    assert not dominates(i2, i1)
    
    # Non-dominated
    i4 = make_dummy_individual(True, 0.0, [2.0, 0.5, 2.0])
    assert not dominates(i1, i4)
    assert not dominates(i4, i1)

def test_fast_non_dominated_sort():
    i1 = make_dummy_individual(True, 0.0, [1.0, 1.0, 1.0]) # Front 0
    i2 = make_dummy_individual(True, 0.0, [2.0, 2.0, 2.0]) # Front 1
    i3 = make_dummy_individual(True, 0.0, [3.0, 3.0, 3.0]) # Front 2
    i4 = make_dummy_individual(True, 0.0, [1.5, 0.5, 1.5]) # Front 0
    
    pop = [i3, i1, i4, i2]
    fronts = fast_non_dominated_sort(pop)
    
    assert len(fronts) == 3
    ids_f0 = [id(x) for x in fronts[0]]
    ids_f1 = [id(x) for x in fronts[1]]
    ids_f2 = [id(x) for x in fronts[2]]
    
    assert all(id(i) in ids_f0 for i in [i1, i4]) and len(ids_f0) == 2
    assert all(id(i) in ids_f1 for i in [i2]) and len(ids_f1) == 1
    assert all(id(i) in ids_f2 for i in [i3]) and len(ids_f2) == 1
    
    assert i1.rank == 0
    assert i2.rank == 1
    assert i3.rank == 2
    assert i4.rank == 0

def test_crowding_distance():
    i1 = make_dummy_individual(True, 0.0, [0.0, 10.0])
    i2 = make_dummy_individual(True, 0.0, [5.0, 5.0])
    i3 = make_dummy_individual(True, 0.0, [10.0, 0.0])
    
    front = [i1, i2, i3]
    calculate_crowding_distance(front)
    
    # Sorted by obj 0 -> i1, i2, i3. Bounds = inf
    # Sorted by obj 1 -> i3, i2, i1. Bounds = inf
    assert i1.crowding_distance == float('inf')
    assert i3.crowding_distance == float('inf')
    
    # i2 dist = (10-0)/10 + (10-0)/10 = 2.0
    assert i2.crowding_distance == 2.0

def test_binary_tournament():
    rng = np.random.default_rng(42)
    i1 = make_dummy_individual(True, 0.0, [1.0, 1.0])
    i1.rank = 0
    i1.crowding_distance = 1.0
    
    i2 = make_dummy_individual(True, 0.0, [2.0, 2.0])
    i2.rank = 1
    i2.crowding_distance = 10.0
    
    # i1 should beat i2 strictly on rank, despite crowding
    pop = [i1, i2]
    win = binary_tournament(pop, rng)
    assert win == i1
    
    i3 = make_dummy_individual(True, 0.0, [1.5, 1.5])
    i3.rank = 0
    i3.crowding_distance = 2.0
    
    # rank tie, i3 beats i1 on crowding
    pop = [i1, i3]
    win = binary_tournament(pop, rng)
    assert win == i3

def test_sbx_bounds():
    rng = np.random.default_rng(42)
    p1 = np.array([0.0, 0.0])
    p2 = np.array([10.0, 10.0])
    lb = np.array([0.0, 0.0])
    ub = np.array([10.0, 10.0])
    
    c1, c2 = sbx_crossover(p1, p2, prob=1.0, eta=2.0, lower_bounds=lb, upper_bounds=ub, rng=rng)
    assert np.all(c1 >= lb)
    assert np.all(c1 <= ub)
    assert np.all(c2 >= lb)
    assert np.all(c2 <= ub)

def test_polynomial_mutation_bounds():
    rng = np.random.default_rng(42)
    p1 = np.array([5.0, 5.0])
    lb = np.array([0.0, 0.0])
    ub = np.array([10.0, 10.0])
    
    c = polynomial_mutation(p1, prob=1.0, eta=2.0, lower_bounds=lb, upper_bounds=ub, rng=rng)
    assert np.all(c >= lb)
    assert np.all(c <= ub)

def test_nsga2_integration_simple_world():
    world = make_3d_static_world(
        bounds_max=(10.0, 10.0, 10.0),
        start=(0.0, 0.0, 0.0),
        goal=(10.0, 10.0, 10.0),
        obstacles=[]
    )
    # Ensure generous limits so initialization is feasible
    world.v_max = 100.0
    world.a_max = 1000.0
    
    config = TrajectoryConfig(8, 3, world.start, world.goal, world.t_max)
    
    # 20 pop size, 100 max evaluations. Meaning init (20) + 4 gens (20*4) = 100 evals.
    opt = NSGA2Optimizer(population_size=20, max_evaluations=100, seed=42)
    final_pop = opt.optimize(world, config)
    
    assert len(final_pop) == 20
    assert opt.evals == 100
    
    # At least some should be feasible in an obstacle-free world with high limits
    feasible_count = sum(1 for p in final_pop if p.evaluation.is_feasible)
    assert feasible_count > 0
    
    # Path length of straight line is approx 17.32
    # Verify no valid trajectory has path length materially below this
    euclidean = np.linalg.norm(world.goal - world.start)
    for p in final_pop:
        assert p.evaluation.path_length >= euclidean - 1e-4
        # Verify finite objectives
        assert np.all(np.isfinite(p.evaluation.objectives))
        # Verify start and goal not mutated
        np.testing.assert_allclose(p.trajectory.control_points[0], world.start)
        np.testing.assert_allclose(p.trajectory.control_points[-1], world.goal)

def test_nsga2_budget_smaller_than_population():
    world = make_3d_static_world(
        bounds_max=(10.0, 10.0, 10.0),
        start=(0.0, 0.0, 0.0),
        goal=(10.0, 10.0, 10.0),
        obstacles=[]
    )
    world.v_max = 100.0
    world.a_max = 1000.0
    config = TrajectoryConfig(8, 3, world.start, world.goal, world.t_max)

    opt = NSGA2Optimizer(population_size=20, max_evaluations=7, seed=42)
    final_pop = opt.optimize(world, config)

    assert opt.evals == 7
    assert len(final_pop) == 7

def test_nsga2_reproducible_with_same_seed():
    world = make_3d_static_world(
        bounds_max=(10.0, 10.0, 10.0),
        start=(0.0, 0.0, 0.0),
        goal=(10.0, 10.0, 10.0),
        obstacles=[]
    )
    world.v_max = 100.0
    world.a_max = 1000.0
    config = TrajectoryConfig(8, 3, world.start, world.goal, world.t_max)

    pop1 = NSGA2Optimizer(population_size=10, max_evaluations=35, seed=123).optimize(world, config)
    pop2 = NSGA2Optimizer(population_size=10, max_evaluations=35, seed=123).optimize(world, config)

    objs1 = np.array([ind.evaluation.objectives for ind in pop1])
    objs2 = np.array([ind.evaluation.objectives for ind in pop2])
    np.testing.assert_allclose(objs1, objs2)

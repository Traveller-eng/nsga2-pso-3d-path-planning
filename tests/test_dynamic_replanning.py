import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from planner.environment.obstacles import SphereObstacle
from planner.environment.world import make_3d_static_world
from planner.evaluation.evaluator import TrajectoryEvaluator
from planner.optimization.dynamic import DynamicWarmStartController, DynamicScenario
from planner.optimization.nsga2 import Individual, NSGA2Optimizer
from planner.representation.trajectory import TrajectoryConfig, create_straight_line_trajectory


def make_scenario():
    base_world = make_3d_static_world(
        bounds_max=(10.0, 10.0, 10.0),
        start=(0.0, 0.0, 0.0),
        goal=(9.0, 9.0, 9.0),
        obstacles=[
            SphereObstacle(center=np.array([5.8, 5.2, 5.2]), radius=1.4, velocity=np.array([0.7, -0.6, 0.2]))
        ],
        v_max=3.0,
        a_max=5.0,
        d_safe=0.5,
        d_comfort=1.5,
        t_max_factor=1.75,
    )
    return DynamicScenario(world=base_world, event_times=[0.0, 2.5, 5.0, 7.5])


def test_obstacle_motion_uses_physical_time():
    obstacle = SphereObstacle(center=np.array([5.5, 5.5, 5.5]), radius=1.4, velocity=np.array([0.7, -0.6, 0.2]))

    assert np.allclose(obstacle.position_at(0.0), np.array([5.5, 5.5, 5.5]))
    assert np.allclose(obstacle.position_at(2.5), np.array([7.25, 4.0, 6.0]))
    assert np.allclose(obstacle.position_at(5.0), np.array([9.0, 2.5, 6.5]))


def test_time_offset_changes_collision_evaluation_for_same_trajectory():
    world = make_3d_static_world(
        bounds_max=(10.0, 10.0, 10.0),
        start=(0.0, 0.0, 0.0),
        goal=(9.0, 9.0, 9.0),
        obstacles=[
            SphereObstacle(center=np.array([4.5, 4.5, 4.5]), radius=0.9, velocity=np.array([0.5, 0.5, 0.5]))
        ],
        v_max=3.0,
        a_max=5.0,
        d_safe=0.5,
        d_comfort=1.5,
        t_max_factor=1.75,
    )
    config = TrajectoryConfig(8, 3, world.start, world.goal, world.t_max)
    traj = create_straight_line_trajectory(config)

    evaluator_zero = TrajectoryEvaluator(world, time_offset=0.0)
    evaluator_event = TrajectoryEvaluator(world, time_offset=2.5)

    res_zero = evaluator_zero.evaluate(traj)
    res_event = evaluator_event.evaluate(traj)

    assert res_zero.total_violation != res_event.total_violation
    assert res_zero.total_violation > res_event.total_violation


def test_nsga2_accepts_initial_population_and_keeps_fixed_endpoints():
    world = make_3d_static_world(
        bounds_max=(10.0, 10.0, 10.0),
        start=(0.0, 0.0, 0.0),
        goal=(9.0, 9.0, 9.0),
        obstacles=[],
        v_max=50.0,
        a_max=200.0,
        d_safe=0.5,
        d_comfort=1.5,
        t_max_factor=1.75,
    )
    config = TrajectoryConfig(8, 3, world.start, world.goal, world.t_max)
    initial = [create_straight_line_trajectory(config)]

    optimizer = NSGA2Optimizer(population_size=10, max_evaluations=20, seed=11)
    population = optimizer.optimize(world, config, initial_population=initial, time_offset=0.0)

    assert len(population) >= 1
    for ind in population:
        np.testing.assert_allclose(ind.trajectory.control_points[0], world.start)
        np.testing.assert_allclose(ind.trajectory.control_points[-1], world.goal)


def test_dynamic_event_controller_enforces_budgets_and_warm_start():
    scenario = make_scenario()
    controller = DynamicWarmStartController(
        population_size=20,
        per_event_budget=100,
        total_budget=400,
        seed=123,
    )

    cold = controller.run_condition("cold", scenario, condition_seed=17)
    warm = controller.run_condition("warm", scenario, condition_seed=17)

    assert cold["total_evaluations"] == 400
    assert warm["total_evaluations"] == 400
    assert cold["event_count"] == 4
    assert warm["event_count"] == 4
    assert cold["events"][-1]["cumulative_evaluations"] == 400
    assert warm["events"][-1]["cumulative_evaluations"] == 400
    assert cold["events"][0]["objective_evaluations"] <= 100
    assert warm["events"][0]["objective_evaluations"] <= 100
    assert warm["events"][0]["accepted_imported_count"] == 0
    for event in cold["events"] + warm["events"]:
        assert event["objective_evaluations"] == 100
        assert event["wall_clock_seconds"] >= 0.0
        assert event["no_feasible_solution"] == (event["feasible_count"] == 0)
        assert event["reevaluated_imported_count"] == event["accepted_imported_count"]


def test_dynamic_event_controller_reuses_only_current_environment_snapshot():
    scenario = make_scenario()
    controller = DynamicWarmStartController(
        population_size=20,
        per_event_budget=100,
        total_budget=400,
        seed=55,
    )

    warm = controller.run_condition("warm", scenario, condition_seed=55)
    assert len(warm["events"]) == 4

    for event in warm["events"]:
        assert "snapshot_id" in event
        assert event["objective_evaluations"] >= 1
        assert event["population_size"] == 20


def test_warm_start_uses_least_violation_front_when_no_feasible_individuals_exist():
    scenario = make_scenario()
    trajectory = create_straight_line_trajectory(scenario.config)
    evaluation = TrajectoryEvaluator(scenario.world).evaluate(trajectory)
    assert not evaluation.is_feasible

    controller = DynamicWarmStartController(population_size=20, per_event_budget=100, total_budget=400)
    imported = controller._warm_start_population([Individual(trajectory, evaluation)])

    assert imported == [trajectory]


def test_piecewise_obstacle_schedule_remains_bounded_and_changes_clearance_across_events():
    waypoints = [
        np.array([4.81472178, 5.17982633, 5.05417576], dtype=float),
        np.array([5.59637246, 4.32515273, 5.71900495], dtype=float),
        np.array([5.38706698, 4.27000335, 5.29863145], dtype=float),
        np.array([4.28149703, 6.5, 5.72593602], dtype=float),
        np.array([5.01472178, 5.27982633, 5.15417576], dtype=float),
    ]
    world = make_3d_static_world(
        bounds_max=(10.0, 10.0, 10.0),
        start=(0.0, 0.0, 0.0),
        goal=(9.0, 9.0, 9.0),
        obstacles=[
            SphereObstacle(
                center=waypoints[0],
                radius=1.4722837237301862,
                path=waypoints,
                path_times=[0.0, 2.5, 5.0, 7.5, 10.0],
            )
        ],
        v_max=3.0,
        a_max=5.0,
        d_safe=0.5,
        d_comfort=1.5,
        t_max_factor=1.75,
    )

    for t in [0.0, 2.5, 5.0, 7.5, 10.0]:
        center = world.obstacles[0].position_at(t)
        assert np.all(center >= 0.0)
        assert np.all(center <= 10.0)

    config = TrajectoryConfig(8, 3, world.start, world.goal, world.t_max)
    traj = create_straight_line_trajectory(config)
    free = traj.free_control_points.copy()
    free[2] = np.array([5.2, 5.2, 5.2], dtype=float)
    traj.genes[:config.n_position_genes] = free.reshape(-1)

    evaluations = [TrajectoryEvaluator(world, time_offset=t).evaluate(traj) for t in [0.0, 2.5, 5.0, 7.5]]
    min_clearances = [ev.min_clearance for ev in evaluations]
    clearance_delta = max(abs(a - b) for a in min_clearances for b in min_clearances)

    assert clearance_delta > 0.75
    assert all(not ev.is_feasible for ev in evaluations)

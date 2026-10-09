import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from planner.environment.obstacles import SphereObstacle
from planner.environment.world import make_3d_static_world
from planner.evaluation.evaluator import TrajectoryEvaluator
from planner.optimization.dynamic import DynamicWarmStartController, DynamicScenario
from planner.optimization.nsga2 import NSGA2Optimizer
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

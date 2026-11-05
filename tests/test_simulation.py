import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from simulation_worker import optimize
from planner.evaluation.evaluator import TrajectoryEvaluator
from planner.representation.trajectory import TrajectoryConfig, create_straight_line_trajectory
from planner.simulation import build_simulation_scenario, scenario_payload, world_at_event


def test_event_shift_adapter_matches_native_evaluator_time_offset():
    scenario = build_simulation_scenario("dynamic", event_index=2)
    config = TrajectoryConfig(8, 3, scenario.world.start, scenario.world.goal, scenario.world.t_max)
    trajectory = create_straight_line_trajectory(config)

    native = TrajectoryEvaluator(scenario.world, time_offset=scenario.time_offset).evaluate(trajectory)
    shifted = TrajectoryEvaluator(world_at_event(scenario.world, scenario.time_offset)).evaluate(trajectory)

    np.testing.assert_allclose(native.objectives, shifted.objectives)
    assert native.total_violation == shifted.total_violation
    assert native.violation_details == shifted.violation_details


def test_dynamic_scenario_payload_uses_backend_obstacle_schedule():
    scenario = build_simulation_scenario("dynamic", event_index=3)
    payload = scenario_payload("dynamic", event_index=3)
    obstacle = scenario.world.obstacles[0]

    np.testing.assert_allclose(payload["world"]["obstacles"][0]["center"], obstacle.position_at(scenario.time_offset))
    assert payload["event_time"] == 7.5
    assert payload["world"]["obstacles"][0]["motion"]["path_times"] == [0.0, 2.5, 5.0, 7.5, 10.0]


def test_custom_endpoints_are_applied_and_rejected_outside_world_bounds():
    scenario = build_simulation_scenario("dynamic", 1, start=(1.0, 2.0, 1.0), goal=(8.0, 8.0, 9.0))

    np.testing.assert_allclose(scenario.world.start, [1.0, 2.0, 1.0])
    np.testing.assert_allclose(scenario.world.goal, [8.0, 8.0, 9.0])
    with pytest.raises(ValueError, match="world bounds"):
        build_simulation_scenario("dynamic", 1, start=(-1.0, 2.0, 1.0))


@pytest.mark.parametrize("algorithm", ["nsga2", "mopso", "hybrid"])
def test_interactive_worker_returns_real_dynamic_candidates_for_each_optimizer(algorithm):
    scenario = build_simulation_scenario("dynamic", event_index=1)
    config = TrajectoryConfig(8, 3, scenario.world.start, scenario.world.goal, scenario.world.t_max)
    seed_trajectory = create_straight_line_trajectory(config)
    request = {
        "algorithm": algorithm,
        "scenario_id": "dynamic",
        "event_index": 1,
        "population_size": 8,
        "max_evaluations": 8,
        "seed": 17,
        "start": scenario.world.start.tolist(),
        "goal": scenario.world.goal.tolist(),
    }
    if algorithm == "nsga2":
        request["initial_population"] = [seed_trajectory.genes.tolist()]

    result = optimize(request)

    assert result["status"] == "complete"
    assert result["event_time"] == 2.5
    assert result["objective_evaluations"] == 8
    assert result["optimization_seconds"] > 0.0
    assert len(result["candidates"]) == 8
    assert all(len(candidate["positions"]) == 160 for candidate in result["candidates"])
    if algorithm == "nsga2":
        assert result["imported_trajectory_count"] == 1
        assert result["accepted_imported_count"] == 1
        assert result["reevaluated_imported_count"] == 1
    else:
        assert result["imported_trajectory_count"] == 0

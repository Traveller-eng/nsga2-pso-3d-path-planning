from planner.simulation_runtime import DroneState, SimulationState


def test_drone_state_tracks_executed_path_and_telemetry():
    state = SimulationState(world_bounds=(10.0, 10.0, 10.0), start=(0.0, 0.0, 0.0), goal=(9.0, 9.0, 9.0))
    drone = state.primary_drone
    drone.sync_plan([(0.0, 0.0, 0.0), (4.0, 4.0, 4.0), (9.0, 9.0, 9.0)])
    drone.status = "active"

    state.advance(0.5)

    assert drone.position != (0.0, 0.0, 0.0)
    assert len(drone.executed_path) >= 2
    assert drone.distance_to_goal > 0.0
    assert drone.status in {"active", "idle"}


def test_simulation_state_records_runtime_events_and_obstacles():
    state = SimulationState(world_bounds=(10.0, 10.0, 10.0), start=(0.0, 0.0, 0.0), goal=(9.0, 9.0, 9.0))
    state.add_runtime_obstacle("obs-1", kind="sphere", center=(5.0, 5.0, 5.0), radius=1.5)
    state.record_event("obstacle_added", {"obstacle_id": "obs-1"})

    assert "obs-1" in state.runtime_obstacles
    assert state.environment_version == 1
    assert any(entry["type"] == "simulation_started" for entry in state.event_log)
    assert any(entry["type"] == "obstacle_added" for entry in state.event_log)
    assert state.event_log[-1]["payload"]["obstacle_id"] == "obs-1"


def test_runtime_collision_detection_flags_path_blocked_by_obstacle():
    state = SimulationState(world_bounds=(10.0, 10.0, 10.0), start=(0.0, 0.0, 0.0), goal=(9.0, 9.0, 9.0))
    state.primary_drone.sync_plan([(0.0, 0.0, 0.0), (5.0, 5.0, 5.0), (9.0, 9.0, 9.0)])
    state.primary_drone.position = (4.7, 4.7, 4.7)
    state.add_runtime_obstacle("obs-1", kind="sphere", center=(5.0, 5.0, 5.0), radius=0.7)

    report = state.check_active_drones_for_runtime_collisions()

    assert report[0]["collision"] is True
    assert report[0]["obstacle_id"] == "obs-1"
    assert report[0]["drone_id"] == "drone-01"


def test_runtime_disruption_marks_replanning_event():
    state = SimulationState(world_bounds=(10.0, 10.0, 10.0), start=(0.0, 0.0, 0.0), goal=(9.0, 9.0, 9.0))
    state.primary_drone.sync_plan([(0.0, 0.0, 0.0), (9.0, 9.0, 9.0)])
    state.trigger_runtime_disruption("static_obstacle", {"kind": "sphere", "center": (4.0, 4.0, 4.0), "radius": 1.0})

    assert state.environment_version >= 2
    assert any(entry["type"] == "replanning_required" for entry in state.event_log)


def test_runtime_collision_detection_flags_drone_to_drone_conflict():
    state = SimulationState(world_bounds=(10.0, 10.0, 10.0), start=(0.0, 0.0, 0.0), goal=(9.0, 9.0, 9.0))
    state.primary_drone.sync_plan([(0.0, 0.0, 0.0), (5.0, 5.0, 5.0), (9.0, 9.0, 9.0)])
    state.primary_drone.position = (4.8, 4.8, 4.8)

    state.add_drone("drone-02", start=(0.0, 0.0, 0.0), goal=(9.0, 9.0, 9.0))
    other = state.drones["drone-02"]
    other.sync_plan([(0.0, 0.0, 0.0), (5.1, 5.1, 5.1), (9.0, 9.0, 9.0)])
    other.position = (5.0, 5.0, 5.0)

    report = state.check_active_drones_for_runtime_collisions()

    assert any(
        entry["drone_id"] == "drone-01" and entry["conflict_type"] == "drone_to_drone"
        for entry in report
    )


def test_runtime_replan_generates_a_new_route_for_blocked_drone():
    state = SimulationState(world_bounds=(10.0, 10.0, 10.0), start=(0.0, 0.0, 0.0), goal=(9.0, 9.0, 9.0))
    state.add_runtime_obstacle("obs-1", kind="sphere", center=(5.0, 5.0, 5.0), radius=1.5)
    state.primary_drone.sync_plan([(0.0, 0.0, 0.0), (4.0, 4.0, 4.0), (9.0, 9.0, 9.0)])
    state.primary_drone.position = (3.8, 3.8, 3.8)

    result = state.replan_drone("drone-01", max_evaluations=20, seed=7)

    assert result["drone_id"] == "drone-01"
    assert result["path_length"] >= 2
    assert state.primary_drone.status == "active"
    assert state.primary_drone.planned_path[0] == tuple(state.primary_drone.position)
    assert state.primary_drone.planned_path[-1] == (9.0, 9.0, 9.0)

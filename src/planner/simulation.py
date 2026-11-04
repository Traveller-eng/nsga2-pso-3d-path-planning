from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np

from planner.environment.obstacles import AABBObstacle, Obstacle, SphereObstacle
from planner.environment.world import WorldConfig, make_3d_static_world

ROOT = Path(__file__).resolve().parents[2]
DYNAMIC_CONFIG = ROOT / "experiments" / "configs" / "dynamic_warm_start.json"


@dataclass(frozen=True)
class SimulationScenario:
    scenario_id: str
    world: WorldConfig
    event_times: tuple[float, ...]
    event_index: int

    @property
    def time_offset(self) -> float:
        return self.event_times[self.event_index] if self.scenario_id == "dynamic" else 0.0


@dataclass(frozen=True)
class TimeShiftedObstacle:
    obstacle: Obstacle
    offset: float

    def position_at(self, t: float) -> Any:
        return self.obstacle.position_at(t + self.offset)

    def distance_to_point(self, point: np.ndarray, t: float = 0.0) -> float:
        return self.obstacle.distance_to_point(point, t + self.offset)

    def is_collision(self, point: np.ndarray, d_safe: float, t: float = 0.0) -> bool:
        return self.obstacle.is_collision(point, d_safe, t + self.offset)


def load_dynamic_config() -> dict[str, Any]:
    return json.loads(DYNAMIC_CONFIG.read_text(encoding="utf-8"))


def _endpoint(name: str, point: tuple[float, ...] | None, bounds_max: tuple[float, ...], default: tuple[float, ...]) -> tuple[float, ...]:
    value = np.asarray(default if point is None else point, dtype=float)
    upper = np.asarray(bounds_max, dtype=float)
    if value.shape != upper.shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain {upper.size} finite coordinates.")
    if np.any(value < 0.0) or np.any(value > upper):
        raise ValueError(f"{name} must remain within the world bounds.")
    return tuple(float(coordinate) for coordinate in value)


def build_simulation_scenario(
    scenario_id: str = "static",
    event_index: int = 0,
    start: tuple[float, ...] | None = None,
    goal: tuple[float, ...] | None = None,
) -> SimulationScenario:
    if scenario_id == "static":
        bounds_max = (10.0, 10.0, 10.0)
        start_point = _endpoint("start", start, bounds_max, (0.0, 0.0, 0.0))
        goal_point = _endpoint("goal", goal, bounds_max, (9.0, 9.0, 9.0))
        if np.allclose(start_point, goal_point):
            raise ValueError("start and goal must be different points.")
        world = make_3d_static_world(start=start_point, goal=goal_point)
        return SimulationScenario("static", world, (0.0,), 0)
    if scenario_id != "dynamic":
        raise ValueError("scenario must be 'static' or 'dynamic'.")

    scenario_config = load_dynamic_config()["scenario"]
    event_times = tuple(float(value) for value in scenario_config["event_times"])
    if event_index < 0 or event_index >= len(event_times):
        raise ValueError(f"event_index must be between 0 and {len(event_times) - 1}.")
    bounds_max = tuple(float(value) for value in scenario_config["bounds_max"])
    start_point = _endpoint("start", start, bounds_max, tuple(scenario_config["start"]))
    goal_point = _endpoint("goal", goal, bounds_max, tuple(scenario_config["goal"]))
    if np.allclose(start_point, goal_point):
        raise ValueError("start and goal must be different points.")
    obstacle_config = scenario_config["obstacle"]
    waypoints = [np.asarray(point, dtype=float) for point in obstacle_config["waypoints"]]
    obstacle = SphereObstacle(
        center=waypoints[0].copy(),
        radius=float(obstacle_config["radius"]),
        path=waypoints,
        path_times=[float(value) for value in obstacle_config["path_times"]],
    )
    world = make_3d_static_world(
        bounds_max=bounds_max,
        start=start_point,
        goal=goal_point,
        obstacles=[obstacle],
        v_max=float(scenario_config["v_max"]),
        a_max=float(scenario_config["a_max"]),
        d_safe=float(scenario_config["d_safe"]),
        d_comfort=float(scenario_config["d_comfort"]),
        t_max_factor=float(scenario_config["t_max_factor"]),
    )
    return SimulationScenario("dynamic", world, event_times, event_index)


def world_at_event(world: WorldConfig, time_offset: float) -> WorldConfig:
    """Adapt optimizers without a time_offset argument to the evaluator's physical-time convention."""
    if time_offset == 0.0:
        return world
    return replace(
        world,
        obstacles=[TimeShiftedObstacle(obstacle, float(time_offset)) for obstacle in world.obstacles],
    )


def _obstacle_payload(obstacle: Obstacle, time_offset: float) -> dict[str, Any]:
    if isinstance(obstacle, SphereObstacle):
        payload: dict[str, Any] = {
            "kind": "sphere",
            "center": obstacle.position_at(time_offset).tolist(),
            "radius": float(obstacle.radius),
        }
        if obstacle.path is not None:
            payload["motion"] = {
                "kind": "piecewise_linear",
                "path": [np.asarray(point, dtype=float).tolist() for point in obstacle.path],
                "path_times": [float(value) for value in obstacle.path_times or []],
            }
        elif obstacle.velocity is not None:
            payload["motion"] = {
                "kind": "linear",
                "initial_center": obstacle.center.tolist(),
                "velocity": obstacle.velocity.tolist(),
            }
        return payload

    lower, upper = obstacle.position_at(time_offset)
    payload = {"kind": "aabb", "lower": lower.tolist(), "upper": upper.tolist()}
    if obstacle.velocity is not None:
        payload["motion"] = {
            "kind": "linear_aabb",
            "initial_lower": obstacle.lower.tolist(),
            "initial_upper": obstacle.upper.tolist(),
            "velocity": obstacle.velocity.tolist(),
        }
    return payload


def scenario_payload(
    scenario_id: str = "static",
    event_index: int = 0,
    start: tuple[float, ...] | None = None,
    goal: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    scenario = build_simulation_scenario(scenario_id, event_index, start, goal)
    world = scenario.world
    return {
        "scenario_id": scenario.scenario_id,
        "event_index": scenario.event_index,
        "event_time": scenario.time_offset,
        "event_times": list(scenario.event_times),
        "world": {
            "bounds_min": world.lower_bounds.tolist(),
            "bounds_max": world.upper_bounds.tolist(),
            "start": world.start.tolist(),
            "goal": world.goal.tolist(),
            "t_max": float(world.t_max),
            "v_max": float(world.v_max),
            "a_max": float(world.a_max),
            "d_safe": float(world.d_safe),
            "d_comfort": float(world.d_comfort),
            "obstacles": [_obstacle_payload(obstacle, scenario.time_offset) for obstacle in world.obstacles],
        },
        "trajectory": {"n_control_points": 8, "degree": 5, "dim": world.dim},
    }

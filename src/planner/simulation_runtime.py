from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from planner.environment.obstacles import AABBObstacle, SphereObstacle
from planner.environment.world import make_3d_static_world
from planner.optimization.nsga2 import NSGA2Optimizer
from planner.representation.trajectory import TrajectoryConfig, create_straight_line_trajectory


@dataclass
class DroneState:
    """Runtime drone state connected to optimizer-produced trajectories."""

    drone_id: str
    start: tuple[float, float, float]
    goal: tuple[float, float, float]
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    heading: float = 0.0
    altitude: float = 0.0
    status: str = "idle"
    planned_path: list[tuple[float, float, float]] = field(default_factory=list)
    executed_path: list[tuple[float, float, float]] = field(default_factory=list)
    trajectory_history: list[list[tuple[float, float, float]]] = field(default_factory=list)
    environment_version: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    _path_index: int = field(default=1, init=False, repr=False)

    def __post_init__(self) -> None:
        self.position = tuple(float(value) for value in self.start)
        if not self.planned_path:
            self.planned_path = [self.start, self.goal]
        self.executed_path = [self.position]
        self.metrics.setdefault("distance_to_goal", self.distance_to_goal)

    @property
    def distance_to_goal(self) -> float:
        return float(np.linalg.norm(np.asarray(self.goal, dtype=float) - np.asarray(self.position, dtype=float)))

    def sync_plan(self, path: list[tuple[float, float, float]] | np.ndarray | None) -> None:
        if path is None:
            return
        converted = [tuple(float(value) for value in point) for point in np.asarray(path, dtype=float).tolist()]
        self.planned_path = converted if converted else [self.start, self.goal]
        self._path_index = 1 if len(self.planned_path) > 1 else 0
        self.status = "active"
        self.trajectory_history.append(self.planned_path)

    def apply_trajectory_step(self, dt: float) -> None:
        if len(self.planned_path) < 2:
            return
        if self._path_index >= len(self.planned_path):
            self.status = "idle"
            return

        current = np.asarray(self.position, dtype=float)
        target = np.asarray(self.planned_path[self._path_index], dtype=float)
        delta = target - current
        distance = float(np.linalg.norm(delta))
        if distance < 1e-9:
            self.position = tuple(float(value) for value in target)
            self.executed_path.append(self.position)
            self._path_index += 1
            self.metrics["distance_to_goal"] = self.distance_to_goal
            if self._path_index >= len(self.planned_path):
                self.status = "idle"
            else:
                self.status = "active"
            return

        step = min(dt, distance)
        direction = delta / distance
        next_position = current + direction * step
        self.position = tuple(float(value) for value in next_position)
        self.executed_path.append(self.position)
        self.velocity = tuple(float(value) for value in direction * (distance / max(dt, 1e-9)))
        self.heading = float(np.arctan2(direction[1], direction[0]))
        self.altitude = float(self.position[2])
        self.metrics["distance_to_goal"] = self.distance_to_goal

        if np.allclose(np.asarray(self.position), target, atol=1e-3):
            self._path_index += 1
            if self._path_index >= len(self.planned_path):
                self.status = "idle"
            else:
                self.status = "active"
        else:
            self.status = "active"


@dataclass
class SimulationState:
    """Central runtime simulation state for the planner and 3D scene."""

    world_bounds: tuple[float, float, float]
    start: tuple[float, float, float]
    goal: tuple[float, float, float]
    primary_drone_id: str = "drone-01"
    simulation_time: float = 0.0
    dt: float = 0.1
    environment_version: int = 0
    event_log: list[dict[str, Any]] = field(default_factory=list)
    runtime_obstacles: dict[str, dict[str, Any]] = field(default_factory=dict)
    drones: dict[str, DroneState] = field(default_factory=dict)
    drone_separation_distance: float = 1.5

    def __post_init__(self) -> None:
        self.drones[self.primary_drone_id] = DroneState(
            drone_id=self.primary_drone_id,
            start=self.start,
            goal=self.goal,
            position=self.start,
            altitude=float(self.start[2]),
            status="idle",
        )
        self.record_event("simulation_started", {"start": self.start, "goal": self.goal})

    @property
    def primary_drone(self) -> DroneState:
        return self.drones[self.primary_drone_id]

    def record_event(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        record = {"time": self.simulation_time, "type": event_type, "payload": payload or {}}
        self.event_log.append(record)

    def add_runtime_obstacle(self, obstacle_id: str, kind: str, **kwargs: Any) -> dict[str, Any]:
        obstacle = {"id": obstacle_id, "kind": kind, **kwargs}
        self.runtime_obstacles[obstacle_id] = obstacle
        self.environment_version += 1
        self.record_event("obstacle_added", {"obstacle_id": obstacle_id, "kind": kind})
        return obstacle

    def remove_runtime_obstacle(self, obstacle_id: str) -> None:
        if obstacle_id in self.runtime_obstacles:
            del self.runtime_obstacles[obstacle_id]
            self.environment_version += 1
            self.record_event("obstacle_removed", {"obstacle_id": obstacle_id})

    def add_drone(self, drone_id: str, start: tuple[float, float, float], goal: tuple[float, float, float]) -> DroneState:
        drone = DroneState(drone_id=drone_id, start=start, goal=goal, position=start, altitude=float(start[2]))
        self.drones[drone_id] = drone
        self.record_event("drone_added", {"drone_id": drone_id, "start": start, "goal": goal})
        return drone

    def check_active_drones_for_runtime_collisions(self) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        drone_ids = list(self.drones.keys())

        for drone_id, drone in self.drones.items():
            for obstacle_id, obstacle in self.runtime_obstacles.items():
                for point in drone.planned_path:
                    if obstacle.get("kind") == "sphere":
                        center = np.asarray(obstacle.get("center", (0.0, 0.0, 0.0)), dtype=float)
                        radius = float(obstacle.get("radius", 0.0))
                        distance = float(np.linalg.norm(np.asarray(point, dtype=float) - center))
                        collision = distance <= radius
                    elif obstacle.get("kind") == "aabb":
                        lower = np.asarray(obstacle.get("lower", (0.0, 0.0, 0.0)), dtype=float)
                        upper = np.asarray(obstacle.get("upper", (0.0, 0.0, 0.0)), dtype=float)
                        point_arr = np.asarray(point, dtype=float)
                        in_box = bool(np.all(point_arr >= lower) and np.all(point_arr <= upper))
                        collision = in_box
                    else:
                        collision = False
                    if collision:
                        reports.append({
                            "drone_id": drone_id,
                            "obstacle_id": obstacle_id,
                            "conflict_type": "obstacle",
                            "collision": True,
                            "point": tuple(float(v) for v in point),
                            "environment_version": self.environment_version,
                        })

        for index, drone_id in enumerate(drone_ids):
            drone = self.drones[drone_id]
            for other_id in drone_ids[index + 1:]:
                other = self.drones[other_id]
                common_steps = min(len(drone.planned_path), len(other.planned_path))
                for step in range(common_steps):
                    point_a = np.asarray(drone.planned_path[step], dtype=float)
                    point_b = np.asarray(other.planned_path[step], dtype=float)
                    separation = float(np.linalg.norm(point_a - point_b))
                    if separation <= self.drone_separation_distance:
                        reports.append({
                            "drone_id": drone_id,
                            "other_drone_id": other_id,
                            "conflict_type": "drone_to_drone",
                            "collision": True,
                            "distance": separation,
                            "point": tuple(float(v) for v in point_a),
                            "other_point": tuple(float(v) for v in point_b),
                            "environment_version": self.environment_version,
                        })
        return reports

    def trigger_runtime_disruption(self, disruption_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.environment_version += 1
        if payload.get("kind") is not None:
            obstacle_id = payload.get("obstacle_id", f"runtime-{self.environment_version}")
            self.add_runtime_obstacle(obstacle_id, payload["kind"], **{k: v for k, v in payload.items() if k not in {"kind", "obstacle_id"}})
        self.record_event("replanning_required", {"type": disruption_type, "payload": payload, "environment_version": self.environment_version})
        drone_id = payload.get("drone_id") or self.primary_drone_id
        if drone_id in self.drones:
            self.replan_drone(drone_id, max_evaluations=20, seed=self.environment_version)
        return {"type": disruption_type, "environment_version": self.environment_version}

    def replan_drone(self, drone_id: str, *, max_evaluations: int = 40, seed: int = 0) -> dict[str, Any]:
        if drone_id not in self.drones:
            raise KeyError(f"Unknown drone '{drone_id}'.")

        drone = self.drones[drone_id]
        runtime_obstacles: list[Any] = []
        for obstacle in self.runtime_obstacles.values():
            kind = obstacle.get("kind")
            if kind == "sphere":
                center = np.asarray(obstacle.get("center", drone.start), dtype=float)
                radius = float(obstacle.get("radius", 0.0))
                motion = obstacle.get("motion") or {}
                velocity = motion.get("velocity")
                path = motion.get("path")
                if path:
                    path_points = [np.asarray(point, dtype=float) for point in path]
                    path_times = [float(value) for value in motion.get("path_times", [])]
                    runtime_obstacles.append(
                        SphereObstacle(
                            center=path_points[0].copy(),
                            radius=radius,
                            path=path_points,
                            path_times=path_times if len(path_times) == len(path_points) else None,
                        )
                    )
                else:
                    runtime_obstacles.append(
                        SphereObstacle(
                            center=center.copy(),
                            radius=radius,
                            velocity=np.asarray(velocity, dtype=float) if velocity is not None else None,
                        )
                    )
            elif kind == "aabb":
                lower = np.asarray(obstacle.get("lower", (0.0, 0.0, 0.0)), dtype=float)
                upper = np.asarray(obstacle.get("upper", (0.0, 0.0, 0.0)), dtype=float)
                motion = obstacle.get("motion") or {}
                velocity = motion.get("velocity")
                runtime_obstacles.append(
                    AABBObstacle(
                        lower=lower.copy(),
                        upper=upper.copy(),
                        velocity=np.asarray(velocity, dtype=float) if velocity is not None else None,
                    )
                )

        world = make_3d_static_world(
            bounds_max=self.world_bounds,
            start=tuple(float(value) for value in drone.position),
            goal=drone.goal,
            obstacles=runtime_obstacles,
            v_max=3.0,
            a_max=5.0,
            d_safe=0.5,
            d_comfort=1.5,
            t_max_factor=1.75,
        )
        config = TrajectoryConfig(
            n_control_points=8,
            dim=3,
            start=np.asarray(drone.position, dtype=float),
            goal=np.asarray(drone.goal, dtype=float),
            t_max=world.t_max,
            degree=5,
            optimize_time=False,
        )
        optimizer = NSGA2Optimizer(
            population_size=max(8, min(20, int(max_evaluations))),
            max_evaluations=max(10, int(max_evaluations)),
            seed=int(seed),
        )
        initial = [create_straight_line_trajectory(config)]
        population = optimizer.optimize(world, config, initial_population=initial, time_offset=0.0)
        feasible = [ind for ind in population if ind.evaluation.is_feasible]
        candidate = min(feasible, key=lambda ind: ind.evaluation.objectives[0], default=min(population, key=lambda ind: ind.evaluation.total_violation))

        path_points = np.asarray(candidate.trajectory.control_points, dtype=float)
        route = [tuple(float(value) for value in point) for point in path_points]
        drone.sync_plan(route)
        self.record_event(
            "drone_replanned",
            {
                "drone_id": drone_id,
                "path_length": len(route),
                "objective_values": [float(value) for value in candidate.evaluation.objectives],
                "environment_version": self.environment_version,
            },
        )
        return {"drone_id": drone_id, "path_length": len(route), "path": route}

    def advance(self, dt: float | None = None) -> None:
        step = self.dt if dt is None else float(dt)
        self.simulation_time += step
        for drone in self.drones.values():
            if drone.status != "idle":
                drone.apply_trajectory_step(step)

        collisions = self.check_active_drones_for_runtime_collisions()
        if collisions:
            self.record_event("runtime_collision_detected", {"collisions": collisions})

    def seed_primary_plan(self, path: list[tuple[float, float, float]] | np.ndarray) -> None:
        self.primary_drone.sync_plan(path)
        self.primary_drone.status = "active"
        self.record_event("route_loaded", {"drone_id": self.primary_drone.drone_id, "point_count": len(self.primary_drone.planned_path)})


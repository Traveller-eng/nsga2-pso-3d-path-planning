from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from planner.environment.world import WorldConfig
from planner.optimization.nsga2 import NSGA2Optimizer, fast_non_dominated_sort
from planner.representation.trajectory import Trajectory, TrajectoryConfig, repair_trajectory


@dataclass
class DynamicScenario:
    """Simple deterministic schedule for a dynamic replanning experiment."""

    world: WorldConfig
    event_times: list[float]
    snapshot_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.event_times:
            raise ValueError("DynamicScenario requires a non-empty event_times list.")
        if not self.snapshot_ids:
            self.snapshot_ids = [f"event_{index}" for index in range(len(self.event_times))]

    @property
    def config(self) -> TrajectoryConfig:
        return TrajectoryConfig(
            n_control_points=8,
            dim=self.world.dim,
            start=self.world.start,
            goal=self.world.goal,
            t_max=self.world.t_max,
            degree=5,
            optimize_time=False,
        )

    def snapshot_id(self, event_index: int) -> str:
        if event_index < 0 or event_index >= len(self.snapshot_ids):
            raise IndexError(f"Event index {event_index} is out of range for {len(self.snapshot_ids)} snapshots.")
        return self.snapshot_ids[event_index]


class DynamicWarmStartController:
    """Minimal warm-start versus cold-start controller for a multi-event replanning smoke test."""

    def __init__(
        self,
        population_size: int = 20,
        per_event_budget: int = 100,
        total_budget: int = 400,
        seed: int = 0,
    ) -> None:
        self.population_size = population_size
        self.per_event_budget = per_event_budget
        self.total_budget = total_budget
        self.seed = seed

    def _warm_start_population(self, population: list[Any]) -> list[Trajectory]:
        if not population:
            return []

        fronts = fast_non_dominated_sort(population)
        if not fronts or not fronts[0]:
            feasible = [ind for ind in population if ind.evaluation.is_feasible]
            if not feasible:
                return []
            return [ind.trajectory for ind in feasible[: self.population_size]]

        front_zero = [ind for ind in fronts[0] if ind.evaluation.is_feasible]
        if not front_zero:
            feasible = [ind for ind in population if ind.evaluation.is_feasible]
            if not feasible:
                return []
            front_zero = feasible

        selected = front_zero[: self.population_size]
        return [ind.trajectory for ind in selected]

    def _validate_imported_population(self, trajectories: list[Trajectory], world: WorldConfig) -> tuple[list[Trajectory], list[str]]:
        valid: list[Trajectory] = []
        reasons: list[str] = []
        for idx, trajectory in enumerate(trajectories):
            try:
                repaired = repair_trajectory(trajectory, world)
            except Exception as exc:  # pragma: no cover - defensive validation
                reasons.append(f"trajectory_{idx}: repair failed ({exc})")
                continue

            start_ok = np.allclose(repaired.control_points[0], world.start)
            goal_ok = np.allclose(repaired.control_points[-1], world.goal)
            if not (start_ok and goal_ok):
                reasons.append(f"trajectory_{idx}: endpoints not fixed after repair")
                continue
            valid.append(repaired)
        return valid, reasons

    def run_condition(self, condition: str, scenario: DynamicScenario, condition_seed: int = 0) -> dict[str, Any]:
        if condition not in {"cold", "warm"}:
            raise ValueError(f"Unsupported condition '{condition}'. Expected 'cold' or 'warm'.")

        cumulative = 0
        event_records: list[dict[str, Any]] = []
        previous_population: list[Any] | None = None

        for event_index, event_time in enumerate(scenario.event_times):
            if condition == "cold" or event_index == 0:
                initial_population = None
                import_count = 0
                rejected_reasons = []
            else:
                imported = self._warm_start_population(previous_population or [])
                initial_population, rejected_reasons = self._validate_imported_population(imported, scenario.world)
                import_count = len(imported)

            optimizer = NSGA2Optimizer(
                population_size=self.population_size,
                max_evaluations=self.per_event_budget,
                seed=self.seed + condition_seed + event_index,
            )
            population = optimizer.optimize(
                scenario.world,
                scenario.config,
                initial_population=initial_population,
                time_offset=event_time,
            )

            if optimizer.evals > self.per_event_budget:
                raise RuntimeError(f"Event {event_index} exceeded per-event budget: {optimizer.evals} > {self.per_event_budget}.")
            if cumulative + optimizer.evals > self.total_budget:
                raise RuntimeError(
                    f"Condition {condition} exceeded total budget at event {event_index}: "
                    f"{cumulative + optimizer.evals} > {self.total_budget}."
                )

            cumulative += optimizer.evals
            fea = [ind for ind in population if ind.evaluation.is_feasible]
            event_record = {
                "condition": condition,
                "seed": self.seed + condition_seed,
                "event_index": event_index,
                "event_time": float(event_time),
                "snapshot_id": scenario.snapshot_id(event_index),
                "objective_evaluations": int(optimizer.evals),
                "cumulative_evaluations": int(cumulative),
                "population_size": len(population),
                "feasible_count": len(fea),
                "feasibility_rate": float(len(fea) / len(population)) if population else 0.0,
                "imported_trajectory_count": int(import_count),
                "accepted_imported_count": len(initial_population) if initial_population is not None else 0,
                "rejected_imported_count": len(rejected_reasons),
                "rejected_imported_reasons": rejected_reasons,
                "best_objectives": np.min(np.array([ind.evaluation.objectives for ind in fea], dtype=float), axis=0).tolist() if fea else None,
            }
            event_records.append(event_record)
            previous_population = population

        return {
            "condition": condition,
            "seed": self.seed + condition_seed,
            "event_count": len(event_records),
            "total_evaluations": int(cumulative),
            "events": event_records,
        }

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from planner.geometry.sampling import parametric_to_physical, sample_trajectory
from planner.optimization.hybrid import HybridNSGA2PSOOptimizer
from planner.optimization.mopso import MOPSOOptimizer
from planner.optimization.nsga2 import Individual, NSGA2Optimizer, fast_non_dominated_sort
from planner.representation.trajectory import TrajectoryConfig, genes_to_trajectory
from planner.simulation import build_simulation_scenario, world_at_event


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


def optimize(request: dict[str, Any]) -> dict[str, Any]:
    algorithm = request["algorithm"]
    if algorithm not in {"nsga2", "mopso", "hybrid"}:
        raise ValueError("algorithm must be nsga2, mopso, or hybrid.")
    scenario = build_simulation_scenario(
        request["scenario_id"],
        int(request["event_index"]),
        request.get("start"),
        request.get("goal"),
    )
    population_size = int(request["population_size"])
    max_evaluations = int(request["max_evaluations"])
    seed = int(request["seed"])
    if not 8 <= population_size <= 80:
        raise ValueError("population_size must be between 8 and 80.")
    if not population_size <= max_evaluations <= 5000:
        raise ValueError("max_evaluations must be between population_size and 5000.")

    world = scenario.world
    config = TrajectoryConfig(
        n_control_points=8,
        dim=world.dim,
        start=world.start,
        goal=world.goal,
        t_max=world.t_max,
        degree=5,
        optimize_time=False,
    )
    initial_genes = request.get("initial_population") or []
    if initial_genes and algorithm != "nsga2":
        raise ValueError("Warm-start populations are currently supported only by NSGA-II.")
    if len(initial_genes) > population_size:
        raise ValueError("initial_population cannot exceed population_size.")
    initial_population = []
    for index, genes in enumerate(initial_genes):
        array = np.asarray(genes, dtype=float)
        if array.shape != (config.n_genes,) or not np.isfinite(array).all():
            raise ValueError(f"initial_population[{index}] must contain {config.n_genes} finite genes.")
        initial_population.append(genes_to_trajectory(array, config))

    optimization_started = time.perf_counter()
    if algorithm == "nsga2":
        optimizer = NSGA2Optimizer(
            population_size=population_size,
            max_evaluations=max_evaluations,
            seed=seed,
        )
        population = optimizer.optimize(
            world,
            config,
            initial_population=initial_population or None,
            time_offset=scenario.time_offset,
        )
        actual_world = world
    elif algorithm == "mopso":
        optimizer = MOPSOOptimizer(
            population_size=population_size,
            max_evaluations=max_evaluations,
            seed=seed,
        )
        actual_world = world_at_event(world, scenario.time_offset)
        population = optimizer.optimize(actual_world, config)
    else:
        optimizer = HybridNSGA2PSOOptimizer(
            population_size=population_size,
            max_evaluations=max_evaluations,
            elite_fraction=0.15,
            min_elites=5,
            pso_iterations=5,
            pso_inertia=0.5,
            pso_cognitive=1.5,
            pso_social=1.5,
            velocity_clamp=0.2,
            neighborhood_size=5,
            writeback_probability=1.0,
            seed=seed,
        )
        actual_world = world_at_event(world, scenario.time_offset)
        population = optimizer.optimize(actual_world, config)
    optimization_seconds = time.perf_counter() - optimization_started

    candidates = [Individual(item.trajectory, item.evaluation) for item in population]
    fronts = fast_non_dominated_sort(candidates) if candidates else []
    for front_index, front in enumerate(fronts):
        for individual in front:
            individual.rank = front_index
    candidates.sort(key=lambda item: (not item.evaluation.is_feasible, item.rank, item.evaluation.total_violation))

    candidate_payloads = []
    for index, individual in enumerate(candidates):
        trajectory = individual.trajectory
        sampled = parametric_to_physical(sample_trajectory(trajectory.to_bspline(), 160), config.t_max)
        evaluation = individual.evaluation
        candidate_payloads.append({
            "index": index,
            "rank": int(individual.rank),
            "genes": trajectory.genes.tolist(),
            "control_points": trajectory.control_points.tolist(),
            "positions": sampled["positions"].tolist(),
            "trajectory_times": sampled["t"].tolist(),
            "objectives": {
                "path_length": float(evaluation.objectives[0]),
                "clearance_risk": float(evaluation.objectives[1]),
                "minimum_snap": float(evaluation.objectives[2]),
            },
            "is_feasible": bool(evaluation.is_feasible),
            "total_violation": float(evaluation.total_violation),
            "violation_details": evaluation.violation_details,
            "minimum_clearance": float(evaluation.min_clearance),
            "max_speed": float(evaluation.max_speed),
            "max_acceleration": float(evaluation.max_accel),
        })

    feasible = [candidate for candidate in candidate_payloads if candidate["is_feasible"]]
    best_index = feasible[0]["index"] if feasible else (candidate_payloads[0]["index"] if candidate_payloads else None)
    history = getattr(optimizer, "history", [])
    evaluations = getattr(optimizer, "evaluation_history", [])
    reevaluated_imported_count = min(len(initial_population), len(evaluations))
    return json_safe({
        "status": "complete",
        "algorithm": algorithm,
        "scenario_id": scenario.scenario_id,
        "event_index": scenario.event_index,
        "event_time": scenario.time_offset,
        "start": world.start.tolist(),
        "goal": world.goal.tolist(),
        "seed": seed,
        "objective_evaluations": int(optimizer.evals),
        "imported_trajectory_count": len(initial_population),
        "accepted_imported_count": len(initial_population),
        "reevaluated_imported_count": reevaluated_imported_count,
        "accepted_imported_feasible_count": sum(
            1 for item in evaluations[:reevaluated_imported_count] if item["is_feasible"]
        ),
        "optimization_seconds": float(optimization_seconds),
        "wall_clock_seconds": None,
        "generation_count": int(getattr(optimizer, "generation", getattr(optimizer, "iteration", 0))),
        "feasible_count": len(feasible),
        "feasibility_rate": float(len(feasible) / len(candidate_payloads)) if candidate_payloads else 0.0,
        "best_candidate_index": best_index,
        "history": history,
        "evaluation_history": evaluations,
        "candidates": candidate_payloads,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    result = optimize(request)
    result["wall_clock_seconds"] = time.perf_counter() - started
    args.output.write_text(json.dumps(json_safe(result), allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()

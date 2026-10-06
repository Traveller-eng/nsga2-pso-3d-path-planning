from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from planner.metrics.hypervolume import calculate_hypervolume
from planner.metrics.igd import calculate_igd
from planner.metrics.pareto import pareto_filter, pooled_reference_front

from planner.environment.world import make_3d_static_world
from planner.optimization.hybrid import HybridNSGA2PSOOptimizer
from planner.optimization.mopso import MOPSOOptimizer
from planner.optimization.nsga2 import NSGA2Optimizer
from planner.representation.trajectory import TrajectoryConfig

DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "baseline_experiment.json"
RESULTS_DIR = ROOT / "experiments" / "results"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_world(cfg: dict):
    world_cfg = cfg["world"]
    return make_3d_static_world(
        bounds_max=tuple(world_cfg["bounds_max"]),
        start=tuple(world_cfg["start"]),
        goal=tuple(world_cfg["goal"]),
        obstacles=[],
        v_max=world_cfg["v_max"],
        a_max=world_cfg["a_max"],
        d_safe=world_cfg["d_safe"],
        d_comfort=world_cfg["d_comfort"],
        t_max_factor=world_cfg["t_max_factor"],
    )


def build_trajectory_config(world, cfg: dict) -> TrajectoryConfig:
    traj_cfg = cfg["trajectory"]
    return TrajectoryConfig(
        n_control_points=traj_cfg["n_control_points"],
        dim=traj_cfg["dim"],
        start=world.start,
        goal=world.goal,
        t_max=world.t_max,
        degree=traj_cfg["degree"],
        optimize_time=traj_cfg["optimize_time"],
    )


def feasible_objectives(population) -> np.ndarray:
    rows = []
    for individual in population:
        if individual.evaluation.is_feasible:
            rows.append(np.asarray(individual.evaluation.objectives, dtype=float))
    if not rows:
        return np.empty((0, 3), dtype=float)
    return np.vstack(rows)


def get_reference_point(config: dict, n_objectives: int = 3) -> np.ndarray:
    """Load the experiment's frozen HV reference point from its canonical config."""
    metrics_config = config.get("metrics", {})
    if "hypervolume_reference_point" not in metrics_config:
        raise ValueError("Experiment config must define metrics.hypervolume_reference_point.")
    reference_point = np.asarray(metrics_config["hypervolume_reference_point"], dtype=float)
    if reference_point.shape != (n_objectives,) or not np.isfinite(reference_point).all():
        raise ValueError(f"HV reference point must contain {n_objectives} finite values.")
    return reference_point


def compute_run_metrics(front: np.ndarray, reference_front: np.ndarray, reference_point: np.ndarray) -> dict:
    front = np.asarray(front, dtype=float)
    if front.size == 0:
        return {
            "hypervolume": 0.0,
            "igd": float("nan"),
            "nondominated_size": 0,
            "best_objective": np.array([np.nan, np.nan, np.nan], dtype=float).tolist(),
        }
    if front.ndim == 1:
        front = front.reshape(1, -1)

    nondominated = pareto_filter(front)
    if nondominated.size == 0:
        return {
            "hypervolume": 0.0,
            "igd": float("nan"),
            "nondominated_size": 0,
            "best_objective": np.array([np.nan, np.nan, np.nan], dtype=float).tolist(),
        }

    ref = np.asarray(reference_front, dtype=float)
    if ref.size == 0:
        ref = nondominated

    hv = calculate_hypervolume(nondominated, np.asarray(reference_point, dtype=float))
    igd = calculate_igd(nondominated, ref)

    return {
        "hypervolume": float(hv),
        "igd": float(igd),
        "nondominated_size": int(nondominated.shape[0]),
        "best_objective": np.min(nondominated, axis=0).tolist(),
    }


def _serialize_history(history: list[dict]) -> list[dict]:
    serialized: list[dict] = []
    for row in history:
        serialized.append(_to_serializable(row))
    return serialized


def _to_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def run_algorithm(name: str, seed: int, cfg: dict, world, traj_cfg):
    start = time.perf_counter()
    if name == "NSGA2":
        optimizer = NSGA2Optimizer(
            population_size=cfg["population_size"],
            max_evaluations=cfg["max_evaluations"],
            seed=seed,
        )
    elif name == "MOPSO":
        optimizer = MOPSOOptimizer(
            population_size=cfg["population_size"],
            max_evaluations=cfg["max_evaluations"],
            seed=seed,
        )
    elif name == "Hybrid":
        optimizer = HybridNSGA2PSOOptimizer(
            population_size=cfg["population_size"],
            max_evaluations=cfg["max_evaluations"],
            elite_fraction=cfg["hybrid"]["elite_fraction"],
            min_elites=cfg["hybrid"]["min_elites"],
            pso_iterations=cfg["hybrid"]["pso_iterations"],
            pso_inertia=cfg["hybrid"]["pso_inertia"],
            pso_cognitive=cfg["hybrid"]["pso_cognitive"],
            pso_social=cfg["hybrid"]["pso_social"],
            velocity_clamp=cfg["hybrid"]["velocity_clamp"],
            neighborhood_size=cfg["hybrid"]["neighborhood_size"],
            writeback_probability=cfg["hybrid"]["writeback_probability"],
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown algorithm: {name}")

    population = optimizer.optimize(world, traj_cfg)
    runtime = time.perf_counter() - start
    return {
        "algorithm": name,
        "seed": seed,
        "population": population,
        "optimizer": optimizer,
        "runtime": runtime,
        "history": optimizer.history,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def aggregate_runs(runs: list[dict], algorithms: list[str]) -> list[dict]:
    metrics = ("hypervolume", "igd", "runtime", "feasible_count")
    aggregates = []
    for algorithm in algorithms:
        subset = [run for run in runs if run["algorithm"] == algorithm]
        aggregates.append({
            "algorithm": algorithm,
            "run_count": len(subset),
            **{
                metric: summarize([run[metric] for run in subset])
                for metric in metrics
            },
        })
    return aggregates


def compare_hybrid_means(aggregates: list[dict]) -> list[dict]:
    by_algorithm = {row["algorithm"]: row for row in aggregates}
    if "Hybrid" not in by_algorithm:
        return []

    comparisons = []
    for baseline in ("NSGA2", "MOPSO"):
        if baseline not in by_algorithm:
            continue
        for metric in ("hypervolume", "igd"):
            baseline_mean = by_algorithm[baseline][metric]["mean"]
            hybrid_mean = by_algorithm["Hybrid"][metric]["mean"]
            difference = hybrid_mean - baseline_mean
            relative = (
                100.0 * difference / abs(baseline_mean)
                if baseline_mean != 0.0 else float("nan")
            )
            comparisons.append({
                "algorithm": "Hybrid",
                "baseline_algorithm": baseline,
                "metric": metric,
                "mean_difference": difference,
                "relative_difference_percent": relative,
            })
    return comparisons


def build_csv_rows(runs: list[dict], aggregates: list[dict], comparisons: list[dict]) -> tuple[list[dict], list[str]]:
    metrics = ("hypervolume", "igd", "runtime", "feasible_count")
    statistic_names = ("mean", "median", "std", "min", "max")
    fields = [
        "record_type", "algorithm", "baseline_algorithm", "seed", "evaluations",
        "feasible_count", "nondominated_size", "hypervolume", "igd", "runtime",
        "metric", "mean_difference", "relative_difference_percent",
    ]
    fields.extend(f"{metric}_{stat}" for metric in metrics for stat in statistic_names)

    rows = []
    for run in runs:
        rows.append({
            "record_type": "per_run",
            **{key: run.get(key) for key in (
                "algorithm", "seed", "evaluations", "feasible_count",
                "nondominated_size", "hypervolume", "igd", "runtime",
            )},
        })
    for aggregate in aggregates:
        row = {"record_type": "cross_seed_aggregate", "algorithm": aggregate["algorithm"]}
        for metric in metrics:
            for stat in statistic_names:
                row[f"{metric}_{stat}"] = aggregate[metric][stat]
        rows.append(row)
    for comparison in comparisons:
        rows.append({"record_type": "mean_difference", **comparison})
    return rows, fields


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small benchmark across NSGA-II, MOPSO, and Hybrid.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--algorithms", nargs="+", default=["NSGA2", "MOPSO", "Hybrid"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    config = load_config(Path(args.config))
    world = build_world(config)
    traj_cfg = build_trajectory_config(world, config)
    results_dir = ROOT / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_outputs = []
    feasible_by_run = []
    for algorithm in args.algorithms:
        for seed in args.seeds:
            run = run_algorithm(algorithm, seed, config, world, traj_cfg)
            front = feasible_objectives(run["population"])
            feasible_by_run.append(front)
            all_outputs.append({
                "algorithm": algorithm,
                "seed": seed,
                "runtime": run["runtime"],
                "evaluations": run["optimizer"].evals,
                "feasible_front_size": int(front.shape[0]),
                "feasible_count": int(front.shape[0]),
                "front": front.tolist(),
                "history": _serialize_history(run["history"]),
            })

    pooled_reference = pooled_reference_front(feasible_by_run, feasible_only=True)
    reference_point = get_reference_point(config)

    for row in all_outputs:
        front = np.asarray(row["front"], dtype=float)
        metrics = compute_run_metrics(front, pooled_reference, reference_point)
        row.update({
            "reference_point": reference_point.tolist(),
            "igd_reference_front": pooled_reference.tolist(),
            "pooled_reference_size": int(pooled_reference.shape[0]) if pooled_reference.size else 0,
            "hypervolume": metrics["hypervolume"],
            "igd": metrics["igd"],
            "nondominated_size": metrics["nondominated_size"],
            "best_objective": metrics["best_objective"],
        })

    aggregates = aggregate_runs(all_outputs, args.algorithms)
    comparisons = compare_hybrid_means(aggregates)
    csv_rows, csv_fields = build_csv_rows(all_outputs, aggregates, comparisons)
    write_csv(results_dir / "benchmark_summary.csv", csv_rows, csv_fields)

    benchmark_artifact = {
        "metadata": {
            "configuration_file": str(Path(args.config)),
            "configuration": config,
            "algorithms": args.algorithms,
            "seeds": args.seeds,
            "expected_run_count": len(args.algorithms) * len(args.seeds),
            "objective_space": config.get("metrics", {}).get("objective_space", "raw"),
            "hypervolume_reference_point": reference_point.tolist(),
            "igd_reference_front": pooled_reference.tolist(),
            "igd_reference_front_size": int(pooled_reference.shape[0]) if pooled_reference.size else 0,
            "igd_reference_front_method": "Pool feasible final objective vectors across all selected algorithms and seeds, remove duplicates, then Pareto-filter for minimization.",
            "igd_normalization": "none; raw objective space",
            "checkpoint_hv_igd": "not available in this benchmark version",
            "evaluations_to_target_hv": "not available in this benchmark version",
        },
        "runs": all_outputs,
        "aggregates": aggregates,
        "hybrid_mean_comparisons": comparisons,
    }
    with (results_dir / "benchmark_runs.json").open("w", encoding="utf-8") as handle:
        json.dump(_to_serializable(benchmark_artifact), handle, indent=2)

    print(json.dumps({
        "algorithms": args.algorithms,
        "seeds": args.seeds,
        "results_dir": str(results_dir),
        "runs": len(all_outputs),
    }, indent=2))


if __name__ == "__main__":
    main()
""

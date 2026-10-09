from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from planner.environment.obstacles import SphereObstacle
from planner.environment.world import make_3d_static_world
from planner.metrics.hypervolume import calculate_hypervolume
from planner.metrics.igd import calculate_igd
from planner.metrics.pareto import pareto_filter
from planner.optimization.dynamic import DynamicScenario, DynamicWarmStartController

METRICS = (
    "wall_clock_seconds",
    "feasibility_rate",
    "best_feasible_path_length",
    "hypervolume",
    "igd",
    "minimum_clearance",
    "minimum_clearance_risk_objective",
    "minimum_snap_objective",
    "minimum_total_violation",
    "mean_total_violation",
)
def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def build_scenario(config: dict) -> DynamicScenario:
    scenario_config = config["scenario"]
    obstacle_config = scenario_config["obstacle"]
    waypoints = [np.asarray(point, dtype=float) for point in obstacle_config["waypoints"]]
    obstacle = SphereObstacle(
        center=waypoints[0].copy(),
        radius=float(obstacle_config["radius"]),
        path=waypoints,
        path_times=[float(value) for value in obstacle_config["path_times"]],
    )
    world = make_3d_static_world(
        bounds_max=tuple(scenario_config["bounds_max"]),
        start=tuple(scenario_config["start"]),
        goal=tuple(scenario_config["goal"]),
        obstacles=[obstacle],
        v_max=float(scenario_config["v_max"]),
        a_max=float(scenario_config["a_max"]),
        d_safe=float(scenario_config["d_safe"]),
        d_comfort=float(scenario_config["d_comfort"]),
        t_max_factor=float(scenario_config["t_max_factor"]),
    )
    return DynamicScenario(
        world=world,
        event_times=[float(value) for value in scenario_config["event_times"]],
        snapshot_ids=list(scenario_config["event_snapshot_ids"]),
    )


def run_conditions(config: dict, scenario: DynamicScenario) -> tuple[list[dict], list[dict]]:
    optimizer_config = config["optimizer"]
    event_count = len(scenario.event_times)
    per_event_budget = int(optimizer_config["objective_evaluations_per_event"])
    total_budget = int(optimizer_config["objective_evaluations_per_condition"])
    if total_budget != per_event_budget * event_count:
        raise ValueError("Condition budget must equal per-event budget times the number of events.")
    if per_event_budget < int(optimizer_config["population_size"]):
        raise ValueError("Per-event budget must be large enough to reevaluate a full imported population.")

    controller = DynamicWarmStartController(
        population_size=int(optimizer_config["population_size"]),
        per_event_budget=per_event_budget,
        total_budget=total_budget,
        seed=int(optimizer_config["optimizer_seed_base"]),
    )
    condition_runs: list[dict] = []
    event_rows: list[dict] = []
    seeds = [int(seed) for seed in config["paired_seeds"]]

    for seed_index, paired_seed in enumerate(seeds):
        order = ["cold", "warm"] if seed_index % 2 == 0 else ["warm", "cold"]
        for condition in order:
            run_start = perf_counter()
            try:
                result = controller.run_condition(condition, scenario, condition_seed=paired_seed)
                run_event_rows = []
                run_record = {
                    "condition": condition,
                    "paired_seed": paired_seed,
                    "status": "success",
                    "total_evaluations": int(result["total_evaluations"]),
                    "event_count": int(result["event_count"]),
                    "condition_wall_clock_seconds": float(perf_counter() - run_start),
                    "error": None,
                }
                for event in result["events"]:
                    row = dict(event)
                    row.update({
                        "condition": condition,
                        "paired_seed": paired_seed,
                        "optimizer_seed": (
                            int(optimizer_config["optimizer_seed_base"]) + paired_seed + int(event["event_index"])
                        ),
                        "status": "success",
                        "error": None,
                    })
                    if row["objective_evaluations"] != per_event_budget:
                        raise RuntimeError(
                            f"{condition} seed {paired_seed} event {row['event_index']} used "
                            f"{row['objective_evaluations']} evaluations, expected {per_event_budget}."
                        )
                    run_event_rows.append(row)
                event_rows.extend(run_event_rows)
                condition_runs.append(run_record)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                condition_runs.append({
                    "condition": condition,
                    "paired_seed": paired_seed,
                    "status": "failed",
                    "total_evaluations": None,
                    "event_count": 0,
                    "condition_wall_clock_seconds": float(perf_counter() - run_start),
                    "error": error,
                })
                for event_index, event_time in enumerate(scenario.event_times):
                    event_rows.append({
                        "condition": condition,
                        "paired_seed": paired_seed,
                        "optimizer_seed": int(optimizer_config["optimizer_seed_base"]) + paired_seed + event_index,
                        "event_index": event_index,
                        "event_time": float(event_time),
                        "snapshot_id": scenario.snapshot_id(event_index),
                        "status": "failed",
                        "error": error,
                        "objective_evaluations": None,
                        "cumulative_evaluations": None,
                        "wall_clock_seconds": None,
                        "population_size": None,
                        "feasible_count": None,
                        "feasibility_rate": None,
                        "no_feasible_solution": None,
                        "imported_trajectory_count": None,
                        "accepted_imported_count": None,
                        "reevaluated_imported_count": None,
                        "accepted_imported_feasible_count": None,
                        "rejected_imported_count": None,
                        "best_feasible_path_length": None,
                        "minimum_clearance": None,
                        "minimum_clearance_risk_objective": None,
                        "minimum_snap_objective": None,
                        "minimum_total_violation": None,
                        "mean_total_violation": None,
                        "minimum_violation_details": None,
                        "population_objectives": None,
                        "population_total_violations": None,
                        "feasible_objectives": [],
                    })
    return condition_runs, event_rows


def add_quality_metrics(config: dict, event_rows: list[dict]) -> dict:
    metric_config = config["metrics"]
    scales = np.asarray(metric_config["objective_normalization_scales"], dtype=float)
    hv_reference = np.asarray(metric_config["hypervolume_reference_point_normalized"], dtype=float)
    reference_fronts: dict[str, dict] = {}

    for event_index in sorted({int(row["event_index"]) for row in event_rows}):
        successful = [
            row for row in event_rows
            if int(row["event_index"]) == event_index and row["status"] == "success"
        ]
        fronts = [np.asarray(row["feasible_objectives"], dtype=float).reshape(-1, 3) for row in successful]
        nonempty = [front for front in fronts if front.size]
        pooled = np.vstack(nonempty) if nonempty else np.empty((0, 3), dtype=float)
        common_front = pareto_filter(pooled) if pooled.size else np.empty((0, 3), dtype=float)
        normalized_front = common_front / scales if common_front.size else np.empty((0, 3), dtype=float)
        reference_fronts[str(event_index)] = {
            "raw_objectives": common_front.tolist(),
            "normalized_objectives": normalized_front.tolist(),
            "size": int(common_front.shape[0]),
            "source": "Pooled feasible final-population objective vectors from both conditions and all successful paired seeds for this event, then Pareto-filtered.",
        }

        for row in event_rows:
            if int(row["event_index"]) != event_index:
                continue
            if row["status"] != "success":
                row["hypervolume"] = None
                row["igd"] = None
                row["common_reference_front_size"] = int(common_front.shape[0])
                continue
            raw_front = np.asarray(row["feasible_objectives"], dtype=float).reshape(-1, 3)
            normalized_approximation = raw_front / scales if raw_front.size else np.empty((0, 3), dtype=float)
            row["hypervolume"] = float(calculate_hypervolume(normalized_approximation, hv_reference))
            row["igd"] = (
                float(calculate_igd(normalized_approximation, normalized_front))
                if raw_front.size and normalized_front.size
                else None
            )
            row["common_reference_front_size"] = int(common_front.shape[0])
    return reference_fronts


def describe(values: list[float], rng: np.random.Generator) -> dict:
    array = np.asarray(values, dtype=float)
    if not array.size:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std_sample": None,
            "q1": None,
            "q3": None,
            "min": None,
            "max": None,
            "mean_ci95": None,
            "median_bootstrap_ci95": None,
        }
    summary = {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std_sample": float(np.std(array, ddof=1)) if array.size > 1 else None,
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean_ci95": None,
        "median_bootstrap_ci95": None,
    }
    if array.size > 1:
        critical = float(stats.t.ppf(0.975, df=array.size - 1))
        margin = critical * float(np.std(array, ddof=1)) / np.sqrt(array.size)
        summary["mean_ci95"] = [summary["mean"] - margin, summary["mean"] + margin]
    if array.size >= 2:
        sample_indices = rng.integers(0, array.size, size=(10000, array.size))
        bootstrap_medians = np.median(array[sample_indices], axis=1)
        summary["median_bootstrap_ci95"] = [
            float(np.quantile(bootstrap_medians, 0.025)),
            float(np.quantile(bootstrap_medians, 0.975)),
        ]
    return summary


def holm_adjust(p_values: list[float | None]) -> list[float | None]:
    adjusted: list[float | None] = [None] * len(p_values)
    valid = [(index, float(value)) for index, value in enumerate(p_values) if value is not None and np.isfinite(value)]
    valid.sort(key=lambda item: item[1])
    running_max = 0.0
    count = len(valid)
    for rank, (index, p_value) in enumerate(valid):
        running_max = max(running_max, (count - rank) * p_value)
        adjusted[index] = float(min(1.0, running_max))
    return adjusted


def paired_analysis(config: dict, event_rows: list[dict]) -> dict:
    analysis_rng = np.random.default_rng(int(config["metrics"].get("analysis_bootstrap_seed", 80421)))
    primary_metrics = set(config["metrics"]["primary_metrics"])
    event_indices = sorted({int(row["event_index"]) for row in event_rows})
    seeds = [int(seed) for seed in config["paired_seeds"]]
    paired_rows: list[dict] = []

    for event_index in event_indices:
        by_condition_seed = {
            (row["condition"], int(row["paired_seed"])): row
            for row in event_rows
            if int(row["event_index"]) == event_index
        }
        for metric in METRICS:
            differences = []
            warm_values = []
            cold_values = []
            missing_warm = 0
            missing_cold = 0
            failed_pairs = 0
            seed_differences = []
            for seed in seeds:
                cold = by_condition_seed.get(("cold", seed))
                warm = by_condition_seed.get(("warm", seed))
                if cold is None or cold["status"] != "success" or warm is None or warm["status"] != "success":
                    failed_pairs += 1
                    seed_differences.append({"paired_seed": seed, "warm_minus_cold": None, "status": "failed_pair"})
                    continue
                cold_value = cold.get(metric)
                warm_value = warm.get(metric)
                if cold_value is None:
                    missing_cold += 1
                if warm_value is None:
                    missing_warm += 1
                if cold_value is None or warm_value is None:
                    seed_differences.append({"paired_seed": seed, "warm_minus_cold": None, "status": "undefined_metric"})
                    continue
                difference = float(warm_value) - float(cold_value)
                cold_values.append(float(cold_value))
                warm_values.append(float(warm_value))
                differences.append(difference)
                seed_differences.append({"paired_seed": seed, "warm_minus_cold": difference, "status": "paired"})

            summary = describe(differences, analysis_rng)
            if len(differences) >= 2:
                if np.allclose(differences, 0.0, rtol=0.0, atol=1e-15):
                    t_p = 1.0
                    wilcoxon_p = 1.0
                else:
                    t_p = (
                        0.0
                        if np.std(differences, ddof=1) == 0.0
                        else float(stats.ttest_rel(warm_values, cold_values).pvalue)
                    )
                    wilcoxon_p = float(stats.wilcoxon(warm_values, cold_values, zero_method="pratt", alternative="two-sided").pvalue)
            else:
                t_p = None
                wilcoxon_p = None
            paired_rows.append({
                "event_index": event_index,
                "metric": metric,
                "direction": "warm_start_minus_cold_start",
                **summary,
                "missing_warm_metric_pairs": missing_warm,
                "missing_cold_metric_pairs": missing_cold,
                "failed_pairs": failed_pairs,
                "paired_t_test_p": t_p,
                "paired_t_test_p_holm": None,
                "wilcoxon_p": wilcoxon_p,
                "wilcoxon_p_holm": None,
                "seed_differences": seed_differences,
            })

    t_adjusted = holm_adjust([row["paired_t_test_p"] for row in paired_rows])
    w_adjusted = holm_adjust([row["wilcoxon_p"] for row in paired_rows])
    for row, t_p, w_p in zip(paired_rows, t_adjusted, w_adjusted):
        row["paired_t_test_p_holm"] = t_p
        row["wilcoxon_p_holm"] = w_p
        row["included_in_primary_test_family"] = row["metric"] in primary_metrics

    return {
        "difference_definition": "warm-start minus cold-start; negative is favorable for lower-is-better outcomes and positive is favorable for higher-is-better outcomes.",
        "confidence_intervals": {
            "mean": "Two-sided 95% Student-t interval on paired differences.",
            "median": "Two-sided 95% percentile bootstrap interval using 10,000 paired-seed resamples and the frozen analysis bootstrap seed.",
        },
        "multiple_comparison_correction": config["metrics"]["multiple_comparison_correction"],
        "primary_metrics": sorted(primary_metrics),
        "paired_differences": paired_rows,
    }


def condition_summaries(config: dict, event_rows: list[dict], rng: np.random.Generator) -> list[dict]:
    summaries = []
    for event_index in sorted({int(row["event_index"]) for row in event_rows}):
        for condition in ("cold", "warm"):
            subset = [
                row for row in event_rows
                if int(row["event_index"]) == event_index and row["condition"] == condition
            ]
            for metric in METRICS:
                values = [
                    float(row[metric]) for row in subset
                    if row["status"] == "success" and row.get(metric) is not None
                ]
                summaries.append({
                    "event_index": event_index,
                    "event_time": float(config["scenario"]["event_times"][event_index]),
                    "condition": condition,
                    "metric": metric,
                    "successful_run_count": sum(1 for row in subset if row["status"] == "success"),
                    "failed_run_count": sum(1 for row in subset if row["status"] != "success"),
                    **describe(values, rng),
                })
    return summaries


def write_json(path: Path, content: dict) -> None:
    path.write_text(json.dumps(json_safe(content), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_summary_csv(path: Path, condition_rows: list[dict], paired_rows: list[dict]) -> None:
    fields = [
        "record_type", "event_index", "event_time", "condition", "metric", "n",
        "successful_run_count", "failed_run_count", "mean", "median", "std_sample",
        "q1", "q3", "min", "max", "mean_ci95_low", "mean_ci95_high",
        "median_bootstrap_ci95_low", "median_bootstrap_ci95_high", "failed_pairs",
        "missing_warm_metric_pairs", "missing_cold_metric_pairs", "paired_t_test_p",
        "paired_t_test_p_holm", "wilcoxon_p", "wilcoxon_p_holm",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in condition_rows:
            ci = row.get("mean_ci95") or [None, None]
            median_ci = row.get("median_bootstrap_ci95") or [None, None]
            writer.writerow({
                "record_type": "condition_summary",
                **{key: row.get(key) for key in ("event_index", "event_time", "condition", "metric", "n", "successful_run_count", "failed_run_count", "mean", "median", "std_sample", "q1", "q3", "min", "max")},
                "mean_ci95_low": ci[0],
                "mean_ci95_high": ci[1],
                "median_bootstrap_ci95_low": median_ci[0],
                "median_bootstrap_ci95_high": median_ci[1],
            })
        for row in paired_rows:
            ci = row.get("mean_ci95") or [None, None]
            median_ci = row.get("median_bootstrap_ci95") or [None, None]
            writer.writerow({
                "record_type": "paired_difference",
                "event_index": row["event_index"],
                "metric": row["metric"],
                "n": row["n"],
                "mean": row["mean"],
                "median": row["median"],
                "std_sample": row["std_sample"],
                "q1": row["q1"],
                "q3": row["q3"],
                "min": row["min"],
                "max": row["max"],
                "mean_ci95_low": ci[0],
                "mean_ci95_high": ci[1],
                "median_bootstrap_ci95_low": median_ci[0],
                "median_bootstrap_ci95_high": median_ci[1],
                "failed_pairs": row["failed_pairs"],
                "missing_warm_metric_pairs": row["missing_warm_metric_pairs"],
                "missing_cold_metric_pairs": row["missing_cold_metric_pairs"],
                "paired_t_test_p": row["paired_t_test_p"],
                "paired_t_test_p_holm": row["paired_t_test_p_holm"],
                "wilcoxon_p": row["wilcoxon_p"],
                "wilcoxon_p_holm": row["wilcoxon_p_holm"],
            })


def run_experiment(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if len(config["paired_seeds"]) < 10:
        raise ValueError("The controlled dynamic experiment requires at least 10 paired seeds.")
    if len(set(config["paired_seeds"])) != len(config["paired_seeds"]):
        raise ValueError("Paired seeds must be unique.")

    scenario = build_scenario(config)
    condition_runs, event_rows = run_conditions(config, scenario)
    reference_fronts = add_quality_metrics(config, event_rows)

    initial_equivalence = []
    for seed in config["paired_seeds"]:
        cold = next(row for row in event_rows if row["event_index"] == 0 and row["paired_seed"] == seed and row["condition"] == "cold")
        warm = next(row for row in event_rows if row["event_index"] == 0 and row["paired_seed"] == seed and row["condition"] == "warm")
        equivalent = cold["status"] == warm["status"] == "success"
        if equivalent:
            equivalent = np.allclose(
                np.asarray(cold["population_objectives"], dtype=float),
                np.asarray(warm["population_objectives"], dtype=float),
                rtol=0.0,
                atol=1e-12,
            ) and np.allclose(
                np.asarray(cold["population_total_violations"], dtype=float),
                np.asarray(warm["population_total_violations"], dtype=float),
                rtol=0.0,
                atol=1e-12,
            )
        initial_equivalence.append({"paired_seed": int(seed), "equivalent_initial_event_result": bool(equivalent)})
        if cold["status"] == warm["status"] == "success" and not equivalent:
            raise RuntimeError(f"Cold and warm event-zero results differ for paired seed {seed} despite identical initialization seeds.")

    bootstrap_seed = int(config["metrics"].get("analysis_bootstrap_seed", 80421))
    condition_summary = condition_summaries(config, event_rows, np.random.default_rng(bootstrap_seed))
    analysis = paired_analysis(config, event_rows)
    failures = [run for run in condition_runs if run["status"] != "success"]
    budget_parity = all(
        row["status"] != "success"
        or row["objective_evaluations"] == int(config["optimizer"]["objective_evaluations_per_event"])
        for row in event_rows
    )
    raw_artifact = {
        "metadata": {
            "experiment_name": config["experiment_name"],
            "configuration_file": str(config_path),
            "configuration": config,
            "reproduction_command": config["reproduction_command"],
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "paired_seed_count": len(config["paired_seeds"]),
            "expected_condition_runs": 2 * len(config["paired_seeds"]),
            "expected_event_records": 2 * len(config["paired_seeds"]) * len(scenario.event_times),
            "budget_parity_verified": bool(budget_parity),
            "event_zero_initialization_equivalence": initial_equivalence,
            "condition_run_order": config["run_order"],
            "common_reference_fronts_by_event": reference_fronts,
            "objective_normalization_scales": config["metrics"]["objective_normalization_scales"],
            "hypervolume_reference_point_normalized": config["metrics"]["hypervolume_reference_point_normalized"],
            "hypervolume_reference_point_raw": config["metrics"]["hypervolume_reference_point_raw"],
            "failure_count": len(failures),
            "undefined_metric_policy": config["metrics"]["no_feasible_solution"],
        },
        "condition_runs": condition_runs,
        "event_results": event_rows,
    }
    summary_artifact = {
        "experiment_name": config["experiment_name"],
        "condition_summary_by_event": condition_summary,
        "failure_count": len(failures),
        "condition_run_count": len(condition_runs),
        "successful_condition_run_count": len(condition_runs) - len(failures),
        "budget_parity_verified": bool(budget_parity),
        "event_zero_initialization_equivalence": initial_equivalence,
        "common_reference_front_sizes_by_event": {
            event: front["size"] for event, front in reference_fronts.items()
        },
    }
    analysis_artifact = {
        "experiment_name": config["experiment_name"],
        "seeds": config["paired_seeds"],
        "paired_tests": analysis,
        "interpretation_note": "These results are descriptive for one deterministic dynamic scenario. Statistical tests do not establish general algorithm superiority; failed runs remain in raw outputs and incomplete or undefined metric pairs are excluded from that metric's paired test with counts reported.",
    }

    output_dir = ROOT / "experiments" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = config["output_basename"]
    write_json(output_dir / f"{basename}_runs.json", raw_artifact)
    write_json(output_dir / f"{basename}_summary.json", summary_artifact)
    write_json(output_dir / f"{basename}_analysis.json", analysis_artifact)
    write_summary_csv(output_dir / f"{basename}_summary.csv", condition_summary, analysis["paired_differences"])
    return {
        "configuration": str(config_path),
        "condition_runs": len(condition_runs),
        "event_records": len(event_rows),
        "successful_condition_runs": len(condition_runs) - len(failures),
        "failed_condition_runs": len(failures),
        "budget_parity_verified": bool(budget_parity),
        "results": [
            str(output_dir / f"{basename}_runs.json"),
            str(output_dir / f"{basename}_summary.json"),
            str(output_dir / f"{basename}_analysis.json"),
            str(output_dir / f"{basename}_summary.csv"),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired cold-start and warm-start dynamic NSGA-II replanning.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "configs" / "dynamic_warm_start.json",
    )
    args = parser.parse_args()
    print(json.dumps(json_safe(run_experiment(args.config)), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

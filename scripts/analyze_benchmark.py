from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "experiments" / "results" / "benchmark_runs.json"
DEFAULT_JSON = ROOT / "experiments" / "results" / "statistical_analysis.json"
DEFAULT_CSV = ROOT / "experiments" / "results" / "statistical_analysis.csv"
BOOTSTRAP_SEED = 20261007
BOOTSTRAP_SAMPLES = 10_000


def bootstrap_mean_ci(
    differences: np.ndarray,
    rng: np.random.Generator,
    samples: int = BOOTSTRAP_SAMPLES,
) -> tuple[float, float] | None:
    if differences.size < 2:
        return None
    indexes = rng.integers(0, differences.size, size=(samples, differences.size))
    means = differences[indexes].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def matched_pairs_rank_biserial(differences: np.ndarray) -> float | None:
    nonzero = differences[differences != 0.0]
    if nonzero.size == 0:
        return None
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(np.sum(ranks[nonzero > 0.0]))
    negative = float(np.sum(ranks[nonzero < 0.0]))
    denominator = positive + negative
    if denominator == 0.0:
        return None
    return (positive - negative) / denominator


def holm_adjust(p_values: list[float | None]) -> list[float | None]:
    adjusted: list[float | None] = [None] * len(p_values)
    valid = [(index, float(value)) for index, value in enumerate(p_values) if value is not None]
    valid.sort(key=lambda item: item[1])
    previous = 0.0
    count = len(valid)
    for rank, (index, p_value) in enumerate(valid):
        previous = max(previous, min(1.0, (count - rank) * p_value))
        adjusted[index] = previous
    return adjusted


def paired_comparison(
    hybrid_values: dict[int, float | None],
    baseline_values: dict[int, float | None],
    rng: np.random.Generator,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict:
    seeds = sorted(set(hybrid_values) & set(baseline_values))
    paired = [
        (seed, float(hybrid_values[seed]), float(baseline_values[seed]))
        for seed in seeds
        if hybrid_values[seed] is not None
        and baseline_values[seed] is not None
        and np.isfinite(hybrid_values[seed])
        and np.isfinite(baseline_values[seed])
    ]
    differences = np.asarray([hybrid - baseline for _, hybrid, baseline in paired], dtype=float)
    result = {
        "n_pairs": int(differences.size),
        "seeds": [seed for seed, _, _ in paired],
        "mean_paired_difference": float(np.mean(differences)) if differences.size else None,
        "median_paired_difference": float(np.median(differences)) if differences.size else None,
        "wilcoxon_p_value": None,
        "wilcoxon_skipped_reason": None,
        "matched_pairs_rank_biserial": matched_pairs_rank_biserial(differences),
        "mean_difference_ci_95": None,
    }

    if differences.size < 2:
        result["wilcoxon_skipped_reason"] = "Fewer than two matched finite seed pairs."
    elif np.count_nonzero(differences) < 2:
        result["wilcoxon_skipped_reason"] = "Fewer than two non-zero paired differences; signed-rank inference is unsuitable."
    else:
        test = wilcoxon(
            differences,
            zero_method="wilcox",
            alternative="two-sided",
            method="auto",
        )
        result["wilcoxon_p_value"] = float(test.pvalue)

    interval = bootstrap_mean_ci(differences, rng, bootstrap_samples)
    if interval is not None:
        result["mean_difference_ci_95"] = {"low": interval[0], "high": interval[1]}
    return result


def describe(values: list[float | None]) -> dict:
    array = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    if array.size == 0:
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array, ddof=0)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def analyze(data: dict, bootstrap_seed: int = BOOTSTRAP_SEED, bootstrap_samples: int = BOOTSTRAP_SAMPLES) -> dict:
    runs = data["runs"]
    algorithms = ("NSGA2", "MOPSO", "Hybrid")
    checkpoints = [int(value) for value in data["metadata"]["checkpoint_evaluations"]]
    by_algorithm_seed = {
        algorithm: {int(run["seed"]): run for run in runs if run["algorithm"] == algorithm}
        for algorithm in algorithms
    }
    rng = np.random.default_rng(bootstrap_seed)

    final_statistics = {}
    checkpoint_statistics = []
    instability = {algorithm: {"checkpoints": []} for algorithm in algorithms}
    for algorithm in algorithms:
        algorithm_runs = list(by_algorithm_seed[algorithm].values())
        final_statistics[algorithm] = {
            "hypervolume": describe([float(run["hypervolume"]) for run in algorithm_runs]),
            "igd": describe([float(run["igd"]) for run in algorithm_runs]),
        }
        instability[algorithm]["final_hypervolume_sd"] = final_statistics[algorithm]["hypervolume"]["std"]
        instability[algorithm]["final_igd_sd"] = final_statistics[algorithm]["igd"]["std"]

        for checkpoint in checkpoints:
            records = {
                int(run["seed"]): record
                for run in algorithm_runs
                for record in run["convergence"]
                if int(record["evaluations"]) == checkpoint
            }
            hv = describe([record["hypervolume"] for record in records.values()])
            igd = describe([record["igd"] for record in records.values()])
            checkpoint_statistics.append({
                "algorithm": algorithm,
                "evaluations": checkpoint,
                "hypervolume": hv,
                "igd": igd,
            })
            instability[algorithm]["checkpoints"].append({
                "evaluations": checkpoint,
                "hypervolume_sd": hv["std"],
                "igd_sd": igd["std"],
                "igd_defined_seed_count": igd["n"],
            })

    comparisons = []
    for baseline in ("NSGA2", "MOPSO"):
        for metric in ("hypervolume", "igd"):
            hybrid_final = {
                seed: float(run[metric])
                for seed, run in by_algorithm_seed["Hybrid"].items()
            }
            baseline_final = {
                seed: float(run[metric])
                for seed, run in by_algorithm_seed[baseline].items()
            }
            comparison = paired_comparison(hybrid_final, baseline_final, rng, bootstrap_samples)
            comparisons.append({
                "comparison": f"Hybrid vs {baseline}",
                "metric": metric,
                "timepoint": "final_retained_population",
                "evaluations": None,
                **comparison,
            })

            for checkpoint in checkpoints:
                hybrid_seed_values = {
                    int(run["seed"]): record[metric]
                    for run in by_algorithm_seed["Hybrid"].values()
                    for record in run["convergence"]
                    if int(record["evaluations"]) == checkpoint
                }
                baseline_seed_values = {
                    int(run["seed"]): record[metric]
                    for run in by_algorithm_seed[baseline].values()
                    for record in run["convergence"]
                    if int(record["evaluations"]) == checkpoint
                }
                comparison = paired_comparison(hybrid_seed_values, baseline_seed_values, rng, bootstrap_samples)
                comparisons.append({
                    "comparison": f"Hybrid vs {baseline}",
                    "metric": metric,
                    "timepoint": "cumulative_discovery",
                    "evaluations": checkpoint,
                    **comparison,
                })

    adjusted = holm_adjust([item["wilcoxon_p_value"] for item in comparisons])
    for item, adjusted_p in zip(comparisons, adjusted):
        item["holm_adjusted_p_value"] = adjusted_p

    hybrid_runs = by_algorithm_seed["Hybrid"]
    hybrid_extremes = {}
    for metric in ("hypervolume", "igd"):
        best_seed = max(hybrid_runs, key=lambda seed: float(hybrid_runs[seed][metric])) if metric == "hypervolume" else min(hybrid_runs, key=lambda seed: float(hybrid_runs[seed][metric]))
        worst_seed = min(hybrid_runs, key=lambda seed: float(hybrid_runs[seed][metric])) if metric == "hypervolume" else max(hybrid_runs, key=lambda seed: float(hybrid_runs[seed][metric]))
        hybrid_extremes[metric] = {
            "best_seed": best_seed,
            "best_value": float(hybrid_runs[best_seed][metric]),
            "worst_seed": worst_seed,
            "worst_value": float(hybrid_runs[worst_seed][metric]),
        }

    return {
        "metadata": {
            "source_artifact": "experiments/results/benchmark_runs.json",
            "run_count": len(runs),
            "paired_seed_design": "Matched seed IDs; differences are Hybrid minus baseline.",
            "difference_interpretation": "Positive HV difference favors Hybrid; positive IGD difference means higher/worse IGD for Hybrid.",
            "wilcoxon": "Two-sided Wilcoxon signed-rank test, zero_method=wilcox, scipy method=auto. Tests are skipped with an explicit reason when fewer than two matched pairs or fewer than two non-zero differences exist.",
            "effect_size": "Matched-pairs rank-biserial correlation computed from signed ranks of non-zero paired differences.",
            "confidence_interval": "Percentile bootstrap 95% interval for mean paired difference, resampling matched seeds with replacement.",
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "multiple_testing": "Holm step-down correction across all valid final and checkpoint tests (both baselines and both metrics); raw p-values are also retained.",
            "hypothesis_test_count": sum(item["wilcoxon_p_value"] is not None for item in comparisons),
            "checkpoint_front_definition": "cumulative feasible discoveries through evaluation E",
            "checkpoint_igd_reference": "Common final pooled reference front, permitted as the same reference for all runs; it is not inserted into checkpoint approximation fronts.",
            "evaluations_to_target_hv": "not available; no defensible common target was specified by the benchmark methodology.",
            "checkpoint_metrics": checkpoint_statistics,
            "final_statistics": final_statistics,
            "hybrid_seed_extremes": hybrid_extremes,
            "instability": instability,
            "paired_comparisons": comparisons,
        },
        "paired_comparisons": comparisons,
        "checkpoint_statistics": checkpoint_statistics,
        "final_statistics": final_statistics,
        "hybrid_seed_extremes": hybrid_extremes,
        "instability": instability,
    }


def csv_rows(analysis: dict) -> tuple[list[dict], list[str]]:
    fields = [
        "record_type", "algorithm", "comparison", "baseline", "metric", "timepoint",
        "evaluations", "n", "mean", "median", "std", "min", "max", "n_pairs",
        "seeds", "mean_paired_difference", "median_paired_difference",
        "wilcoxon_p_value", "holm_adjusted_p_value", "wilcoxon_skipped_reason",
        "matched_pairs_rank_biserial", "ci_95_low", "ci_95_high",
        "igd_defined_seed_count", "bootstrap_samples", "bootstrap_seed", "name", "value",
    ]
    rows = []
    metadata = analysis["metadata"]
    for key in (
        "source_artifact", "run_count", "paired_seed_design", "difference_interpretation",
        "wilcoxon", "effect_size", "confidence_interval", "bootstrap_samples",
        "bootstrap_seed", "multiple_testing", "hypothesis_test_count",
        "checkpoint_front_definition", "checkpoint_igd_reference", "evaluations_to_target_hv",
    ):
        rows.append({"record_type": "metadata", "name": key, "value": json.dumps(metadata[key])})

    for algorithm, metric_map in analysis["final_statistics"].items():
        for metric, stats in metric_map.items():
            rows.append({
                "record_type": "final_summary",
                "algorithm": algorithm,
                "metric": metric,
                **stats,
            })
    for item in analysis["checkpoint_statistics"]:
        rows.append({
            "record_type": "checkpoint_summary",
            "algorithm": item["algorithm"],
            "evaluations": item["evaluations"],
            "metric": "hypervolume",
            **item["hypervolume"],
        })
        rows.append({
            "record_type": "checkpoint_summary",
            "algorithm": item["algorithm"],
            "evaluations": item["evaluations"],
            "metric": "igd",
            **item["igd"],
            "igd_defined_seed_count": item["igd"]["n"],
        })
    for item in analysis["paired_comparisons"]:
        ci = item["mean_difference_ci_95"] or {}
        rows.append({
            "record_type": "paired_comparison",
            "comparison": item["comparison"],
            "baseline": item["comparison"].replace("Hybrid vs ", ""),
            "metric": item["metric"],
            "timepoint": item["timepoint"],
            "evaluations": item["evaluations"],
            "n_pairs": item["n_pairs"],
            "seeds": ",".join(map(str, item["seeds"])),
            "mean_paired_difference": item["mean_paired_difference"],
            "median_paired_difference": item["median_paired_difference"],
            "wilcoxon_p_value": item["wilcoxon_p_value"],
            "holm_adjusted_p_value": item["holm_adjusted_p_value"],
            "wilcoxon_skipped_reason": item["wilcoxon_skipped_reason"],
            "matched_pairs_rank_biserial": item["matched_pairs_rank_biserial"],
            "ci_95_low": ci.get("low"),
            "ci_95_high": ci.get("high"),
            "bootstrap_samples": metadata["bootstrap_samples"],
            "bootstrap_seed": metadata["bootstrap_seed"],
        })
    return rows, fields


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the saved matched-seed optimization benchmark.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as handle:
        source = json.load(handle)
    result = analyze(source, args.bootstrap_seed, args.bootstrap_samples)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    rows, fields = csv_rows(result)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "input": str(args.input),
        "output_json": str(args.output_json),
        "output_csv": str(args.output_csv),
        "runs_analyzed": result["metadata"]["run_count"],
        "hypothesis_tests": result["metadata"]["hypothesis_test_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
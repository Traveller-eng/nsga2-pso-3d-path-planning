from __future__ import annotations

import math

import numpy as np

from planner.metrics.hypervolume import calculate_hypervolume
from planner.metrics.igd import calculate_igd
from planner.metrics.pareto import pareto_filter


def compute_checkpoints(max_evaluations: int, fractions=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)) -> np.ndarray:
    values = []
    for fraction in fractions:
        values.append(int(round(fraction * max_evaluations)))
    values = sorted({max(0, min(v, max_evaluations)) for v in values})
    if values[-1] != max_evaluations:
        values.append(max_evaluations)
    return np.asarray(values, dtype=int)


def evaluations_to_target_hv(history: list[dict], target_hv: float) -> float:
    for record in history:
        if not isinstance(record, dict):
            continue
        hv = float(record.get("hypervolume", math.nan))
        if math.isnan(hv):
            continue
        if hv >= target_hv:
            return float(record.get("evaluations", math.nan))
    return float("nan")


def compute_checkpoint_metrics(
    evaluation_history: list[dict],
    checkpoints: np.ndarray,
    reference_point: np.ndarray,
    reference_front: np.ndarray,
) -> list[dict]:
    """Compute cumulative evaluated-so-far metrics without admitting future records."""
    checkpoints = np.asarray(checkpoints, dtype=int)
    if checkpoints.ndim != 1 or np.any(np.diff(checkpoints) <= 0):
        raise ValueError("checkpoints must be a strictly increasing one-dimensional sequence.")

    history = sorted(evaluation_history, key=lambda record: int(record["evaluations"]))
    evaluation_counts = [int(record["evaluations"]) for record in history]
    if any(right <= left for left, right in zip(evaluation_counts, evaluation_counts[1:])):
        raise ValueError("evaluation_history must have unique increasing evaluation counts.")

    reference_point = np.asarray(reference_point, dtype=float)
    reference_front = np.asarray(reference_front, dtype=float)
    records = []
    for checkpoint in checkpoints:
        available = [record for record in history if int(record["evaluations"]) <= checkpoint]
        feasible_records = [record for record in available if record["is_feasible"]]
        objective_vectors = [np.asarray(record["objectives"], dtype=float) for record in feasible_records]
        if objective_vectors:
            approximation = pareto_filter(np.vstack(objective_vectors))
        else:
            approximation = np.empty((0, reference_point.size), dtype=float)

        igd = calculate_igd(approximation, reference_front) if approximation.shape[0] else None
        records.append({
            "evaluations": int(checkpoint),
            "feasible_count": len(feasible_records),
            "front_size": int(approximation.shape[0]),
            "hypervolume": float(calculate_hypervolume(approximation, reference_point)),
            "igd": None if igd is None else float(igd),
        })
    return records


def aggregate_checkpoint_metrics(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for record in records:
        key = (record["algorithm"], int(record["evaluations"]))
        grouped.setdefault(key, []).append(record)

    aggregates = []
    for (algorithm, evaluations), group in sorted(grouped.items()):
        hv = np.asarray([record["hypervolume"] for record in group], dtype=float)
        igd = np.asarray(
            [record["igd"] for record in group if record["igd"] is not None],
            dtype=float,
        )
        aggregates.append({
            "algorithm": algorithm,
            "evaluations": evaluations,
            "hypervolume_mean": float(np.mean(hv)),
            "hypervolume_median": float(np.median(hv)),
            "hypervolume_std": float(np.std(hv, ddof=0)),
            "igd_sample_count": int(igd.size),
            "igd_mean": float(np.mean(igd)) if igd.size else None,
            "igd_median": float(np.median(igd)) if igd.size else None,
            "igd_std": float(np.std(igd, ddof=0)) if igd.size else None,
        })
    return aggregates

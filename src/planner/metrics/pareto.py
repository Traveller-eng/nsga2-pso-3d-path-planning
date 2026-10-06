from __future__ import annotations

import numpy as np


def remove_duplicates(objectives: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    arr = np.asarray(objectives, dtype=float)
    if arr.size == 0:
        return arr.reshape((0,))
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] == 0:
        return arr.copy()
    rounded = np.round(arr / tol, 0) * tol
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    return arr[np.sort(unique_idx)]


def pareto_filter(objectives: np.ndarray, *, minimize: bool = True) -> np.ndarray:
    """Return the non-dominated subset of an objective matrix."""
    arr = np.asarray(objectives, dtype=float)
    if arr.size == 0:
        return arr.reshape((0, 0))
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] == 0:
        return arr.copy()

    unique = remove_duplicates(arr)
    keep: list[int] = []
    for i, candidate in enumerate(unique):
        dominated = False
        for j, other in enumerate(unique):
            if i == j:
                continue
            if minimize:
                better_or_equal = np.all(other <= candidate + 1e-12)
                strictly_better = np.any(other < candidate - 1e-12)
            else:
                better_or_equal = np.all(other >= candidate - 1e-12)
                strictly_better = np.any(other > candidate + 1e-12)
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return unique[keep]


def extract_feasible_objectives(population: list | np.ndarray) -> np.ndarray:
    if isinstance(population, np.ndarray):
        arr = np.asarray(population, dtype=float)
        if arr.size == 0:
            return arr.reshape((0, 0))
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr

    feasible: list[np.ndarray] = []
    for item in population:
        if hasattr(item, "evaluation"):
            evaluation = item.evaluation
            if getattr(evaluation, "is_feasible", False):
                feasible.append(np.asarray(evaluation.objectives, dtype=float))
        elif hasattr(item, "objective"):
            feasible.append(np.asarray(item.objective, dtype=float))
        elif isinstance(item, (list, tuple, np.ndarray)):
            feasible.append(np.asarray(item, dtype=float))
    if not feasible:
        return np.empty((0, 0), dtype=float)
    return np.vstack(feasible)


def pooled_reference_front(algorithm_results, *, feasible_only: bool = True) -> np.ndarray:
    collected: list[np.ndarray] = []
    for entry in algorithm_results:
        arr = np.asarray(entry, dtype=float)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] == 0:
            continue
        if feasible_only:
            arr = arr[np.isfinite(arr).all(axis=1)]
        if arr.size:
            collected.append(arr)
    if not collected:
        return np.empty((0, 0), dtype=float)
    merged = np.vstack(collected)
    merged = remove_duplicates(merged)
    return pareto_filter(merged)

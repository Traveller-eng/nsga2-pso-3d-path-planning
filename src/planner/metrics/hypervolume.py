from __future__ import annotations

import numpy as np


def _remove_dominated(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return arr.reshape((0, 0))
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] == 0:
        return arr.copy()
    arr = np.unique(np.round(arr, 12), axis=0)
    keep = []
    for i, candidate in enumerate(arr):
        dominated = False
        for j, other in enumerate(arr):
            if i == j:
                continue
            if np.all(other <= candidate + 1e-12) and np.any(other < candidate - 1e-12):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return arr[keep]


def _hv_2d(points: np.ndarray, reference: np.ndarray) -> float:
    if points.size == 0:
        return 0.0
    arr = np.asarray(points, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] == 0:
        return 0.0
    arr = arr[np.isfinite(arr).all(axis=1)]
    arr = arr[np.all(arr <= reference + 1e-12, axis=1)]
    arr = _remove_dominated(arr)
    if arr.shape[0] == 0:
        return 0.0

    y_values = np.unique(arr[:, 0])
    hv = 0.0
    for idx, y_val in enumerate(y_values):
        current = arr[arr[:, 0] <= y_val]
        min_z = float(np.min(current[:, 1]))
        next_y = reference[0] if idx == len(y_values) - 1 else y_values[idx + 1]
        hv += (next_y - y_val) * (reference[1] - min_z)
    return max(hv, 0.0)


def calculate_hypervolume(objectives: np.ndarray, reference_point: np.ndarray) -> float:
    """Exact hypervolume for 2D/3D minimization problems with a fixed reference point."""
    arr = np.asarray(objectives, dtype=float)
    ref = np.asarray(reference_point, dtype=float)

    if arr.size == 0:
        return 0.0
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != ref.shape[0]:
        raise ValueError("Objective dimensionality must match reference point dimensionality.")

    arr = arr[np.isfinite(arr).all(axis=1)]
    if arr.shape[0] == 0:
        return 0.0

    arr = arr[np.all(arr <= ref + 1e-12, axis=1)]
    if arr.shape[0] == 0:
        return 0.0

    arr = _remove_dominated(arr)
    if arr.shape[0] == 0:
        return 0.0

    n_obj = arr.shape[1]
    if n_obj == 1:
        return max(ref[0] - np.min(arr[:, 0]), 0.0)
    if n_obj == 2:
        return float(_hv_2d(arr, ref))
    if n_obj == 3:
        x_values = np.unique(arr[:, 0])
        hv = 0.0
        for idx, x_val in enumerate(x_values):
            current = arr[arr[:, 0] <= x_val]
            yz_hv = _hv_2d(current[:, 1:3], ref[1:])
            next_x = ref[0] if idx == len(x_values) - 1 else x_values[idx + 1]
            hv += (next_x - x_val) * yz_hv
        return max(hv, 0.0)

    raise ValueError("Only 1D, 2D, and 3D problems are supported.")

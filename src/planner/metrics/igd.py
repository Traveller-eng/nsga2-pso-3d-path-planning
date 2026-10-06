from __future__ import annotations

import numpy as np


def _normalize_fronts(approximation: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    all_points = np.vstack([approximation, reference])
    mins = np.min(all_points, axis=0)
    maxs = np.max(all_points, axis=0)
    spread = maxs - mins
    spread[spread == 0.0] = 1.0
    return (approximation - mins) / spread, (reference - mins) / spread


def calculate_igd(
    approximation: np.ndarray,
    reference: np.ndarray,
    normalize: bool = False,
) -> float:
    """Inverted generational distance between an approximation front and a reference front."""
    approx = np.asarray(approximation, dtype=float)
    ref = np.asarray(reference, dtype=float)

    if approx.size == 0:
        return float("nan")
    if ref.size == 0:
        return 0.0
    if approx.ndim == 1:
        approx = approx.reshape(1, -1)
    if ref.ndim == 1:
        ref = ref.reshape(1, -1)

    if normalize:
        approx, ref = _normalize_fronts(approx, ref)

    distances = np.array([
        np.min(np.linalg.norm(approx - point, axis=1))
        for point in ref
    ], dtype=float)
    return float(np.mean(distances))

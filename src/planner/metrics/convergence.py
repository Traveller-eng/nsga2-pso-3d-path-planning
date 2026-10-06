from __future__ import annotations

import math

import numpy as np


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

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_experiment import build_reference_point, compute_run_metrics


def test_build_reference_point_and_run_metrics_are_finite():
    fronts = [
        np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
        np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]]),
    ]

    reference = build_reference_point(fronts)
    metrics = compute_run_metrics(fronts[0], fronts[1], reference)

    assert reference.shape == (3,)
    assert np.all(np.isfinite(reference))
    assert np.isfinite(metrics["hypervolume"])
    assert np.isfinite(metrics["igd"])
    assert metrics["hypervolume"] >= 0.0
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_experiment import compute_run_metrics, get_reference_point


def test_configured_reference_point_is_fixed_and_run_metrics_are_finite():
    fronts = [
        np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
        np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]]),
    ]

    config = {
        "metrics": {
            "hypervolume_reference_point": [5.0, 5.0, 5.0],
        },
    }
    reference = get_reference_point(config)
    metrics = compute_run_metrics(fronts[0], fronts[1], reference)

    assert np.array_equal(reference, np.array([5.0, 5.0, 5.0]))
    assert np.isfinite(metrics["hypervolume"])
    assert np.isfinite(metrics["igd"])
    assert metrics["hypervolume"] >= 0.0


def test_reference_point_is_required_in_experiment_config():
    import pytest

    with pytest.raises(ValueError, match="must define"):
        get_reference_point({})
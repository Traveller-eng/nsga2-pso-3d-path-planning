from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_experiment import (
    aggregate_runs,
    compare_hybrid_means,
    compute_run_metrics,
    get_reference_point,
)


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


def test_aggregates_and_hybrid_mean_differences_cover_requested_metrics():
    runs = [
        {"algorithm": "NSGA2", "hypervolume": 2.0, "igd": 4.0, "runtime": 3.0, "feasible_count": 5},
        {"algorithm": "NSGA2", "hypervolume": 4.0, "igd": 2.0, "runtime": 5.0, "feasible_count": 7},
        {"algorithm": "Hybrid", "hypervolume": 5.0, "igd": 1.0, "runtime": 2.0, "feasible_count": 8},
    ]

    aggregates = aggregate_runs(runs, ["NSGA2", "Hybrid"])
    comparisons = compare_hybrid_means(aggregates)

    nsga2 = aggregates[0]
    assert nsga2["hypervolume"] == {
        "mean": 3.0, "median": 3.0, "std": 1.0, "min": 2.0, "max": 4.0,
    }
    assert next(c for c in comparisons if c["metric"] == "hypervolume")["mean_difference"] == 2.0
    assert next(c for c in comparisons if c["metric"] == "igd")["mean_difference"] == -2.0
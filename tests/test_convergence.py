from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from planner.metrics.convergence import (
    aggregate_checkpoint_metrics,
    compute_checkpoint_metrics,
    compute_checkpoints,
)


def test_checkpoints_are_exact_monotonic_and_bounded():
    checkpoints = compute_checkpoints(300)

    assert checkpoints.tolist() == list(range(30, 301, 30))
    assert np.all(np.diff(checkpoints) > 0)
    assert np.all(checkpoints <= 300)


def test_checkpoint_metrics_exclude_future_objectives_and_recompute_hv_igd():
    history = [
        {"evaluations": 30, "objectives": [8.0, 8.0, 8.0], "is_feasible": True},
        {"evaluations": 31, "objectives": [1.0, 1.0, 1.0], "is_feasible": True},
    ]
    reference_point = np.array([10.0, 10.0, 10.0])
    reference_front = np.array([[1.0, 1.0, 1.0]])
    checkpoints = np.array([30, 31])

    records = compute_checkpoint_metrics(history, checkpoints, reference_point, reference_front)

    early = records[0]
    assert early["evaluations"] == 30
    assert early["feasible_count"] == 1
    assert early["front_size"] == 1
    assert early["hypervolume"] == 8.0
    assert np.isclose(early["igd"], np.sqrt(147.0))

    later = records[1]
    assert later["evaluations"] == 31
    assert later["feasible_count"] == 2
    assert later["front_size"] == 1
    assert later["hypervolume"] == 729.0
    assert later["igd"] == 0.0


def test_checkpoint_metrics_are_reproducible_and_json_serializable():
    history = [
        {"evaluations": 30, "objectives": [2.0, 2.0, 2.0], "is_feasible": True},
        {"evaluations": 60, "objectives": [1.0, 1.0, 1.0], "is_feasible": True},
    ]
    reference_point = np.array([3.0, 3.0, 3.0])
    reference_front = np.array([[1.0, 1.0, 1.0]])
    checkpoints = np.array([30, 60])

    first = compute_checkpoint_metrics(history, checkpoints, reference_point, reference_front)
    second = compute_checkpoint_metrics(history, checkpoints, reference_point, reference_front)

    assert first == second
    assert [record["evaluations"] for record in first] == checkpoints.tolist()
    assert json.loads(json.dumps(first)) == first


def test_igd_is_null_when_checkpoint_has_no_feasible_approximation():
    history = [
        {"evaluations": 30, "objectives": [2.0, 2.0, 2.0], "is_feasible": False},
    ]
    record = compute_checkpoint_metrics(
        history,
        np.array([30]),
        np.array([3.0, 3.0, 3.0]),
        np.array([[1.0, 1.0, 1.0]]),
    )[0]

    assert record["feasible_count"] == 0
    assert record["front_size"] == 0
    assert record["hypervolume"] == 0.0
    assert record["igd"] is None
    assert json.loads(json.dumps(record))["igd"] is None


def test_checkpoint_igd_aggregate_reports_defined_seed_count():
    records = [
        {"algorithm": "NSGA2", "evaluations": 30, "hypervolume": 0.0, "igd": None},
        {"algorithm": "NSGA2", "evaluations": 30, "hypervolume": 2.0, "igd": 0.5},
        {"algorithm": "NSGA2", "evaluations": 30, "hypervolume": 4.0, "igd": 1.5},
    ]

    aggregate = aggregate_checkpoint_metrics(records)[0]

    assert aggregate["igd_sample_count"] == 2
    assert aggregate["igd_mean"] == 1.0
    assert aggregate["igd_median"] == 1.0
    assert aggregate["igd_std"] == 0.5
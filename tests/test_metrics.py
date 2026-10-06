import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from planner.metrics.hypervolume import calculate_hypervolume
from planner.metrics.igd import calculate_igd
from planner.metrics.pareto import pareto_filter, pooled_reference_front, remove_duplicates


def test_hypervolume_2d_known_value():
    points = np.array([[1.0, 1.0], [2.0, 3.0]])
    reference = np.array([5.0, 5.0])
    hv = calculate_hypervolume(points, reference)
    assert np.isclose(hv, 16.0)


def test_hypervolume_3d_known_value_and_duplicates():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    reference = np.array([2.0, 2.0, 2.0])
    hv = calculate_hypervolume(points, reference)
    assert np.isclose(hv, 8.0)


def test_hypervolume_empty_and_dominated_points():
    assert calculate_hypervolume(np.empty((0, 3)), np.array([2.0, 2.0, 2.0])) == 0.0
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    hv = calculate_hypervolume(points, np.array([3.0, 3.0, 3.0]))
    assert hv > 0.0


def test_igd_zero_for_identical_fronts():
    approx = np.array([[0.0, 0.0], [1.0, 1.0]])
    reference = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert calculate_igd(approx, reference) == 0.0


def test_igd_averages_reference_to_approximation_distances():
    approximation = np.array([[0.0, 0.0]])
    reference = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert np.isclose(calculate_igd(approximation, reference), 1.0)


def test_igd_handles_empty_approximation():
    approx = np.empty((0, 3))
    reference = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert np.isnan(calculate_igd(approx, reference))


def test_pareto_filter_removes_dominated_and_duplicates():
    points = np.array([
        [1.0, 4.0],
        [2.0, 3.0],
        [3.0, 2.0],
        [2.0, 3.0],
        [5.0, 5.0],
    ])
    filtered = pareto_filter(points)
    assert filtered.shape[0] == 3
    assert np.all(np.any(filtered == np.array([1.0, 4.0]), axis=1) | np.any(filtered == np.array([2.0, 3.0]), axis=1) | np.any(filtered == np.array([3.0, 2.0]), axis=1))


def test_reference_front_pooled_and_feasible_only():
    all_points = [
        np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        np.array([[0.5, 2.5], [2.0, 2.0], [1.7, 1.7]]),
    ]
    pooled = pooled_reference_front(all_points, feasible_only=True)
    assert pooled.size > 0
    assert np.all(np.isfinite(pooled))
    assert len(remove_duplicates(pooled)) == len(remove_duplicates(pooled))

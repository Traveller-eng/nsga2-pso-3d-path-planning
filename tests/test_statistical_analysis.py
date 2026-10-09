from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_benchmark import holm_adjust, paired_comparison


def test_paired_comparison_uses_hybrid_minus_baseline_and_rank_biserial():
    hybrid = {0: 2.0, 1: 3.0, 2: 4.0, 3: 5.0}
    baseline = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}

    result = paired_comparison(hybrid, baseline, np.random.default_rng(7), bootstrap_samples=1000)

    assert result["n_pairs"] == 4
    assert result["mean_paired_difference"] == 2.5
    assert result["median_paired_difference"] == 2.5
    assert result["matched_pairs_rank_biserial"] == 1.0
    assert result["wilcoxon_p_value"] is not None
    assert result["mean_difference_ci_95"]["low"] <= 2.5
    assert result["mean_difference_ci_95"]["high"] >= 2.5


def test_paired_comparison_skips_all_zero_signed_ranks():
    values = {0: 2.0, 1: 3.0, 2: 4.0}

    result = paired_comparison(values, values, np.random.default_rng(7), bootstrap_samples=1000)

    assert result["wilcoxon_p_value"] is None
    assert result["matched_pairs_rank_biserial"] is None
    assert "non-zero" in result["wilcoxon_skipped_reason"]


def test_bootstrap_interval_is_reproducible_and_holm_adjustment_is_step_down():
    hybrid = {0: 2.0, 1: None, 2: 4.0, 3: 5.0}
    baseline = {0: 1.0, 1: 2.0, 2: 1.0, 3: 1.0}
    first = paired_comparison(hybrid, baseline, np.random.default_rng(17), bootstrap_samples=2000)
    second = paired_comparison(hybrid, baseline, np.random.default_rng(17), bootstrap_samples=2000)

    assert first == second
    assert first["seeds"] == [0, 2, 3]
    assert holm_adjust([0.01, 0.04, None, 0.03]) == [0.03, 0.06, None, 0.06]
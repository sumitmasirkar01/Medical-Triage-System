"""Selective prediction keeps a fixed threshold and exposes referral tradeoffs."""

import numpy as np
import pytest

from stat_triage.selective import (
    ABSTAIN, NEGATIVE, POSITIVE, abstention_boundaries,
    grouped_selective_bootstrap_confidence_intervals,
    select_abstention_operating_point, selective_metrics, three_way_decisions,
)


def test_zero_margin_reproduces_locked_binary_decisions():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.49, 0.5, 0.9])
    decisions = three_way_decisions(scores, threshold=0.5, margin_fraction=0.0)
    assert decisions.tolist() == [NEGATIVE, NEGATIVE, POSITIVE, POSITIVE]
    measured = selective_metrics(y, scores, 0.5, 0.0)
    assert measured.coverage == 1.0
    assert measured.abstained == 0
    assert measured.selective_accuracy == 1.0


def test_normalized_margin_abstains_on_both_sides_of_low_threshold():
    lower, upper = abstention_boundaries(0.1, 0.2)
    assert lower == pytest.approx(0.08)
    assert upper == pytest.approx(0.28)
    decisions = three_way_decisions(np.array([0.07, 0.08, 0.2, 0.28]), 0.1, 0.2)
    assert decisions.tolist() == [NEGATIVE, ABSTAIN, ABSTAIN, POSITIVE]


def test_metrics_distinguish_referrals_from_residual_automatic_errors():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.05, 0.45, 0.8, 0.2, 0.55, 0.9])
    measured = selective_metrics(y, scores, 0.5, 0.2)
    assert measured.abstained == 2
    assert measured.automatic == 4
    assert measured.tp == 1 and measured.tn == 1
    assert measured.fp == 1 and measured.fn == 1
    assert measured.automatic_false_negative_rate_all_positives == pytest.approx(1 / 3)


def test_selector_uses_smallest_margin_meeting_safety_and_coverage_constraints():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.05, 0.1, 0.2, 0.8, 0.3, 0.55, 0.7, 0.9])
    selected = select_abstention_operating_point(
        y, scores, 0.5, minimum_coverage=0.5,
        maximum_automatic_false_negative_rate=0.0,
        candidate_margins=np.array([0.0, 0.2, 0.4]),
    )
    assert selected.margin_fraction == pytest.approx(0.4)
    assert selected.metrics.coverage == pytest.approx(0.75)
    assert selected.metrics.fn == 0


def test_selector_rejects_infeasible_or_invalid_policies():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.1, 0.9])
    with pytest.raises(ValueError, match="no candidate"):
        select_abstention_operating_point(
            y, scores, 0.5, minimum_coverage=1.0,
            maximum_automatic_false_negative_rate=0.0,
            candidate_margins=np.array([0.0, 0.2]),
        )
    with pytest.raises(ValueError, match="margin_fraction"):
        three_way_decisions(scores, 0.5, 1.0)


def test_selective_bootstrap_is_grouped_and_deterministic():
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.7, 0.9, 0.15, 0.3, 0.6, 0.85])
    groups = np.array(["A", "A", "B", "B", "C", "C", "D", "D"])
    first = grouped_selective_bootstrap_confidence_intervals(
        y, scores, groups, 0.5, 0.1, replicates=50, seed=7,
        minimum_valid_fraction=0.5, sampling_unit="exam_id",
    )
    second = grouped_selective_bootstrap_confidence_intervals(
        y, scores, groups, 0.5, 0.1, replicates=50, seed=7,
        minimum_valid_fraction=0.5, sampling_unit="exam_id",
    )
    assert first.to_dict() == second.to_dict()
    assert first.groups == 4
    assert first.sampling_unit == "exam_id"

"""Grouped bootstrap resamples complete patients/exams deterministically."""

import numpy as np
import pytest

from stat_triage.bootstrap import grouped_bootstrap_confidence_intervals


def cohort():
    # Patients A and C have two images; these rows must remain atomic.
    return (
        np.array([1, 1, 0, 0, 1, 0]),
        np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.1]),
        np.array(["A", "A", "B", "C", "C", "D"]),
    )


def test_grouped_bootstrap_is_deterministic_and_reports_units():
    targets, scores, groups = cohort()
    first = grouped_bootstrap_confidence_intervals(
        targets, scores, groups, 0.5, replicates=500, seed=7,
        minimum_valid_fraction=0.5,
    )
    second = grouped_bootstrap_confidence_intervals(
        targets, scores, groups, 0.5, replicates=500, seed=7,
        minimum_valid_fraction=0.5,
    )

    assert first == second
    assert first.groups == 4
    assert first.rows == 6
    assert first.requested_replicates == 500
    assert set(first.intervals) == {
        "roc_auc", "average_precision", "sensitivity", "specificity",
        "ppv", "npv", "accuracy",
    }


def test_different_seed_changes_a_non_degenerate_interval():
    targets, scores, groups = cohort()
    first = grouped_bootstrap_confidence_intervals(
        targets, scores, groups, 0.5, replicates=301, seed=1,
        confidence_level=0.8, minimum_valid_fraction=0.5,
    )
    second = grouped_bootstrap_confidence_intervals(
        targets, scores, groups, 0.5, replicates=301, seed=2,
        confidence_level=0.8, minimum_valid_fraction=0.5,
    )

    assert first.to_dict() != second.to_dict()


def test_group_identifiers_must_match_and_be_present():
    with pytest.raises(ValueError, match="matching targets"):
        grouped_bootstrap_confidence_intervals([0, 1], [0.1, 0.9], ["A"], 0.5)
    with pytest.raises(ValueError, match="missing or empty"):
        grouped_bootstrap_confidence_intervals([0, 1], [0.1, 0.9], ["", "B"], 0.5)


def test_excess_single_class_replicates_are_rejected():
    with pytest.raises(ValueError, match="too few valid bootstrap"):
        grouped_bootstrap_confidence_intervals(
            [0, 1], [0.1, 0.9], ["NEG", "POS"], 0.5,
            replicates=100, seed=4, minimum_valid_fraction=0.9,
        )


"""Temperature scaling is held-out, deterministic, and allowed to do nothing."""

import numpy as np
import pytest

from stat_triage.calibration import (
    calibration_metrics,
    fit_affine_logistic_scaling,
    fit_temperature_scaling,
)


def test_temperature_scaling_improves_overconfident_predictions():
    targets = np.array([0, 0, 0, 1, 1, 1])
    logits = np.array([-8.0, -5.0, 4.0, -4.0, 5.0, 8.0])

    fitted = fit_temperature_scaling(targets, logits)

    assert fitted.selected_method == "temperature_scaling"
    assert fitted.temperature > 1.0
    assert fitted.after.negative_log_likelihood < fitted.before.negative_log_likelihood
    assert fitted.nll_improvement > 0


def test_identity_is_retained_when_temperature_cannot_improve_nll():
    targets = np.array([0, 0, 1, 1])
    logits = np.zeros(4)

    fitted = fit_temperature_scaling(targets, logits)

    assert fitted.selected_method == "identity"
    assert fitted.temperature == 1.0
    assert fitted.after == fitted.before


def test_calibration_metrics_have_known_values():
    measured = calibration_metrics([0, 1], [0.25, 0.75], bins=2)

    assert measured.negative_log_likelihood == pytest.approx(-np.log(0.75))
    assert measured.brier_score == pytest.approx(0.0625)
    assert measured.expected_calibration_error == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("targets", "logits", "message"),
    [
        ([0, 2], [0.0, 1.0], "only 0 and 1"),
        ([1, 1], [0.0, 1.0], "both target classes"),
        ([0, 1], [0.0, float("nan")], "finite"),
    ],
)
def test_invalid_calibration_cohorts_are_rejected(targets, logits, message):
    with pytest.raises(ValueError, match=message):
        fit_temperature_scaling(targets, logits)


def test_affine_scaling_corrects_a_weighted_loss_intercept_shift():
    targets = np.array([0] * 90 + [1] * 10)
    true_probabilities = np.array([0.05] * 90 + [0.65] * 10)
    logits = np.log(true_probabilities / (1 - true_probabilities)) + 2.0

    fitted = fit_affine_logistic_scaling(targets, logits)

    assert fitted.selected_method == "affine_logistic_scaling"
    assert fitted.intercept < 0
    assert fitted.after.negative_log_likelihood < fitted.before.negative_log_likelihood


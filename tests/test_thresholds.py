"""Operating points are calibration-only and sensitivity-constrained."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stat_triage.thresholds import select_sensitivity_threshold


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts" / "rsna_baseline" / "rsna_baseline_bundle"


def test_selector_reproduces_rsna_calibration_operating_point():
    rows = pd.read_csv(BUNDLE / "calibration_predictions.csv")
    locked = json.loads((BUNDLE / "operating_point.json").read_text(encoding="utf-8"))

    selected = select_sensitivity_threshold(
        rows["target_any"], rows["score_any"], locked["target_sensitivity"]
    )

    # The CSV stores fewer decimal places than the JSON threshold; its selected
    # score therefore agrees within serialization precision rather than bitwise.
    assert selected.threshold == pytest.approx(locked["threshold"], abs=2e-8)
    assert selected.sensitivity == locked["calibration_sensitivity"]
    assert selected.specificity == locked["calibration_specificity"]
    assert {name: getattr(selected, name) for name in ("tp", "tn", "fp", "fn")} == (
        locked["calibration_confusion_counts"]
    )


def test_selected_threshold_is_maximal_with_tied_positive_scores():
    targets = np.array([1, 1, 1, 0, 0])
    scores = np.array([0.9, 0.7, 0.7, 0.8, 0.2])

    selected = select_sensitivity_threshold(targets, scores, 2 / 3)

    assert selected.threshold == 0.7
    assert selected.sensitivity == 1.0
    assert selected.specificity == 0.5
    threshold_above = np.nextafter(selected.threshold, 1.0)
    sensitivity_above = np.mean(scores[targets == 1] >= threshold_above)
    assert sensitivity_above < 2 / 3


@pytest.mark.parametrize("target", [0.0, -0.1, 1.1, float("nan")])
def test_invalid_sensitivity_target_is_rejected(target):
    with pytest.raises(ValueError, match="target_sensitivity"):
        select_sensitivity_threshold([0, 1], [0.1, 0.9], target)


def test_both_classes_are_required_for_operating_point_selection():
    with pytest.raises(ValueError, match="both target classes"):
        select_sensitivity_threshold([1, 1], [0.8, 0.9], 0.9)


"""Chest X-ray calibration is per target and calibration-only."""

import numpy as np
import pandas as pd

from stat_triage.cxr_calibration import calibrate_cxr_predictions
from stat_triage.datasets.nih_training import NIH_TARGET_ORDER


def test_calibration_produces_scores_and_sensitivity_constrained_thresholds():
    logits = np.array([-3.0, -1.0, 0.5, 2.0, 3.0, -0.5])
    targets = np.array([0, 0, 1, 1, 1, 0])
    rows = {}
    for index, target in enumerate(NIH_TARGET_ORDER):
        rows[f"target_{target}"] = np.roll(targets, index)
        rows[f"logit_{target}"] = np.roll(logits, index)
    predictions, fitted = calibrate_cxr_predictions(
        pd.DataFrame(rows),
        target_sensitivities={target: 2 / 3 for target in NIH_TARGET_ORDER},
    )

    assert fitted.examples == 6
    assert set(fitted.targets) == set(NIH_TARGET_ORDER)
    for target in NIH_TARGET_ORDER:
        scores = predictions[f"calibrated_score_{target}"]
        assert scores.between(0, 1).all()
        operating_point = fitted.targets[target]["operating_point"]
        assert operating_point["sensitivity"] >= 2 / 3
        assert operating_point["selection_rule"] == (
            "largest_threshold_with_sensitivity_at_least_target"
        )

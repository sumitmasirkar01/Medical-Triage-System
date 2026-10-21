import numpy as np
import pandas as pd

from stat_triage.datasets.mrnet import MRNET_TARGETS
from stat_triage.mrnet_calibration import calibrate_mrnet_predictions


def test_mrnet_calibration_fits_each_target_and_locks_thresholds():
    predictions = pd.DataFrame({"exam_id": [f"{i:04d}" for i in range(12)]})
    labels = np.array([0, 1] * 6)
    for index, target in enumerate(MRNET_TARGETS):
        predictions[f"target_{target}"] = np.roll(labels, index)
        predictions[f"logit_{target}"] = np.linspace(-2, 2, 12) + index * 0.1
    calibrated, result = calibrate_mrnet_predictions(
        predictions,
        target_sensitivities={target: 0.8 for target in MRNET_TARGETS},
    )
    assert result.exams == 12
    assert set(result.targets) == set(MRNET_TARGETS)
    for target in MRNET_TARGETS:
        scores = calibrated[f"calibrated_score_{target}"]
        assert scores.between(0, 1).all()
        assert result.targets[target]["operating_point"]["sensitivity"] >= 0.8


def test_mrnet_calibration_rejects_incomplete_policy():
    predictions = pd.DataFrame({"exam_id": ["0000"]})
    try:
        calibrate_mrnet_predictions(predictions, target_sensitivities={"acl": 0.9})
    except ValueError as error:
        assert "exactly match" in str(error)
    else:
        raise AssertionError("incomplete target policy was accepted")

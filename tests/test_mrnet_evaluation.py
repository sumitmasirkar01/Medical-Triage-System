import numpy as np
import pandas as pd

from stat_triage.datasets.mrnet import MRNET_TARGETS
from stat_triage.mrnet_evaluation import evaluate_locked_mrnet_predictions


def _locks():
    return {
        target: {
            "calibration": {
                "selected_method": "affine_logistic_scaling",
                "slope": 1.0,
                "intercept": 0.0,
            },
            "operating_point": {"threshold": 0.5},
        }
        for target in MRNET_TARGETS
    }


def test_locked_mrnet_evaluation_applies_thresholds_and_bootstraps_exams():
    frame = pd.DataFrame({"exam_id": [f"{i:04d}" for i in range(20)]})
    labels = np.array([0, 1] * 10)
    for index, target in enumerate(MRNET_TARGETS):
        frame[f"target_{target}"] = np.roll(labels, index)
        frame[f"logit_{target}"] = np.where(frame[f"target_{target}"] == 1, 2.0, -2.0)
    evaluated, results = evaluate_locked_mrnet_predictions(
        frame, _locks(), bootstrap_replicates=50
    )
    assert set(results) == set(MRNET_TARGETS)
    assert results["acl"]["point_estimates"]["roc_auc"] == 1.0
    assert results["acl"]["exam_grouped_bootstrap"]["sampling_unit"] == "exam_id"
    assert "predicted_meniscus" in evaluated


def test_locked_mrnet_evaluation_rejects_duplicate_exams():
    frame = pd.DataFrame({"exam_id": ["0001", "0001"]})
    for target in MRNET_TARGETS:
        frame[f"target_{target}"] = [0, 1]
        frame[f"logit_{target}"] = [-1.0, 1.0]
    try:
        evaluate_locked_mrnet_predictions(frame, _locks(), bootstrap_replicates=2)
    except ValueError as error:
        assert "unique exam" in str(error)
    else:
        raise AssertionError("duplicate exams were accepted")

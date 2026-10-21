"""Locked CXR evaluation applies frozen calibration by patient."""

import numpy as np
import pandas as pd

from stat_triage.cxr_evaluation import evaluate_locked_cxr_predictions
from stat_triage.datasets.nih_training import NIH_TARGET_ORDER


def test_locked_multilabel_evaluation_uses_declared_calibrators_and_thresholds():
    rows = {"image_id": [f"I{i}" for i in range(8)],
            "patient_id": [f"P{i}" for i in range(8)]}
    locked = {}
    base_targets = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    base_logits = np.array([-3, -2, -1, 0, 0.5, 1, 2, 3], dtype=float)
    for index, target in enumerate(NIH_TARGET_ORDER):
        rows[f"target_{target}"] = np.roll(base_targets, index)
        rows[f"logit_{target}"] = np.roll(base_logits, index)
        locked[target] = {
            "calibration": {"selected_method": "affine_logistic_scaling", "slope": 0.8, "intercept": -0.2},
            "operating_point": {"threshold": 0.45},
        }

    predictions, metrics = evaluate_locked_cxr_predictions(
        pd.DataFrame(rows), locked, bootstrap_replicates=100, bootstrap_seed=7
    )

    assert set(metrics) == set(NIH_TARGET_ORDER)
    for target in NIH_TARGET_ORDER:
        assert predictions[f"calibrated_score_{target}"].between(0, 1).all()
        assert set(predictions[f"predicted_{target}"].unique()) <= {0, 1}
        assert metrics[target]["patient_grouped_bootstrap"]["groups"] == 8

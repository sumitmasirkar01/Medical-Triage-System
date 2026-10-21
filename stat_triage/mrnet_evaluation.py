"""Locked exam-level evaluation for the Stanford MRNet vertical slice."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from .bootstrap import grouped_bootstrap_confidence_intervals
from .calibration import calibration_metrics, sigmoid
from .datasets.mrnet import MRNET_TARGETS
from .evaluation import binary_metrics


def evaluate_locked_mrnet_predictions(
    predictions: pd.DataFrame,
    locked_targets: Mapping[str, Mapping[str, object]],
    *,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply frozen calibrators and thresholds, then bootstrap complete exams."""
    required = {"exam_id"}
    for target in MRNET_TARGETS:
        required.update({f"target_{target}", f"logit_{target}"})
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing MRNet test prediction columns: {sorted(missing)}")
    if predictions.empty or predictions["exam_id"].duplicated().any():
        raise ValueError("MRNet predictions must have unique exam IDs")
    if set(locked_targets) != set(MRNET_TARGETS):
        raise ValueError("Locked targets do not match the MRNet contract")

    output = predictions.copy()
    results: dict[str, object] = {}
    for index, target in enumerate(MRNET_TARGETS):
        locked = locked_targets[target]
        calibration = locked["calibration"]
        operating = locked["operating_point"]
        if calibration.get("selected_method") != "affine_logistic_scaling":
            raise ValueError(f"Unsupported locked calibrator for {target}")
        scores = sigmoid(
            float(calibration["slope"]) * output[f"logit_{target}"].to_numpy()
            + float(calibration["intercept"])
        )
        threshold = float(operating["threshold"])
        output[f"calibrated_score_{target}"] = scores
        output[f"predicted_{target}"] = (scores >= threshold).astype("int8")
        point = binary_metrics(output[f"target_{target}"], scores, threshold)
        calibration_quality = calibration_metrics(output[f"target_{target}"], scores)
        bootstrap = grouped_bootstrap_confidence_intervals(
            output[f"target_{target}"], scores, output["exam_id"], threshold,
            replicates=bootstrap_replicates, seed=bootstrap_seed + index,
            sampling_unit="exam_id",
        )
        results[target] = {
            "threshold": threshold,
            "point_estimates": point.to_dict(),
            "calibration_metrics": calibration_quality.to_dict(),
            "exam_grouped_bootstrap": bootstrap.to_dict(),
        }
    return output, results

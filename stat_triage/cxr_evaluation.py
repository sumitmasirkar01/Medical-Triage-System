"""Locked multi-label evaluation for the NIH ChestX-ray14 vertical slice."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from .bootstrap import grouped_bootstrap_confidence_intervals
from .calibration import calibration_metrics, sigmoid
from .datasets.nih_training import NIH_TARGET_ORDER
from .evaluation import binary_metrics


def evaluate_locked_cxr_predictions(
    predictions: pd.DataFrame,
    locked_targets: Mapping[str, Mapping[str, object]],
    *,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply frozen affine calibrators and thresholds, then evaluate by patient."""

    required = {"image_id", "patient_id"}
    for target in NIH_TARGET_ORDER:
        required.update({f"target_{target}", f"logit_{target}"})
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing test prediction columns: {sorted(missing)}")
    if predictions.empty or predictions["image_id"].duplicated().any():
        raise ValueError("Test predictions must be non-empty with unique image IDs")
    if set(locked_targets) != set(NIH_TARGET_ORDER):
        raise ValueError("Locked targets do not match the NIH target contract")

    output = predictions.copy()
    results: dict[str, object] = {}
    for index, target in enumerate(NIH_TARGET_ORDER):
        locked = locked_targets[target]
        calibration = locked["calibration"]
        operating = locked["operating_point"]
        if calibration.get("selected_method") != "affine_logistic_scaling":
            raise ValueError(f"Unsupported locked calibrator for {target}")
        slope = float(calibration["slope"])
        intercept = float(calibration["intercept"])
        threshold = float(operating["threshold"])
        scores = sigmoid(
            slope * output[f"logit_{target}"].to_numpy() + intercept
        )
        output[f"calibrated_score_{target}"] = scores
        output[f"predicted_{target}"] = (scores >= threshold).astype("int8")
        point = binary_metrics(output[f"target_{target}"], scores, threshold)
        calibration_quality = calibration_metrics(
            output[f"target_{target}"], scores
        )
        bootstrap = grouped_bootstrap_confidence_intervals(
            output[f"target_{target}"],
            scores,
            output["patient_id"],
            threshold,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + index,
            sampling_unit="patient_id",
        )
        results[target] = {
            "threshold": threshold,
            "point_estimates": point.to_dict(),
            "calibration_metrics": calibration_quality.to_dict(),
            "patient_grouped_bootstrap": bootstrap.to_dict(),
        }
    return output, results

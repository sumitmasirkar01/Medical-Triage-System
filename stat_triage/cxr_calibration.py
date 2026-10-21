"""Held-out calibration and operating points for the NIH CXR endpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .calibration import fit_affine_logistic_scaling
from .datasets.nih_training import NIH_TARGET_ORDER
from .thresholds import select_sensitivity_threshold


DEFAULT_TARGET_SENSITIVITIES: dict[str, float] = {
    "pneumothorax": 0.90,
    "pleural_effusion": 0.85,
    "consolidation": 0.85,
    "atelectasis": 0.85,
}


@dataclass(frozen=True)
class CXRCalibrationResult:
    targets: dict[str, dict[str, object]]
    examples: int

    def to_dict(self) -> dict[str, object]:
        return {"examples": self.examples, "targets": self.targets}


def calibrate_cxr_predictions(
    predictions: pd.DataFrame,
    *,
    target_sensitivities: Mapping[str, float] = DEFAULT_TARGET_SENSITIVITIES,
) -> tuple[pd.DataFrame, CXRCalibrationResult]:
    """Fit per-label temperatures and thresholds on calibration predictions."""

    if predictions.empty:
        raise ValueError("Calibration predictions cannot be empty")
    if set(target_sensitivities) != set(NIH_TARGET_ORDER):
        raise ValueError("Sensitivity targets must exactly match the NIH target set")

    output = predictions.copy()
    results: dict[str, dict[str, object]] = {}
    for target in NIH_TARGET_ORDER:
        target_column = f"target_{target}"
        logit_column = f"logit_{target}"
        missing = {target_column, logit_column}.difference(output.columns)
        if missing:
            raise ValueError(f"Missing calibration columns for {target}: {sorted(missing)}")

        fitted = fit_affine_logistic_scaling(output[target_column], output[logit_column])
        calibrated = fitted.transform_logits(output[logit_column].to_numpy())
        calibrated_column = f"calibrated_score_{target}"
        output[calibrated_column] = calibrated
        operating_point = select_sensitivity_threshold(
            output[target_column], calibrated, target_sensitivities[target]
        )
        results[target] = {
            "calibration": fitted.to_dict(),
            "operating_point": operating_point.to_dict(),
        }

    return output, CXRCalibrationResult(results, len(output))

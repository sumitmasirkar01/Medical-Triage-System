"""Held-out probability calibration and operating points for MRNet targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from .calibration import fit_affine_logistic_scaling
from .datasets.mrnet import MRNET_TARGETS
from .thresholds import select_sensitivity_threshold


DEFAULT_MRNET_TARGET_SENSITIVITIES: dict[str, float] = {
    "abnormal": 0.90,
    "acl": 0.90,
    "meniscus": 0.85,
}


@dataclass(frozen=True)
class MRNetCalibrationResult:
    targets: dict[str, dict[str, object]]
    exams: int

    def to_dict(self) -> dict[str, object]:
        return {"exams": self.exams, "targets": self.targets}


def calibrate_mrnet_predictions(
    predictions: pd.DataFrame,
    *,
    target_sensitivities: Mapping[str, float] = DEFAULT_MRNET_TARGET_SENSITIVITIES,
) -> tuple[pd.DataFrame, MRNetCalibrationResult]:
    """Fit per-target affine logistic scalers and sensitivity thresholds."""
    if predictions.empty:
        raise ValueError("MRNet calibration predictions cannot be empty")
    if set(target_sensitivities) != set(MRNET_TARGETS):
        raise ValueError("Sensitivity targets must exactly match the MRNet targets")
    output = predictions.copy()
    results: dict[str, dict[str, object]] = {}
    for target in MRNET_TARGETS:
        target_column = f"target_{target}"
        logit_column = f"logit_{target}"
        missing = {target_column, logit_column}.difference(output.columns)
        if missing:
            raise ValueError(f"Missing calibration columns for {target}: {sorted(missing)}")
        fitted = fit_affine_logistic_scaling(output[target_column], output[logit_column])
        calibrated = fitted.transform_logits(output[logit_column].to_numpy())
        output[f"calibrated_score_{target}"] = calibrated
        operating_point = select_sensitivity_threshold(
            output[target_column], calibrated, target_sensitivities[target]
        )
        results[target] = {
            "calibration": fitted.to_dict(),
            "operating_point": operating_point.to_dict(),
        }
    return output, MRNetCalibrationResult(results, len(output))

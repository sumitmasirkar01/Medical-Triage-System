"""Post-test operating-point robustness for the locked Stanford MRNet model."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .bootstrap import grouped_bootstrap_confidence_intervals
from .datasets.mrnet import MRNET_TARGETS
from .evaluation import binary_metrics


class MRNetRobustnessError(ValueError):
    """MRNet robustness inputs violate the locked evaluation contract."""


def lock_threshold_perturbations(
    locked_calibration: Mapping[str, object],
    *,
    multipliers: Sequence[float] = (0.8, 0.9, 1.0, 1.1, 1.2),
) -> dict[str, object]:
    """Record a post-test exploratory grid before its mechanical evaluation."""

    if locked_calibration.get("status") != "locked":
        raise MRNetRobustnessError("Calibration artifact is not locked")
    if locked_calibration.get("test_partition_accessed") is not False:
        raise MRNetRobustnessError("Calibration artifact accessed the test partition")
    targets = locked_calibration.get("targets", {})
    if set(targets) != set(MRNET_TARGETS):
        raise MRNetRobustnessError("Locked targets do not match the MRNet contract")
    values = tuple(float(value) for value in multipliers)
    if (
        not values
        or not np.isfinite(values).all()
        or any(value <= 0 for value in values)
        or 1.0 not in values
        or tuple(sorted(set(values))) != values
    ):
        raise MRNetRobustnessError(
            "Multipliers must be unique, increasing, positive, finite, and include 1.0"
        )

    target_thresholds: dict[str, object] = {}
    for target in MRNET_TARGETS:
        locked_threshold = float(targets[target]["operating_point"]["threshold"])
        target_thresholds[target] = {
            "locked_threshold": locked_threshold,
            "perturbations": [
                {
                    "multiplier": multiplier,
                    "threshold": float(np.clip(locked_threshold * multiplier, 0, 1)),
                }
                for multiplier in values
            ],
        }
    return {
        "status": "locked",
        "analysis_status": "post_test_exploratory",
        "classification_reason": (
            "the perturbation analysis was designed after the locked MRNet test "
            "results were known; locking the grid before mechanical evaluation does "
            "not make the analysis prospective"
        ),
        "selection_partition": "calibration",
        "evaluation_partition": "test_not_loaded",
        "method": "multiplicative_threshold_perturbation",
        "multipliers": list(values),
        "targets": target_thresholds,
        "test_rows_present": 0,
        "test_evaluated": False,
    }


def evaluate_threshold_robustness(
    test_predictions: pd.DataFrame,
    perturbation_lock: Mapping[str, object],
    *,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 42,
) -> dict[str, object]:
    """Evaluate locked and perturbed thresholds on complete held-out exams."""

    required = {"exam_id"}
    for target in MRNET_TARGETS:
        required.update({f"target_{target}", f"calibrated_score_{target}"})
    if missing := required.difference(test_predictions.columns):
        raise MRNetRobustnessError(
            f"Test predictions missing columns: {sorted(missing)}"
        )
    if test_predictions.empty or test_predictions["exam_id"].duplicated().any():
        raise MRNetRobustnessError("Test predictions require unique exam IDs")
    if (
        perturbation_lock.get("status") != "locked"
        or perturbation_lock.get("test_evaluated") is not False
    ):
        raise MRNetRobustnessError("Threshold perturbations are not cleanly locked")
    if set(perturbation_lock.get("targets", {})) != set(MRNET_TARGETS):
        raise MRNetRobustnessError("Perturbation targets do not match MRNet")

    results: dict[str, object] = {}
    endpoint_index = 0
    for target in MRNET_TARGETS:
        target_lock = perturbation_lock["targets"][target]
        locked_threshold = float(target_lock["locked_threshold"])
        scores = test_predictions[f"calibrated_score_{target}"].to_numpy(dtype=float)
        labels = test_predictions[f"target_{target}"].to_numpy(dtype=int)
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise MRNetRobustnessError(f"Invalid calibrated scores for {target}")
        locked_decisions = scores >= locked_threshold
        profiles: list[dict[str, object]] = []
        for perturbation in target_lock["perturbations"]:
            threshold = float(perturbation["threshold"])
            decisions = scores >= threshold
            metrics = binary_metrics(labels, scores, threshold)
            bootstrap = grouped_bootstrap_confidence_intervals(
                labels,
                scores,
                test_predictions["exam_id"],
                threshold,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + endpoint_index,
                sampling_unit="exam_id",
            )
            profiles.append({
                "multiplier": float(perturbation["multiplier"]),
                "threshold": threshold,
                "changed_decisions_from_locked": int(
                    np.count_nonzero(decisions != locked_decisions)
                ),
                "changed_decision_fraction": float(
                    np.mean(decisions != locked_decisions)
                ),
                "metrics": metrics.to_dict(),
                "exam_grouped_bootstrap": bootstrap.to_dict(),
            })
            endpoint_index += 1
        results[target] = {
            "exams": int(len(test_predictions)),
            "positive_exams": int(labels.sum()),
            "negative_exams": int(len(labels) - labels.sum()),
            "locked_threshold": locked_threshold,
            "profiles": profiles,
        }

    return {
        "status": "complete",
        "analysis_status": "post_test_exploratory",
        "perturbation_lock": dict(perturbation_lock),
        "test_exams": int(len(test_predictions)),
        "sampling_unit": "exam_id",
        "patient_identifiers_available": False,
        "targets": results,
        "bootstrap_replicates": bootstrap_replicates,
        "multiplicity_adjustment": "none",
        "interpretation": (
            "operating-point sensitivity analysis; confidence intervals are wide and "
            "exam independence cannot be verified because patient identifiers are unavailable"
        ),
    }

"""Development-defined study-size robustness analysis for RSNA Phase 2 CT."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .bootstrap import grouped_bootstrap_confidence_intervals
from .evaluation import binary_metrics


class CTRobustnessError(ValueError):
    """Study-size robustness inputs are invalid or underpowered."""


def lock_slice_count_strata(calibration: pd.DataFrame) -> dict[str, object]:
    """Lock approximate tertile cut points using calibration studies only."""

    required = {"patient_id", "study_uid", "split", "slice_count"}
    if missing := required.difference(calibration.columns):
        raise CTRobustnessError(f"Calibration rows missing columns: {sorted(missing)}")
    if set(calibration["split"]) != {"calibration"}:
        raise CTRobustnessError("Stratum locking requires calibration studies only")
    counts = calibration["slice_count"].to_numpy(dtype=float)
    if not np.isfinite(counts).all() or np.any(counts < 1):
        raise CTRobustnessError("slice_count must contain positive finite values")
    lower, upper = np.quantile(counts, [1 / 3, 2 / 3], method="nearest")
    lower, upper = int(lower), int(upper)
    if lower >= upper:
        raise CTRobustnessError("Calibration slice counts cannot define three strata")
    return {
        "status": "locked", "selection_partition": "calibration",
        "evaluation_partition": "test_not_loaded", "method": "calibration_tertiles_nearest",
        "lower_cutpoint": lower, "upper_cutpoint": upper,
        "labels": [f"low_<=_{lower}", f"medium_{lower + 1}_to_{upper}", f"high_>_{upper}"],
        "calibration_studies": len(calibration), "test_rows_present": 0,
        "test_evaluated": False,
    }


def assign_slice_count_strata(slice_counts: pd.Series, lock: dict[str, object]) -> pd.Series:
    lower, upper = int(lock["lower_cutpoint"]), int(lock["upper_cutpoint"])
    labels = list(lock["labels"])
    return pd.cut(
        slice_counts, bins=[-math.inf, lower, upper, math.inf], labels=labels,
        right=True, ordered=True,
    ).astype(str)


def evaluate_locked_slice_count_strata(
    test: pd.DataFrame,
    lock: dict[str, object],
    threshold: float,
    *,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 42,
    minimum_group_size: int = 30,
    minimum_class_size: int = 10,
) -> dict[str, object]:
    """Evaluate locked test predictions within calibration-defined size strata."""

    required = {
        "patient_id", "study_uid", "split", "slice_count", "target_any", "score_any"
    }
    if missing := required.difference(test.columns):
        raise CTRobustnessError(f"Test rows missing columns: {sorted(missing)}")
    if set(test["split"]) != {"test"}:
        raise CTRobustnessError("Robustness evaluation requires test studies only")
    if lock.get("status") != "locked" or lock.get("test_evaluated") is not False:
        raise CTRobustnessError("Study-size strata are not cleanly locked")
    rows = test.copy()
    rows["slice_count_stratum"] = assign_slice_count_strata(rows["slice_count"], lock)
    results: dict[str, object] = {}
    for index, label in enumerate(lock["labels"]):
        group = rows[rows["slice_count_stratum"] == label]
        positives = int(group["target_any"].sum())
        negatives = len(group) - positives
        if len(group) < minimum_group_size or min(positives, negatives) < minimum_class_size:
            results[str(label)] = {
                "status": "underpowered", "studies": len(group),
                "positive_studies": positives, "negative_studies": negatives,
                "minimum_group_size": minimum_group_size,
                "minimum_class_size": minimum_class_size,
            }
            continue
        metrics = binary_metrics(group["target_any"], group["score_any"], threshold)
        bootstrap = grouped_bootstrap_confidence_intervals(
            group["target_any"], group["score_any"], group["patient_id"], threshold,
            replicates=bootstrap_replicates, seed=bootstrap_seed + index,
            sampling_unit="patient_id",
        )
        results[str(label)] = {
            "status": "reported_exploratory", "studies": len(group),
            "positive_studies": positives, "negative_studies": negatives,
            "slice_count": {
                "minimum": int(group["slice_count"].min()),
                "median": float(group["slice_count"].median()),
                "maximum": int(group["slice_count"].max()),
            },
            "metrics": metrics.to_dict(),
            "patient_grouped_bootstrap": bootstrap.to_dict(),
        }
    return {
        "status": "complete", "analysis_status": "exploratory_subgroup_analysis",
        "stratification": lock, "threshold": threshold, "test_studies": len(rows),
        "groups": results, "multiplicity_adjustment": "none",
        "interpretation": "descriptive robustness analysis; no causal or fairness claims",
    }

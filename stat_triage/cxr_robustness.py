"""Calibration-defined subgroup robustness analysis for NIH ChestX-ray14."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd

from .bootstrap import grouped_bootstrap_confidence_intervals
from .datasets.nih_training import NIH_TARGET_ORDER
from .evaluation import binary_metrics


class CXRRobustnessError(ValueError):
    """Chest X-ray subgroup robustness inputs violate the locked contract."""


def lock_cxr_subgroups(calibration_manifest: pd.DataFrame) -> dict[str, object]:
    """Lock view, recorded-sex, and age strata using calibration rows only."""

    required = {"image_id", "patient_id", "split", "view", "gender", "age"}
    if missing := required.difference(calibration_manifest.columns):
        raise CXRRobustnessError(
            f"Calibration manifest missing columns: {sorted(missing)}"
        )
    if set(calibration_manifest["split"]) != {"calibration"}:
        raise CXRRobustnessError("Subgroup locking requires calibration rows only")
    if calibration_manifest.empty or calibration_manifest["image_id"].duplicated().any():
        raise CXRRobustnessError(
            "Calibration manifest must be non-empty with unique image IDs"
        )

    ages = pd.to_numeric(calibration_manifest["age"], errors="coerce")
    valid_age = ages.between(0, 120, inclusive="both")
    if int(valid_age.sum()) < 3:
        raise CXRRobustnessError("Insufficient valid calibration ages")
    lower, upper = np.quantile(
        ages[valid_age].to_numpy(dtype=float), [1 / 3, 2 / 3], method="nearest"
    )
    lower, upper = int(lower), int(upper)
    if lower >= upper:
        raise CXRRobustnessError("Calibration ages cannot define three strata")

    return {
        "status": "locked",
        "selection_partition": "calibration",
        "evaluation_partition": "test_not_loaded",
        "definitions": {
            "view": {"column": "view", "groups": ["AP", "PA"]},
            "recorded_sex": {"column": "gender", "groups": ["F", "M"]},
            "age": {
                "column": "age",
                "method": "calibration_tertiles_nearest",
                "valid_range_years": [0, 120],
                "lower_cutpoint": lower,
                "upper_cutpoint": upper,
                "groups": [
                    f"young_<={lower}",
                    f"middle_{lower + 1}_to_{upper}",
                    f"older_>{upper}",
                ],
            },
        },
        "calibration_images": int(len(calibration_manifest)),
        "calibration_patients": int(calibration_manifest["patient_id"].nunique()),
        "calibration_images_with_invalid_age": int((~valid_age).sum()),
        "test_rows_present": 0,
        "test_evaluated": False,
    }


def _assign_age_group(ages: pd.Series, lock: Mapping[str, object]) -> pd.Series:
    definition = lock["definitions"]["age"]
    lower = int(definition["lower_cutpoint"])
    upper = int(definition["upper_cutpoint"])
    labels = list(definition["groups"])
    numeric = pd.to_numeric(ages, errors="coerce")
    valid = numeric.between(0, 120, inclusive="both")
    output = pd.Series(pd.NA, index=ages.index, dtype="object")
    output.loc[valid] = pd.cut(
        numeric.loc[valid],
        bins=[-math.inf, lower, upper, math.inf],
        labels=labels,
        right=True,
        ordered=True,
    ).astype(str)
    return output


def evaluate_locked_cxr_subgroups(
    test_manifest: pd.DataFrame,
    test_predictions: pd.DataFrame,
    lock: Mapping[str, object],
    locked_targets: Mapping[str, Mapping[str, object]],
    *,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 42,
    minimum_images: int = 100,
    minimum_positive_images: int = 30,
    minimum_negative_images: int = 30,
) -> dict[str, object]:
    """Evaluate frozen scores and thresholds in prespecified test subgroups."""

    manifest_required = {"image_id", "patient_id", "split", "view", "gender", "age"}
    if missing := manifest_required.difference(test_manifest.columns):
        raise CXRRobustnessError(f"Test manifest missing columns: {sorted(missing)}")
    if set(test_manifest["split"]) != {"test"}:
        raise CXRRobustnessError("Robustness evaluation requires test rows only")
    if lock.get("status") != "locked" or lock.get("test_evaluated") is not False:
        raise CXRRobustnessError("Subgroups are not cleanly locked")
    if set(locked_targets) != set(NIH_TARGET_ORDER):
        raise CXRRobustnessError("Locked targets do not match the NIH target contract")

    prediction_required = {"image_id", "patient_id"}
    for target in NIH_TARGET_ORDER:
        prediction_required.update(
            {f"target_{target}", f"calibrated_score_{target}"}
        )
    if missing := prediction_required.difference(test_predictions.columns):
        raise CXRRobustnessError(
            f"Test predictions missing columns: {sorted(missing)}"
        )
    for frame, name in ((test_manifest, "manifest"), (test_predictions, "predictions")):
        if frame.empty or frame["image_id"].duplicated().any():
            raise CXRRobustnessError(
                f"Test {name} must be non-empty with unique image IDs"
            )
    if set(test_manifest["image_id"]) != set(test_predictions["image_id"]):
        raise CXRRobustnessError("Test manifest and predictions contain different images")

    metadata = test_manifest[
        ["image_id", "patient_id", "view", "gender", "age"]
    ].rename(columns={"patient_id": "manifest_patient_id"})
    rows = test_predictions.merge(metadata, on="image_id", how="inner", validate="one_to_one")
    if not rows["patient_id"].astype(str).equals(
        rows["manifest_patient_id"].astype(str)
    ):
        raise CXRRobustnessError("Patient identifiers disagree between evidence files")
    rows["age_group"] = _assign_age_group(rows["age"], lock)

    dimensions = {
        "view": ("view", list(lock["definitions"]["view"]["groups"])),
        "recorded_sex": (
            "gender", list(lock["definitions"]["recorded_sex"]["groups"])
        ),
        "age": ("age_group", list(lock["definitions"]["age"]["groups"])),
    }
    results: dict[str, object] = {}
    endpoint_index = 0
    for dimension, (column, labels) in dimensions.items():
        dimension_results: dict[str, object] = {}
        for label in labels:
            group = rows[rows[column] == label]
            target_results: dict[str, object] = {}
            for target in NIH_TARGET_ORDER:
                positives = int(group[f"target_{target}"].sum())
                negatives = int(len(group) - positives)
                common = {
                    "images": int(len(group)),
                    "patients": int(group["patient_id"].nunique()),
                    "positive_images": positives,
                    "negative_images": negatives,
                }
                if (
                    len(group) < minimum_images
                    or positives < minimum_positive_images
                    or negatives < minimum_negative_images
                ):
                    target_results[target] = {
                        "status": "underpowered",
                        **common,
                        "reporting_requirements": {
                            "minimum_images": minimum_images,
                            "minimum_positive_images": minimum_positive_images,
                            "minimum_negative_images": minimum_negative_images,
                        },
                    }
                else:
                    threshold = float(
                        locked_targets[target]["operating_point"]["threshold"]
                    )
                    metrics = binary_metrics(
                        group[f"target_{target}"],
                        group[f"calibrated_score_{target}"],
                        threshold,
                    )
                    bootstrap = grouped_bootstrap_confidence_intervals(
                        group[f"target_{target}"],
                        group[f"calibrated_score_{target}"],
                        group["patient_id"],
                        threshold,
                        replicates=bootstrap_replicates,
                        seed=bootstrap_seed + endpoint_index,
                        sampling_unit="patient_id",
                    )
                    target_results[target] = {
                        "status": "reported_exploratory",
                        **common,
                        "threshold": threshold,
                        "metrics": metrics.to_dict(),
                        "patient_grouped_bootstrap": bootstrap.to_dict(),
                    }
                endpoint_index += 1
            dimension_results[str(label)] = {"targets": target_results}
        results[dimension] = dimension_results

    return {
        "status": "complete",
        "analysis_status": "exploratory_subgroup_analysis",
        "subgroup_lock": dict(lock),
        "test_images": int(len(rows)),
        "test_patients": int(rows["patient_id"].nunique()),
        "test_images_excluded_from_age_analysis": int(rows["age_group"].isna().sum()),
        "dimensions": results,
        "bootstrap_replicates": bootstrap_replicates,
        "multiplicity_adjustment": "none",
        "interpretation": (
            "descriptive robustness analysis; subgroup comparisons are correlated and "
            "support no causal, biological, or fairness claims"
        ),
    }

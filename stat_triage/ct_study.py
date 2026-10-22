"""Leakage-safe study-level aggregation for ordered head CT slice predictions."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .bootstrap import grouped_bootstrap_confidence_intervals
from .evaluation import binary_metrics
from .thresholds import select_sensitivity_threshold


POOLING_METHODS = ("max", "mean", "top_k_mean", "noisy_or")


class StudyAggregationError(ValueError):
    """Slice predictions cannot support a valid study-level experiment."""


def _pool(values: np.ndarray, method: str, top_k: int) -> float:
    if method == "max":
        return float(values.max())
    if method == "mean":
        return float(values.mean())
    if method == "top_k_mean":
        return float(np.sort(values)[-min(top_k, len(values)):].mean())
    if method == "noisy_or":
        return float(1.0 - np.prod(1.0 - values))
    raise StudyAggregationError(f"Unsupported pooling method: {method}")


def aggregate_study_predictions(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    method: str,
    top_k: int = 3,
    minimum_median_slices: int = 10,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Join slice predictions to provenance and produce one row per study."""

    prediction_columns = {"image_id", "score_any", "target_any"}
    manifest_columns = {"image_id", "patient_id", "study_uid", "z_position", "split"}
    if missing := prediction_columns.difference(predictions.columns):
        raise StudyAggregationError(f"Predictions missing columns: {sorted(missing)}")
    if missing := manifest_columns.difference(manifest.columns):
        raise StudyAggregationError(f"Manifest missing columns: {sorted(missing)}")
    if method not in POOLING_METHODS:
        raise StudyAggregationError(f"Unsupported pooling method: {method}")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise StudyAggregationError("top_k must be a positive integer")
    if predictions["image_id"].duplicated().any() or manifest["image_id"].duplicated().any():
        raise StudyAggregationError("image_id must be unique in predictions and manifest")
    rows = predictions[list(prediction_columns)].merge(
        manifest[list(manifest_columns)], on="image_id", how="left", validate="one_to_one"
    )
    if rows[list(manifest_columns - {"image_id"})].isna().any().any():
        raise StudyAggregationError("Some prediction rows lack manifest provenance")
    if not np.isin(rows["target_any"], (0, 1)).all():
        raise StudyAggregationError("target_any must be binary")
    scores = rows["score_any"].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise StudyAggregationError("score_any must be finite and in [0, 1]")
    consistency = rows.groupby("study_uid").agg(
        patient_count=("patient_id", "nunique"), split_count=("split", "nunique")
    )
    if (consistency[["patient_count", "split_count"]] != 1).any().any():
        raise StudyAggregationError("A study cannot span patients or splits")
    counts = rows.groupby("study_uid").size()
    median_slices = float(counts.median())
    if median_slices < minimum_median_slices:
        raise StudyAggregationError(
            f"Sparse study coverage: median {median_slices:g} slices; "
            f"requires at least {minimum_median_slices}"
        )
    rows = rows.sort_values(["study_uid", "z_position", "image_id"], kind="stable")
    aggregated = []
    for study_uid, group in rows.groupby("study_uid", sort=False):
        aggregated.append(
            {
                "study_uid": study_uid,
                "patient_id": group["patient_id"].iloc[0],
                "split": group["split"].iloc[0],
                "slice_count": len(group),
                "target_any": int(group["target_any"].max()),
                "score_any": _pool(group["score_any"].to_numpy(float), method, top_k),
            }
        )
    output = pd.DataFrame(aggregated)
    audit = {
        "method": method,
        "top_k": top_k if method == "top_k_mean" else None,
        "slices": len(rows),
        "studies": len(output),
        "patients": int(output["patient_id"].nunique()),
        "median_slices_per_study": median_slices,
        "minimum_slices_per_study": int(counts.min()),
        "maximum_slices_per_study": int(counts.max()),
    }
    return output, audit


def evaluate_study_pooling_protocol(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    methods: Iterable[str] = POOLING_METHODS,
    top_k: int = 3,
    target_sensitivity: float = 0.9,
    minimum_median_slices: int = 10,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 42,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Select pooling on validation, threshold on calibration, then test once."""

    candidates: dict[str, tuple[pd.DataFrame, dict[str, object], float]] = {}
    for method in methods:
        studies, audit = aggregate_study_predictions(
            predictions, manifest, method=method, top_k=top_k,
            minimum_median_slices=minimum_median_slices,
        )
        validation = studies[studies["split"] == "validation"]
        if set(validation["target_any"]) != {0, 1}:
            raise StudyAggregationError("Validation studies require both target classes")
        auc = float(roc_auc_score(validation["target_any"], validation["score_any"]))
        candidates[method] = (studies, audit, auc)
    selected_method = sorted(candidates, key=lambda name: (-candidates[name][2], name))[0]
    studies, audit, validation_auc = candidates[selected_method]
    calibration = studies[studies["split"] == "calibration"]
    test = studies[studies["split"] == "test"].copy()
    if set(calibration["target_any"]) != {0, 1} or set(test["target_any"]) != {0, 1}:
        raise StudyAggregationError("Calibration and test studies require both target classes")
    operating = select_sensitivity_threshold(
        calibration["target_any"], calibration["score_any"], target_sensitivity
    )
    test["predicted_any"] = (test["score_any"] >= operating.threshold).astype(int)
    metrics = binary_metrics(test["target_any"], test["score_any"], operating.threshold)
    bootstrap = grouped_bootstrap_confidence_intervals(
        test["target_any"], test["score_any"], test["patient_id"], operating.threshold,
        replicates=bootstrap_replicates, seed=bootstrap_seed, sampling_unit="patient_id",
    )
    report = {
        "endpoint": "study_level_any_hemorrhage",
        "selection_partition": "validation",
        "threshold_partition": "calibration",
        "evaluation_partition": "test",
        "selected_method": selected_method,
        "validation_auc_by_method": {name: value[2] for name, value in candidates.items()},
        "validation_auc": validation_auc,
        "operating_point": operating.to_dict(),
        "test_metrics": metrics.to_dict(),
        "patient_grouped_bootstrap": bootstrap.to_dict(),
        "coverage": audit,
        "test_not_used_for_selection": True,
    }
    return report, test


def lock_study_pooling_protocol(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    methods: Iterable[str] = POOLING_METHODS,
    top_k: int = 3,
    target_sensitivity: float = 0.9,
    minimum_median_slices: int = 10,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Select pooling and threshold using development partitions only."""

    prediction_splits = set(predictions["split"]) if "split" in predictions else set()
    manifest_splits = set(manifest["split"]) if "split" in manifest else set()
    expected = {"validation", "calibration"}
    if prediction_splits != expected or manifest_splits != expected:
        raise StudyAggregationError(
            "Protocol locking requires exactly validation and calibration partitions"
        )
    candidates: dict[str, tuple[pd.DataFrame, dict[str, object], float]] = {}
    for method in methods:
        studies, audit = aggregate_study_predictions(
            predictions, manifest, method=method, top_k=top_k,
            minimum_median_slices=minimum_median_slices,
        )
        validation = studies[studies["split"] == "validation"]
        if set(validation["target_any"]) != {0, 1}:
            raise StudyAggregationError("Validation studies require both target classes")
        auc = float(roc_auc_score(validation["target_any"], validation["score_any"]))
        candidates[method] = (studies, audit, auc)
    selected_method = sorted(candidates, key=lambda name: (-candidates[name][2], name))[0]
    studies, audit, validation_auc = candidates[selected_method]
    calibration = studies[studies["split"] == "calibration"].copy()
    if set(calibration["target_any"]) != {0, 1}:
        raise StudyAggregationError("Calibration studies require both target classes")
    operating = select_sensitivity_threshold(
        calibration["target_any"], calibration["score_any"], target_sensitivity
    )
    calibration["predicted_any"] = (
        calibration["score_any"] >= operating.threshold
    ).astype(int)
    report = {
        "status": "locked", "endpoint": "study_level_any_hemorrhage",
        "selection_partition": "validation", "threshold_partition": "calibration",
        "evaluation_partition": "test_not_loaded", "selected_method": selected_method,
        "top_k": top_k if selected_method == "top_k_mean" else None,
        "validation_auc_by_method": {
            name: value[2] for name, value in candidates.items()
        },
        "validation_auc": validation_auc, "operating_point": operating.to_dict(),
        "coverage": audit, "development_slices": len(predictions),
        "development_studies": len(studies), "test_rows_present": 0,
        "test_evaluated": False,
    }
    return report, calibration.reset_index(drop=True)

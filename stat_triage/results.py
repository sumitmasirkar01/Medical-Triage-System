"""Consolidate locked multimodal metrics without comparing unlike evaluation units."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


RESULT_COLUMNS = (
    "modality", "dataset", "evaluation_level", "target", "test_units",
    "test_patients", "positive_units", "negative_units", "prevalence",
    "threshold", "roc_auc", "roc_auc_ci_lower", "roc_auc_ci_upper",
    "average_precision", "average_precision_ci_lower", "average_precision_ci_upper",
    "sensitivity", "sensitivity_ci_lower", "sensitivity_ci_upper",
    "specificity", "specificity_ci_lower", "specificity_ci_upper",
    "ppv", "ppv_ci_lower", "ppv_ci_upper", "npv", "npv_ci_lower",
    "npv_ci_upper", "accuracy", "accuracy_ci_lower", "accuracy_ci_upper",
    "calibration_nll", "calibration_brier", "calibration_ece",
    "mean_inference_ms_per_unit", "bootstrap_sampling_unit",
    "bootstrap_replicates", "checkpoint_sha256",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _interval(intervals: dict[str, Any], metric: str) -> tuple[float | None, float | None]:
    value = intervals.get(metric)
    if not isinstance(value, dict):
        return None, None
    return value.get("lower"), value.get("upper")


def _row(
    *,
    modality: str,
    dataset: str,
    evaluation_level: str,
    target: str,
    test_units: int,
    test_patients: int | None,
    threshold: float,
    point: dict[str, Any],
    intervals: dict[str, Any],
    calibration: dict[str, Any] | None,
    mean_ms: float,
    sampling_unit: str,
    replicates: int,
    checkpoint: str,
) -> dict[str, Any]:
    positives = int(point["positive_images"])
    negatives = int(point["negative_images"])
    row: dict[str, Any] = {
        "modality": modality, "dataset": dataset,
        "evaluation_level": evaluation_level, "target": target,
        "test_units": test_units, "test_patients": test_patients,
        "positive_units": positives, "negative_units": negatives,
        "prevalence": positives / (positives + negatives), "threshold": threshold,
        "calibration_nll": calibration.get("negative_log_likelihood") if calibration else None,
        "calibration_brier": calibration.get("brier_score") if calibration else None,
        "calibration_ece": calibration.get("expected_calibration_error") if calibration else None,
        "mean_inference_ms_per_unit": mean_ms,
        "bootstrap_sampling_unit": sampling_unit,
        "bootstrap_replicates": replicates,
        "checkpoint_sha256": checkpoint,
    }
    for metric in (
        "roc_auc", "average_precision", "sensitivity", "specificity",
        "ppv", "npv", "accuracy",
    ):
        lower, upper = _interval(intervals, metric)
        row[metric] = point[metric]
        row[f"{metric}_ci_lower"] = lower
        row[f"{metric}_ci_upper"] = upper
    return {column: row.get(column) for column in RESULT_COLUMNS}


def build_consolidated_results(project_root: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return long-form locked results and a structured provenance summary."""
    root = Path(project_root).resolve()
    rsna_dir = root / "artifacts/rsna_baseline/rsna_baseline_bundle"
    cxr_dir = root / "artifacts/chest_xray/nih_cxr14_locked_test_bundle"
    mri_dir = root / "artifacts/mrnet_baseline"
    rsna = _read(rsna_dir / "test_metrics.json")
    cxr = _read(cxr_dir / "nih_densenet_test/test_metrics.json")
    mri = _read(mri_dir / "mrnet_test/test_metrics.json")
    rows = [
        _row(
            modality="head_ct", dataset="RSNA ICH 2019", evaluation_level="slice",
            target="any", test_units=rsna["test_images"],
            test_patients=rsna["test_patients"], threshold=rsna["threshold"],
            point=rsna["primary_metrics"], intervals=rsna["patient_bootstrap_95_ci"],
            calibration=None,
            mean_ms=1000 * rsna["timing"]["mean_seconds_per_image_at_batch_size"],
            sampling_unit="patient_id", replicates=rsna["bootstrap_replicates_requested"],
            checkpoint=rsna["model_checkpoint_sha256"],
        )
    ]
    for target, result in cxr["targets"].items():
        bootstrap = result["patient_grouped_bootstrap"]
        rows.append(
            _row(
                modality="chest_xray", dataset="NIH ChestX-ray14",
                evaluation_level="image", target=target,
                test_units=cxr["test_images"], test_patients=cxr["test_patients"],
                threshold=result["threshold"], point=result["point_estimates"],
                intervals=bootstrap["intervals"], calibration=result["calibration_metrics"],
                mean_ms=cxr["mean_inference_ms_per_image"],
                sampling_unit=bootstrap["sampling_unit"],
                replicates=bootstrap["requested_replicates"],
                checkpoint=cxr["checkpoint_sha256"],
            )
        )
    for target, result in mri["targets"].items():
        bootstrap = result["exam_grouped_bootstrap"]
        rows.append(
            _row(
                modality="knee_mri", dataset="Stanford MRNet",
                evaluation_level="exam", target=target,
                test_units=mri["test_exams"], test_patients=None,
                threshold=result["threshold"], point=result["point_estimates"],
                intervals=bootstrap["intervals"], calibration=result["calibration_metrics"],
                mean_ms=mri["mean_inference_ms_per_exam"],
                sampling_unit=bootstrap["sampling_unit"],
                replicates=bootstrap["requested_replicates"],
                checkpoint=mri["checkpoint_sha256"],
            )
        )
    frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    summary = {
        "schema_version": "1.0",
        "status": "locked_results_consolidated",
        "rows": len(frame),
        "modalities": sorted(frame["modality"].unique()),
        "comparison_warning": (
            "Metrics are not direct cross-modality comparisons: Head CT is slice-level, "
            "Chest X-ray is image-level, and Knee MRI is exam-level, with different "
            "datasets, prevalences, labels, and operating policies."
        ),
        "metric_definitions": {
            "roc_auc": "threshold-independent discrimination",
            "average_precision": "prevalence-sensitive precision-recall summary",
            "sensitivity_specificity_ppv_npv_accuracy": "locked-threshold point estimates",
            "calibration_nll_brier_ece": "test-set calibration diagnostics; absent for CT",
            "confidence_intervals": "95% percentile grouped bootstrap intervals",
        },
        "evaluation_units": {
            "head_ct": "slice", "chest_xray": "image", "knee_mri": "exam"
        },
        "patient_independence": {
            "head_ct": "patient-grouped",
            "chest_xray": "patient-grouped",
            "knee_mri": "patient identifiers unavailable; exam-grouped",
        },
        "records": frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records"),
    }
    return frame, summary


def render_results_markdown(frame: pd.DataFrame) -> str:
    """Render a compact human-readable table without implying unit equivalence."""
    lines = [
        "# Locked multimodal results",
        "",
        "> Results use different evaluation units and cohorts and are not direct model comparisons.",
        "",
        "| Modality | Unit | Target | N | ROC-AUC (95% CI) | AP | Sensitivity | Specificity |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in frame.to_dict(orient="records"):
        auc = f"{row['roc_auc']:.3f} ({row['roc_auc_ci_lower']:.3f}–{row['roc_auc_ci_upper']:.3f})"
        lines.append(
            f"| {row['modality']} | {row['evaluation_level']} | {row['target']} | "
            f"{row['test_units']} | {auc} | {row['average_precision']:.3f} | "
            f"{row['sensitivity']:.3f} | {row['specificity']:.3f} |"
        )
    return "\n".join(lines) + "\n"

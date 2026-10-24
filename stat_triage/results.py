"""Consolidate locked multimodal metrics without comparing unlike evaluation units."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd


RESULT_COLUMNS = (
    "modality", "dataset", "evaluation_level", "evidence_classification",
    "target", "test_units",
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


def _read_zip_json(path: Path, member: str) -> dict[str, Any]:
    with ZipFile(path) as archive:
        matches = [name for name in archive.namelist() if name == member]
        if len(matches) != 1:
            raise ValueError(f"Evidence bundle must contain exactly one {member}")
        value = json.loads(archive.read(member))
    if not isinstance(value, dict):
        raise ValueError(f"{member} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


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
    evidence_classification: str,
    target: str,
    test_units: int,
    test_patients: int | None,
    threshold: float,
    point: dict[str, Any],
    intervals: dict[str, Any],
    calibration: dict[str, Any] | None,
    mean_ms: float | None,
    sampling_unit: str,
    replicates: int,
    checkpoint: str,
) -> dict[str, Any]:
    positives = int(point["positive_images"])
    negatives = int(point["negative_images"])
    row: dict[str, Any] = {
        "modality": modality, "dataset": dataset,
        "evaluation_level": evaluation_level,
        "evidence_classification": evidence_classification,
        "target": target,
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
    phase2_bundle = root / "artifacts/rsna_phase2/rsna_phase2_final_evidence_bundle.zip"
    phase2 = _read_zip_json(phase2_bundle, "study_test_metrics.json")
    phase2_export = _read_zip_json(
        phase2_bundle, "rsna_phase2_prediction_export.json"
    )
    rows = [
        _row(
            modality="head_ct", dataset="RSNA ICH 2019", evaluation_level="slice",
            evidence_classification="retrospective_locked_baseline",
            target="any", test_units=rsna["test_images"],
            test_patients=rsna["test_patients"], threshold=rsna["threshold"],
            point=rsna["primary_metrics"], intervals=rsna["patient_bootstrap_95_ci"],
            calibration=None,
            mean_ms=1000 * rsna["timing"]["mean_seconds_per_image_at_batch_size"],
            sampling_unit="patient_id", replicates=rsna["bootstrap_replicates_requested"],
            checkpoint=rsna["model_checkpoint_sha256"],
        ),
        _row(
            modality="head_ct",
            dataset="RSNA ICH 2019 Phase 2",
            evaluation_level="study",
            evidence_classification="confirmatory_prespecified_locked",
            target="any",
            test_units=phase2["coverage"]["studies"],
            test_patients=phase2["coverage"]["patients"],
            threshold=phase2["threshold"],
            point=phase2["test_metrics"],
            intervals=phase2["patient_grouped_bootstrap"]["intervals"],
            calibration=None,
            mean_ms=None,
            sampling_unit=phase2["patient_grouped_bootstrap"]["sampling_unit"],
            replicates=phase2["patient_grouped_bootstrap"]["requested_replicates"],
            checkpoint=phase2_export["checkpoint_sha256"],
        ),
    ]
    for target, result in cxr["targets"].items():
        bootstrap = result["patient_grouped_bootstrap"]
        rows.append(
            _row(
                modality="chest_xray", dataset="NIH ChestX-ray14",
                evaluation_level="image", target=target,
                evidence_classification="retrospective_locked_baseline",
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
                evidence_classification="retrospective_locked_baseline",
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
            "Metrics are not direct comparisons: rows use different cohorts, evaluation "
            "units, prevalences, labels, and operating policies. The CT slice pilot and "
            "CT study evaluation are also distinct evidence sets."
        ),
        "metric_definitions": {
            "roc_auc": "threshold-independent discrimination",
            "average_precision": "prevalence-sensitive precision-recall summary",
            "sensitivity_specificity_ppv_npv_accuracy": "locked-threshold point estimates",
            "calibration_nll_brier_ece": "test-set calibration diagnostics; absent for CT",
            "confidence_intervals": "95% percentile grouped bootstrap intervals",
        },
        "evaluation_units": {
            "head_ct": ["slice", "study"],
            "chest_xray": ["image"],
            "knee_mri": ["exam"],
        },
        "patient_independence": {
            "head_ct": "patient-grouped",
            "chest_xray": "patient-grouped",
            "knee_mri": "patient identifiers unavailable; exam-grouped",
        },
        "evidence_hierarchy": {
            "confirmatory_prespecified_locked": (
                "CT Phase 2 study-level evaluation specified before final test evaluation"
            ),
            "retrospective_locked_baseline": (
                "baseline evaluation archived before the formal protocol was frozen"
            ),
        },
        "source_artifacts": {
            "head_ct_slice_metrics": _source(root, rsna_dir / "test_metrics.json"),
            "head_ct_study_bundle": _source(root, phase2_bundle),
            "chest_xray_metrics": _source(
                root, cxr_dir / "nih_densenet_test/test_metrics.json"
            ),
            "knee_mri_metrics": _source(root, mri_dir / "mrnet_test/test_metrics.json"),
        },
        "records": frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records"),
    }
    return frame, summary


def render_results_markdown(frame: pd.DataFrame) -> str:
    """Render a compact human-readable table without implying unit equivalence."""
    lines = [
        "# Locked multimodal results",
        "",
        "> Results use different evaluation units and cohorts and are not direct model comparisons. Evidence classification follows the Phase 2 protocol amendment.",
        "",
        "| Modality | Unit | Evidence | Target | N | ROC-AUC (95% CI) | AP | Sensitivity | Specificity |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in frame.to_dict(orient="records"):
        auc = f"{row['roc_auc']:.3f} ({row['roc_auc_ci_lower']:.3f}–{row['roc_auc_ci_upper']:.3f})"
        lines.append(
            f"| {row['modality']} | {row['evaluation_level']} | "
            f"{row['evidence_classification']} | {row['target']} | "
            f"{row['test_units']} | {auc} | {row['average_precision']:.3f} | "
            f"{row['sensitivity']:.3f} | {row['specificity']:.3f} |"
        )
    return "\n".join(lines) + "\n"

"""End-to-end validation of a supported model release and its evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .bootstrap import grouped_bootstrap_confidence_intervals
from .calibration import fit_temperature_scaling
from .evaluation import (
    EvidenceBundleError,
    evaluate_locked_predictions,
    sha256,
    validate_evidence_bundle,
)
from .rsna import RSNA_LABELS
from .thresholds import select_sensitivity_threshold


class ReleaseValidationError(ValueError):
    """A release claim cannot be reproduced from its declared evidence."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseValidationError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseValidationError(f"{path} must contain a JSON object")
    return value


def _safe_project_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ReleaseValidationError(f"Path escapes project root: {relative}") from exc
    return candidate


def _validate_artifact_declaration(
    root: Path, declaration: object, label: str
) -> Path:
    if not isinstance(declaration, dict):
        raise ReleaseValidationError(f"Invalid artifact declaration: {label}")
    artifact = _safe_project_path(root, str(declaration.get("path", "")))
    if (
        not artifact.is_file()
        or artifact.stat().st_size != declaration.get("bytes")
        or sha256(artifact) != declaration.get("sha256")
    ):
        raise ReleaseValidationError(
            f"Artifact size or SHA-256 mismatch for {label}: {artifact}"
        )
    return artifact


def validate_phase2_release_artifacts(
    manifest: dict[str, Any], project_root: str | Path
) -> dict[str, object]:
    """Verify every Phase 2 declaration, lock/report pair, and source provenance."""

    root = Path(project_root).resolve()
    modalities = manifest.get("modalities")
    if not isinstance(modalities, dict):
        raise ReleaseValidationError("release manifest has no modality declarations")

    head = modalities.get("head_ct", {}).get("phase_2_study_level", {})
    for name in ("evidence_registry", "validation_report"):
        _validate_artifact_declaration(root, head.get(name), f"head_ct.{name}")
    bundles = head.get("bundles")
    if not isinstance(bundles, dict) or set(bundles) != {
        "cohort", "development", "locked_protocol", "test_slices", "final_evaluation"
    }:
        raise ReleaseValidationError("head CT Phase 2 bundle declarations are incomplete")
    for name, declaration in bundles.items():
        _validate_artifact_declaration(root, declaration, f"head_ct.bundle.{name}")

    lock_pairs = (
        (
            "head_ct.selective_prediction",
            head.get("selective_prediction", {}),
            "policy", "report", None,
        ),
        (
            "head_ct.slice_count_robustness",
            head.get("slice_count_robustness", {}),
            "strata_lock", "report", "stratification",
        ),
        (
            "chest_xray.subgroup_robustness",
            modalities.get("chest_xray", {}).get("subgroup_robustness", {}),
            "subgroup_lock", "report", "subgroup_lock",
        ),
        (
            "knee_mri.threshold_robustness",
            modalities.get("knee_mri", {}).get("threshold_robustness", {}),
            "perturbation_lock", "report", "perturbation_lock",
        ),
    )
    verified: dict[str, object] = {}
    for label, section, lock_key, report_key, embedded_key in lock_pairs:
        if not isinstance(section, dict):
            raise ReleaseValidationError(f"Missing Phase 2 section: {label}")
        lock_path = _validate_artifact_declaration(
            root, section.get(lock_key), f"{label}.{lock_key}"
        )
        report_path = _validate_artifact_declaration(
            root, section.get(report_key), f"{label}.{report_key}"
        )
        locked = _read_json(lock_path)
        report = _read_json(report_path)
        if embedded_key is None:
            if report.get("provenance", {}).get("selective_policy_sha256") != sha256(
                lock_path
            ):
                raise ReleaseValidationError(f"Lock/report mismatch for {label}")
        elif report.get(embedded_key) != locked:
            raise ReleaseValidationError(f"Lock/report mismatch for {label}")
        if report.get("analysis_status") != section.get("analysis_status"):
            raise ReleaseValidationError(f"Analysis-status mismatch for {label}")
        verified[label] = {
            "analysis_status": report.get("analysis_status"),
            "lock_sha256": sha256(lock_path),
            "report_sha256": sha256(report_path),
        }

    cxr_report = _read_json(
        root / "artifacts/chest_xray/robustness_analysis/subgroup_robustness.json"
    )
    cxr_sources = {
        "manifest_sha256": root / "artifacts/chest_xray/nih_cxr14_locked_test_bundle/nih_cxr14_manifest_split.csv",
        "locked_calibration_sha256": root / "artifacts/chest_xray/nih_cxr14_locked_test_bundle/nih_densenet_calibration/locked_calibration.json",
        "test_predictions_sha256": root / "artifacts/chest_xray/nih_cxr14_locked_test_bundle/nih_densenet_test/test_predictions.csv",
    }
    mrnet_report = _read_json(
        root / "artifacts/mrnet_robustness/threshold_robustness.json"
    )
    mrnet_sources = {
        "locked_calibration_sha256": root / "artifacts/mrnet_baseline/mrnet_calibration/locked_calibration.json",
        "test_predictions_sha256": root / "artifacts/mrnet_baseline/mrnet_test/test_predictions.csv",
    }
    for label, report, sources in (
        ("chest_xray", cxr_report, cxr_sources),
        ("knee_mri", mrnet_report, mrnet_sources),
    ):
        provenance = report.get("provenance", {})
        for key, source in sources.items():
            if provenance.get(key) != sha256(source):
                raise ReleaseValidationError(
                    f"Source provenance mismatch for {label}.{key}"
                )
    return {"status": "passed", "verified_analyses": verified}


def validate_project_release_manifest(
    release_manifest_path: str | Path,
    project_root: str | Path,
) -> dict[str, object]:
    """Verify the generated release manifest's source and report hashes."""

    path = Path(release_manifest_path).resolve()
    root = Path(project_root).resolve()
    manifest = _read_json(path)
    deliverables = manifest.get("deliverables")
    if not isinstance(deliverables, dict):
        raise ReleaseValidationError("release manifest has no deliverables")
    source_hashes = deliverables.get("supported_source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ReleaseValidationError("release manifest has no supported source registry")
    for relative, expected_hash in source_hashes.items():
        source = _safe_project_path(root, relative)
        if not source.is_file():
            raise ReleaseValidationError(f"Missing supported source: {relative}")
        if sha256(source) != expected_hash:
            raise ReleaseValidationError(f"Supported source hash mismatch: {relative}")

    report = deliverables.get("report")
    if not isinstance(report, dict):
        raise ReleaseValidationError("release manifest has no report declaration")
    report_path = _safe_project_path(root, report.get("path", ""))
    if not report_path.is_file():
        raise ReleaseValidationError(f"Missing declared report: {report_path}")
    if report_path.stat().st_size != report.get("bytes") or sha256(report_path) != report.get("sha256"):
        raise ReleaseValidationError("Declared report size or SHA-256 does not match")

    multimodal = manifest.get("multimodal_evidence")
    if not isinstance(multimodal, dict):
        raise ReleaseValidationError("release manifest has no multimodal evidence declaration")
    declarations = [
        multimodal.get("registry"),
        multimodal.get("validation_report"),
        multimodal.get("smoke_report"),
        *(multimodal.get("bundle_manifests") or {}).values(),
        *(multimodal.get("consolidated_results") or {}).values(),
        *(multimodal.get("paper_assets") or {}).values(),
    ]
    for declaration in declarations:
        if not isinstance(declaration, dict):
            raise ReleaseValidationError("invalid multimodal artifact declaration")
        artifact = _safe_project_path(root, declaration.get("path", ""))
        if (
            not artifact.is_file()
            or artifact.stat().st_size != declaration.get("bytes")
            or sha256(artifact) != declaration.get("sha256")
        ):
            raise ReleaseValidationError(
                f"Multimodal artifact size or SHA-256 mismatch: {artifact}"
            )
    modality_checkpoints = {
        name: values.get("checkpoint_sha256")
        for name, values in (manifest.get("modalities") or {}).items()
    }
    if set(modality_checkpoints) != {"head_ct", "chest_xray", "knee_mri"}:
        raise ReleaseValidationError("release manifest does not declare three modalities")
    phase2 = validate_phase2_release_artifacts(manifest, root)

    return {
        "name": manifest.get("release", {}).get("name"),
        "version": manifest.get("release", {}).get("version"),
        "status": manifest.get("release", {}).get("status"),
        "supported_source_files": len(source_hashes),
        "report_sha256": report["sha256"],
        "expected_tests": manifest.get("verification", {}).get("expected_tests"),
        "checkpoint_sha256": manifest.get("supported_scope", {}).get("checkpoint_sha256"),
        "schema_version": manifest.get("schema_version"),
        "modality_checkpoint_sha256": modality_checkpoints,
        "phase2_artifact_validation": phase2,
    }


def _require_columns(rows: pd.DataFrame, required: set[str], filename: str) -> None:
    missing = required.difference(rows.columns)
    if missing:
        raise ReleaseValidationError(
            f"{filename} is missing columns: {', '.join(sorted(missing))}"
        )


def _require_metric_agreement(actual: dict[str, float | int], expected: dict[str, Any]) -> None:
    if set(actual) != set(expected):
        raise ReleaseValidationError("Saved and recomputed primary metric names disagree")
    for name, value in actual.items():
        saved = expected[name]
        if isinstance(value, int):
            agrees = value == saved
        else:
            agrees = math.isclose(float(value), float(saved), rel_tol=0.0, abs_tol=1e-12)
        if not agrees:
            raise ReleaseValidationError(
                f"Locked metric mismatch for {name}: recomputed={value}, saved={saved}"
            )


def evaluate_rsna_release(
    bundle_dir: str | Path,
    *,
    release_manifest_path: str | Path | None = None,
    project_root: str | Path | None = None,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 42,
) -> dict[str, object]:
    """Run every evidence gate without tuning on the locked test partition."""

    bundle = Path(bundle_dir).resolve()
    try:
        bundle_summary = validate_evidence_bundle(bundle)
        recomputed = evaluate_locked_predictions(bundle)
    except EvidenceBundleError as exc:
        raise ReleaseValidationError(str(exc)) from exc

    operating = _read_json(bundle / "operating_point.json")
    saved_metrics = _read_json(bundle / "test_metrics.json")
    calibration_rows = pd.read_csv(bundle / "calibration_predictions.csv")
    test_rows = pd.read_csv(bundle / "test_predictions.csv")
    required_calibration = {"image_id", "patient_id"}
    for label in RSNA_LABELS:
        required_calibration.update(
            {f"target_{label}", f"logit_{label}", f"score_{label}"}
        )
    _require_columns(calibration_rows, required_calibration, "calibration_predictions.csv")
    _require_columns(
        test_rows,
        {"image_id", "patient_id", "target_any", "score_any", "predicted_any"},
        "test_predictions.csv",
    )
    if calibration_rows["image_id"].duplicated().any() or test_rows["image_id"].duplicated().any():
        raise ReleaseValidationError("Prediction artifacts contain duplicate image IDs")
    patient_overlap = set(calibration_rows["patient_id"]) & set(test_rows["patient_id"])
    if patient_overlap:
        raise ReleaseValidationError(
            f"Calibration and test partitions share {len(patient_overlap)} patients"
        )
    if len(calibration_rows) != operating.get("calibration_images"):
        raise ReleaseValidationError("Calibration row count disagrees with operating point")
    if len(test_rows) != saved_metrics.get("test_images"):
        raise ReleaseValidationError("Test row count disagrees with saved metrics")

    actual_metrics = recomputed.to_dict()
    _require_metric_agreement(actual_metrics, saved_metrics.get("primary_metrics", {}))

    selected = select_sensitivity_threshold(
        calibration_rows["target_any"],
        calibration_rows["score_any"],
        operating["target_sensitivity"],
    )
    # The CSV has fewer decimals than the JSON operating-point artifact.
    if not math.isclose(selected.threshold, operating["threshold"], rel_tol=0.0, abs_tol=2e-8):
        raise ReleaseValidationError("Calibration rows do not reproduce the locked threshold")
    expected_counts = operating.get("calibration_confusion_counts", {})
    actual_counts = {name: getattr(selected, name) for name in ("tp", "tn", "fp", "fn")}
    if actual_counts != expected_counts:
        raise ReleaseValidationError("Calibration confusion counts do not reproduce")

    calibration_audit: dict[str, object] = {}
    for label in RSNA_LABELS:
        calibration_audit[label] = fit_temperature_scaling(
            calibration_rows[f"target_{label}"], calibration_rows[f"logit_{label}"]
        ).to_dict()

    bootstrap = grouped_bootstrap_confidence_intervals(
        test_rows["target_any"],
        test_rows["score_any"],
        test_rows["patient_id"],
        operating["threshold"],
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
        sampling_unit="patient_id",
    )

    release_summary = None
    if release_manifest_path is not None:
        if project_root is None:
            raise ReleaseValidationError(
                "project_root is required when validating a release manifest"
            )
        release_summary = validate_project_release_manifest(
            release_manifest_path, project_root
        )
        if release_summary["checkpoint_sha256"] != bundle_summary["checkpoint_sha256"]:
            raise ReleaseValidationError(
                "Release manifest and evidence bundle refer to different checkpoints"
            )

    return {
        "schema_version": "1.0",
        "status": "passed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "modality": "head_ct",
            "evaluation_level": "slice",
            "endpoint": bundle_summary["endpoint"],
            "locked_test_untouched": True,
        },
        "release_manifest_validation": release_summary,
        "bundle_validation": bundle_summary,
        "cohorts": {
            "calibration_rows": len(calibration_rows),
            "calibration_patients": int(calibration_rows["patient_id"].nunique()),
            "test_rows": len(test_rows),
            "test_patients": int(test_rows["patient_id"].nunique()),
            "patient_overlap": 0,
        },
        "locked_test_metrics": actual_metrics,
        "threshold_audit": selected.to_dict(),
        "calibration_audit": calibration_audit,
        "grouped_bootstrap_audit": bootstrap.to_dict(),
        "policy": {
            "calibration_applied_to_locked_release": False,
            "threshold_retuned_on_test": False,
            "bootstrap_changes_point_estimates": False,
        },
        "limitations": saved_metrics.get("limitations", []),
    }


"""Validation of the complete locked RSNA Phase 2 study-level evidence chain."""

from __future__ import annotations

from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
import zipfile

import pandas as pd

from .bootstrap import grouped_bootstrap_confidence_intervals
from .evaluation import binary_metrics


class CTPhase2EvidenceError(ValueError):
    """Phase 2 evidence is missing, modified, or internally inconsistent."""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_zip(path: Path) -> dict[str, bytes]:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or any(
                Path(name).name != name or "\\" in name for name in names
            ):
                raise CTPhase2EvidenceError(f"Unsafe or duplicate member in {path.name}")
            return {name: archive.read(name) for name in names}
    except (OSError, zipfile.BadZipFile) as exc:
        raise CTPhase2EvidenceError(f"Cannot read {path.name}: {exc}") from exc


def _json(files: dict[str, bytes], name: str) -> dict[str, object]:
    try:
        value = json.loads(files[name])
    except (KeyError, json.JSONDecodeError) as exc:
        raise CTPhase2EvidenceError(f"Cannot read {name}") from exc
    if not isinstance(value, dict):
        raise CTPhase2EvidenceError(f"{name} must contain a JSON object")
    return value


def _csv(files: dict[str, bytes], name: str) -> pd.DataFrame:
    try:
        return pd.read_csv(BytesIO(files[name]))
    except (KeyError, ValueError) as exc:
        raise CTPhase2EvidenceError(f"Cannot read {name}") from exc


def _require_hash(actual: bytes, expected: object, label: str) -> None:
    if not isinstance(expected, str) or _digest(actual) != expected:
        raise CTPhase2EvidenceError(f"SHA-256 mismatch: {label}")


def validate_ct_phase2_evidence(
    artifact_dir: str | Path,
    registry_path: str | Path,
) -> dict[str, object]:
    """Verify artifacts, provenance links, point metrics, and bootstrap intervals."""

    root = Path(artifact_dir)
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    bundles = registry.get("bundles")
    if not isinstance(bundles, dict) or len(bundles) != 5:
        raise CTPhase2EvidenceError("Evidence registry must declare five bundles")
    loaded: dict[str, dict[str, bytes]] = {}
    for role, declaration in bundles.items():
        if not isinstance(declaration, dict):
            raise CTPhase2EvidenceError(f"Invalid registry declaration: {role}")
        path = root / str(declaration.get("file"))
        if not path.is_file():
            raise CTPhase2EvidenceError(f"Missing Phase 2 bundle: {path.name}")
        if path.stat().st_size != declaration.get("bytes"):
            raise CTPhase2EvidenceError(f"Byte count mismatch: {path.name}")
        _require_hash(path.read_bytes(), declaration.get("sha256"), path.name)
        loaded[role] = _read_zip(path)

    cohort, development = loaded["cohort"], loaded["development"]
    locked, test_slices, final = (
        loaded["locked_protocol"], loaded["test_slices"], loaded["final_evaluation"]
    )
    for name, source in (
        ("locked_study_protocol.json", locked),
        ("study_calibration_predictions.csv", locked),
        ("rsna_phase2_prediction_export.json", test_slices),
        ("rsna_phase2_test_manifest.csv", test_slices),
        ("rsna_phase2_cohort_validation.json", cohort),
    ):
        if final.get(name) != source.get(name):
            raise CTPhase2EvidenceError(f"Final bundle copy disagrees: {name}")

    summary = _json(cohort, "rsna_phase2_cohort_summary.json")
    cohort_validation = _json(cohort, "rsna_phase2_cohort_validation.json")
    cohort_manifest = cohort["rsna_phase2_study_manifest.csv"]
    _require_hash(cohort_manifest, summary.get("manifest", {}).get("sha256"), "cohort manifest")
    _require_hash(cohort_manifest, cohort_validation.get("manifest_sha256"), "validated cohort manifest")
    if cohort_validation.get("status") != "passed":
        raise CTPhase2EvidenceError("Cohort validation did not pass")

    dev_export = _json(development, "rsna_phase2_prediction_export.json")
    dev_predictions = development["rsna_phase2_slice_predictions.csv"]
    dev_manifest = development["rsna_phase2_development_manifest.csv"]
    _require_hash(dev_predictions, dev_export.get("predictions_sha256"), "development predictions")
    _require_hash(dev_manifest, dev_export.get("manifest_sha256"), "development manifest")

    protocol = _json(locked, "locked_study_protocol.json")
    if protocol.get("status") != "locked" or protocol.get("test_evaluated") is not False:
        raise CTPhase2EvidenceError("Development protocol is not cleanly locked")
    provenance = protocol.get("provenance", {})
    _require_hash(dev_predictions, provenance.get("slice_predictions_sha256"), "protocol development predictions")
    _require_hash(dev_manifest, provenance.get("development_manifest_sha256"), "protocol development manifest")
    _require_hash(
        locked["study_calibration_predictions.csv"],
        provenance.get("calibration_predictions_sha256"), "calibration predictions",
    )

    test_export = _json(test_slices, "rsna_phase2_prediction_export.json")
    test_predictions = test_slices["rsna_phase2_slice_predictions.csv"]
    test_manifest = test_slices["rsna_phase2_test_manifest.csv"]
    _require_hash(test_predictions, test_export.get("predictions_sha256"), "test slice predictions")
    _require_hash(test_manifest, test_export.get("manifest_sha256"), "test manifest")
    expected_checkpoint = registry.get("checkpoint_sha256")
    if {
        dev_export.get("checkpoint_sha256"), test_export.get("checkpoint_sha256")
    } != {expected_checkpoint}:
        raise CTPhase2EvidenceError("Checkpoint provenance disagrees")

    metrics = _json(final, "study_test_metrics.json")
    final_provenance = metrics.get("provenance", {})
    _require_hash(test_predictions, final_provenance.get("slice_predictions_sha256"), "final test slices")
    _require_hash(test_manifest, final_provenance.get("test_manifest_sha256"), "final test manifest")
    _require_hash(
        locked["locked_study_protocol.json"],
        final_provenance.get("locked_protocol_sha256"), "final locked protocol",
    )
    study_bytes = final["study_test_predictions.csv"]
    _require_hash(study_bytes, final_provenance.get("study_test_predictions_sha256"), "study predictions")
    if not all(metrics.get(flag) is expected for flag, expected in (
        ("pooling_reselected", False), ("threshold_reselected", False),
        ("test_evaluated_once", True),
    )):
        raise CTPhase2EvidenceError("Final evaluation lock flags are invalid")
    if metrics.get("selected_method") != protocol.get("selected_method"):
        raise CTPhase2EvidenceError("Final pooling method differs from locked protocol")
    threshold = float(protocol["operating_point"]["threshold"])
    if float(metrics.get("threshold")) != threshold:
        raise CTPhase2EvidenceError("Final threshold differs from locked protocol")

    studies = _csv(final, "study_test_predictions.csv")
    measured = binary_metrics(studies["target_any"], studies["score_any"], threshold)
    for name, value in measured.to_dict().items():
        saved = metrics.get("test_metrics", {}).get(name)
        if not isinstance(saved, (int, float)) or not math.isclose(
            float(value), float(saved), rel_tol=0.0, abs_tol=1e-12
        ):
            raise CTPhase2EvidenceError(f"Recomputed metric disagrees: {name}")
    bootstrap_saved = metrics.get("patient_grouped_bootstrap", {})
    bootstrap = grouped_bootstrap_confidence_intervals(
        studies["target_any"], studies["score_any"], studies["patient_id"], threshold,
        replicates=int(bootstrap_saved.get("requested_replicates", 0)),
        confidence_level=float(bootstrap_saved.get("confidence_level", 0)),
        seed=int(bootstrap_saved.get("seed", -1)), sampling_unit="patient_id",
    ).to_dict()
    if bootstrap != bootstrap_saved:
        raise CTPhase2EvidenceError("Recomputed grouped bootstrap intervals disagree")
    return {
        "status": "passed", "endpoint": metrics.get("endpoint"),
        "checkpoint_sha256": expected_checkpoint, "patients": len(studies),
        "positive_studies": int(studies["target_any"].sum()),
        "selected_method": metrics.get("selected_method"), "threshold": threshold,
        "test_metrics": measured.to_dict(),
        "bootstrap_replicates": bootstrap["requested_replicates"],
        "outer_bundle_sha256": {
            role: declaration["sha256"] for role, declaration in bundles.items()
        },
    }

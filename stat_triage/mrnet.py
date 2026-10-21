"""Locked Stanford MRNet evidence validation and artifact registration."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from .calibration import sigmoid
from .contracts import Modality, ModelEvidence
from .datasets.mrnet import MRNET_PLANES, MRNET_TARGETS
from .datasets.mrnet_preprocessing import MRNetVolumeError, preprocess_mrnet_volume
from .evaluation import binary_metrics
from .models import MRNetMultiPlaneModel


def _default_bundle_dir() -> Path:
    configured = os.environ.get("STAT_TRIAGE_MRNET_BUNDLE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    relative = Path("artifacts/mrnet_baseline")
    working = Path.cwd() / relative
    if working.exists():
        return working.resolve()
    return (Path(__file__).resolve().parents[1] / relative).resolve()


DEFAULT_MRNET_BUNDLE_DIR = _default_bundle_dir()


class MRNetEvidenceError(ValueError):
    """The locked MRNet evidence bundle is invalid or inconsistent."""


class MRNetInputError(ValueError):
    """A knee MRI input cannot satisfy the three-plane MRNet contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MRNetEvidenceError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MRNetEvidenceError(f"{path.name} must contain a JSON object")
    return value


def validate_mrnet_evidence_bundle(
    bundle_dir: str | Path = DEFAULT_MRNET_BUNDLE_DIR,
    *,
    recompute_metrics: bool = False,
) -> dict[str, object]:
    """Verify hashes, frozen-policy references, checkpoint, and saved metrics."""
    bundle = Path(bundle_dir).resolve()
    registry = _read_json(bundle / "mrnet_bundle_manifest.json")
    declarations = registry.get("files")
    if not isinstance(declarations, dict) or not declarations:
        raise MRNetEvidenceError("MRNet bundle registry has no files")
    for relative, declaration in declarations.items():
        if not isinstance(relative, str) or not isinstance(declaration, dict):
            raise MRNetEvidenceError("MRNet registry contains an invalid declaration")
        path = (bundle / relative).resolve()
        try:
            path.relative_to(bundle)
        except ValueError as exc:
            raise MRNetEvidenceError(f"Bundle path escapes its root: {relative}") from exc
        if not path.is_file():
            raise MRNetEvidenceError(f"Missing MRNet artifact: {relative}")
        if path.stat().st_size != declaration.get("bytes"):
            raise MRNetEvidenceError(f"Byte count mismatch: {relative}")
        if _sha256(path) != declaration.get("sha256"):
            raise MRNetEvidenceError(f"SHA-256 mismatch: {relative}")

    checkpoint_name = "mrnet_baseline/best_mrnet_resnet18.pt"
    manifest_name = "mrnet_manifest/mrnet_manifest_split.csv"
    calibration_name = "mrnet_calibration/locked_calibration.json"
    predictions_name = "mrnet_test/test_predictions.csv"
    metrics_name = "mrnet_test/test_metrics.json"
    for required in (
        checkpoint_name, manifest_name, calibration_name, predictions_name, metrics_name
    ):
        if required not in declarations:
            raise MRNetEvidenceError(f"Required artifact is undeclared: {required}")

    checkpoint_hash = str(declarations[checkpoint_name]["sha256"])
    manifest_hash = str(declarations[manifest_name]["sha256"])
    calibration_hash = str(declarations[calibration_name]["sha256"])
    locked = _read_json(bundle / calibration_name)
    metrics = _read_json(bundle / metrics_name)
    if registry.get("checkpoint_sha256") != checkpoint_hash:
        raise MRNetEvidenceError("Registry checkpoint reference does not match")
    if registry.get("manifest_sha256") != manifest_hash:
        raise MRNetEvidenceError("Registry data-manifest reference does not match")
    if locked.get("checkpoint", {}).get("sha256") != checkpoint_hash:
        raise MRNetEvidenceError("Calibration refers to a different checkpoint")
    if locked.get("manifest_sha256") != manifest_hash:
        raise MRNetEvidenceError("Calibration refers to a different data manifest")
    if metrics.get("checkpoint_sha256") != checkpoint_hash:
        raise MRNetEvidenceError("Test metrics refer to a different checkpoint")
    if metrics.get("manifest_sha256") != manifest_hash:
        raise MRNetEvidenceError("Test metrics refer to a different data manifest")
    if metrics.get("locked_calibration_sha256") != calibration_hash:
        raise MRNetEvidenceError("Test metrics refer to a different calibration artifact")
    if metrics.get("threshold_retuned_on_test") is not False:
        raise MRNetEvidenceError("Test artifact does not prohibit threshold retuning")
    if metrics.get("calibrator_refit_on_test") is not False:
        raise MRNetEvidenceError("Test artifact does not prohibit calibrator refitting")
    if metrics.get("test_exams") != registry.get("test_exams"):
        raise MRNetEvidenceError("Test exam count disagrees across artifacts")
    if set(locked.get("targets", {})) != set(MRNET_TARGETS):
        raise MRNetEvidenceError("Locked target set does not match MRNet")

    checkpoint = torch.load(
        bundle / checkpoint_name, map_location="cpu", weights_only=True
    )
    if tuple(checkpoint.get("targets", ())) != MRNET_TARGETS:
        raise MRNetEvidenceError("Checkpoint target order does not match MRNet")
    if tuple(checkpoint.get("planes", ())) != MRNET_PLANES:
        raise MRNetEvidenceError("Checkpoint plane order does not match MRNet")
    if int(checkpoint.get("epoch", -1)) != int(locked.get("checkpoint", {}).get("epoch", -2)):
        raise MRNetEvidenceError("Checkpoint epoch disagrees with calibration")

    if recompute_metrics:
        rows = pd.read_csv(bundle / predictions_name, dtype={"exam_id": str})
        if len(rows) != metrics["test_exams"] or rows["exam_id"].duplicated().any():
            raise MRNetEvidenceError("Saved test prediction cohort is inconsistent")
        for target in MRNET_TARGETS:
            threshold = float(locked["targets"][target]["operating_point"]["threshold"])
            if not math.isclose(
                threshold, float(metrics["targets"][target]["threshold"]),
                rel_tol=0, abs_tol=1e-15,
            ):
                raise MRNetEvidenceError(f"Threshold mismatch for {target}")
            scores = rows[f"calibrated_score_{target}"]
            decisions = (scores >= threshold).astype("int8")
            if not np.array_equal(decisions, rows[f"predicted_{target}"].to_numpy()):
                raise MRNetEvidenceError(f"Saved decisions mismatch for {target}")
            actual = binary_metrics(rows[f"target_{target}"], scores, threshold).to_dict()
            expected = metrics["targets"][target]["point_estimates"]
            for name, value in actual.items():
                agrees = (
                    math.isclose(value, float(expected[name]), rel_tol=0, abs_tol=1e-12)
                    if isinstance(value, float) else value == expected[name]
                )
                if not agrees:
                    raise MRNetEvidenceError(f"Metric mismatch for {target}.{name}")

    return {
        "status": metrics.get("status"),
        "artifact_count": len(declarations),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "labels": MRNET_TARGETS,
        "planes": MRNET_PLANES,
        "test_exams": metrics.get("test_exams"),
        "patient_identifiers_available": metrics.get("patient_identifiers_available"),
        "metrics_recomputed": recompute_metrics,
    }


class MRNetInferenceAdapter:
    """Run the verified three-plane model with frozen calibration and thresholds."""

    def __init__(
        self,
        bundle_dir: str | Path = DEFAULT_MRNET_BUNDLE_DIR,
        device: str | torch.device | None = None,
        *,
        slice_chunk_size: int = 32,
    ) -> None:
        self.bundle_dir = Path(bundle_dir).resolve()
        summary = validate_mrnet_evidence_bundle(self.bundle_dir)
        self.model_sha256 = str(summary["checkpoint_sha256"])
        self.model_epoch = int(summary["checkpoint_epoch"])
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.locked = _read_json(
            self.bundle_dir / "mrnet_calibration/locked_calibration.json"
        )
        self.calibrators: dict[str, tuple[float, float]] = {}
        self.thresholds: dict[str, float] = {}
        for target in MRNET_TARGETS:
            target_lock = self.locked["targets"][target]
            calibration = target_lock["calibration"]
            if calibration.get("selected_method") != "affine_logistic_scaling":
                raise MRNetEvidenceError(f"Unsupported calibrator for {target}")
            slope = float(calibration["slope"])
            intercept = float(calibration["intercept"])
            threshold = float(target_lock["operating_point"]["threshold"])
            if not math.isfinite(slope) or slope <= 0 or not math.isfinite(intercept):
                raise MRNetEvidenceError(f"Invalid affine calibrator for {target}")
            if not 0 <= threshold <= 1:
                raise MRNetEvidenceError(f"Invalid threshold for {target}")
            self.calibrators[target] = (slope, intercept)
            self.thresholds[target] = threshold
        checkpoint = torch.load(
            self.bundle_dir / "mrnet_baseline/best_mrnet_resnet18.pt",
            map_location="cpu",
            weights_only=True,
        )
        self.model = MRNetMultiPlaneModel(
            pretrained=False, slice_chunk_size=slice_chunk_size
        )
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.to(self.device).eval()

    def _preprocess_planes(
        self, paths: dict[str, Path]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        volumes: dict[str, torch.Tensor] = {}
        masks: dict[str, torch.Tensor] = {}
        for plane in MRNET_PLANES:
            try:
                tensor, _, _ = preprocess_mrnet_volume(paths[plane], size=224)
            except (MRNetVolumeError, OSError, ValueError) as exc:
                raise MRNetInputError(f"Invalid {plane} volume: {exc}") from exc
            volumes[plane] = tensor.unsqueeze(0).to(self.device)
            masks[plane] = torch.ones(
                (1, tensor.shape[0]), dtype=torch.bool, device=self.device
            )
        return volumes, masks

    @torch.inference_mode()
    def predict_exam(
        self,
        *,
        axial: str | Path,
        coronal: str | Path,
        sagittal: str | Path,
        exam_id: str,
        patient_id: str | None = None,
    ) -> ModelEvidence:
        paths = {
            "axial": Path(axial),
            "coronal": Path(coronal),
            "sagittal": Path(sagittal),
        }
        started = time.perf_counter()
        volumes, masks = self._preprocess_planes(paths)
        logits = self.model(volumes, masks).squeeze(0).float().cpu().numpy()
        raw_values = sigmoid(logits)
        calibrated_values = np.array(
            [
                sigmoid(
                    np.array(
                        self.calibrators[target][0] * logits[index]
                        + self.calibrators[target][1]
                    )
                ).item()
                for index, target in enumerate(MRNET_TARGETS)
            ]
        )
        raw_scores = {
            target: float(raw_values[index])
            for index, target in enumerate(MRNET_TARGETS)
        }
        calibrated_scores = {
            target: float(calibrated_values[index])
            for index, target in enumerate(MRNET_TARGETS)
        }
        decisions = {
            target: calibrated_scores[target] >= self.thresholds[target]
            for target in MRNET_TARGETS
        }
        return ModelEvidence.created_now(
            exam_id=exam_id,
            patient_id=patient_id or exam_id,
            modality=Modality.KNEE_MRI,
            endpoint="mrnet_three_target_exam_triage",
            model_id=f"shared_resnet18_three_plane_epoch_{self.model_epoch}",
            model_sha256=self.model_sha256,
            preprocess_version="mrnet_uint8_resize224_imagenet_v1",
            raw_scores=raw_scores,
            calibrated_scores=calibrated_scores,
            thresholds=self.thresholds,
            decisions=decisions,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            input_hashes={
                f"{plane}/{path.name}": _sha256(path) for plane, path in paths.items()
            },
            warnings=(
                "research_triage_pilot_not_for_clinical_use",
                "input_plane_identity_is_caller_supplied_and_not_encoded_in_npy",
                "mrnet_patient_identifiers_unavailable_for_split_verification",
                "high_sensitivity_operating_points_have_low_specificity",
            ),
        )

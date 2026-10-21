"""Locked NIH ChestX-ray14 inference adapter and evidence validation."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time

import pandas as pd
import torch

from .calibration import sigmoid
from .contracts import Modality, ModelEvidence
from .datasets.nih_preprocessing import preprocess_nih_radiograph
from .datasets.nih_training import NIH_TARGET_ORDER
from .evaluation import binary_metrics
from .models import build_densenet121_multilabel


def _default_bundle_dir() -> Path:
    configured = os.environ.get("STAT_TRIAGE_CXR_BUNDLE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    relative = Path("artifacts/chest_xray/nih_cxr14_locked_test_bundle")
    working = Path.cwd() / relative
    if working.exists():
        return working.resolve()
    return (Path(__file__).resolve().parents[1] / relative).resolve()


DEFAULT_CXR_BUNDLE_DIR = _default_bundle_dir()


class CXREvidenceError(ValueError):
    """The locked Chest X-ray evidence bundle is invalid or inconsistent."""


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
        raise CXREvidenceError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CXREvidenceError(f"{path.name} must contain a JSON object")
    return value


def validate_cxr_evidence_bundle(
    bundle_dir: str | Path = DEFAULT_CXR_BUNDLE_DIR,
    *,
    recompute_metrics: bool = False,
) -> dict[str, object]:
    """Verify the registry and optionally reproduce all locked point estimates."""

    bundle = Path(bundle_dir).resolve()
    registry = _read_json(bundle / "bundle_manifest.json")
    declarations = registry.get("files")
    if not isinstance(declarations, dict) or not declarations:
        raise CXREvidenceError("Bundle registry has no files")
    for relative, declaration in declarations.items():
        if not isinstance(relative, str) or not isinstance(declaration, dict):
            raise CXREvidenceError("Bundle registry contains an invalid declaration")
        path = (bundle / relative).resolve()
        try:
            path.relative_to(bundle)
        except ValueError as exc:
            raise CXREvidenceError(f"Bundle path escapes its root: {relative}") from exc
        if not path.is_file():
            raise CXREvidenceError(f"Missing bundle artifact: {relative}")
        if path.stat().st_size != declaration.get("bytes"):
            raise CXREvidenceError(f"Byte count mismatch: {relative}")
        if _sha256(path) != declaration.get("sha256"):
            raise CXREvidenceError(f"SHA-256 mismatch: {relative}")

    checkpoint_name = "nih_densenet/best_densenet121.pt"
    calibration_name = "nih_densenet_calibration/locked_calibration.json"
    metrics_name = "nih_densenet_test/test_metrics.json"
    locked = _read_json(bundle / calibration_name)
    metrics = _read_json(bundle / metrics_name)
    checkpoint_hash = declarations[checkpoint_name]["sha256"]
    calibration_hash = declarations[calibration_name]["sha256"]
    if locked.get("checkpoint", {}).get("sha256") != checkpoint_hash:
        raise CXREvidenceError("Calibration refers to a different checkpoint")
    if metrics.get("checkpoint_sha256") != checkpoint_hash:
        raise CXREvidenceError("Test metrics refer to a different checkpoint")
    if metrics.get("locked_calibration_sha256") != calibration_hash:
        raise CXREvidenceError("Test metrics refer to a different calibration artifact")
    if metrics.get("threshold_retuned_on_test") is not False:
        raise CXREvidenceError("Test artifact does not prohibit threshold retuning")
    if metrics.get("calibrator_refit_on_test") is not False:
        raise CXREvidenceError("Test artifact does not prohibit calibrator refitting")

    if recompute_metrics:
        rows = pd.read_csv(bundle / "nih_densenet_test/test_predictions.csv")
        for target in NIH_TARGET_ORDER:
            threshold = float(locked["targets"][target]["operating_point"]["threshold"])
            actual = binary_metrics(
                rows[f"target_{target}"], rows[f"calibrated_score_{target}"], threshold
            ).to_dict()
            expected = metrics["targets"][target]["point_estimates"]
            for name, value in actual.items():
                if isinstance(value, float):
                    agrees = math.isclose(value, float(expected[name]), rel_tol=0, abs_tol=1e-12)
                else:
                    agrees = value == expected[name]
                if not agrees:
                    raise CXREvidenceError(f"Metric mismatch for {target}.{name}")

    return {
        "status": registry.get("status"),
        "artifact_count": len(declarations),
        "checkpoint_sha256": checkpoint_hash,
        "labels": NIH_TARGET_ORDER,
        "test_images": metrics.get("test_images"),
        "test_patients": metrics.get("test_patients"),
        "metrics_recomputed": recompute_metrics,
    }


class NIHChestXrayAdapter:
    """Run the verified four-output DenseNet with frozen calibration and thresholds."""

    def __init__(
        self,
        bundle_dir: str | Path = DEFAULT_CXR_BUNDLE_DIR,
        device: str | torch.device | None = None,
    ) -> None:
        self.bundle_dir = Path(bundle_dir).resolve()
        summary = validate_cxr_evidence_bundle(self.bundle_dir)
        self.model_sha256 = str(summary["checkpoint_sha256"])
        self.locked = _read_json(
            self.bundle_dir / "nih_densenet_calibration/locked_calibration.json"
        )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.calibrators: dict[str, tuple[float, float]] = {}
        self.thresholds: dict[str, float] = {}
        for target in NIH_TARGET_ORDER:
            target_lock = self.locked["targets"][target]
            calibration = target_lock["calibration"]
            if calibration.get("selected_method") != "affine_logistic_scaling":
                raise CXREvidenceError(f"Unsupported calibrator for {target}")
            slope = float(calibration["slope"])
            intercept = float(calibration["intercept"])
            threshold = float(target_lock["operating_point"]["threshold"])
            if not math.isfinite(slope) or slope <= 0 or not math.isfinite(intercept):
                raise CXREvidenceError(f"Invalid affine calibrator for {target}")
            if not 0 <= threshold <= 1:
                raise CXREvidenceError(f"Invalid threshold for {target}")
            self.calibrators[target] = (slope, intercept)
            self.thresholds[target] = threshold
        self.model, self.model_epoch = self._load_model()

    def _load_model(self) -> tuple[torch.nn.Module, int]:
        checkpoint = torch.load(
            self.bundle_dir / "nih_densenet/best_densenet121.pt",
            map_location="cpu",
            weights_only=True,
        )
        if tuple(checkpoint.get("labels", ())) != NIH_TARGET_ORDER:
            raise CXREvidenceError("Checkpoint label order does not match")
        if checkpoint.get("architecture") != "torchvision_densenet121":
            raise CXREvidenceError("Checkpoint architecture does not match")
        model = build_densenet121_multilabel(len(NIH_TARGET_ORDER), pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(self.device).eval()
        return model, int(checkpoint["epoch"])

    @torch.inference_mode()
    def predict_file(
        self,
        path: str | Path,
        *,
        patient_id: str | None = None,
        exam_id: str | None = None,
    ) -> ModelEvidence:
        source = Path(path)
        started = time.perf_counter()
        tensor, _, _ = preprocess_nih_radiograph(source, size=224)
        logits = self.model(tensor.unsqueeze(0).to(self.device)).squeeze(0).cpu().numpy()
        raw_values = sigmoid(logits)
        raw_scores = {
            target: float(raw_values[index]) for index, target in enumerate(NIH_TARGET_ORDER)
        }
        calibrated_scores = {
            target: float(sigmoid(slope * logits[index] + intercept))
            for index, target in enumerate(NIH_TARGET_ORDER)
            for slope, intercept in (self.calibrators[target],)
        }
        decisions = {
            target: calibrated_scores[target] >= self.thresholds[target]
            for target in NIH_TARGET_ORDER
        }
        latency_ms = (time.perf_counter() - started) * 1000.0
        return ModelEvidence.created_now(
            exam_id=exam_id or source.stem,
            patient_id=patient_id or source.stem,
            modality=Modality.CHEST_XRAY,
            endpoint="nih_cxr14_four_target_triage",
            model_id=f"densenet121_epoch_{self.model_epoch}",
            model_sha256=self.model_sha256,
            preprocess_version="nih_grayscale_aspect_pad_imagenet_v1",
            raw_scores=raw_scores,
            calibrated_scores=calibrated_scores,
            thresholds=self.thresholds,
            decisions=decisions,
            latency_ms=latency_ms,
            input_hashes={source.name: _sha256(source)},
            warnings=(
                "research_triage_pilot_not_for_clinical_use",
                "input_frontal_view_cannot_be_verified_from_pixels_alone",
            ),
        )

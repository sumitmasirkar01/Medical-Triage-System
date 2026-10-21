"""Artifact validation and reproducible binary evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


class EvidenceBundleError(ValueError):
    """A release artifact is missing, modified, or internally inconsistent."""


@dataclass(frozen=True)
class BinaryMetrics:
    roc_auc: float
    average_precision: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    accuracy: float
    tp: int
    tn: int
    fp: int
    fn: int
    positive_images: int
    negative_images: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceBundleError(f"Cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceBundleError(f"{path.name} must contain a JSON object")
    return value


def validate_evidence_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    """Verify declared bytes/hashes and the locked model/threshold references."""

    bundle = Path(bundle_dir).resolve()
    manifest = _read_json(bundle / "bundle_manifest.json")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise EvidenceBundleError("bundle_manifest.json has no file registry")

    for name, declaration in files.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise EvidenceBundleError(f"Unsafe artifact name in manifest: {name!r}")
        if not isinstance(declaration, dict):
            raise EvidenceBundleError(f"Invalid declaration for {name}")
        path = bundle / name
        if not path.is_file():
            raise EvidenceBundleError(f"Missing declared artifact: {name}")
        expected_bytes = declaration.get("bytes")
        expected_hash = declaration.get("sha256")
        if path.stat().st_size != expected_bytes:
            raise EvidenceBundleError(f"Byte count mismatch for {name}")
        if not isinstance(expected_hash, str) or sha256(path) != expected_hash:
            raise EvidenceBundleError(f"SHA-256 mismatch for {name}")

    operating = _read_json(bundle / "operating_point.json")
    metrics = _read_json(bundle / "test_metrics.json")
    checkpoint_hash = files.get("best_efficientnet_b0.pt", {}).get("sha256")
    references = {
        operating.get("model_checkpoint_sha256"),
        metrics.get("model_checkpoint_sha256"),
    }
    if references != {checkpoint_hash}:
        raise EvidenceBundleError("Checkpoint references disagree across artifacts")
    if metrics.get("threshold") != operating.get("threshold"):
        raise EvidenceBundleError("Test metrics do not use the locked threshold")
    if metrics.get("endpoint") != operating.get("endpoint"):
        raise EvidenceBundleError("Endpoint disagrees across artifacts")

    return {
        "bundle": manifest.get("bundle"),
        "artifact_count": len(files),
        "checkpoint_sha256": checkpoint_hash,
        "threshold": operating["threshold"],
        "endpoint": operating["endpoint"],
    }


def binary_metrics(
    targets: np.ndarray | pd.Series,
    scores: np.ndarray | pd.Series,
    threshold: float,
) -> BinaryMetrics:
    """Compute the locked binary metrics without modifying scores or threshold."""

    y = np.asarray(targets)
    p = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError("targets and scores must be non-empty one-dimensional arrays")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("targets must contain only 0 and 1")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("scores must be finite and in [0, 1]")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    if len(np.unique(y)) != 2:
        raise ValueError("both target classes are required")

    predicted = p >= threshold
    positive = y == 1
    negative = ~positive
    tp = int(np.sum(predicted & positive))
    tn = int(np.sum(~predicted & negative))
    fp = int(np.sum(predicted & negative))
    fn = int(np.sum(~predicted & positive))

    return BinaryMetrics(
        roc_auc=float(roc_auc_score(y, p)),
        average_precision=float(average_precision_score(y, p)),
        sensitivity=tp / (tp + fn),
        specificity=tn / (tn + fp),
        ppv=tp / (tp + fp) if tp + fp else float("nan"),
        npv=tn / (tn + fn) if tn + fn else float("nan"),
        accuracy=(tp + tn) / len(y),
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        positive_images=int(np.sum(positive)),
        negative_images=int(np.sum(negative)),
    )


def evaluate_locked_predictions(bundle_dir: str | Path) -> BinaryMetrics:
    """Validate a bundle and recompute its primary endpoint from saved rows."""

    bundle = Path(bundle_dir).resolve()
    summary = validate_evidence_bundle(bundle)
    predictions = pd.read_csv(bundle / "test_predictions.csv")
    required = {"target_any", "score_any", "predicted_any"}
    missing = required.difference(predictions.columns)
    if missing:
        raise EvidenceBundleError(
            f"test_predictions.csv is missing columns: {', '.join(sorted(missing))}"
        )

    metrics = binary_metrics(
        predictions["target_any"], predictions["score_any"], summary["threshold"]
    )
    expected_decisions = (predictions["score_any"] >= summary["threshold"]).astype(int)
    if not np.array_equal(expected_decisions, predictions["predicted_any"].to_numpy()):
        raise EvidenceBundleError("Saved decisions do not use the locked threshold")
    return metrics


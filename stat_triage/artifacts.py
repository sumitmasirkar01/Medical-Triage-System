"""Unified registry and validation for every locked modality release."""

from __future__ import annotations

import json
from pathlib import Path

from .contracts import Modality
from .cxr import validate_cxr_evidence_bundle
from .evaluation import evaluate_locked_predictions, validate_evidence_bundle
from .mrnet import validate_mrnet_evidence_bundle
from .multimodal import MODALITY_REGISTRY


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MULTIMODAL_REGISTRY = PROJECT_ROOT / "artifacts/multimodal_registry.json"


class MultimodalRegistryError(ValueError):
    """The unified artifact registry or one of its bundles is inconsistent."""


def _read_registry(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MultimodalRegistryError(f"Cannot read artifact registry: {exc}") from exc
    if not isinstance(value, dict):
        raise MultimodalRegistryError("Artifact registry must contain a JSON object")
    return value


def _safe_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise MultimodalRegistryError("Bundle path must be a non-empty relative path")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise MultimodalRegistryError(f"Bundle path escapes project root: {relative}") from exc
    return candidate


def validate_multimodal_evidence_registry(
    registry_path: str | Path = DEFAULT_MULTIMODAL_REGISTRY,
    *,
    project_root: str | Path = PROJECT_ROOT,
    recompute_metrics: bool = True,
) -> dict[str, object]:
    """Validate registry contracts, all bundle hashes, and locked point estimates."""
    root = Path(project_root).resolve()
    registry = _read_registry(Path(registry_path).resolve())
    if registry.get("schema_version") != "1.0":
        raise MultimodalRegistryError("Unsupported artifact registry schema")
    entries = registry.get("modalities")
    if not isinstance(entries, dict) or set(entries) != {m.value for m in Modality}:
        raise MultimodalRegistryError("Registry must contain exactly the supported modalities")

    results: dict[str, object] = {}
    for modality in Modality:
        entry = entries[modality.value]
        if not isinstance(entry, dict):
            raise MultimodalRegistryError(f"Invalid registry entry for {modality.value}")
        definition = MODALITY_REGISTRY[modality]
        if entry.get("endpoint") != definition.endpoint:
            raise MultimodalRegistryError(f"Endpoint mismatch for {modality.value}")
        if entry.get("evaluation_level") != definition.evaluation_level:
            raise MultimodalRegistryError(f"Evaluation level mismatch for {modality.value}")
        if tuple(entry.get("labels", ())) != definition.labels:
            raise MultimodalRegistryError(f"Label order mismatch for {modality.value}")
        bundle = _safe_path(root, entry.get("bundle_path"))
        try:
            if modality is Modality.HEAD_CT:
                summary = validate_evidence_bundle(bundle)
                if recompute_metrics:
                    evaluate_locked_predictions(bundle)
                summary = summary | {"metrics_recomputed": recompute_metrics}
            elif modality is Modality.CHEST_XRAY:
                summary = validate_cxr_evidence_bundle(
                    bundle, recompute_metrics=recompute_metrics
                )
            else:
                summary = validate_mrnet_evidence_bundle(
                    bundle, recompute_metrics=recompute_metrics
                )
        except (OSError, ValueError) as exc:
            raise MultimodalRegistryError(
                f"{modality.value} evidence failed validation: {exc}"
            ) from exc
        if summary.get("checkpoint_sha256") != entry.get("checkpoint_sha256"):
            raise MultimodalRegistryError(
                f"Checkpoint registry mismatch for {modality.value}"
            )
        results[modality.value] = {
            "bundle_path": str(bundle),
            "endpoint": definition.endpoint,
            "evaluation_level": definition.evaluation_level,
            "labels": list(definition.labels),
            **summary,
        }

    return {
        "schema_version": registry["schema_version"],
        "status": "passed",
        "release_status": registry.get("status"),
        "modality_count": len(results),
        "metrics_recomputed": recompute_metrics,
        "modalities": results,
    }

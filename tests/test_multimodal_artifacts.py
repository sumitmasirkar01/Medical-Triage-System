import json
from pathlib import Path

import pytest

from stat_triage.artifacts import (
    DEFAULT_MULTIMODAL_REGISTRY,
    MultimodalRegistryError,
    validate_multimodal_evidence_registry,
)


ROOT = Path(__file__).resolve().parents[1]


def test_unified_registry_validates_all_bundles_and_recomputes_metrics():
    report = validate_multimodal_evidence_registry(recompute_metrics=True)
    assert report["status"] == "passed"
    assert report["modality_count"] == 3
    assert set(report["modalities"]) == {"head_ct", "chest_xray", "knee_mri"}
    assert all(item["metrics_recomputed"] for item in report["modalities"].values())


def test_unified_registry_matches_the_three_model_contract():
    report = validate_multimodal_evidence_registry(recompute_metrics=False)
    assert report["modalities"]["head_ct"]["evaluation_level"] == "slice"
    assert report["modalities"]["chest_xray"]["evaluation_level"] == "image"
    assert report["modalities"]["knee_mri"]["evaluation_level"] == "exam"


def test_unified_registry_rejects_checkpoint_substitution(tmp_path):
    registry = json.loads(DEFAULT_MULTIMODAL_REGISTRY.read_text(encoding="utf-8"))
    registry["modalities"]["knee_mri"]["checkpoint_sha256"] = "0" * 64
    modified = tmp_path / "registry.json"
    modified.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(MultimodalRegistryError, match="Checkpoint registry mismatch"):
        validate_multimodal_evidence_registry(
            modified, project_root=ROOT, recompute_metrics=False
        )

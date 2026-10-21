import json
from pathlib import Path

import pytest

from stat_triage.contracts import Modality
from stat_triage.experiments import (
    ExperimentRegistryError, build_run_record, initialize_registry,
    register_run, validate_experiment_registry,
)


def _setup(tmp_path: Path):
    protocol = tmp_path / "protocol.json"
    split = tmp_path / "split.csv"
    config = tmp_path / "config.json"
    result = tmp_path / "result.json"
    protocol.write_text('{"version":"1"}', encoding="utf-8")
    split.write_text("id,split\n1,train\n", encoding="utf-8")
    config.write_text('{"lr":0.001}', encoding="utf-8")
    result.write_text('{"auc":0.8}', encoding="utf-8")
    registry = tmp_path / "registry.json"
    initialize_registry(registry, protocol, tmp_path)
    record = build_run_record(
        project_root=tmp_path, protocol_path=protocol, run_id="ct-ablation-001",
        modality=Modality.HEAD_CT, purpose="pooling ablation", dataset_id="fixture-v1",
        seed=42, split_manifest=split, config=config, artifacts=[("metrics", result)],
        code_commit="a" * 40,
    )
    return registry, record, result


def test_registry_round_trip_and_duplicate_rejection(tmp_path):
    registry, record, _ = _setup(tmp_path)
    register_run(registry, record, tmp_path)
    assert validate_experiment_registry(registry, tmp_path)["runs"][0]["seed"] == 42
    with pytest.raises(ExperimentRegistryError, match="Duplicate"):
        register_run(registry, record, tmp_path)


def test_registry_detects_artifact_tampering(tmp_path):
    registry, record, result = _setup(tmp_path)
    register_run(registry, record, tmp_path)
    result.write_text('{"auc":0.9}', encoding="utf-8")
    with pytest.raises(ExperimentRegistryError, match="changed"):
        validate_experiment_registry(registry, tmp_path)


def test_registry_rejects_path_escape(tmp_path):
    outside = tmp_path.parent / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    registry, _, _ = _setup(tmp_path)
    payload = json.loads(registry.read_text(encoding="utf-8"))
    payload["protocol"]["path"] = str(outside)
    registry.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ExperimentRegistryError, match="escapes"):
        validate_experiment_registry(registry, tmp_path)

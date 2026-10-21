"""The supported evaluator must reproduce and protect locked evidence."""

import json
from pathlib import Path
import shutil

import pytest

from stat_triage.evaluation import (
    EvidenceBundleError,
    evaluate_locked_predictions,
    validate_evidence_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts" / "rsna_baseline" / "rsna_baseline_bundle"


def test_bundle_registry_verifies_all_locked_artifacts():
    summary = validate_evidence_bundle(BUNDLE)

    assert summary["artifact_count"] == 9
    assert summary["checkpoint_sha256"] == (
        "4a15c605bb1481de7f12a4881f9c9b5adb36d9386ecac0aa6eb3ae7c3ae805b1"
    )
    assert summary["threshold"] == 0.2538610100746155


def test_evaluator_reproduces_locked_rsna_metrics():
    saved = json.loads((BUNDLE / "test_metrics.json").read_text(encoding="utf-8"))
    actual = evaluate_locked_predictions(BUNDLE).to_dict()

    for name, expected in saved["primary_metrics"].items():
        assert actual[name] == pytest.approx(expected)


def test_bundle_validator_rejects_modified_artifact(tmp_path):
    copied = tmp_path / "bundle"
    shutil.copytree(BUNDLE, copied)
    with (copied / "test_predictions.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(EvidenceBundleError, match="Byte count mismatch"):
        validate_evidence_bundle(copied)


from pathlib import Path

import pytest

from stat_triage.mrnet import MRNetEvidenceError, validate_mrnet_evidence_bundle


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts" / "mrnet_baseline"


def test_mrnet_bundle_hashes_references_checkpoint_and_metrics_are_valid():
    summary = validate_mrnet_evidence_bundle(BUNDLE, recompute_metrics=True)
    assert summary["status"] == "complete"
    assert summary["artifact_count"] == 9
    assert summary["checkpoint_epoch"] == 8
    assert summary["test_exams"] == 120
    assert summary["patient_identifiers_available"] is False
    assert summary["metrics_recomputed"] is True


def test_mrnet_bundle_validator_rejects_missing_registry(tmp_path):
    with pytest.raises(MRNetEvidenceError, match="Cannot read"):
        validate_mrnet_evidence_bundle(tmp_path)

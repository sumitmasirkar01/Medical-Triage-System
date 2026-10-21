"""Common evidence is immutable, internally consistent, and modality-neutral."""

from datetime import datetime, timezone

import pytest

from stat_triage.contracts import EvidenceContractError, Modality, ModelEvidence


DIGEST = "a" * 64


def evidence(**overrides):
    values = {
        "exam_id": "EXAM-1",
        "patient_id": "PATIENT-1",
        "modality": Modality.CHEST_XRAY,
        "endpoint": "pneumothorax",
        "model_id": "model-v1",
        "model_sha256": DIGEST,
        "preprocess_version": "cxr-v1",
        "raw_scores": {"pneumothorax": 0.8},
        "calibrated_scores": {"pneumothorax": 0.7},
        "thresholds": {"pneumothorax": 0.6},
        "decisions": {"pneumothorax": True},
        "latency_ms": 12.5,
        "input_hashes": {"frontal.png": DIGEST},
        "created_at": datetime(2026, 10, 20, tzinfo=timezone.utc),
    }
    values.update(overrides)
    return ModelEvidence(**values)


def test_common_evidence_is_immutable_and_validated():
    result = evidence()

    assert result.modality is Modality.CHEST_XRAY
    with pytest.raises(TypeError):
        result.raw_scores["pneumothorax"] = 0.1


def test_decision_must_follow_calibrated_score_and_threshold():
    with pytest.raises(EvidenceContractError, match="disagrees"):
        evidence(decisions={"pneumothorax": False})


def test_score_targets_must_match():
    with pytest.raises(EvidenceContractError, match="same targets"):
        evidence(calibrated_scores={"effusion": 0.7})


def test_audit_timestamp_must_have_timezone():
    with pytest.raises(EvidenceContractError, match="timezone"):
        evidence(created_at=datetime(2026, 10, 20))


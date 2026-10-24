"""Phase 3 outcomes separate supported findings from workflow disposition."""

from datetime import datetime, timezone

import pytest

from stat_triage.contracts import Modality, ModelEvidence
from stat_triage.multimodal_worklist import MultimodalTriagePolicy
from stat_triage.triage_outcome import (
    FindingAssessment,
    TriageDisposition,
    TriageOutcome,
    TriageOutcomeError,
)


def evidence(score: float, *, modality: Modality = Modality.HEAD_CT) -> ModelEvidence:
    target = "any" if modality is Modality.HEAD_CT else "acl"
    return ModelEvidence(
        exam_id="EXAM-1",
        patient_id="PATIENT-1",
        modality=modality,
        endpoint="endpoint",
        model_id="model",
        model_sha256="a" * 64,
        preprocess_version="v1",
        raw_scores={target: score},
        calibrated_scores={target: score},
        thresholds={target: 0.5},
        decisions={target: score >= 0.5},
        latency_ms=1.0,
        input_hashes={"input": "b" * 64},
        created_at=datetime(2026, 10, 24, tzinfo=timezone.utc),
    )


def test_positive_urgent_finding_is_red_and_requires_review():
    item = evidence(0.8)
    decision = MultimodalTriagePolicy().evaluate(item, 0)
    outcome = TriageOutcome.from_policy(item, decision)

    assert outcome.disposition is TriageDisposition.RED
    assert outcome.assessment is FindingAssessment.SUPPORTED_FINDING_DETECTED
    assert outcome.positive_targets == ("any",)
    assert outcome.requires_human_review is True


def test_green_means_no_supported_finding_not_normal():
    item = evidence(0.1)
    outcome = TriageOutcome.from_policy(
        item, MultimodalTriagePolicy().evaluate(item, 0)
    )

    assert outcome.disposition is TriageDisposition.GREEN
    assert outcome.assessment is FindingAssessment.NO_SUPPORTED_FINDING_DETECTED
    assert "normal" not in outcome.message.lower()


def test_near_threshold_yellow_can_have_no_positive_finding():
    item = evidence(0.4, modality=Modality.KNEE_MRI)
    outcome = TriageOutcome.from_policy(
        item, MultimodalTriagePolicy().evaluate(item, 0)
    )

    assert outcome.disposition is TriageDisposition.YELLOW
    assert outcome.assessment is FindingAssessment.NO_SUPPORTED_FINDING_DETECTED
    assert outcome.positive_targets == ()


def test_unsupported_and_failed_are_indeterminate_without_model_evidence():
    unsupported = TriageOutcome.unsupported(
        "unsupported_anatomy", requested_modality=Modality.HEAD_CT
    )
    failed = TriageOutcome.failed("pixel_decode_failed")

    assert unsupported.disposition is TriageDisposition.UNSUPPORTED
    assert failed.disposition is TriageDisposition.FAILED
    assert unsupported.assessment is failed.assessment is FindingAssessment.INDETERMINATE
    assert unsupported.evidence is failed.evidence is None


def test_failure_contract_rejects_false_clinical_certainty():
    with pytest.raises(TriageOutcomeError, match="indeterminate"):
        TriageOutcome(
            disposition=TriageDisposition.FAILED,
            assessment=FindingAssessment.NO_SUPPORTED_FINDING_DETECTED,
            message="No finding.",
            reasons=("decode_failed",),
            requires_human_review=True,
            created_at=datetime.now(timezone.utc),
        )

"""CXR routing uses explicit modality-specific decisions."""

from datetime import datetime, timezone

import pytest

from stat_triage.contracts import Modality, ModelEvidence
from stat_triage.cxr_policy import CXRWorklistPolicy
from stat_triage.datasets.nih_training import NIH_TARGET_ORDER
from stat_triage.worklist import PriorityZone


def evidence(positive_targets=()):
    thresholds = {target: 0.2 for target in NIH_TARGET_ORDER}
    scores = {
        target: (0.8 if target in positive_targets else 0.1)
        for target in NIH_TARGET_ORDER
    }
    return ModelEvidence(
        exam_id="EXAM", patient_id="PATIENT", modality=Modality.CHEST_XRAY,
        endpoint="nih_cxr14_four_target_triage", model_id="model",
        model_sha256="a" * 64, preprocess_version="v1",
        raw_scores=scores, calibrated_scores=scores, thresholds=thresholds,
        decisions={target: target in positive_targets for target in NIH_TARGET_ORDER},
        latency_ms=1.0, input_hashes={"image.png": "b" * 64},
        created_at=datetime.now(timezone.utc),
    )


@pytest.mark.parametrize(
    ("positives", "zone"),
    [
        (("pneumothorax",), PriorityZone.RED),
        (("pleural_effusion",), PriorityZone.YELLOW),
        ((), PriorityZone.GREEN),
    ],
)
def test_cxr_policy_routes_named_locked_decisions(positives, zone):
    routed = CXRWorklistPolicy().route(evidence(positives))

    assert routed.zone is zone
    assert 0 <= routed.within_zone_score <= 1
    assert routed.rule_id == "nih_cxr14_pilot_v1"


def test_cxr_policy_rejects_other_modalities():
    original = evidence()
    values = {**original.__dict__, "modality": Modality.HEAD_CT}
    with pytest.raises(ValueError, match="chest_xray"):
        CXRWorklistPolicy().route(ModelEvidence(**values))

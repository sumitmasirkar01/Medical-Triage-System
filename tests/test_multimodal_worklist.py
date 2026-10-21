from datetime import datetime, timezone

from stat_triage.contracts import Modality, ModelEvidence
from stat_triage.multimodal_worklist import (
    MultimodalTriagePolicy,
    UnifiedWorklistService,
)
from stat_triage.worklist import PriorityZone


def make_evidence(modality, scores, thresholds, *, exam_id="E1"):
    decisions = {target: scores[target] >= value for target, value in thresholds.items()}
    return ModelEvidence(
        exam_id=exam_id, patient_id="P1", modality=modality,
        endpoint="endpoint", model_id="model", model_sha256="a" * 64,
        preprocess_version="v1", raw_scores=scores, calibrated_scores=scores,
        thresholds=thresholds, decisions=decisions, latency_ms=1.0,
        input_hashes={"input": "b" * 64}, created_at=datetime.now(timezone.utc),
    )


def test_ct_any_positive_is_red():
    item = make_evidence(Modality.HEAD_CT, {"any": 0.8}, {"any": 0.5})
    result = MultimodalTriagePolicy().evaluate(item, 0)
    assert result.zone is PriorityZone.RED
    assert result.urgent_targets == ("any",)


def test_cxr_pneumothorax_is_red_but_other_positive_is_yellow():
    urgent = make_evidence(
        Modality.CHEST_XRAY,
        {"pneumothorax": 0.8, "pleural_effusion": 0.2},
        {"pneumothorax": 0.5, "pleural_effusion": 0.5},
    )
    nonurgent = make_evidence(
        Modality.CHEST_XRAY,
        {"pneumothorax": 0.2, "pleural_effusion": 0.8},
        {"pneumothorax": 0.5, "pleural_effusion": 0.5},
    )
    policy = MultimodalTriagePolicy()
    assert policy.evaluate(urgent, 0).zone is PriorityZone.RED
    assert policy.evaluate(nonurgent, 0).zone is PriorityZone.YELLOW


def test_mri_positive_is_yellow_and_not_emergency_red():
    item = make_evidence(Modality.KNEE_MRI, {"acl": 0.8}, {"acl": 0.5})
    result = MultimodalTriagePolicy().evaluate(item, 0)
    assert result.zone is PriorityZone.YELLOW
    assert result.urgent_targets == ()


def test_near_threshold_and_waiting_rules_are_explicit():
    item = make_evidence(Modality.KNEE_MRI, {"acl": 0.38}, {"acl": 0.5})
    policy = MultimodalTriagePolicy(near_threshold_fraction=0.75)
    result = policy.evaluate(item, 0)
    assert result.zone is PriorityZone.YELLOW
    assert result.near_threshold_targets == ("acl",)

    low = make_evidence(Modality.KNEE_MRI, {"acl": 0.1}, {"acl": 0.5})
    assert policy.evaluate(low, 0).zone is PriorityZone.GREEN
    assert policy.evaluate(low, 60).zone is PriorityZone.YELLOW


def test_unified_worklist_orders_overrides_completes_and_audits():
    clock = [1_000.0]
    service = UnifiedWorklistService(clock=lambda: clock[0])
    green = make_evidence(
        Modality.KNEE_MRI, {"acl": 0.1}, {"acl": 0.5}, exam_id="GREEN"
    )
    red = make_evidence(
        Modality.HEAD_CT, {"any": 0.9}, {"any": 0.5}, exam_id="RED"
    )
    service.enqueue(green)
    service.enqueue(red)
    assert [row.case_id for row in service.active_worklist()] == ["RED", "GREEN"]

    row = service.override("GREEN", PriorityZone.RED, "clinical concern", actor="DR1")
    assert row.effective_zone is PriorityZone.RED
    assert row.overridden is True
    assert [event.action for event in service.audit_history("GREEN")] == [
        "ENQUEUED", "OVERRIDDEN"
    ]
    service.complete("GREEN", actor="DR1")
    assert [row.case_id for row in service.active_worklist()] == ["RED"]

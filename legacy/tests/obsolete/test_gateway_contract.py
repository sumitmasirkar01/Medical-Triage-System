"""Queue contract regressions; no checkpoints or inference downloads required.

These cover enqueueing and stored override state. Elapsed-time score refresh and
override ranking policy are separate planned work (PROJECT_PLAN.md, M1.5).
"""

import pytest

from core.base import ModalityTriageOutput, ModalityType, TriagePriorityZone
from core.gateway import DynamicPriorityGateway


def output(probability):
    return ModalityTriageOutput(
        patient_id="PAT-1",
        modality=ModalityType.HEAD_CT,
        p_critical=probability,
        uncertainty_score=0.0,
        conformal_set=["Normal"],
        conformal_guarantee=0.0,
    )


def test_enqueue_preserves_identity_output_and_initial_wait(monkeypatch):
    monkeypatch.setattr("core.gateway.time.time", lambda: 10_000.0)
    gateway = DynamicPriorityGateway()
    prediction = output(0.9)

    item = gateway.enqueue("STUDY-1", "PAT-1", ModalityType.HEAD_CT, 15, prediction)

    assert gateway.get_sorted_worklist() == [item]
    assert item.study_uid == "STUDY-1"
    assert item.patient_id == "PAT-1"
    assert item.modality == ModalityType.HEAD_CT
    assert item.triage_output is prediction
    assert item.t_wait_minutes == pytest.approx(15)
    assert item.arrival_time == 9_100.0
    assert item.dynamic_score == pytest.approx(0.87)
    assert item.priority_zone == TriagePriorityZone.RED
    assert not item.clinician_overridden
    assert item.override_reason == ""


def test_worklist_orders_enqueued_studies_by_score():
    gateway = DynamicPriorityGateway()
    low = gateway.enqueue("LOW", "PAT-1", ModalityType.HEAD_CT, 0, output(0.1))
    high = gateway.enqueue("HIGH", "PAT-1", ModalityType.HEAD_CT, 0, output(0.95))

    assert [item.study_uid for item in gateway.get_sorted_worklist()] == ["HIGH", "LOW"]
    assert high.dynamic_score > low.dynamic_score


def test_override_updates_shared_zone_and_records_reason():
    gateway = DynamicPriorityGateway()
    item = gateway.enqueue("STUDY-1", "PAT-1", ModalityType.HEAD_CT, 0, output(0.1))

    assert gateway.override_item("STUDY-1", TriagePriorityZone.RED, "Manual review requested")
    assert item.priority_zone == item.urgency_zone == TriagePriorityZone.RED
    assert item.clinician_overridden
    assert item.override_reason == "Manual review requested"
    assert not gateway.override_item("MISSING", TriagePriorityZone.RED, "Unknown study")

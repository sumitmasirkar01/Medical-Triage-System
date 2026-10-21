"""Regression tests for the supported evidence-to-worklist policy."""

import pytest

from stat_triage.rsna import RSNAEvidence, RSNA_LABELS
from stat_triage.worklist import PriorityZone, WorklistService


class Clock:
    def __init__(self, value: float = 10_000.0):
        self.value = value

    def __call__(self) -> float:
        return self.value


def evidence(image_id: str, any_score: float, threshold: float = 0.25) -> RSNAEvidence:
    scores = {label: 0.01 for label in RSNA_LABELS}
    scores["any"] = any_score
    return RSNAEvidence(
        image_id=image_id,
        patient_id=f"PAT-{image_id}",
        study_instance_uid=f"STUDY-{image_id}",
        series_instance_uid=f"SERIES-{image_id}",
        scores=scores,
        any_threshold=threshold,
        predicted_any=any_score >= threshold,
        model_sha256="abc123",
        model_epoch=4,
    )


def test_elapsed_time_refreshes_score_and_promotes_waiting_case():
    clock = Clock()
    service = WorklistService(clock=clock, aging_per_minute=0.002)
    initial = service.enqueue(evidence("A", 0.05), arrival_time=clock.value)

    assert initial.wait_minutes == 0
    assert initial.priority_score == pytest.approx(0.05)
    assert initial.effective_zone is PriorityZone.GREEN

    clock.value += 61 * 60
    refreshed = service.active_worklist()[0]
    assert refreshed.wait_minutes == pytest.approx(61)
    assert refreshed.priority_score == pytest.approx(0.172)
    assert refreshed.automatic_zone is PriorityZone.YELLOW


def test_ordering_uses_zone_then_score_then_arrival_then_case_id():
    clock = Clock()
    service = WorklistService(clock=clock, aging_per_minute=0)
    service.enqueue(evidence("GREEN", 0.05), arrival_time=9_900)
    service.enqueue(evidence("RED-B", 0.8), arrival_time=9_900)
    service.enqueue(evidence("RED-A", 0.8), arrival_time=9_900)
    service.enqueue(evidence("YELLOW", 0.2), arrival_time=9_900)

    assert [row.case_id for row in service.active_worklist()] == [
        "RED-A",
        "RED-B",
        "YELLOW",
        "GREEN",
    ]


def test_override_controls_zone_until_explicitly_cleared():
    clock = Clock()
    service = WorklistService(clock=clock)
    service.enqueue(evidence("POSITIVE", 0.9))

    overridden = service.override(
        "POSITIVE", PriorityZone.GREEN, "Known duplicate acquisition", actor="radiologist-1"
    )
    assert overridden.automatic_zone is PriorityZone.RED
    assert overridden.effective_zone is PriorityZone.GREEN
    assert overridden.overridden

    clock.value += 24 * 60 * 60
    assert service.active_worklist()[0].effective_zone is PriorityZone.GREEN

    cleared = service.clear_override(
        "POSITIVE", actor="radiologist-2", reason="Duplicate status corrected"
    )
    assert cleared.effective_zone is PriorityZone.RED
    assert not cleared.overridden
    assert [event.action for event in service.audit_history("POSITIVE")] == [
        "ENQUEUED",
        "OVERRIDDEN",
        "OVERRIDE_CLEARED",
    ]


def test_completion_removes_case_from_active_list_but_retains_history():
    service = WorklistService(clock=Clock())
    original_evidence = evidence("A", 0.3)
    service.enqueue(original_evidence)
    service.complete("A", actor="radiologist-1")

    assert service.active_worklist() == ()
    assert service.evidence_for("A") is original_evidence
    assert [event.action for event in service.audit_history("A")] == ["ENQUEUED", "COMPLETED"]
    with pytest.raises(ValueError, match="not active"):
        service.complete("A", actor="radiologist-1")


def test_duplicate_case_and_incomplete_override_are_rejected():
    service = WorklistService(clock=Clock())
    service.enqueue(evidence("A", 0.3))

    with pytest.raises(ValueError, match="already exists"):
        service.enqueue(evidence("A", 0.4))
    with pytest.raises(ValueError, match="reason"):
        service.override("A", PriorityZone.RED, "", actor="radiologist-1")

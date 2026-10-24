"""Structured and role-aware Phase 3 application boundary."""

from datetime import datetime, timezone
import sqlite3

from stat_triage.application import (
    ActorRole,
    ApplicationActor,
    TriageApplication,
)
from stat_triage.contracts import Modality, ModelEvidence
from stat_triage.multimodal import ChestXrayRequest
from stat_triage.multimodal_worklist import MultimodalTriagePolicy
from stat_triage.persistent_cases import PersistentCaseStore, PersistentTriageCoordinator
from stat_triage.triage_outcome import TriageOutcome
from stat_triage.worklist import PriorityZone


def outcome(score=0.8):
    evidence = ModelEvidence(
        exam_id="EXAM", patient_id="PATIENT", modality=Modality.CHEST_XRAY,
        endpoint="cxr", model_id="model", model_sha256="a" * 64,
        preprocess_version="v1", raw_scores={"pneumothorax": score},
        calibrated_scores={"pneumothorax": score}, thresholds={"pneumothorax": 0.5},
        decisions={"pneumothorax": score >= 0.5}, latency_ms=1.0,
        input_hashes={"patient-name.png": "b" * 64},
        created_at=datetime.now(timezone.utc),
    )
    return TriageOutcome.from_policy(
        evidence, MultimodalTriagePolicy().evaluate(evidence, 0)
    )


class TriageStub:
    def __init__(self, result):
        self.result = result

    def triage(self, _request):
        return self.result


def application(tmp_path, result=None):
    store = PersistentCaseStore(tmp_path / "cases.db", clock=lambda: 1000.0)
    coordinator = PersistentTriageCoordinator(
        store, triage=TriageStub(result or outcome()), clock=lambda: 1000.0,
    )
    return TriageApplication(coordinator)


CLINICIAN = ApplicationActor("doctor-1", ActorRole.CLINICIAN)
TECHNICIAN = ApplicationActor("tech-1", ActorRole.TECHNICIAN)
AUDITOR = ApplicationActor("audit-1", ActorRole.AUDITOR)


def test_submit_returns_structured_case_without_input_filename(tmp_path):
    response = application(tmp_path).submit(
        ChestXrayRequest(tmp_path / "patient-name.png"),
        case_id="CASE-1", actor=TECHNICIAN,
    )
    payload = response.to_dict()
    assert payload["ok"] is True
    assert payload["code"] == "case_submitted"
    assert payload["data"]["case"]["outcome"]["disposition"] == "RED"
    assert "patient-name.png" not in str(payload)


def test_unsupported_request_is_rejected_before_orchestration(tmp_path):
    response = application(tmp_path).submit(object(), case_id="CASE-1", actor=TECHNICIAN)
    assert response.ok is False
    assert response.code == "unsupported_request_type"


def test_case_identifier_contract_blocks_unsafe_values(tmp_path):
    response = application(tmp_path).submit(
        ChestXrayRequest("x.png"), case_id="../../patient", actor=TECHNICIAN
    )
    assert response.code == "invalid_case_id"


def test_duplicate_submission_returns_conflict_without_exception_detail(tmp_path):
    app = application(tmp_path)
    request = ChestXrayRequest("x.png")
    assert app.submit(request, case_id="CASE", actor=TECHNICIAN).ok
    response = app.submit(request, case_id="CASE", actor=TECHNICIAN)
    assert response.to_dict() == {"ok": False, "code": "case_conflict", "data": {}}


def test_only_clinician_can_override_or_complete(tmp_path):
    app = application(tmp_path)
    app.submit(ChestXrayRequest("x.png"), case_id="CASE", actor=TECHNICIAN)
    assert app.override("CASE", PriorityZone.GREEN, "review", actor=TECHNICIAN).code == "forbidden"
    assert app.complete("CASE", actor=TECHNICIAN).code == "forbidden"
    assert app.override("CASE", PriorityZone.GREEN, "review", actor=CLINICIAN).ok
    assert app.complete("CASE", actor=CLINICIAN).ok


def test_worklist_reflects_override_and_completion(tmp_path):
    app = application(tmp_path)
    app.submit(ChestXrayRequest("x.png"), case_id="CASE", actor=TECHNICIAN)
    assert app.worklist(actor=TECHNICIAN).data["cases"][0]["effective_zone"] == "RED"
    app.override("CASE", PriorityZone.GREEN, "reviewed", actor=CLINICIAN)
    assert app.worklist(actor=CLINICIAN).data["cases"][0]["effective_zone"] == "GREEN"
    app.complete("CASE", actor=CLINICIAN)
    assert app.worklist(actor=TECHNICIAN).data["count"] == 0


def test_failed_submission_is_queryable_but_absent_from_worklist(tmp_path):
    app = application(tmp_path, TriageOutcome.failed("decode_failed"))
    submitted = app.submit(ChestXrayRequest("x.png"), case_id="FAILED", actor=TECHNICIAN)
    assert submitted.data["case"]["status"] == "REJECTED"
    assert app.case("FAILED", actor=AUDITOR).data["case"]["outcome"]["disposition"] == "FAILED"
    assert app.worklist(actor=TECHNICIAN).data["count"] == 0


def test_audit_access_is_role_restricted_and_structured(tmp_path):
    app = application(tmp_path)
    app.submit(ChestXrayRequest("x.png"), case_id="CASE", actor=TECHNICIAN)
    assert app.audit(actor=TECHNICIAN).code == "forbidden"
    response = app.audit(actor=AUDITOR, case_id="CASE")
    assert response.ok
    assert response.data["events"][0]["action"] == "SUBMITTED"
    assert len(response.data["events"][0]["event_hash"]) == 64


def test_unknown_case_and_invalid_transition_use_stable_codes(tmp_path):
    app = application(tmp_path)
    assert app.case("MISSING", actor=CLINICIAN).code == "case_not_found"
    app.submit(ChestXrayRequest("x.png"), case_id="CASE", actor=TECHNICIAN)
    assert app.clear_override("CASE", "none", actor=CLINICIAN).code == "invalid_case_transition"


def test_health_fails_closed_when_integrity_is_broken(tmp_path):
    app = application(tmp_path)
    app.submit(ChestXrayRequest("x.png"), case_id="CASE", actor=TECHNICIAN)
    assert app.health().code == "healthy"
    with sqlite3.connect(app.store.path) as connection:
        connection.execute("DROP TRIGGER audit_events_no_update")
        connection.execute("UPDATE audit_events SET actor='tampered'")
    response = app.health()
    assert response.ok is False
    assert response.code == "unhealthy"


def test_audit_retrieval_fails_closed_when_history_is_deleted(tmp_path):
    app = application(tmp_path)
    app.submit(ChestXrayRequest("x.png"), case_id="CASE", actor=TECHNICIAN)
    with sqlite3.connect(app.store.path) as connection:
        connection.execute("DROP TRIGGER audit_events_no_delete")
        connection.execute("DELETE FROM audit_events")
    response = app.audit(actor=AUDITOR, case_id="CASE")
    assert response.ok is False
    assert response.code == "case_integrity_failure"

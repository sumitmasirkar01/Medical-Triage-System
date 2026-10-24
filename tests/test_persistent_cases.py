"""Restart-safe Phase 3 case orchestration and audit integrity."""

from datetime import datetime, timezone
import hashlib
import json
import sqlite3

import pytest

from stat_triage.contracts import Modality, ModelEvidence
from stat_triage.multimodal import ChestXrayRequest
from stat_triage.persistent_cases import (
    CaseStoreError,
    PersistentCaseStore,
    PersistentTriageCoordinator,
)
from stat_triage.triage_outcome import TriageOutcome
from stat_triage.worklist import PriorityZone


def evidence(score=0.8, exam_id="E1", input_name="image"):
    return ModelEvidence(
        exam_id=exam_id, patient_id="P1", modality=Modality.CHEST_XRAY,
        endpoint="cxr", model_id="model", model_sha256="a" * 64,
        preprocess_version="v1", raw_scores={"pneumothorax": score},
        calibrated_scores={"pneumothorax": score}, thresholds={"pneumothorax": 0.5},
        decisions={"pneumothorax": score >= 0.5}, latency_ms=1.0,
        input_hashes={input_name: "b" * 64}, created_at=datetime.now(timezone.utc),
    )


class TriageStub:
    def __init__(self, outcome):
        self.outcome = outcome

    def triage(self, _request):
        return self.outcome


def routed_outcome(score=0.8, exam_id="E1", input_name="image"):
    from stat_triage.multimodal_worklist import MultimodalTriagePolicy

    item = evidence(score, exam_id, input_name)
    return TriageOutcome.from_policy(item, MultimodalTriagePolicy().evaluate(item, 0))


def test_case_and_audit_survive_store_restart(tmp_path):
    clock = [1000.0]
    path = tmp_path / "cases.sqlite3"
    store = PersistentCaseStore(path, clock=lambda: clock[0])
    store.create("CASE-1", routed_outcome(), actor="gateway")

    reopened = PersistentCaseStore(path, clock=lambda: clock[0])
    case = reopened.get("CASE-1")
    assert case.status == "ACTIVE"
    assert case.outcome.evidence.exam_id == "E1"
    assert [event.action for event in reopened.audit_history("CASE-1")] == ["SUBMITTED"]
    assert reopened.validate_integrity() == {
        "status": "passed", "cases": 1, "audit_events": 1,
    }


def test_failed_submission_is_persisted_but_not_queued(tmp_path):
    store = PersistentCaseStore(tmp_path / "cases.db", clock=lambda: 1000.0)
    coordinator = PersistentTriageCoordinator(
        store, triage=TriageStub(TriageOutcome.failed("decode_failed")),
        clock=lambda: 1000.0,
    )
    case = coordinator.submit(object(), case_id="FAILED-1", actor="gateway")
    assert case.status == "REJECTED"
    assert coordinator.active_worklist() == ()
    assert store.audit_history()[0].details["disposition"] == "FAILED"


def test_persistent_worklist_orders_and_ages_after_restart(tmp_path):
    clock = [1000.0]
    store = PersistentCaseStore(tmp_path / "cases.db", clock=lambda: clock[0])
    store.create("GREEN", routed_outcome(0.1, "GREEN"))
    clock[0] = 1010.0
    store.create("RED", routed_outcome(0.8, "RED"))
    clock[0] = 1060.0
    coordinator = PersistentTriageCoordinator(store, clock=lambda: clock[0])
    rows = coordinator.active_worklist()
    assert [row.case_id for row in rows] == ["RED", "GREEN"]
    assert rows[1].wait_minutes == pytest.approx(1.0)


def test_override_clear_and_completion_are_atomic_and_audited(tmp_path):
    store = PersistentCaseStore(tmp_path / "cases.db", clock=lambda: 1000.0)
    store.create("CASE", routed_outcome(0.1), actor="gateway")
    assert store.override("CASE", PriorityZone.RED, "clinical concern", actor="DR1").override_zone is PriorityZone.RED
    assert store.clear_override("CASE", actor="DR1", reason="reviewed").override_zone is None
    completed = store.complete("CASE", actor="DR1")
    assert completed.status == "COMPLETED"
    history = store.audit_history("CASE")
    assert [event.action for event in history] == [
        "SUBMITTED", "OVERRIDDEN", "OVERRIDE_CLEARED", "COMPLETED",
    ]
    assert history[2].details["previous_zone"] == "RED"
    with pytest.raises(CaseStoreError, match="not active"):
        store.override("CASE", PriorityZone.GREEN, "late", actor="DR1")


def test_duplicate_case_identifier_is_rejected(tmp_path):
    store = PersistentCaseStore(tmp_path / "cases.db")
    store.create("CASE", routed_outcome())
    with pytest.raises(CaseStoreError, match="already exists"):
        store.create("CASE", routed_outcome())


def test_case_tampering_is_detected(tmp_path):
    path = tmp_path / "cases.db"
    store = PersistentCaseStore(path)
    store.create("CASE", routed_outcome())
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE cases SET status='COMPLETED' WHERE case_id='CASE'")
    with pytest.raises(CaseStoreError, match="case integrity mismatch"):
        store.get("CASE")


def test_audit_tampering_is_detected(tmp_path):
    path = tmp_path / "cases.db"
    store = PersistentCaseStore(path)
    store.create("CASE", routed_outcome())
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TRIGGER audit_events_no_update")
        connection.execute("UPDATE audit_events SET actor='attacker' WHERE sequence=1")
    with pytest.raises(CaseStoreError, match="audit event hash mismatch"):
        store.validate_integrity()


def test_audit_rows_are_immutable_through_normal_database_access(tmp_path):
    path = tmp_path / "cases.db"
    store = PersistentCaseStore(path)
    store.create("CASE", routed_outcome())
    with sqlite3.connect(path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="audit events are immutable"):
            connection.execute("UPDATE audit_events SET actor='attacker'")
        with pytest.raises(sqlite3.IntegrityError, match="audit events are immutable"):
            connection.execute("DELETE FROM audit_events")
    assert store.validate_integrity()["audit_events"] == 1


def test_deleted_entire_audit_log_is_detected_by_anchor(tmp_path):
    path = tmp_path / "cases.db"
    store = PersistentCaseStore(path)
    store.create("CASE", routed_outcome())
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TRIGGER audit_events_no_delete")
        connection.execute("DELETE FROM audit_events")
    with pytest.raises(CaseStoreError, match="audit event count mismatch"):
        store.validate_integrity()
    with pytest.raises(CaseStoreError, match="audit event count mismatch"):
        store.audit_history("CASE")


def test_deleted_audit_tail_is_detected_and_blocks_new_writes(tmp_path):
    path = tmp_path / "cases.db"
    store = PersistentCaseStore(path)
    store.create("FIRST", routed_outcome(exam_id="FIRST"))
    store.create("SECOND", routed_outcome(exam_id="SECOND"))
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TRIGGER audit_events_no_delete")
        connection.execute("DELETE FROM audit_events WHERE sequence=2")
    with pytest.raises(CaseStoreError, match="audit event count mismatch"):
        store.validate_integrity()
    with pytest.raises(CaseStoreError, match="audit anchor mismatch before append"):
        store.create("THIRD", routed_outcome(exam_id="THIRD"))
    with pytest.raises(KeyError):
        store.get("THIRD")


def test_missing_initial_submission_is_detected_even_with_reanchored_chain(tmp_path):
    path = tmp_path / "cases.db"
    store = PersistentCaseStore(path)
    store.create("CASE", routed_outcome())
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("DROP TRIGGER audit_events_no_update")
        row = connection.execute("SELECT * FROM audit_events WHERE sequence=1").fetchone()
        details = '{"disposition":"RED","status":"ACTIVE"}'
        payload = {
            "timestamp": row["timestamp"], "case_id": "CASE", "action": "COMPLETED",
            "actor": row["actor"], "details": json.loads(details),
            "previous_hash": "0" * 64,
        }
        event_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        connection.execute(
            "UPDATE audit_events SET action='COMPLETED',details_json=?,event_hash=? WHERE sequence=1",
            (details, event_hash),
        )
        connection.execute(
            "UPDATE metadata SET value=? WHERE key='audit_chain_head'", (event_hash,)
        )
    with pytest.raises(CaseStoreError, match="exactly one initial submission"):
        store.validate_integrity()


def test_version_one_store_is_validated_and_migrated(tmp_path):
    path = tmp_path / "cases.db"
    store = PersistentCaseStore(path)
    store.create("CASE", routed_outcome())
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TRIGGER audit_events_no_update")
        connection.execute("DROP TRIGGER audit_events_no_delete")
        connection.execute("UPDATE metadata SET value='1' WHERE key='schema_version'")
        connection.execute(
            "DELETE FROM metadata WHERE key IN ('audit_event_count','audit_chain_head')"
        )
    migrated = PersistentCaseStore(path)
    assert migrated.validate_integrity() == {
        "status": "passed", "cases": 1, "audit_events": 1,
    }
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()[0] == "2"


def test_coordinator_persists_service_outcome_without_input_paths(tmp_path):
    source = tmp_path / "patient-name.png"
    source.touch()
    store = PersistentCaseStore(tmp_path / "cases.db", clock=lambda: 1000.0)
    coordinator = PersistentTriageCoordinator(
        store,
        triage=TriageStub(routed_outcome(input_name="patient-name.png")),
        clock=lambda: 1000.0,
    )
    coordinator.submit(ChestXrayRequest(source), case_id="ANON-1")
    with sqlite3.connect(store.path) as connection:
        serialized = connection.execute(
            "SELECT outcome_json FROM cases WHERE case_id='ANON-1'"
        ).fetchone()[0]
    assert "patient-name.png" not in serialized
    assert "input_0001" in serialized

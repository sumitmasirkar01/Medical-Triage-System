"""Transactional persistence and orchestration for Phase 3 triage cases."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time
from types import MappingProxyType
from typing import Callable, Mapping

from .contracts import Modality, ModelEvidence
from .multimodal_worklist import MultimodalTriagePolicy, UnifiedWorklistRow
from .triage_outcome import TriageDisposition, TriageOutcome
from .triage_service import UnifiedTriageService
from .worklist import CaseStatus, PriorityZone


SCHEMA_VERSION = 1
_ZONE_BASE = {PriorityZone.GREEN: 0.0, PriorityZone.YELLOW: 10.0, PriorityZone.RED: 20.0}
_ZONE_RANK = {PriorityZone.GREEN: 0, PriorityZone.YELLOW: 1, PriorityZone.RED: 2}


class CaseStoreError(RuntimeError):
    """Persistent case data is invalid, inconsistent, or cannot be changed."""


class CaseConflictError(CaseStoreError):
    """A case identifier already exists."""


class CaseTransitionError(CaseStoreError):
    """A requested case state transition is not permitted."""


class CaseIntegrityError(CaseStoreError):
    """Stored case or audit data failed integrity validation."""


@dataclass(frozen=True)
class StoredCase:
    case_id: str
    outcome: TriageOutcome
    status: str
    arrival_time: float
    override_zone: PriorityZone | None = None
    override_reason: str | None = None
    completed_at: float | None = None


@dataclass(frozen=True)
class PersistentAuditEvent:
    sequence: int
    timestamp: float
    case_id: str
    action: str
    actor: str
    details: Mapping[str, str]
    previous_hash: str
    event_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _evidence_dict(evidence: ModelEvidence | None) -> dict[str, object] | None:
    if evidence is None:
        return None
    return {
        "exam_id": evidence.exam_id, "patient_id": evidence.patient_id,
        "modality": evidence.modality.value, "endpoint": evidence.endpoint,
        "model_id": evidence.model_id, "model_sha256": evidence.model_sha256,
        "preprocess_version": evidence.preprocess_version,
        "raw_scores": dict(evidence.raw_scores),
        "calibrated_scores": dict(evidence.calibrated_scores),
        "thresholds": dict(evidence.thresholds), "decisions": dict(evidence.decisions),
        "latency_ms": evidence.latency_ms,
        "input_hashes": {
            f"input_{index:04d}": digest
            for index, digest in enumerate(evidence.input_hashes.values(), start=1)
        },
        "created_at": evidence.created_at.isoformat(), "warnings": list(evidence.warnings),
        "study_instance_uid": evidence.study_instance_uid,
        "series_instance_uid": evidence.series_instance_uid,
    }


def _evidence_from_dict(value: Mapping[str, object] | None) -> ModelEvidence | None:
    if value is None:
        return None
    return ModelEvidence(
        exam_id=str(value["exam_id"]), patient_id=str(value["patient_id"]),
        modality=Modality(str(value["modality"])), endpoint=str(value["endpoint"]),
        model_id=str(value["model_id"]), model_sha256=str(value["model_sha256"]),
        preprocess_version=str(value["preprocess_version"]),
        raw_scores=dict(value["raw_scores"]), calibrated_scores=dict(value["calibrated_scores"]),
        thresholds=dict(value["thresholds"]), decisions=dict(value["decisions"]),
        latency_ms=float(value["latency_ms"]), input_hashes=dict(value["input_hashes"]),
        created_at=datetime.fromisoformat(str(value["created_at"])),
        warnings=tuple(value.get("warnings", ())),
        study_instance_uid=value.get("study_instance_uid"),
        series_instance_uid=value.get("series_instance_uid"),
    )


def _outcome_dict(outcome: TriageOutcome) -> dict[str, object]:
    value = outcome.to_dict()
    value["evidence"] = _evidence_dict(outcome.evidence)
    return value


def _outcome_from_dict(value: Mapping[str, object]) -> TriageOutcome:
    from .triage_outcome import FindingAssessment

    modality = value.get("requested_modality")
    return TriageOutcome(
        disposition=TriageDisposition(str(value["disposition"])),
        assessment=FindingAssessment(str(value["assessment"])),
        message=str(value["message"]), reasons=tuple(value["reasons"]),
        requires_human_review=bool(value["requires_human_review"]),
        created_at=datetime.fromisoformat(str(value["created_at"])),
        evidence=_evidence_from_dict(value.get("evidence")),
        priority_score=value.get("priority_score"),
        positive_targets=tuple(value.get("positive_targets", ())),
        requested_modality=Modality(str(modality)) if modality else None,
    )


def _case_payload(
    case_id: str, outcome_json: str, status: str, arrival_time: float,
    override_zone: str | None, override_reason: str | None, completed_at: float | None,
) -> dict[str, object]:
    return {
        "case_id": case_id, "outcome": json.loads(outcome_json), "status": status,
        "arrival_time": arrival_time, "override_zone": override_zone,
        "override_reason": override_reason, "completed_at": completed_at,
    }


class PersistentCaseStore:
    """SQLite case store with atomic mutations and a tamper-evident audit chain."""

    def __init__(self, path: str | Path, *, clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    outcome_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('ACTIVE','COMPLETED','REJECTED')),
                    arrival_time REAL NOT NULL,
                    override_zone TEXT CHECK(override_zone IN ('RED','YELLOW','GREEN')),
                    override_reason TEXT,
                    completed_at REAL,
                    integrity_sha256 TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    case_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL UNIQUE,
                    FOREIGN KEY(case_id) REFERENCES cases(case_id)
                );
            """)
            row = connection.execute(
                "SELECT value FROM metadata WHERE key='schema_version'"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO metadata(key,value) VALUES('schema_version',?)",
                    (str(SCHEMA_VERSION),),
                )
            elif row["value"] != str(SCHEMA_VERSION):
                raise CaseStoreError("unsupported case store schema version")

    def create(self, case_id: str, outcome: TriageOutcome, *, actor: str = "system") -> StoredCase:
        if not isinstance(case_id, str) or not case_id.strip() or not actor.strip():
            raise ValueError("case_id and actor must be non-empty")
        now = float(self._clock())
        if not math.isfinite(now):
            raise ValueError("clock must return a finite timestamp")
        status = "ACTIVE" if outcome.evidence is not None else "REJECTED"
        outcome_json = _canonical(_outcome_dict(outcome))
        payload = _case_payload(case_id, outcome_json, status, now, None, None, None)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "INSERT INTO cases VALUES(?,?,?,?,?,?,?,?)",
                    (case_id, outcome_json, status, now, None, None, None, _digest(payload)),
                )
            except sqlite3.IntegrityError as exc:
                raise CaseConflictError(f"case already exists: {case_id}") from exc
            self._append_event(connection, case_id, "SUBMITTED", actor, {
                "disposition": outcome.disposition.value, "status": status,
            }, now)
        return self.get(case_id)

    def get(self, case_id: str) -> StoredCase:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM cases WHERE case_id=?", (case_id,)).fetchone()
        if row is None:
            raise KeyError(case_id)
        return self._decode_case(row)

    def list_cases(self, *, status: str | None = None) -> tuple[StoredCase, ...]:
        if status is not None and status not in {"ACTIVE", "COMPLETED", "REJECTED"}:
            raise ValueError("invalid case status")
        query = "SELECT * FROM cases" + (" WHERE status=?" if status else "") + " ORDER BY arrival_time, case_id"
        with self._connect() as connection:
            rows = connection.execute(query, (status,) if status else ()).fetchall()
        return tuple(self._decode_case(row) for row in rows)

    def override(self, case_id: str, zone: PriorityZone, reason: str, *, actor: str) -> StoredCase:
        resolved = PriorityZone(zone)
        if not reason.strip() or not actor.strip():
            raise ValueError("override actor and reason must be non-empty")
        return self._mutate(case_id, "OVERRIDDEN", actor, resolved.value, reason.strip(), False)

    def clear_override(self, case_id: str, *, actor: str, reason: str) -> StoredCase:
        if not actor.strip() or not reason.strip():
            raise ValueError("actor and reason must be non-empty")
        return self._mutate(case_id, "OVERRIDE_CLEARED", actor, None, None, False, reason)

    def complete(self, case_id: str, *, actor: str) -> StoredCase:
        if not actor.strip():
            raise ValueError("actor must be non-empty")
        return self._mutate(case_id, "COMPLETED", actor, None, None, True)

    def _mutate(self, case_id, action, actor, zone, override_reason, complete, audit_reason=None):
        now = float(self._clock())
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM cases WHERE case_id=?", (case_id,)).fetchone()
            if row is None:
                raise KeyError(case_id)
            current = self._decode_case(row)
            if current.status != "ACTIVE":
                raise CaseTransitionError(f"case is not active: {case_id}")
            if action == "OVERRIDE_CLEARED" and current.override_zone is None:
                raise CaseTransitionError(f"case has no override: {case_id}")
            status = "COMPLETED" if complete else "ACTIVE"
            completed_at = now if complete else None
            if complete:
                zone, override_reason = (
                    current.override_zone.value if current.override_zone else None,
                    current.override_reason,
                )
            payload = _case_payload(
                case_id, row["outcome_json"], status, row["arrival_time"],
                zone, override_reason, completed_at,
            )
            connection.execute(
                "UPDATE cases SET status=?,override_zone=?,override_reason=?,completed_at=?,integrity_sha256=? WHERE case_id=?",
                (status, zone, override_reason, completed_at, _digest(payload), case_id),
            )
            details = {"zone": zone, "reason": override_reason} if action == "OVERRIDDEN" else {}
            if action == "OVERRIDE_CLEARED" and current.override_zone is not None:
                details["previous_zone"] = current.override_zone.value
            if audit_reason is not None:
                details["reason"] = audit_reason
            self._append_event(connection, case_id, action, actor, details, now)
        return self.get(case_id)

    def audit_history(self, case_id: str | None = None) -> tuple[PersistentAuditEvent, ...]:
        query = "SELECT * FROM audit_events" + (" WHERE case_id=?" if case_id else "") + " ORDER BY sequence"
        with self._connect() as connection:
            if case_id and connection.execute("SELECT 1 FROM cases WHERE case_id=?", (case_id,)).fetchone() is None:
                raise KeyError(case_id)
            rows = connection.execute(query, (case_id,) if case_id else ()).fetchall()
        return tuple(self._decode_event(row) for row in rows)

    def validate_integrity(self) -> dict[str, int | str]:
        cases = self.list_cases()
        events = self.audit_history()
        previous = "0" * 64
        for event in events:
            if event.previous_hash != previous:
                raise CaseIntegrityError("audit chain previous hash mismatch")
            payload = {
                "timestamp": event.timestamp, "case_id": event.case_id,
                "action": event.action, "actor": event.actor,
                "details": dict(event.details), "previous_hash": previous,
            }
            if _digest(payload) != event.event_hash:
                raise CaseIntegrityError("audit event hash mismatch")
            previous = event.event_hash
        return {"status": "passed", "cases": len(cases), "audit_events": len(events)}

    def _decode_case(self, row: sqlite3.Row) -> StoredCase:
        payload = _case_payload(
            row["case_id"], row["outcome_json"], row["status"], row["arrival_time"],
            row["override_zone"], row["override_reason"], row["completed_at"],
        )
        if _digest(payload) != row["integrity_sha256"]:
            raise CaseIntegrityError(f"case integrity mismatch: {row['case_id']}")
        try:
            outcome = _outcome_from_dict(json.loads(row["outcome_json"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CaseIntegrityError(f"invalid stored outcome: {row['case_id']}") from exc
        return StoredCase(
            case_id=row["case_id"], outcome=outcome, status=row["status"],
            arrival_time=float(row["arrival_time"]),
            override_zone=PriorityZone(row["override_zone"]) if row["override_zone"] else None,
            override_reason=row["override_reason"], completed_at=row["completed_at"],
        )

    def _append_event(self, connection, case_id, action, actor, details, timestamp):
        last = connection.execute(
            "SELECT event_hash FROM audit_events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        previous = last["event_hash"] if last else "0" * 64
        clean_details = {str(key): str(value) for key, value in details.items() if value is not None}
        payload = {
            "timestamp": timestamp, "case_id": case_id, "action": action,
            "actor": actor, "details": clean_details, "previous_hash": previous,
        }
        connection.execute(
            "INSERT INTO audit_events(timestamp,case_id,action,actor,details_json,previous_hash,event_hash) VALUES(?,?,?,?,?,?,?)",
            (timestamp, case_id, action, actor, _canonical(clean_details), previous, _digest(payload)),
        )

    @staticmethod
    def _decode_event(row: sqlite3.Row) -> PersistentAuditEvent:
        return PersistentAuditEvent(
            sequence=row["sequence"], timestamp=row["timestamp"], case_id=row["case_id"],
            action=row["action"], actor=row["actor"], details=json.loads(row["details_json"]),
            previous_hash=row["previous_hash"], event_hash=row["event_hash"],
        )


class PersistentTriageCoordinator:
    """Execute triage, persist outcomes, and expose a restart-safe active worklist."""

    def __init__(
        self, store: PersistentCaseStore, *, triage: UnifiedTriageService | None = None,
        policy: MultimodalTriagePolicy | None = None, clock: Callable[[], float] = time.time,
    ) -> None:
        self.store = store
        self.triage_service = triage or UnifiedTriageService()
        self.policy = policy or MultimodalTriagePolicy()
        self._clock = clock

    def submit(self, request: object, *, case_id: str, actor: str = "system") -> StoredCase:
        return self.store.create(case_id, self.triage_service.triage(request), actor=actor)

    def active_worklist(self) -> tuple[UnifiedWorklistRow, ...]:
        now = self._clock()
        rows = []
        for case in self.store.list_cases(status="ACTIVE"):
            evidence = case.outcome.evidence
            if evidence is None:
                raise CaseStoreError(f"active case has no evidence: {case.case_id}")
            wait = max(0.0, (now - case.arrival_time) / 60.0)
            decision = self.policy.evaluate(evidence, wait)
            zone = case.override_zone or decision.zone
            rows.append(UnifiedWorklistRow(
                case_id=case.case_id, exam_id=evidence.exam_id, patient_id=evidence.patient_id,
                modality=evidence.modality, endpoint=evidence.endpoint,
                positive_targets=decision.positive_targets, rationale=decision.rationale,
                wait_minutes=wait,
                priority_score=_ZONE_BASE[zone] + self.policy.aging_per_minute * wait,
                automatic_zone=decision.zone, effective_zone=zone,
                overridden=case.override_zone is not None, override_reason=case.override_reason,
                model_sha256=evidence.model_sha256,
            ))
        return tuple(sorted(rows, key=lambda row: (-_ZONE_RANK[row.effective_zone], self.store.get(row.case_id).arrival_time, row.case_id)))

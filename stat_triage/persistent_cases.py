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


SCHEMA_VERSION = 2
_GENESIS_HASH = "0" * 64
_AUDIT_COUNT_KEY = "audit_event_count"
_AUDIT_HEAD_KEY = "audit_chain_head"
_AUDIT_ACTIONS = {"SUBMITTED", "OVERRIDDEN", "OVERRIDE_CLEARED", "COMPLETED"}
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
                populated = connection.execute(
                    "SELECT (SELECT COUNT(*) FROM cases) + "
                    "(SELECT COUNT(*) FROM audit_events) AS total"
                ).fetchone()["total"]
                if populated:
                    raise CaseStoreError("case store schema metadata is missing")
                connection.executemany(
                    "INSERT INTO metadata(key,value) VALUES(?,?)",
                    (
                        ("schema_version", str(SCHEMA_VERSION)),
                        (_AUDIT_COUNT_KEY, "0"),
                        (_AUDIT_HEAD_KEY, _GENESIS_HASH),
                    ),
                )
            elif row["value"] == "1":
                self._migrate_v1(connection)
            elif row["value"] != str(SCHEMA_VERSION):
                raise CaseStoreError("unsupported case store schema version")
            self._require_audit_anchor(connection)
            self._install_audit_guards(connection)

    @staticmethod
    def _install_audit_guards(connection: sqlite3.Connection) -> None:
        connection.executescript("""
            CREATE TRIGGER IF NOT EXISTS audit_events_no_update
            BEFORE UPDATE ON audit_events
            BEGIN
                SELECT RAISE(ABORT, 'audit events are immutable');
            END;
            CREATE TRIGGER IF NOT EXISTS audit_events_no_delete
            BEFORE DELETE ON audit_events
            BEGIN
                SELECT RAISE(ABORT, 'audit events are immutable');
            END;
        """)

    def _migrate_v1(self, connection: sqlite3.Connection) -> None:
        """Validate a legacy chain before establishing version-2 anchors."""
        cases = tuple(self._decode_case(row) for row in connection.execute(
            "SELECT * FROM cases ORDER BY arrival_time, case_id"
        ).fetchall())
        events = self._events_from_connection(connection)
        head = self._validate_event_chain(events)
        self._validate_case_lifecycles(cases, events)
        connection.executemany(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
            (
                (_AUDIT_COUNT_KEY, str(len(events))),
                (_AUDIT_HEAD_KEY, head),
                ("schema_version", str(SCHEMA_VERSION)),
            ),
        )

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
        with self._connect() as connection:
            connection.execute("BEGIN")
            cases = tuple(self._decode_case(row) for row in connection.execute(
                "SELECT * FROM cases ORDER BY arrival_time, case_id"
            ).fetchall())
            events = self._events_from_connection(connection)
            expected_count, expected_head = self._require_audit_anchor(connection)
            self._validate_snapshot(cases, events, expected_count, expected_head)
            if case_id and all(case.case_id != case_id for case in cases):
                raise KeyError(case_id)
        return tuple(event for event in events if case_id is None or event.case_id == case_id)

    def validate_integrity(self) -> dict[str, int | str]:
        with self._connect() as connection:
            cases = tuple(self._decode_case(row) for row in connection.execute(
                "SELECT * FROM cases ORDER BY arrival_time, case_id"
            ).fetchall())
            events = self._events_from_connection(connection)
            expected_count, expected_head = self._require_audit_anchor(connection)
        self._validate_snapshot(cases, events, expected_count, expected_head)
        return {"status": "passed", "cases": len(cases), "audit_events": len(events)}

    @classmethod
    def _validate_snapshot(
        cls,
        cases: tuple[StoredCase, ...],
        events: tuple[PersistentAuditEvent, ...],
        expected_count: int,
        expected_head: str,
    ) -> None:
        head = cls._validate_event_chain(events)
        if len(events) != expected_count:
            raise CaseIntegrityError("audit event count mismatch")
        if head != expected_head:
            raise CaseIntegrityError("audit chain head mismatch")
        cls._validate_case_lifecycles(cases, events)

    @staticmethod
    def _validate_event_chain(events: tuple[PersistentAuditEvent, ...]) -> str:
        previous = _GENESIS_HASH
        expected_sequence = 1
        for event in events:
            if event.sequence != expected_sequence:
                raise CaseIntegrityError("audit event sequence mismatch")
            if event.previous_hash != previous:
                raise CaseIntegrityError("audit chain previous hash mismatch")
            if event.action not in _AUDIT_ACTIONS:
                raise CaseIntegrityError("unknown audit action")
            if not event.actor.strip() or not math.isfinite(event.timestamp):
                raise CaseIntegrityError("invalid audit event fields")
            payload = {
                "timestamp": event.timestamp, "case_id": event.case_id,
                "action": event.action, "actor": event.actor,
                "details": dict(event.details), "previous_hash": previous,
            }
            if _digest(payload) != event.event_hash:
                raise CaseIntegrityError("audit event hash mismatch")
            previous = event.event_hash
            expected_sequence += 1
        return previous

    @staticmethod
    def _validate_case_lifecycles(
        cases: tuple[StoredCase, ...], events: tuple[PersistentAuditEvent, ...]
    ) -> None:
        by_case: dict[str, list[PersistentAuditEvent]] = {case.case_id: [] for case in cases}
        for event in events:
            if event.case_id not in by_case:
                raise CaseIntegrityError(f"audit event references unknown case: {event.case_id}")
            by_case[event.case_id].append(event)

        for case in cases:
            history = by_case[case.case_id]
            submitted = [event for event in history if event.action == "SUBMITTED"]
            if len(submitted) != 1 or not history or history[0].action != "SUBMITTED":
                raise CaseIntegrityError(f"case must have exactly one initial submission: {case.case_id}")
            first = history[0]
            expected_initial = "ACTIVE" if case.outcome.evidence is not None else "REJECTED"
            if first.timestamp != case.arrival_time:
                raise CaseIntegrityError(f"case submission timestamp mismatch: {case.case_id}")
            if first.details.get("status") != expected_initial:
                raise CaseIntegrityError(f"case submission status mismatch: {case.case_id}")
            if first.details.get("disposition") != case.outcome.disposition.value:
                raise CaseIntegrityError(f"case submission disposition mismatch: {case.case_id}")

            status = expected_initial
            override_zone = None
            override_reason = None
            completed_at = None
            for event in history[1:]:
                if status != "ACTIVE" or event.action == "SUBMITTED":
                    raise CaseIntegrityError(f"invalid case event transition: {case.case_id}")
                if event.action == "OVERRIDDEN":
                    try:
                        override_zone = PriorityZone(event.details["zone"])
                    except (KeyError, ValueError) as exc:
                        raise CaseIntegrityError(
                            f"invalid override audit event: {case.case_id}"
                        ) from exc
                    override_reason = event.details.get("reason")
                    if not override_reason:
                        raise CaseIntegrityError(f"missing override reason: {case.case_id}")
                elif event.action == "OVERRIDE_CLEARED":
                    if override_zone is None or event.details.get("previous_zone") != override_zone.value:
                        raise CaseIntegrityError(f"invalid override clear event: {case.case_id}")
                    if not event.details.get("reason"):
                        raise CaseIntegrityError(f"missing override clear reason: {case.case_id}")
                    override_zone = None
                    override_reason = None
                elif event.action == "COMPLETED":
                    status = "COMPLETED"
                    completed_at = event.timestamp
                else:
                    raise CaseIntegrityError(f"invalid case event transition: {case.case_id}")

            if (
                case.status != status
                or case.override_zone != override_zone
                or case.override_reason != override_reason
                or case.completed_at != completed_at
            ):
                raise CaseIntegrityError(f"case lifecycle state mismatch: {case.case_id}")

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
        expected_count, expected_head = self._require_audit_anchor(connection)
        actual = connection.execute(
            "SELECT COUNT(*) AS count, COALESCE((SELECT event_hash FROM audit_events "
            "ORDER BY sequence DESC LIMIT 1), ?) AS head FROM audit_events",
            (_GENESIS_HASH,),
        ).fetchone()
        if actual["count"] != expected_count or actual["head"] != expected_head:
            raise CaseIntegrityError("audit anchor mismatch before append")
        previous = expected_head
        clean_details = {str(key): str(value) for key, value in details.items() if value is not None}
        payload = {
            "timestamp": timestamp, "case_id": case_id, "action": action,
            "actor": actor, "details": clean_details, "previous_hash": previous,
        }
        event_hash = _digest(payload)
        connection.execute(
            "INSERT INTO audit_events(timestamp,case_id,action,actor,details_json,previous_hash,event_hash) VALUES(?,?,?,?,?,?,?)",
            (timestamp, case_id, action, actor, _canonical(clean_details), previous, event_hash),
        )
        connection.executemany(
            "UPDATE metadata SET value=? WHERE key=?",
            ((str(expected_count + 1), _AUDIT_COUNT_KEY), (event_hash, _AUDIT_HEAD_KEY)),
        )

    @staticmethod
    def _require_audit_anchor(connection: sqlite3.Connection) -> tuple[int, str]:
        rows = connection.execute(
            "SELECT key,value FROM metadata WHERE key IN (?,?)",
            (_AUDIT_COUNT_KEY, _AUDIT_HEAD_KEY),
        ).fetchall()
        values = {row["key"]: row["value"] for row in rows}
        try:
            count = int(values[_AUDIT_COUNT_KEY])
            head = values[_AUDIT_HEAD_KEY]
        except (KeyError, TypeError, ValueError) as exc:
            raise CaseIntegrityError("audit anchor metadata is missing or invalid") from exc
        if count < 0 or len(head) != 64 or any(char not in "0123456789abcdef" for char in head):
            raise CaseIntegrityError("audit anchor metadata is invalid")
        if count == 0 and head != _GENESIS_HASH:
            raise CaseIntegrityError("empty audit anchor has a non-genesis head")
        if count > 0 and head == _GENESIS_HASH:
            raise CaseIntegrityError("non-empty audit anchor has a genesis head")
        return count, head

    @classmethod
    def _events_from_connection(
        cls, connection: sqlite3.Connection
    ) -> tuple[PersistentAuditEvent, ...]:
        rows = connection.execute("SELECT * FROM audit_events ORDER BY sequence").fetchall()
        try:
            return tuple(cls._decode_event(row) for row in rows)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CaseIntegrityError("invalid audit event encoding") from exc

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

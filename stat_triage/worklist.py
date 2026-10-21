"""Deterministic, auditable worklist policy for supported RSNA evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from types import MappingProxyType
from typing import Callable, Mapping

from .rsna import RSNAEvidence


class PriorityZone(str, Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"


class CaseStatus(str, Enum):
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"


_ZONE_RANK = {
    PriorityZone.GREEN: 0,
    PriorityZone.YELLOW: 1,
    PriorityZone.RED: 2,
}


@dataclass(frozen=True)
class AuditEvent:
    timestamp: float
    case_id: str
    action: str
    actor: str
    details: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


@dataclass
class WorklistCase:
    case_id: str
    evidence: RSNAEvidence
    arrival_time: float
    status: CaseStatus = CaseStatus.ACTIVE
    override_zone: PriorityZone | None = None
    override_reason: str | None = None
    override_actor: str | None = None
    completed_at: float | None = None


@dataclass(frozen=True)
class WorklistRow:
    case_id: str
    patient_id: str
    study_instance_uid: str
    image_id: str
    any_score: float
    threshold: float
    predicted_any: bool
    wait_minutes: float
    priority_score: float
    automatic_zone: PriorityZone
    effective_zone: PriorityZone
    overridden: bool
    override_reason: str | None
    model_sha256: str


class WorklistService:
    """Maintain active cases and evaluate a transparent priority policy.

    `priority_score` is a workflow score, not a calibrated medical probability.
    The automatic zone is RED for a threshold-positive slice, YELLOW for a score
    close to the locked threshold or a case exceeding the routine wait limit, and
    GREEN otherwise. A clinician override controls the effective zone until it is
    explicitly cleared or the case is completed.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.time,
        aging_per_minute: float = 0.002,
        yellow_threshold_fraction: float = 0.5,
        max_green_wait_minutes: float = 60.0,
    ) -> None:
        if aging_per_minute < 0:
            raise ValueError("aging_per_minute must be non-negative")
        if not 0.0 <= yellow_threshold_fraction <= 1.0:
            raise ValueError("yellow_threshold_fraction must be in [0, 1]")
        if max_green_wait_minutes < 0:
            raise ValueError("max_green_wait_minutes must be non-negative")
        self._clock = clock
        self.aging_per_minute = aging_per_minute
        self.yellow_threshold_fraction = yellow_threshold_fraction
        self.max_green_wait_minutes = max_green_wait_minutes
        self._cases: dict[str, WorklistCase] = {}
        self._audit: list[AuditEvent] = []

    def enqueue(
        self,
        evidence: RSNAEvidence,
        *,
        case_id: str | None = None,
        arrival_time: float | None = None,
        actor: str = "system",
    ) -> WorklistRow:
        resolved_id = case_id or evidence.image_id
        if not resolved_id.strip():
            raise ValueError("case_id must not be empty")
        if resolved_id in self._cases:
            raise ValueError(f"case already exists: {resolved_id}")
        now = self._clock()
        arrived = now if arrival_time is None else float(arrival_time)
        if arrived > now:
            raise ValueError("arrival_time cannot be in the future")
        case = WorklistCase(resolved_id, evidence, arrived)
        self._cases[resolved_id] = case
        self._record(case, "ENQUEUED", actor)
        return self._row(case, now)

    def active_worklist(self) -> tuple[WorklistRow, ...]:
        now = self._clock()
        rows = [
            self._row(case, now)
            for case in self._cases.values()
            if case.status is CaseStatus.ACTIVE
        ]
        rows.sort(
            key=lambda row: (
                -_ZONE_RANK[row.effective_zone],
                -row.priority_score,
                self._cases[row.case_id].arrival_time,
                row.case_id,
            )
        )
        return tuple(rows)

    def override(
        self,
        case_id: str,
        zone: PriorityZone,
        reason: str,
        *,
        actor: str,
    ) -> WorklistRow:
        case = self._active_case(case_id)
        if not reason.strip():
            raise ValueError("override reason must not be empty")
        if not actor.strip():
            raise ValueError("override actor must not be empty")
        case.override_zone = PriorityZone(zone)
        case.override_reason = reason.strip()
        case.override_actor = actor.strip()
        self._record(
            case,
            "OVERRIDDEN",
            actor,
            {"zone": case.override_zone.value, "reason": case.override_reason},
        )
        return self._row(case, self._clock())

    def clear_override(self, case_id: str, *, actor: str, reason: str) -> WorklistRow:
        case = self._active_case(case_id)
        if case.override_zone is None:
            raise ValueError(f"case has no override: {case_id}")
        if not actor.strip() or not reason.strip():
            raise ValueError("actor and reason must not be empty")
        previous_zone = case.override_zone.value
        case.override_zone = None
        case.override_reason = None
        case.override_actor = None
        self._record(
            case,
            "OVERRIDE_CLEARED",
            actor,
            {"previous_zone": previous_zone, "reason": reason.strip()},
        )
        return self._row(case, self._clock())

    def complete(self, case_id: str, *, actor: str) -> None:
        case = self._active_case(case_id)
        if not actor.strip():
            raise ValueError("actor must not be empty")
        case.status = CaseStatus.COMPLETED
        case.completed_at = self._clock()
        self._record(case, "COMPLETED", actor)

    def audit_history(self, case_id: str | None = None) -> tuple[AuditEvent, ...]:
        if case_id is None:
            return tuple(self._audit)
        if case_id not in self._cases:
            raise KeyError(case_id)
        return tuple(event for event in self._audit if event.case_id == case_id)

    def evidence_for(self, case_id: str) -> RSNAEvidence:
        """Return immutable evidence for an existing case, including completed cases."""
        try:
            return self._cases[case_id].evidence
        except KeyError:
            raise KeyError(case_id) from None

    def _active_case(self, case_id: str) -> WorklistCase:
        try:
            case = self._cases[case_id]
        except KeyError:
            raise KeyError(case_id) from None
        if case.status is not CaseStatus.ACTIVE:
            raise ValueError(f"case is not active: {case_id}")
        return case

    def _automatic_zone(self, evidence: RSNAEvidence, wait_minutes: float) -> PriorityZone:
        if evidence.predicted_any:
            return PriorityZone.RED
        if (
            evidence.any_score >= evidence.any_threshold * self.yellow_threshold_fraction
            or wait_minutes >= self.max_green_wait_minutes
        ):
            return PriorityZone.YELLOW
        return PriorityZone.GREEN

    def _row(self, case: WorklistCase, now: float) -> WorklistRow:
        wait = max(0.0, (now - case.arrival_time) / 60.0)
        automatic_zone = self._automatic_zone(case.evidence, wait)
        effective_zone = case.override_zone or automatic_zone
        score = case.evidence.any_score + self.aging_per_minute * wait
        return WorklistRow(
            case_id=case.case_id,
            patient_id=case.evidence.patient_id,
            study_instance_uid=case.evidence.study_instance_uid,
            image_id=case.evidence.image_id,
            any_score=case.evidence.any_score,
            threshold=case.evidence.any_threshold,
            predicted_any=case.evidence.predicted_any,
            wait_minutes=wait,
            priority_score=score,
            automatic_zone=automatic_zone,
            effective_zone=effective_zone,
            overridden=case.override_zone is not None,
            override_reason=case.override_reason,
            model_sha256=case.evidence.model_sha256,
        )

    def _record(
        self,
        case: WorklistCase,
        action: str,
        actor: str,
        details: Mapping[str, str] | None = None,
    ) -> None:
        self._audit.append(
            AuditEvent(
                timestamp=self._clock(),
                case_id=case.case_id,
                action=action,
                actor=actor,
                details=details or {},
            )
        )

"""Transparent modality-aware triage policy and auditable unified worklist."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

from .contracts import Modality, ModelEvidence
from .worklist import AuditEvent, CaseStatus, PriorityZone


RED_TARGETS: dict[Modality, frozenset[str]] = {
    Modality.HEAD_CT: frozenset({"any"}),
    Modality.CHEST_XRAY: frozenset({"pneumothorax"}),
    Modality.KNEE_MRI: frozenset(),
}


@dataclass(frozen=True)
class TriagePolicyDecision:
    zone: PriorityZone
    positive_targets: tuple[str, ...]
    urgent_targets: tuple[str, ...]
    near_threshold_targets: tuple[str, ...]
    aged: bool
    priority_score: float
    rationale: tuple[str, ...]


class MultimodalTriagePolicy:
    """Map locked target decisions to workflow urgency without changing them."""

    def __init__(
        self,
        *,
        near_threshold_fraction: float = 0.75,
        max_green_wait_minutes: float = 60.0,
        aging_per_minute: float = 0.002,
    ) -> None:
        if not 0 <= near_threshold_fraction <= 1:
            raise ValueError("near_threshold_fraction must be in [0, 1]")
        if max_green_wait_minutes < 0 or aging_per_minute < 0:
            raise ValueError("wait and aging parameters must be non-negative")
        self.near_threshold_fraction = near_threshold_fraction
        self.max_green_wait_minutes = max_green_wait_minutes
        self.aging_per_minute = aging_per_minute

    def evaluate(self, evidence: ModelEvidence, wait_minutes: float) -> TriagePolicyDecision:
        if wait_minutes < 0:
            raise ValueError("wait_minutes must be non-negative")
        positive = tuple(target for target, value in evidence.decisions.items() if value)
        urgent = tuple(target for target in positive if target in RED_TARGETS[evidence.modality])
        near = tuple(
            target
            for target, threshold in evidence.thresholds.items()
            if not evidence.decisions[target]
            and evidence.calibrated_scores[target]
            >= threshold * self.near_threshold_fraction
        )
        aged = wait_minutes >= self.max_green_wait_minutes
        if urgent:
            zone = PriorityZone.RED
        elif positive or near or aged:
            zone = PriorityZone.YELLOW
        else:
            zone = PriorityZone.GREEN

        ratios = [
            evidence.calibrated_scores[target] / threshold
            for target, threshold in evidence.thresholds.items()
            if threshold > 0
        ]
        signal = min(max(ratios, default=0.0), 2.0)
        zone_base = {
            PriorityZone.GREEN: 0.0,
            PriorityZone.YELLOW: 10.0,
            PriorityZone.RED: 20.0,
        }[zone]
        rationale = []
        if urgent:
            rationale.append("urgent_locked_decision:" + ",".join(urgent))
        if positive and not urgent:
            rationale.append("positive_locked_decision:" + ",".join(positive))
        if near:
            rationale.append("near_locked_threshold:" + ",".join(near))
        if aged:
            rationale.append("routine_wait_limit_reached")
        if not rationale:
            rationale.append("no_locked_or_near_threshold_finding")
        return TriagePolicyDecision(
            zone=zone,
            positive_targets=positive,
            urgent_targets=urgent,
            near_threshold_targets=near,
            aged=aged,
            priority_score=zone_base + signal + self.aging_per_minute * wait_minutes,
            rationale=tuple(rationale),
        )


@dataclass
class UnifiedWorklistCase:
    case_id: str
    evidence: ModelEvidence
    arrival_time: float
    status: CaseStatus = CaseStatus.ACTIVE
    override_zone: PriorityZone | None = None
    override_reason: str | None = None
    completed_at: float | None = None


@dataclass(frozen=True)
class UnifiedWorklistRow:
    case_id: str
    exam_id: str
    patient_id: str
    modality: Modality
    endpoint: str
    positive_targets: tuple[str, ...]
    rationale: tuple[str, ...]
    wait_minutes: float
    priority_score: float
    automatic_zone: PriorityZone
    effective_zone: PriorityZone
    overridden: bool
    override_reason: str | None
    model_sha256: str


class UnifiedWorklistService:
    """Maintain modality-neutral cases with immutable model evidence and audit events."""

    def __init__(
        self,
        *,
        policy: MultimodalTriagePolicy | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.policy = policy or MultimodalTriagePolicy()
        self._clock = clock
        self._cases: dict[str, UnifiedWorklistCase] = {}
        self._audit: list[AuditEvent] = []

    def enqueue(
        self,
        evidence: ModelEvidence,
        *,
        case_id: str | None = None,
        arrival_time: float | None = None,
        actor: str = "system",
    ) -> UnifiedWorklistRow:
        resolved = case_id or evidence.exam_id
        if not resolved.strip() or not actor.strip():
            raise ValueError("case_id and actor must be non-empty")
        if resolved in self._cases:
            raise ValueError(f"case already exists: {resolved}")
        now = self._clock()
        arrived = now if arrival_time is None else float(arrival_time)
        if arrived > now:
            raise ValueError("arrival_time cannot be in the future")
        case = UnifiedWorklistCase(resolved, evidence, arrived)
        self._cases[resolved] = case
        self._record(case, "ENQUEUED", actor, {"modality": evidence.modality.value})
        return self._row(case, now)

    def active_worklist(self) -> tuple[UnifiedWorklistRow, ...]:
        now = self._clock()
        rows = [
            self._row(case, now)
            for case in self._cases.values()
            if case.status is CaseStatus.ACTIVE
        ]
        return tuple(sorted(rows, key=lambda row: (-row.priority_score, row.case_id)))

    def override(
        self, case_id: str, zone: PriorityZone, reason: str, *, actor: str
    ) -> UnifiedWorklistRow:
        case = self._active(case_id)
        if not reason.strip() or not actor.strip():
            raise ValueError("override actor and reason must be non-empty")
        case.override_zone = PriorityZone(zone)
        case.override_reason = reason.strip()
        self._record(
            case, "OVERRIDDEN", actor,
            {"zone": case.override_zone.value, "reason": case.override_reason},
        )
        return self._row(case, self._clock())

    def clear_override(self, case_id: str, *, actor: str, reason: str) -> UnifiedWorklistRow:
        case = self._active(case_id)
        if case.override_zone is None:
            raise ValueError(f"case has no override: {case_id}")
        if not actor.strip() or not reason.strip():
            raise ValueError("actor and reason must be non-empty")
        previous = case.override_zone.value
        case.override_zone = None
        case.override_reason = None
        self._record(
            case, "OVERRIDE_CLEARED", actor,
            {"previous_zone": previous, "reason": reason.strip()},
        )
        return self._row(case, self._clock())

    def complete(self, case_id: str, *, actor: str) -> None:
        case = self._active(case_id)
        if not actor.strip():
            raise ValueError("actor must be non-empty")
        case.status = CaseStatus.COMPLETED
        case.completed_at = self._clock()
        self._record(case, "COMPLETED", actor)

    def evidence_for(self, case_id: str) -> ModelEvidence:
        try:
            return self._cases[case_id].evidence
        except KeyError:
            raise KeyError(case_id) from None

    def audit_history(self, case_id: str | None = None) -> tuple[AuditEvent, ...]:
        if case_id is None:
            return tuple(self._audit)
        if case_id not in self._cases:
            raise KeyError(case_id)
        return tuple(event for event in self._audit if event.case_id == case_id)

    def _active(self, case_id: str) -> UnifiedWorklistCase:
        try:
            case = self._cases[case_id]
        except KeyError:
            raise KeyError(case_id) from None
        if case.status is not CaseStatus.ACTIVE:
            raise ValueError(f"case is not active: {case_id}")
        return case

    def _row(self, case: UnifiedWorklistCase, now: float) -> UnifiedWorklistRow:
        wait = max(0.0, (now - case.arrival_time) / 60.0)
        decision = self.policy.evaluate(case.evidence, wait)
        return UnifiedWorklistRow(
            case_id=case.case_id,
            exam_id=case.evidence.exam_id,
            patient_id=case.evidence.patient_id,
            modality=case.evidence.modality,
            endpoint=case.evidence.endpoint,
            positive_targets=decision.positive_targets,
            rationale=decision.rationale,
            wait_minutes=wait,
            priority_score=decision.priority_score,
            automatic_zone=decision.zone,
            effective_zone=case.override_zone or decision.zone,
            overridden=case.override_zone is not None,
            override_reason=case.override_reason,
            model_sha256=case.evidence.model_sha256,
        )

    def _record(
        self,
        case: UnifiedWorklistCase,
        action: str,
        actor: str,
        details: dict[str, str] | None = None,
    ) -> None:
        self._audit.append(
            AuditEvent(
                timestamp=self._clock(), case_id=case.case_id,
                action=action, actor=actor, details=details or {},
            )
        )

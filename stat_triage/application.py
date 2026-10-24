"""Structured application boundary for the persistent Phase 3 workflow."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re
from types import MappingProxyType
from typing import Mapping

from .multimodal import (
    ChestXrayRequest,
    HeadCTRequest,
    HeadCTStudyRequest,
    KneeMRIRequest,
)
from .persistent_cases import (
    CaseConflictError,
    CaseIntegrityError,
    CaseStoreError,
    CaseTransitionError,
    PersistentAuditEvent,
    PersistentCaseStore,
    PersistentTriageCoordinator,
    StoredCase,
)
from .worklist import PriorityZone


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_REQUEST_TYPES = (HeadCTRequest, HeadCTStudyRequest, ChestXrayRequest, KneeMRIRequest)


class ActorRole(str, Enum):
    SYSTEM = "system"
    TECHNICIAN = "technician"
    CLINICIAN = "clinician"
    AUDITOR = "auditor"


@dataclass(frozen=True)
class ApplicationActor:
    actor_id: str
    role: ActorRole

    def __post_init__(self) -> None:
        if not isinstance(self.actor_id, str) or not _IDENTIFIER.fullmatch(self.actor_id):
            raise ValueError("actor_id must be a safe non-empty identifier")
        if not isinstance(self.role, ActorRole):
            raise ValueError("role must be an ActorRole")


@dataclass(frozen=True)
class ApplicationResponse:
    ok: bool
    code: str
    data: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.ok, bool):
            raise ValueError("ok must be boolean")
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError("code must be non-empty")
        object.__setattr__(self, "data", MappingProxyType(dict(self.data)))

    def to_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "code": self.code, "data": dict(self.data)}


def _valid_identifier(value: object) -> bool:
    return isinstance(value, str) and _IDENTIFIER.fullmatch(value) is not None


def _case_payload(case: StoredCase) -> dict[str, object]:
    evidence = case.outcome.evidence
    return {
        "case_id": case.case_id,
        "status": case.status,
        "arrival_time": case.arrival_time,
        "completed_at": case.completed_at,
        "override_zone": case.override_zone.value if case.override_zone else None,
        "override_reason": case.override_reason,
        "outcome": case.outcome.to_dict(),
        "evidence": None if evidence is None else {
            "exam_id": evidence.exam_id,
            "patient_id": evidence.patient_id,
            "modality": evidence.modality.value,
            "endpoint": evidence.endpoint,
            "model_id": evidence.model_id,
            "model_sha256": evidence.model_sha256,
            "preprocess_version": evidence.preprocess_version,
            "calibrated_scores": dict(evidence.calibrated_scores),
            "thresholds": dict(evidence.thresholds),
            "decisions": dict(evidence.decisions),
            "latency_ms": evidence.latency_ms,
            "created_at": evidence.created_at.isoformat(),
            "warnings": list(evidence.warnings),
        },
    }


def _audit_payload(event: PersistentAuditEvent) -> dict[str, object]:
    return {
        "sequence": event.sequence, "timestamp": event.timestamp,
        "case_id": event.case_id, "action": event.action, "actor": event.actor,
        "details": dict(event.details), "previous_hash": event.previous_hash,
        "event_hash": event.event_hash,
    }


class TriageApplication:
    """Authorization and response boundary over inference and persistent workflow."""

    def __init__(self, coordinator: PersistentTriageCoordinator) -> None:
        self.coordinator = coordinator
        self.store: PersistentCaseStore = coordinator.store

    def submit(
        self, request: object, *, case_id: str, actor: ApplicationActor
    ) -> ApplicationResponse:
        if actor.role not in {ActorRole.SYSTEM, ActorRole.TECHNICIAN, ActorRole.CLINICIAN}:
            return self._error("forbidden")
        if not isinstance(request, _REQUEST_TYPES):
            return self._error("unsupported_request_type")
        if not _valid_identifier(case_id):
            return self._error("invalid_case_id")
        try:
            case = self.coordinator.submit(
                request, case_id=case_id, actor=actor.actor_id
            )
            return self._success("case_submitted", {"case": _case_payload(case)})
        except CaseConflictError:
            return self._error("case_conflict")
        except CaseStoreError:
            return self._error("case_store_failure")
        except Exception:
            return self._error("submission_failure")

    def case(self, case_id: str, *, actor: ApplicationActor) -> ApplicationResponse:
        if not _valid_identifier(case_id):
            return self._error("invalid_case_id")
        try:
            return self._success("case_retrieved", {"case": _case_payload(self.store.get(case_id))})
        except KeyError:
            return self._error("case_not_found")
        except CaseIntegrityError:
            return self._error("case_integrity_failure")

    def worklist(self, *, actor: ApplicationActor) -> ApplicationResponse:
        if actor.role is ActorRole.AUDITOR:
            return self._error("forbidden")
        try:
            rows = [
                {
                    "case_id": row.case_id, "exam_id": row.exam_id,
                    "patient_id": row.patient_id, "modality": row.modality.value,
                    "endpoint": row.endpoint, "positive_targets": list(row.positive_targets),
                    "rationale": list(row.rationale), "wait_minutes": row.wait_minutes,
                    "priority_score": row.priority_score,
                    "automatic_zone": row.automatic_zone.value,
                    "effective_zone": row.effective_zone.value,
                    "overridden": row.overridden, "override_reason": row.override_reason,
                    "model_sha256": row.model_sha256,
                }
                for row in self.coordinator.active_worklist()
            ]
            return self._success("worklist_retrieved", {"cases": rows, "count": len(rows)})
        except CaseIntegrityError:
            return self._error("case_integrity_failure")
        except Exception:
            return self._error("worklist_failure")

    def override(
        self, case_id: str, zone: PriorityZone, reason: str, *, actor: ApplicationActor
    ) -> ApplicationResponse:
        if actor.role is not ActorRole.CLINICIAN:
            return self._error("forbidden")
        if not _valid_identifier(case_id) or not isinstance(reason, str) or not reason.strip():
            return self._error("invalid_override")
        try:
            resolved = PriorityZone(zone)
        except (TypeError, ValueError):
            return self._error("invalid_override")
        try:
            case = self.store.override(
                case_id, resolved, reason.strip(), actor=actor.actor_id
            )
            return self._success("case_overridden", {"case": _case_payload(case)})
        except KeyError:
            return self._error("case_not_found")
        except CaseTransitionError:
            return self._error("invalid_case_transition")
        except CaseIntegrityError:
            return self._error("case_integrity_failure")

    def clear_override(
        self, case_id: str, reason: str, *, actor: ApplicationActor
    ) -> ApplicationResponse:
        if actor.role is not ActorRole.CLINICIAN:
            return self._error("forbidden")
        if not _valid_identifier(case_id) or not isinstance(reason, str) or not reason.strip():
            return self._error("invalid_override")
        try:
            case = self.store.clear_override(
                case_id, actor=actor.actor_id, reason=reason.strip()
            )
            return self._success("override_cleared", {"case": _case_payload(case)})
        except KeyError:
            return self._error("case_not_found")
        except CaseTransitionError:
            return self._error("invalid_case_transition")
        except CaseIntegrityError:
            return self._error("case_integrity_failure")

    def complete(self, case_id: str, *, actor: ApplicationActor) -> ApplicationResponse:
        if actor.role is not ActorRole.CLINICIAN:
            return self._error("forbidden")
        if not _valid_identifier(case_id):
            return self._error("invalid_case_id")
        try:
            case = self.store.complete(case_id, actor=actor.actor_id)
            return self._success("case_completed", {"case": _case_payload(case)})
        except KeyError:
            return self._error("case_not_found")
        except CaseTransitionError:
            return self._error("invalid_case_transition")
        except CaseIntegrityError:
            return self._error("case_integrity_failure")

    def audit(
        self, *, actor: ApplicationActor, case_id: str | None = None
    ) -> ApplicationResponse:
        if actor.role not in {ActorRole.CLINICIAN, ActorRole.AUDITOR}:
            return self._error("forbidden")
        if case_id is not None and not _valid_identifier(case_id):
            return self._error("invalid_case_id")
        try:
            events = [_audit_payload(event) for event in self.store.audit_history(case_id)]
            return self._success("audit_retrieved", {"events": events, "count": len(events)})
        except KeyError:
            return self._error("case_not_found")
        except CaseIntegrityError:
            return self._error("case_integrity_failure")

    def health(self) -> ApplicationResponse:
        try:
            integrity = self.store.validate_integrity()
            if not math.isfinite(float(integrity["cases"])):
                raise CaseStoreError("invalid integrity result")
            return self._success("healthy", {"case_store": integrity})
        except Exception:
            return self._error("unhealthy")

    @staticmethod
    def _success(code: str, data: Mapping[str, object]) -> ApplicationResponse:
        return ApplicationResponse(True, code, data)

    @staticmethod
    def _error(code: str) -> ApplicationResponse:
        return ApplicationResponse(False, code, {})

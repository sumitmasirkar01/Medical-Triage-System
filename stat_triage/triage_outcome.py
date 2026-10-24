"""Phase 3 output contract separating model findings from workflow disposition."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import math

from .contracts import Modality, ModelEvidence
from .multimodal_worklist import TriagePolicyDecision


class TriageDisposition(str, Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"
    UNSUPPORTED = "UNSUPPORTED"
    FAILED = "FAILED"


class FindingAssessment(str, Enum):
    SUPPORTED_FINDING_DETECTED = "supported_finding_detected"
    NO_SUPPORTED_FINDING_DETECTED = "no_supported_finding_detected"
    INDETERMINATE = "indeterminate"


class TriageOutcomeError(ValueError):
    """A Phase 3 triage outcome is internally inconsistent."""


@dataclass(frozen=True)
class TriageOutcome:
    """One final, auditable result returned by the multimodal triage boundary.

    ``NO_SUPPORTED_FINDING_DETECTED`` is deliberately narrower than "normal".
    Every disposition requires human review in this research demonstration.
    """

    disposition: TriageDisposition
    assessment: FindingAssessment
    message: str
    reasons: tuple[str, ...]
    requires_human_review: bool
    created_at: datetime
    evidence: ModelEvidence | None = None
    priority_score: float | None = None
    positive_targets: tuple[str, ...] = ()
    requested_modality: Modality | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, TriageDisposition):
            raise TriageOutcomeError("disposition must be a TriageDisposition")
        if not isinstance(self.assessment, FindingAssessment):
            raise TriageOutcomeError("assessment must be a FindingAssessment")
        if not isinstance(self.message, str) or not self.message.strip():
            raise TriageOutcomeError("message must be a non-empty string")
        if not isinstance(self.reasons, tuple) or not self.reasons or not all(
            isinstance(reason, str) and reason.strip() for reason in self.reasons
        ):
            raise TriageOutcomeError("reasons must be a tuple of non-empty strings")
        if self.requires_human_review is not True:
            raise TriageOutcomeError("research outcomes must require human review")
        if not isinstance(self.created_at, datetime) or (
            self.created_at.tzinfo is None or self.created_at.utcoffset() is None
        ):
            raise TriageOutcomeError("created_at must include a timezone")
        if not isinstance(self.positive_targets, tuple) or not all(
            isinstance(target, str) and target.strip() for target in self.positive_targets
        ):
            raise TriageOutcomeError("positive_targets must be a tuple of non-empty strings")
        if len(set(self.positive_targets)) != len(self.positive_targets):
            raise TriageOutcomeError("positive_targets cannot contain duplicates")
        if self.requested_modality is not None and not isinstance(
            self.requested_modality, Modality
        ):
            raise TriageOutcomeError("requested_modality must be a Modality value")

        is_success = self.disposition in {
            TriageDisposition.RED,
            TriageDisposition.YELLOW,
            TriageDisposition.GREEN,
        }
        if is_success:
            if not isinstance(self.evidence, ModelEvidence):
                raise TriageOutcomeError("routed outcomes require model evidence")
            if (
                isinstance(self.priority_score, bool)
                or not isinstance(self.priority_score, (int, float))
                or not math.isfinite(float(self.priority_score))
            ):
                raise TriageOutcomeError("routed outcomes require a finite priority score")
            if self.requested_modality not in (None, self.evidence.modality):
                raise TriageOutcomeError("requested modality disagrees with model evidence")
            invalid_targets = set(self.positive_targets).difference(self.evidence.decisions)
            if invalid_targets:
                raise TriageOutcomeError("positive targets are absent from model decisions")
            if any(not self.evidence.decisions[target] for target in self.positive_targets):
                raise TriageOutcomeError("positive targets must have locked positive decisions")
            expected_assessment = (
                FindingAssessment.SUPPORTED_FINDING_DETECTED
                if self.positive_targets
                else FindingAssessment.NO_SUPPORTED_FINDING_DETECTED
            )
            if self.assessment is not expected_assessment:
                raise TriageOutcomeError("assessment disagrees with locked target decisions")
        else:
            if self.assessment is not FindingAssessment.INDETERMINATE:
                raise TriageOutcomeError("unsupported and failed outcomes are indeterminate")
            if self.evidence is not None or self.priority_score is not None:
                raise TriageOutcomeError("unsupported and failed outcomes cannot contain evidence")
            if self.positive_targets:
                raise TriageOutcomeError("unsupported and failed outcomes cannot be positive")

    @classmethod
    def from_policy(
        cls,
        evidence: ModelEvidence,
        decision: TriagePolicyDecision,
        *,
        created_at: datetime | None = None,
    ) -> "TriageOutcome":
        disposition = TriageDisposition(decision.zone.value)
        assessment = (
            FindingAssessment.SUPPORTED_FINDING_DETECTED
            if decision.positive_targets
            else FindingAssessment.NO_SUPPORTED_FINDING_DETECTED
        )
        message = (
            "One or more supported findings crossed a locked operating threshold."
            if decision.positive_targets
            else "No supported finding crossed a locked operating threshold."
        )
        return cls(
            disposition=disposition,
            assessment=assessment,
            message=message,
            reasons=decision.rationale,
            requires_human_review=True,
            created_at=created_at or datetime.now(timezone.utc),
            evidence=evidence,
            priority_score=decision.priority_score,
            positive_targets=decision.positive_targets,
            requested_modality=evidence.modality,
        )

    @classmethod
    def unsupported(
        cls,
        reason: str,
        *,
        requested_modality: Modality | None = None,
        created_at: datetime | None = None,
    ) -> "TriageOutcome":
        return cls(
            disposition=TriageDisposition.UNSUPPORTED,
            assessment=FindingAssessment.INDETERMINATE,
            message="The input is outside the supported model contract.",
            reasons=(reason,),
            requires_human_review=True,
            created_at=created_at or datetime.now(timezone.utc),
            requested_modality=requested_modality,
        )

    @classmethod
    def failed(
        cls,
        reason: str,
        *,
        requested_modality: Modality | None = None,
        created_at: datetime | None = None,
    ) -> "TriageOutcome":
        return cls(
            disposition=TriageDisposition.FAILED,
            assessment=FindingAssessment.INDETERMINATE,
            message="The input could not be processed safely.",
            reasons=(reason,),
            requires_human_review=True,
            created_at=created_at or datetime.now(timezone.utc),
            requested_modality=requested_modality,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "disposition": self.disposition.value,
            "assessment": self.assessment.value,
            "message": self.message,
            "reasons": list(self.reasons),
            "requires_human_review": self.requires_human_review,
            "created_at": self.created_at.isoformat(),
            "priority_score": self.priority_score,
            "positive_targets": list(self.positive_targets),
            "requested_modality": (
                self.requested_modality.value if self.requested_modality else None
            ),
        }

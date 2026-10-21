"""Shared immutable contracts for every supported imaging adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Mapping

import math


class Modality(str, Enum):
    HEAD_CT = "head_ct"
    CHEST_XRAY = "chest_xray"
    KNEE_MRI = "knee_mri"


class EvidenceContractError(ValueError):
    """An adapter output violates the common evidence contract."""


def _immutable_mapping(values: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(values))


def _validate_identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceContractError(f"{name} must be a non-empty string")


def _validate_probability_mapping(name: str, values: Mapping[str, float]) -> None:
    if not values:
        raise EvidenceContractError(f"{name} cannot be empty")
    for target, value in values.items():
        _validate_identifier(f"{name} target", target)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EvidenceContractError(f"{name}[{target!r}] must be numeric")
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise EvidenceContractError(f"{name}[{target!r}] must be finite and in [0, 1]")


@dataclass(frozen=True)
class ModelEvidence:
    """One model execution, suitable for audit and downstream policy decisions.

    Scores are model outputs. They are not assumed to be comparable across
    modalities. Only targets with a locked operating point appear in
    ``thresholds`` and ``decisions``.
    """

    exam_id: str
    patient_id: str
    modality: Modality
    endpoint: str
    model_id: str
    model_sha256: str
    preprocess_version: str
    raw_scores: Mapping[str, float]
    calibrated_scores: Mapping[str, float]
    thresholds: Mapping[str, float]
    decisions: Mapping[str, bool]
    latency_ms: float
    input_hashes: Mapping[str, str]
    created_at: datetime
    warnings: tuple[str, ...] = ()
    study_instance_uid: str | None = None
    series_instance_uid: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "exam_id",
            "patient_id",
            "endpoint",
            "model_id",
            "model_sha256",
            "preprocess_version",
        ):
            _validate_identifier(name, getattr(self, name))
        if not isinstance(self.modality, Modality):
            raise EvidenceContractError("modality must be a Modality value")
        if len(self.model_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.model_sha256
        ):
            raise EvidenceContractError("model_sha256 must be a lowercase SHA-256 digest")

        _validate_probability_mapping("raw_scores", self.raw_scores)
        if set(self.calibrated_scores) != set(self.raw_scores):
            raise EvidenceContractError("calibrated_scores must have the same targets as raw_scores")
        _validate_probability_mapping("calibrated_scores", self.calibrated_scores)
        _validate_probability_mapping("thresholds", self.thresholds)
        if set(self.decisions) != set(self.thresholds):
            raise EvidenceContractError("decisions and thresholds must contain the same targets")
        if not set(self.thresholds).issubset(self.raw_scores):
            raise EvidenceContractError("threshold targets must be present in scores")
        for target, decision in self.decisions.items():
            if not isinstance(decision, bool):
                raise EvidenceContractError(f"decisions[{target!r}] must be boolean")
            expected = self.calibrated_scores[target] >= self.thresholds[target]
            if decision != expected:
                raise EvidenceContractError(
                    f"decisions[{target!r}] disagrees with its calibrated score and threshold"
                )

        if isinstance(self.latency_ms, bool) or not isinstance(self.latency_ms, (int, float)):
            raise EvidenceContractError("latency_ms must be numeric")
        if not math.isfinite(float(self.latency_ms)) or self.latency_ms < 0:
            raise EvidenceContractError("latency_ms must be finite and non-negative")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise EvidenceContractError("created_at must include a timezone")
        if not isinstance(self.warnings, tuple) or not all(
            isinstance(item, str) and item.strip() for item in self.warnings
        ):
            raise EvidenceContractError("warnings must be a tuple of non-empty strings")
        if not self.input_hashes:
            raise EvidenceContractError("input_hashes cannot be empty")
        for name, digest in self.input_hashes.items():
            _validate_identifier("input hash name", name)
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise EvidenceContractError(f"input hash for {name!r} is not a lowercase SHA-256")

        for name in ("study_instance_uid", "series_instance_uid"):
            value = getattr(self, name)
            if value is not None:
                _validate_identifier(name, value)

        object.__setattr__(self, "raw_scores", _immutable_mapping(self.raw_scores))
        object.__setattr__(self, "calibrated_scores", _immutable_mapping(self.calibrated_scores))
        object.__setattr__(self, "thresholds", _immutable_mapping(self.thresholds))
        object.__setattr__(self, "decisions", _immutable_mapping(self.decisions))
        object.__setattr__(self, "input_hashes", _immutable_mapping(self.input_hashes))

    @classmethod
    def created_now(cls, **values: object) -> "ModelEvidence":
        """Construct evidence with an explicit UTC audit timestamp."""

        return cls(created_at=datetime.now(timezone.utc), **values)


"""Fail-closed Phase 3 inference and triage boundary."""

from __future__ import annotations

import math

from .contracts import Modality
from .multimodal import (
    ChestXrayRequest,
    HeadCTRequest,
    HeadCTStudyRequest,
    InferenceIssue,
    KneeMRIRequest,
    UnifiedInferenceError,
    UnifiedInferenceRouter,
)
from .multimodal_worklist import MultimodalTriagePolicy
from .triage_outcome import TriageOutcome


def requested_modality(request: object) -> Modality | None:
    if isinstance(request, (HeadCTRequest, HeadCTStudyRequest)):
        return Modality.HEAD_CT
    if isinstance(request, ChestXrayRequest):
        return Modality.CHEST_XRAY
    if isinstance(request, KneeMRIRequest):
        return Modality.KNEE_MRI
    return None


def _input_paths(request: object) -> tuple[object, ...]:
    if isinstance(request, HeadCTRequest):
        return (request.dicom_path,)
    if isinstance(request, HeadCTStudyRequest):
        return request.dicom_paths
    if isinstance(request, ChestXrayRequest):
        return (request.image_path,)
    if isinstance(request, KneeMRIRequest):
        return (request.axial_path, request.coronal_path, request.sagittal_path)
    return ()


class UnifiedTriageService:
    """Return one safe outcome for every supported or rejected inference request."""

    def __init__(
        self,
        *,
        router: UnifiedInferenceRouter | None = None,
        policy: MultimodalTriagePolicy | None = None,
        device: str | None = None,
    ) -> None:
        self.router = router or UnifiedInferenceRouter(device=device)
        self.policy = policy or MultimodalTriagePolicy()

    def triage(self, request: object, *, wait_minutes: float = 0.0) -> TriageOutcome:
        modality = requested_modality(request)
        if isinstance(wait_minutes, bool) or not isinstance(wait_minutes, (int, float)):
            return TriageOutcome.failed(
                "invalid_wait_minutes", requested_modality=modality
            )
        wait = float(wait_minutes)
        if not math.isfinite(wait) or wait < 0:
            return TriageOutcome.failed(
                "invalid_wait_minutes", requested_modality=modality
            )
        paths = _input_paths(request)
        if paths and any(not path.is_file() for path in paths):
            return TriageOutcome.failed(
                "input_file_missing", requested_modality=modality
            )
        try:
            evidence = self.router.predict(request)  # type: ignore[arg-type]
            decision = self.policy.evaluate(evidence, wait)
            return TriageOutcome.from_policy(evidence, decision)
        except UnifiedInferenceError as exc:
            if exc.issue is InferenceIssue.UNSUPPORTED:
                return TriageOutcome.unsupported(
                    exc.code, requested_modality=modality
                )
            return TriageOutcome.failed(exc.code, requested_modality=modality)
        except Exception:
            # The public contract must never leak decoder, path, or runtime details.
            return TriageOutcome.failed(
                "unexpected_inference_failure", requested_modality=modality
            )

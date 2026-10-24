"""The Phase 3 service always produces a safe, stable triage outcome."""

from datetime import datetime, timezone

from stat_triage.contracts import Modality, ModelEvidence
from stat_triage.multimodal import (
    ChestXrayRequest,
    HeadCTStudyRequest,
    InferenceIssue,
    UnifiedInferenceError,
    UnifiedInferenceRouter,
)
from stat_triage.triage_outcome import TriageDisposition
from stat_triage.triage_service import UnifiedTriageService


def evidence(score: float = 0.8) -> ModelEvidence:
    return ModelEvidence(
        exam_id="CT-STUDY", patient_id="P1", modality=Modality.HEAD_CT,
        endpoint="study_level_any_hemorrhage", model_id="locked-model",
        model_sha256="a" * 64, preprocess_version="v1",
        raw_scores={"any": score}, calibrated_scores={"any": score},
        thresholds={"any": 0.2}, decisions={"any": score >= 0.2},
        latency_ms=1.0, input_hashes={"slice": "b" * 64},
        created_at=datetime.now(timezone.utc),
    )


class RouterStub:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def predict(self, _request):
        if self.error:
            raise self.error
        return self.result


class StudyStub:
    def predict_series(self, paths):
        assert tuple(path.name for path in paths) == ("a.dcm", "b.dcm")
        return evidence()


def test_successful_request_returns_policy_outcome_with_evidence(tmp_path):
    (tmp_path / "a.dcm").touch()
    (tmp_path / "b.dcm").touch()
    service = UnifiedTriageService(router=RouterStub(result=evidence()))
    outcome = service.triage(
        HeadCTStudyRequest([tmp_path / "a.dcm", tmp_path / "b.dcm"])
    )
    assert outcome.disposition is TriageDisposition.RED
    assert outcome.evidence is not None
    assert outcome.requested_modality is Modality.HEAD_CT


def test_router_routes_explicit_ct_series_request(tmp_path):
    router = UnifiedInferenceRouter(study_adapter_factory=StudyStub)
    result = router.predict(
        HeadCTStudyRequest([tmp_path / "a.dcm", tmp_path / "b.dcm"])
    )
    assert result.endpoint == "study_level_any_hemorrhage"
    assert router.unload(Modality.HEAD_CT) == (Modality.HEAD_CT,)


def test_unsupported_error_maps_to_stable_code_without_detail(tmp_path):
    error = UnifiedInferenceError(
        "private/path/bad.dcm has the wrong anatomy",
        issue=InferenceIssue.UNSUPPORTED,
        code="unsupported_anatomy",
    )
    source = tmp_path / "private.png"
    source.touch()
    outcome = UnifiedTriageService(router=RouterStub(error=error)).triage(
        ChestXrayRequest(source)
    )
    assert outcome.disposition is TriageDisposition.UNSUPPORTED
    assert outcome.reasons == ("unsupported_anatomy",)
    assert "private" not in outcome.message


def test_runtime_error_maps_to_failed_stable_code():
    error = UnifiedInferenceError(
        "GPU details", issue=InferenceIssue.FAILED, code="model_execution_failed"
    )
    outcome = UnifiedTriageService(router=RouterStub(error=error)).triage(object())
    assert outcome.disposition is TriageDisposition.FAILED
    assert outcome.reasons == ("model_execution_failed",)


def test_unknown_request_is_unsupported_by_real_router():
    outcome = UnifiedTriageService(router=UnifiedInferenceRouter()).triage(object())
    assert outcome.disposition is TriageDisposition.UNSUPPORTED
    assert outcome.reasons == ("unsupported_request_type",)
    assert outcome.requested_modality is None


def test_unexpected_exception_is_contained(tmp_path):
    source = tmp_path / "x.png"
    source.touch()
    outcome = UnifiedTriageService(
        router=RouterStub(error=TypeError("private implementation detail"))
    ).triage(ChestXrayRequest(source))
    assert outcome.disposition is TriageDisposition.FAILED
    assert outcome.reasons == ("unexpected_inference_failure",)


def test_invalid_wait_time_fails_before_inference():
    outcome = UnifiedTriageService(router=RouterStub(result=evidence())).triage(
        ChestXrayRequest("x.png"), wait_minutes=float("nan")
    )
    assert outcome.disposition is TriageDisposition.FAILED
    assert outcome.reasons == ("invalid_wait_minutes",)


def test_missing_input_is_failed_without_exposing_path(tmp_path):
    outcome = UnifiedTriageService(router=UnifiedInferenceRouter()).triage(
        ChestXrayRequest(tmp_path / "private-patient-name.png")
    )
    assert outcome.disposition is TriageDisposition.FAILED
    assert outcome.reasons == ("input_file_missing",)
    assert "private-patient-name" not in outcome.message

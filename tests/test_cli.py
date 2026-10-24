from datetime import datetime, timezone

from stat_triage.cli import parse_arguments, request_from_arguments, run_inference
from stat_triage.contracts import Modality, ModelEvidence
from stat_triage.multimodal import ChestXrayRequest, HeadCTRequest, KneeMRIRequest


class StubRouter:
    def predict(self, request):
        modality = {
            HeadCTRequest: Modality.HEAD_CT,
            ChestXrayRequest: Modality.CHEST_XRAY,
            KneeMRIRequest: Modality.KNEE_MRI,
        }[type(request)]
        return ModelEvidence(
            exam_id=getattr(request, "exam_id", None) or "CT-1",
            patient_id="P-1", modality=modality, endpoint="endpoint",
            model_id="model", model_sha256="a" * 64, preprocess_version="v1",
            raw_scores={"finding": 0.8}, calibrated_scores={"finding": 0.7},
            thresholds={"finding": 0.5}, decisions={"finding": True},
            latency_ms=1.0, input_hashes={"input": "b" * 64},
            created_at=datetime.now(timezone.utc),
        )


def touch(path):
    path.write_bytes(b"fixture")
    return path


def test_cli_builds_explicit_request_for_each_modality(tmp_path):
    ct = touch(tmp_path / "ct.dcm")
    cxr = touch(tmp_path / "cxr.png")
    axial = touch(tmp_path / "a.npy")
    coronal = touch(tmp_path / "c.npy")
    sagittal = touch(tmp_path / "s.npy")
    assert isinstance(request_from_arguments(parse_arguments(["head-ct", "--dicom", str(ct)])), HeadCTRequest)
    assert isinstance(request_from_arguments(parse_arguments(["chest-xray", "--image", str(cxr)])), ChestXrayRequest)
    assert isinstance(
        request_from_arguments(
            parse_arguments([
                "knee-mri", "--axial", str(axial), "--coronal", str(coronal),
                "--sagittal", str(sagittal), "--exam-id", "MRI-1",
            ])
        ),
        KneeMRIRequest,
    )


def test_cli_returns_strict_common_evidence_and_optional_triage(tmp_path):
    source = touch(tmp_path / "cxr.png")
    arguments = parse_arguments([
        "--with-triage", "chest-xray", "--image", str(source), "--exam-id", "CXR-1"
    ])
    result = run_inference(arguments, router=StubRouter())
    assert result["status"] == "complete"
    assert result["evidence"]["modality"] == "chest_xray"
    assert result["evidence"]["exam_id"] == "CXR-1"
    assert result["triage"]["disposition"] == "YELLOW"
    assert result["triage"]["assessment"] == "supported_finding_detected"
    assert result["triage"]["requires_human_review"] is True


def test_cli_rejects_missing_input_before_loading_model(tmp_path):
    arguments = parse_arguments(["head-ct", "--dicom", str(tmp_path / "missing.dcm")])
    try:
        request_from_arguments(arguments)
    except ValueError as error:
        assert "does not exist" in str(error)
    else:
        raise AssertionError("missing CLI input was accepted")


def test_cli_defaults_cxr_exam_id_to_filename(tmp_path):
    source = touch(tmp_path / "case-42.png")
    request = request_from_arguments(
        parse_arguments(["chest-xray", "--image", str(source)])
    )
    assert request.exam_id == "case-42"

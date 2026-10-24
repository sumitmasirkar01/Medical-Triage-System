import json
from datetime import datetime, timezone

import pytest

from scripts.infer_multimodal import main
from stat_triage.cli import parse_arguments, request_from_arguments, run_inference
from stat_triage.contracts import Modality, ModelEvidence
from stat_triage.multimodal import (
    ChestXrayRequest,
    HeadCTRequest,
    HeadCTStudyRequest,
    InferenceIssue,
    KneeMRIRequest,
    UnifiedInferenceError,
)
from stat_triage.triage_service import UnifiedTriageService


class StubRouter:
    def predict(self, request):
        modality = {
            HeadCTRequest: Modality.HEAD_CT,
            HeadCTStudyRequest: Modality.HEAD_CT,
            ChestXrayRequest: Modality.CHEST_XRAY,
            KneeMRIRequest: Modality.KNEE_MRI,
        }[type(request)]
        return ModelEvidence(
            exam_id=getattr(request, "exam_id", None) or "CT-1",
            patient_id="P-1",
            modality=modality,
            endpoint="endpoint",
            model_id="model",
            model_sha256="a" * 64,
            preprocess_version="v1",
            raw_scores={"finding": 0.8},
            calibrated_scores={"finding": 0.7},
            thresholds={"finding": 0.5},
            decisions={"finding": True},
            latency_ms=1.0,
            input_hashes={"private-case-name.png": "b" * 64},
            created_at=datetime.now(timezone.utc),
        )


class RouterMustNotRun:
    def predict(self, request):
        raise AssertionError("router was called for a missing input")


class UnsupportedRouter:
    def predict(self, request):
        raise UnifiedInferenceError(
            "private decoder detail",
            issue=InferenceIssue.UNSUPPORTED,
            code="unsupported_input_format",
        )


def touch(path):
    path.write_bytes(b"fixture")
    return path


def test_cli_builds_explicit_request_for_every_supported_input(tmp_path):
    ct_one = touch(tmp_path / "ct-1.dcm")
    ct_two = touch(tmp_path / "ct-2.dcm")
    cxr = touch(tmp_path / "cxr.png")
    axial = touch(tmp_path / "a.npy")
    coronal = touch(tmp_path / "c.npy")
    sagittal = touch(tmp_path / "s.npy")

    slice_request = request_from_arguments(
        parse_arguments(["head-ct", "--dicom", str(ct_one)])
    )
    study_request = request_from_arguments(
        parse_arguments(
            ["head-ct-series", "--dicom", str(ct_one), str(ct_two)]
        )
    )
    cxr_request = request_from_arguments(
        parse_arguments(
            ["chest-xray", "--image", str(cxr), "--exam-id", "CXR-1"]
        )
    )
    mri_request = request_from_arguments(
        parse_arguments(
            [
                "knee-mri",
                "--axial",
                str(axial),
                "--coronal",
                str(coronal),
                "--sagittal",
                str(sagittal),
                "--exam-id",
                "MRI-1",
            ]
        )
    )

    assert isinstance(slice_request, HeadCTRequest)
    assert isinstance(study_request, HeadCTStudyRequest)
    assert study_request.dicom_paths == (ct_one.resolve(), ct_two.resolve())
    assert isinstance(cxr_request, ChestXrayRequest)
    assert cxr_request.patient_id == "CXR-1"
    assert isinstance(mri_request, KneeMRIRequest)


def test_cli_always_returns_triage_and_aliases_input_hash_names(tmp_path):
    source = touch(tmp_path / "private-case-name.png")
    arguments = parse_arguments(
        ["chest-xray", "--image", str(source), "--exam-id", "CXR-1"]
    )
    result = run_inference(arguments, router=StubRouter())

    assert result["schema_version"] == "2.0"
    assert result["status"] == "complete"
    assert result["outcome"]["disposition"] == "YELLOW"
    assert result["outcome"]["assessment"] == "supported_finding_detected"
    assert result["outcome"]["requires_human_review"] is True
    assert result["evidence"]["modality"] == "chest_xray"
    assert result["evidence"]["input_hashes"] == {"input_0001": "b" * 64}
    assert "private-case-name" not in json.dumps(result)


def test_cli_missing_input_fails_closed_without_path_or_model_load(tmp_path):
    private_name = "patient-smith-missing.dcm"
    arguments = parse_arguments(
        ["head-ct", "--dicom", str(tmp_path / private_name)]
    )
    result = run_inference(arguments, router=RouterMustNotRun())
    rendered = json.dumps(result)

    assert result["outcome"]["disposition"] == "FAILED"
    assert result["outcome"]["reasons"] == ["input_file_missing"]
    assert result["evidence"] is None
    assert private_name not in rendered
    assert str(tmp_path) not in rendered


def test_chest_xray_requires_caller_supplied_exam_identifier(tmp_path):
    source = touch(tmp_path / "private-case-name.png")
    with pytest.raises(SystemExit):
        parse_arguments(["chest-xray", "--image", str(source)])


def test_entrypoint_emits_json_and_meaningful_exit_codes(tmp_path, capsys):
    source = touch(tmp_path / "cxr.png")
    service = UnifiedTriageService(router=StubRouter())

    success = main(
        ["chest-xray", "--image", str(source), "--exam-id", "CXR-1"],
        service=service,
    )
    success_payload = json.loads(capsys.readouterr().out)
    assert success == 0
    assert success_payload["outcome"]["disposition"] == "YELLOW"

    missing_name = "private-missing.dcm"
    failure = main(
        ["head-ct", "--dicom", str(tmp_path / missing_name)],
        service=UnifiedTriageService(router=RouterMustNotRun()),
    )
    failure_output = capsys.readouterr().out
    failure_payload = json.loads(failure_output)
    assert failure == 2
    assert failure_payload["outcome"]["disposition"] == "FAILED"
    assert missing_name not in failure_output

    unsupported = main(
        ["chest-xray", "--image", str(source), "--exam-id", "CXR-1"],
        service=UnifiedTriageService(router=UnsupportedRouter()),
    )
    unsupported_output = capsys.readouterr().out
    unsupported_payload = json.loads(unsupported_output)
    assert unsupported == 3
    assert unsupported_payload["outcome"]["disposition"] == "UNSUPPORTED"
    assert "private decoder detail" not in unsupported_output


def test_entrypoint_converts_output_write_error_to_safe_failure(tmp_path, capsys):
    source = touch(tmp_path / "cxr.png")
    output_directory = tmp_path / "existing-directory"
    output_directory.mkdir()

    exit_code = main(
        [
            "--output",
            str(output_directory),
            "chest-xray",
            "--image",
            str(source),
            "--exam-id",
            "CXR-1",
        ],
        service=UnifiedTriageService(router=StubRouter()),
    )
    rendered = capsys.readouterr().out
    payload = json.loads(rendered)
    assert exit_code == 2
    assert payload["outcome"]["reasons"] == ["output_write_failed"]
    assert str(output_directory) not in rendered


def test_legacy_with_triage_flag_is_a_hidden_no_op(tmp_path):
    source = touch(tmp_path / "cxr.png")
    arguments = parse_arguments(
        [
            "--with-triage",
            "chest-xray",
            "--image",
            str(source),
            "--exam-id",
            "CXR-1",
        ]
    )
    result = run_inference(arguments, router=StubRouter())
    assert result["outcome"]["disposition"] == "YELLOW"

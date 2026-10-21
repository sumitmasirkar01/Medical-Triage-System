"""Fail-closed integration checks across the multimodal inference boundary."""

from pathlib import Path

import numpy as np
import pydicom
import pytest

from scripts.create_demo_dicom import create_demo_ct
from stat_triage.contracts import Modality, ModelEvidence
from stat_triage.multimodal import (
    ChestXrayRequest,
    HeadCTRequest,
    KneeMRIRequest,
    UnifiedInferenceError,
    UnifiedInferenceRouter,
)
from stat_triage.multimodal_worklist import UnifiedWorklistService


@pytest.fixture(scope="module")
def real_router():
    instance = UnifiedInferenceRouter(device="cpu")
    yield instance
    instance.unload()


def test_corrupt_dicom_fails_closed(real_router, tmp_path):
    source = tmp_path / "corrupt.dcm"
    source.write_bytes(b"not a dicom")
    with pytest.raises(UnifiedInferenceError):
        real_router.predict(HeadCTRequest(source))


def test_non_ct_dicom_is_rejected_by_head_ct_contract(real_router, tmp_path):
    source = tmp_path / "wrong_modality.dcm"
    create_demo_ct(source)
    dataset = pydicom.dcmread(source)
    dataset.Modality = "MR"
    dataset.save_as(source, enforce_file_format=True)
    with pytest.raises(UnifiedInferenceError, match="Expected a CT"):
        real_router.predict(HeadCTRequest(source))


def test_unreadable_chest_xray_is_rejected(real_router, tmp_path):
    source = tmp_path / "corrupt.png"
    source.write_bytes(b"not an image")
    with pytest.raises(UnifiedInferenceError):
        real_router.predict(ChestXrayRequest(source))


def test_missing_mri_plane_fails_with_plane_context(real_router, tmp_path):
    valid = np.zeros((3, 32, 32), dtype=np.uint8)
    coronal = tmp_path / "coronal.npy"
    sagittal = tmp_path / "sagittal.npy"
    np.save(coronal, valid)
    np.save(sagittal, valid)
    with pytest.raises(UnifiedInferenceError, match="Invalid axial volume"):
        real_router.predict(
            KneeMRIRequest(
                axial_path=tmp_path / "missing.npy",
                coronal_path=coronal,
                sagittal_path=sagittal,
                exam_id="INCOMPLETE",
            )
        )


def test_malformed_mri_shape_fails_closed(real_router, tmp_path):
    axial = tmp_path / "axial.npy"
    coronal = tmp_path / "coronal.npy"
    sagittal = tmp_path / "sagittal.npy"
    np.save(axial, np.zeros((32, 32), dtype=np.uint8))
    np.save(coronal, np.zeros((3, 32, 32), dtype=np.uint8))
    np.save(sagittal, np.zeros((3, 32, 32), dtype=np.uint8))
    with pytest.raises(UnifiedInferenceError, match=r"Expected \[slices,height,width\]"):
        real_router.predict(
            KneeMRIRequest(
                axial_path=axial, coronal_path=coronal, sagittal_path=sagittal,
                exam_id="MALFORMED",
            )
        )


def test_unsupported_request_object_is_rejected_without_guessing(real_router):
    with pytest.raises(UnifiedInferenceError, match="Unsupported inference request type"):
        real_router.predict(object())


def test_duplicate_worklist_case_is_rejected(valid_model_evidence):
    service = UnifiedWorklistService()
    service.enqueue(valid_model_evidence)
    with pytest.raises(ValueError, match="case already exists"):
        service.enqueue(valid_model_evidence)


@pytest.fixture
def valid_model_evidence():
    from datetime import datetime, timezone

    return ModelEvidence(
        exam_id="DUPLICATE", patient_id="P1", modality=Modality.KNEE_MRI,
        endpoint="endpoint", model_id="model", model_sha256="a" * 64,
        preprocess_version="v1", raw_scores={"acl": 0.1},
        calibrated_scores={"acl": 0.1}, thresholds={"acl": 0.5},
        decisions={"acl": False}, latency_ms=1.0,
        input_hashes={"input": "b" * 64}, created_at=datetime.now(timezone.utc),
    )

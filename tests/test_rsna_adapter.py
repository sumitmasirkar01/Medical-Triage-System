"""Contract tests for the supported RSNA slice-level inference path."""

from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import pytest
import torch

import stat_triage.rsna as rsna
from stat_triage.rsna import (
    DEFAULT_BUNDLE_DIR,
    RSNA_LABELS,
    RSNAInferenceAdapter,
    RSNAInputError,
)
from stat_triage.worklist import PriorityZone, WorklistService


def ct_dataset() -> FileDataset:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.Modality = "CT"
    dataset.PatientID = "PATIENT-1"
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Rows = 32
    dataset.Columns = 32
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.RescaleSlope = 1.0
    dataset.RescaleIntercept = 0.0
    pixels = np.linspace(-1000, 1800, 32 * 32, dtype=np.int16).reshape(32, 32)
    dataset.PixelData = pixels.tobytes()
    return dataset


@pytest.fixture(scope="module")
def adapter() -> RSNAInferenceAdapter:
    return RSNAInferenceAdapter(DEFAULT_BUNDLE_DIR, device="cpu")


def test_preprocessing_reproduces_configured_shape_and_normalization(adapter):
    tensor = adapter.preprocessor.transform_dataset(ct_dataset())

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()
    assert tensor.min().item() >= -2.120
    assert tensor.max().item() <= 2.642


def test_checkpoint_and_operating_point_share_the_locked_contract(adapter):
    assert adapter.model_sha256 == "4a15c605bb1481de7f12a4881f9c9b5adb36d9386ecac0aa6eb3ae7c3ae805b1"
    assert adapter.model_epoch == 4
    assert adapter.any_threshold == pytest.approx(0.2538610100746155)
    assert tuple(adapter.run_config["labels"]) == RSNA_LABELS


def test_input_contract_rejects_other_modalities_and_missing_physics(adapter):
    wrong_modality = ct_dataset()
    wrong_modality.Modality = "CR"
    with pytest.raises(RSNAInputError, match="Expected a CT"):
        adapter.preprocessor.transform_dataset(wrong_modality)

    missing_rescale = ct_dataset()
    del missing_rescale.RescaleSlope
    with pytest.raises(RSNAInputError, match="RescaleSlope"):
        adapter.preprocessor.transform_dataset(missing_rescale)


def test_input_contract_rejects_unsafe_dimensions_before_pixel_decode(adapter):
    oversized = ct_dataset()
    oversized.Rows = rsna.MAX_CT_ROWS + 1
    with pytest.raises(RSNAInputError, match="dimensions exceed"):
        adapter.preprocessor.transform_dataset(oversized)


def test_input_contract_rejects_multiframe_ct_before_pixel_decode(adapter):
    multiframe = ct_dataset()
    multiframe.NumberOfFrames = 2
    with pytest.raises(RSNAInputError, match="single-frame"):
        adapter.preprocessor.transform_dataset(multiframe)


def test_input_contract_rejects_excessive_decoded_allocation(adapter, monkeypatch):
    dataset = ct_dataset()
    monkeypatch.setattr(rsna, "MAX_CT_DECODED_BYTES", 1024)
    with pytest.raises(RSNAInputError, match="decoded pixel data exceeds"):
        adapter.preprocessor.transform_dataset(dataset)


def test_inference_returns_named_auditable_evidence(adapter):
    evidence = adapter.predict_dataset(ct_dataset(), image_id="SYNTHETIC-1")

    assert evidence.image_id == "SYNTHETIC-1"
    assert evidence.patient_id == "PATIENT-1"
    assert tuple(evidence.scores) == RSNA_LABELS
    assert all(0.0 <= score <= 1.0 for score in evidence.scores.values())
    assert evidence.predicted_any == (evidence.any_score >= evidence.any_threshold)
    assert evidence.score_interpretation == "sigmoid_model_output_not_population_calibrated"

    common = evidence.to_model_evidence(
        latency_ms=1.0,
        input_hashes={"synthetic.dcm": "a" * 64},
    )
    assert common.modality.value == "head_ct"
    assert common.decisions["any"] == evidence.predicted_any

    worklist = WorklistService(clock=lambda: 1_000.0)
    row = worklist.enqueue(evidence)
    if evidence.predicted_any:
        expected_zone = PriorityZone.RED
    elif evidence.any_score >= evidence.any_threshold * 0.5:
        expected_zone = PriorityZone.YELLOW
    else:
        expected_zone = PriorityZone.GREEN
    assert row.effective_zone is expected_zone
    assert row.model_sha256 == adapter.model_sha256

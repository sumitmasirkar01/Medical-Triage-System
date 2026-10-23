"""Upload-boundary tests for the supported DICOM ingestion service."""

from io import BytesIO

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import pytest

import stat_triage.ingestion as ingestion
from stat_triage.ingestion import DICOMIngestionError, ingest_dicom_bytes
from stat_triage.rsna import DEFAULT_BUNDLE_DIR, RSNAInferenceAdapter
from stat_triage.worklist import WorklistService


def dicom_bytes(modality: str = "CT") -> bytes:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.Modality = modality
    dataset.PatientID = "UPLOAD-PATIENT"
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.Rows = dataset.Columns = 16
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.RescaleSlope = 1.0
    dataset.RescaleIntercept = -1024.0
    dataset.PixelData = np.full((16, 16), 1064, dtype=np.int16).tobytes()
    stream = BytesIO()
    dataset.save_as(stream, enforce_file_format=True)
    return stream.getvalue()


@pytest.fixture(scope="module")
def adapter() -> RSNAInferenceAdapter:
    return RSNAInferenceAdapter(DEFAULT_BUNDLE_DIR, device="cpu")


def test_valid_upload_reaches_worklist(adapter):
    worklist = WorklistService(clock=lambda: 1_000.0)
    row = ingest_dicom_bytes(dicom_bytes(), "uploaded_case.dcm", adapter, worklist)

    assert row.case_id == "uploaded_case"
    assert row.patient_id == "UPLOAD-PATIENT"
    assert worklist.active_worklist() == (row,)


@pytest.mark.parametrize(
    ("payload", "filename", "message"),
    [
        (b"", "empty.dcm", "empty"),
        (b"plain text", "fake.dcm", "not a valid DICOM"),
    ],
)
def test_invalid_uploads_have_actionable_errors(adapter, payload, filename, message):
    with pytest.raises(DICOMIngestionError, match=message):
        ingest_dicom_bytes(payload, filename, adapter, WorklistService())


def test_non_ct_upload_is_rejected_before_enqueue(adapter):
    worklist = WorklistService()
    with pytest.raises(DICOMIngestionError, match="Expected a CT"):
        ingest_dicom_bytes(dicom_bytes("CR"), "chest.dcm", adapter, worklist)
    assert worklist.active_worklist() == ()


def test_oversized_upload_is_rejected_before_dicom_parsing(adapter, monkeypatch):
    monkeypatch.setattr(ingestion, "MAX_DICOM_UPLOAD_BYTES", 16)
    with pytest.raises(DICOMIngestionError, match="upload limit"):
        ingest_dicom_bytes(b"x" * 17, "oversized.dcm", adapter, WorklistService())

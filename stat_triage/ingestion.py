"""Application service for converting uploaded bytes into worklist evidence."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pydicom
from pydicom.errors import InvalidDicomError

from .rsna import RSNAInferenceAdapter, RSNAInputError
from .worklist import WorklistRow, WorklistService


class DICOMIngestionError(ValueError):
    """User-facing failure while reading or validating an uploaded DICOM."""


MAX_DICOM_UPLOAD_BYTES = 32 * 1024 * 1024


def ingest_dicom_bytes(
    payload: bytes,
    filename: str,
    adapter: RSNAInferenceAdapter,
    worklist: WorklistService,
) -> WorklistRow:
    if not payload:
        raise DICOMIngestionError("The uploaded file is empty.")
    if len(payload) > MAX_DICOM_UPLOAD_BYTES:
        raise DICOMIngestionError(
            "The uploaded DICOM exceeds the 32 MiB upload limit "
            f"({len(payload):,} bytes received)."
        )
    try:
        dataset = pydicom.dcmread(BytesIO(payload))
    except InvalidDicomError as exc:
        raise DICOMIngestionError(
            "The uploaded file is not a valid DICOM Part 10 file."
        ) from exc
    except Exception as exc:
        raise DICOMIngestionError(f"The DICOM header could not be read: {exc}") from exc

    try:
        evidence = adapter.predict_dataset(dataset, image_id=Path(filename).stem)
        return worklist.enqueue(evidence)
    except RSNAInputError as exc:
        raise DICOMIngestionError(str(exc)) from exc

# core/dicom_parser.py
"""
STAT-Triage AI: Native DICOM Ingestion & Physical Radiodensity Engine.
Extracts 16-bit physical Hounsfield Units (HU), verifies photometric polarity,
and generates multi-band diagnostic window tensors for neural foundation models.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
import io
import numpy as np
import pydicom
from pydicom.dataset import FileDataset
import torch


@dataclass
class DICOMMetadata:
    """Clinical metadata extracted from DICOM headers."""
    patient_id: str
    study_instance_uid: str
    series_instance_uid: str
    sop_instance_uid: str
    modality: str
    rows: int
    columns: int
    slice_thickness_mm: float
    pixel_spacing_mm: Tuple[float, float]
    rescale_slope: float
    rescale_intercept: float
    photometric_interpretation: str
    window_center: Optional[float] = None
    window_width: Optional[float] = None


class PhysicalDICOMParser:
    """
    Physical DICOM Ingestion Engine.
    Converts raw 12-bit / 16-bit CT & X-Ray scans into physically calibrated HU tensors.
    """

    # Multi-band diagnostic CT window definitions (Window Level / Center, Window Width)
    CT_WINDOWS = {
        "brain": (40.0, 80.0),       # Range: [0, 80] HU (Parenchyma, acute ischemic stroke)
        "subdural": (75.0, 215.0),   # Range: [-32.5, 182.5] HU (Acute & subacute hemorrhage)
        "bone": (600.0, 2800.0),     # Range: [-800, 2000] HU (Calvarial & skull base fractures)
    }

    @staticmethod
    def apply_window(hu_array: np.ndarray, center: float, width: float) -> np.ndarray:
        """
        Maps continuous HU values into normalized viewing range [0.0, 1.0]:
        val = clip((hu - (center - width/2)) / width, 0.0, 1.0)
        """
        lower = center - (width / 2.0)
        upper = center + (width / 2.0)
        windowed = (hu_array - lower) / (upper - lower + 1e-8)
        return np.clip(windowed, 0.0, 1.0).astype(np.float32)

    def parse_file(self, dicom_source: Union[str, Path, bytes, io.BytesIO]) -> Tuple[torch.Tensor, DICOMMetadata]:
        """
        Parses a DICOM file/stream, applies linear HU recalibration,
        photometric polarity correction, and generates a multi-band 3-channel tensor (1, 3, H, W).
        """
        if isinstance(dicom_source, (str, Path)):
            ds = pydicom.dcmread(str(dicom_source), force=True)
        else:
            if isinstance(dicom_source, bytes):
                dicom_source = io.BytesIO(dicom_source)
            ds = pydicom.dcmread(dicom_source, force=True)

        # 1. Extract physical metadata
        meta = self._extract_metadata(ds)

        # 2. Extract raw sensor array
        raw_pixels = ds.pixel_array.astype(np.float32)

        # 3. Apply linear physical HU rescaling: HU = PixelValue * Slope + Intercept
        hu_array = (raw_pixels * meta.rescale_slope) + meta.rescale_intercept

        # 4. Photometric Interpretation Polarity Check
        # MONOCHROME1: Minimum intensity is rendered as white -> invert to standard MONOCHROME2
        if meta.photometric_interpretation == "MONOCHROME1":
            hu_array = hu_array.max() - hu_array

        # 5. Modality-specific tensor projection
        if meta.modality == "CT":
            ch_brain = self.apply_window(hu_array, *self.CT_WINDOWS["brain"])
            ch_subdural = self.apply_window(hu_array, *self.CT_WINDOWS["subdural"])
            ch_bone = self.apply_window(hu_array, *self.CT_WINDOWS["bone"])
            composite = np.stack([ch_brain, ch_subdural, ch_bone], axis=0)  # (3, H, W)
        else:
            p_low, p_high = np.percentile(hu_array, (0.5, 99.5))
            norm_xray = np.clip((hu_array - p_low) / (p_high - p_low + 1e-8), 0.0, 1.0)
            composite = np.repeat(norm_xray[np.newaxis, :, :], 3, axis=0)   # (3, H, W)

        tensor = torch.from_numpy(composite).unsqueeze(0).float()  # (1, 3, H, W)
        return tensor, meta

    def _extract_metadata(self, ds: FileDataset) -> DICOMMetadata:
        """Safely extracts header tags with robust fallback defaults."""
        patient_id = str(getattr(ds, "PatientID", "UNKNOWN_PATIENT"))
        study_uid = str(getattr(ds, "StudyInstanceUID", "1.2.840.10008.1"))
        series_uid = str(getattr(ds, "SeriesInstanceUID", "1.2.840.10008.2"))
        sop_uid = str(getattr(ds, "SOPInstanceUID", "1.2.840.10008.3"))
        modality = str(getattr(ds, "Modality", "CT")).upper()

        rows = int(getattr(ds, "Rows", 512))
        cols = int(getattr(ds, "Columns", 512))

        thickness = float(getattr(ds, "SliceThickness", 1.0))
        spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
        pixel_spacing = (float(spacing[0]), float(spacing[1]))

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).strip()

        wc = getattr(ds, "WindowCenter", None)
        ww = getattr(ds, "WindowWidth", None)
        if isinstance(wc, (list, pydicom.multival.MultiValue)):
            wc = float(wc[0])
        if isinstance(ww, (list, pydicom.multival.MultiValue)):
            ww = float(ww[0])

        return DICOMMetadata(
            patient_id=patient_id,
            study_instance_uid=study_uid,
            series_instance_uid=series_uid,
            sop_instance_uid=sop_uid,
            modality=modality,
            rows=rows,
            columns=cols,
            slice_thickness_mm=thickness,
            pixel_spacing_mm=pixel_spacing,
            rescale_slope=slope,
            rescale_intercept=intercept,
            photometric_interpretation=photometric,
            window_center=float(wc) if wc is not None else None,
            window_width=float(ww) if ww is not None else None,
        )
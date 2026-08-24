# core/preprocessing.py
"""
Physics-Informed DICOM Preprocessing Engine.
Calibrates 16-bit CT attenuations to Hounsfield Units (HU),
constructs 3-channel composite windows, and sorts volumetric stacks by anatomical Z-axis.
"""

import os
from pathlib import Path
from typing import Union, List, Tuple, Optional
import numpy as np
import pydicom
import torch
import torch.nn.functional as F


class DICOMPhysicsEngine:
    """Physics-calibrated CT attenuation and multi-window projection engine."""

    @staticmethod
    def get_calibrated_hu(dcm: pydicom.dataset.FileDataset) -> np.ndarray:
        """
        Converts raw 12-bit / 16-bit CT pixel arrays to calibrated Hounsfield Units (HU).
        Handles PhotometricInterpretation inversion and protects against corrupted NaN/Inf values.
        """
        pixel_array = dcm.pixel_array.astype(np.float32)

        # Handle MONOCHROME1 inverted photometric interpretation
        if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
            pixel_array = np.amax(pixel_array) - pixel_array

        # Extract rescale metadata
        slope = float(getattr(dcm, "RescaleSlope", 1.0))
        intercept = float(getattr(dcm, "RescaleIntercept", -1024.0))

        # Linear HU calibration: HU = pixel_value * slope + intercept
        if slope != 1.0:
            pixel_array = pixel_array * slope
        pixel_array = pixel_array + intercept

        # Clamp physical limits [-1024 (air) to +3071 (dense cortical bone)]
        return np.nan_to_num(pixel_array, nan=-1024.0, posinf=3071.0, neginf=-1024.0)

    @staticmethod
    def apply_window(hu_img: np.ndarray, center: float, width: float) -> np.ndarray:
        """Clips and normalizes HU pixel intensities into a [0, 1] clinical window."""
        lower = center - width / 2.0
        upper = center + width / 2.0
        windowed = np.clip(hu_img, lower, upper)
        return (windowed - lower) / (upper - lower + 1e-7)

    @classmethod
    def create_tri_window_composite(cls, dcm: pydicom.dataset.FileDataset) -> np.ndarray:
        """
        Constructs a 3-channel clinical tensor (3, H, W):
          - Channel 0: Brain Window (Center: 40, Width: 80) -> Soft tissue & acute intraparenchymal bleeds
          - Channel 1: Subdural Window (Center: 75, Width: 215) -> Subdural / epidural hyperdense collections
          - Channel 2: Bone Window (Center: 600, Width: 2800) -> Calvarial fractures & skull base continuity
        """
        hu = cls.get_calibrated_hu(dcm)
        c0 = cls.apply_window(hu, 40.0, 80.0)
        c1 = cls.apply_window(hu, 75.0, 215.0)
        c2 = cls.apply_window(hu, 600.0, 2800.0)
        return np.stack([c0, c1, c2], axis=0).astype(np.float32)


class DICOMPipelineAdapter:
    """High-throughput volumetric loader and preprocessor for complete DICOM studies."""

    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.target_size = target_size
        self.physics_engine = DICOMPhysicsEngine()

    def process_single_slice(self, dcm_input: Union[str, Path, pydicom.dataset.FileDataset]) -> torch.Tensor:
        """
        Reads a DICOM file or object, converts to tri-window composite, and resizes to target spatial size.
        Returns tensor of shape (3, H, W).
        """
        if isinstance(dcm_input, (str, Path)):
            dcm = pydicom.dcmread(str(dcm_input), stop_before_pixels=False)
        else:
            dcm = dcm_input

        composite = self.physics_engine.create_tri_window_composite(dcm)
        tensor = torch.from_numpy(composite)  # (3, H, W)

        if tensor.shape[1:] != self.target_size:
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=self.target_size,
                mode="bilinear",
                align_corners=False
            ).squeeze(0)

        return tensor

    def load_and_sort_study(self, study_dir: Union[str, Path]) -> Tuple[torch.Tensor, List[str]]:
        """
        Scans a study directory, reads DICOM headers to sort slices along the anatomical Z-axis,
        and constructs a standardized 4D volumetric tensor (Seq_Len, 3, 256, 256).
        """
        study_path = Path(study_dir)
        dicom_files = list(study_path.glob("*.dcm"))
        if not dicom_files:
            raise FileNotFoundError(f"No DICOM files found in directory: {study_dir}")

        slice_metadata = []
        for fpath in dicom_files:
            try:
                hdr = pydicom.dcmread(str(fpath), stop_before_pixels=True)
                z_pos = 0.0
                if hasattr(hdr, "ImagePositionPatient") and len(hdr.ImagePositionPatient) >= 3:
                    z_pos = float(hdr.ImagePositionPatient[2])
                elif hasattr(hdr, "SliceLocation"):
                    z_pos = float(hdr.SliceLocation)
                elif hasattr(hdr, "InstanceNumber"):
                    z_pos = float(hdr.InstanceNumber)

                slice_metadata.append((z_pos, fpath))
            except Exception:
                continue

        # Sort inferior-to-superior along anatomical Z-axis
        slice_metadata.sort(key=lambda x: x[0])
        sorted_filepaths = [item[1] for item in slice_metadata]

        # Stack into 4D volumetric tensor
        processed_slices = [self.process_single_slice(fp) for fp in sorted_filepaths]
        volume_tensor = torch.stack(processed_slices, dim=0)  # Shape: (S, 3, 256, 256)

        return volume_tensor, [str(fp.name) for fp in sorted_filepaths]
import hashlib
import numpy as np
import torch
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass(frozen=True)
class CTWindow:
    center: float
    width: float

    @property
    def bounds(self) -> Tuple[float, float]:
        if self.width <= 0:
            raise ValueError(f"Window width must be strictly positive, got {self.width}")
        return self.center - self.width / 2.0, self.center + self.width / 2.0


class ClinicalCTPreprocessor:
    """
    HIPAA-compliant, clinically calibrated CT normalization engine.
    Applies multi-channel Hounsfield Unit (HU) windowing:
      - Channel 0: Brain Window (40, 80)
      - Channel 1: Subdural Window (75, 215)
      - Channel 2: Bone Window (600, 2800)
    """
    BRAIN_WINDOW = CTWindow(center=40.0, width=80.0)
    SUBDURAL_WINDOW = CTWindow(center=75.0, width=215.0)
    BONE_WINDOW = CTWindow(center=600.0, width=2800.0)

    @staticmethod
    def anonymize_phi(patient_id: str, salt: str = "RSNA_TRIAGE_SECURE_SALT") -> str:
        """One-way cryptographic pseudonymization for HIPAA Safe Harbor compliance."""
        digest = hashlib.sha256(f"{patient_id}_{salt}".encode("utf-8")).hexdigest()
        return f"ANON_{digest[:12].upper()}"

    @staticmethod
    def raw_to_hu(pixel_array: np.ndarray, slope: float = 1.0, intercept: float = -1024.0) -> np.ndarray:
        if not isinstance(pixel_array, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(pixel_array)}")
        if np.isnan(pixel_array).any() or np.isinf(pixel_array).any():
            pixel_array = np.nan_to_num(pixel_array, nan=-1024.0, posinf=3071.0, neginf=-1024.0)
        return pixel_array.astype(np.float32) * slope + intercept

    @classmethod
    def apply_window(cls, hu_array: np.ndarray, window: CTWindow) -> np.ndarray:
        lower, upper = window.bounds
        windowed = np.clip(hu_array, lower, upper)
        return (windowed - lower) / (upper - lower + 1e-7)

    @classmethod
    def create_composite_tensor(
        cls, 
        hu_array: np.ndarray, 
        target_size: Tuple[int, int] = (512, 512)
    ) -> torch.Tensor:
        """Converts 2D HU matrix into (3, H, W) normalized tensor with strict shape verification."""
        if hu_array.ndim != 2:
            raise ValueError(f"Expected 2D slice matrix (H, W), got shape {hu_array.shape}")

        c0 = cls.apply_window(hu_array, cls.BRAIN_WINDOW)
        c1 = cls.apply_window(hu_array, cls.SUBDURAL_WINDOW)
        c2 = cls.apply_window(hu_array, cls.BONE_WINDOW)
        
        composite = np.stack([c0, c1, c2], axis=0) # (3, H, W)
        tensor = torch.from_numpy(composite).float()
        
        if tensor.shape[1:] != target_size:
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False
            ).squeeze(0)
            
        return tensor

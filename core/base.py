# core/base.py
"""
Abstract Polymorphic Base Contract for Multi-Modal Clinical Triage.
Defines standard interfaces and strongly-typed dataclasses for all imaging modalities.
"""

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np
import torch


class ModalityType(Enum):
    """Supported emergency imaging modalities."""
    HEAD_CT = "HEAD_CT"        # 3D Volumetric Axial CT Sequence
    CHEST_XRAY = "CHEST_XRAY"  # 2D High-Resolution Radiograph Projection
    KNEE_MRI = "KNEE_MRI"      # Multi-Slice Soft-Tissue MRI Series


@dataclass
class ModalityTriageOutput:
    """Standardized output contract returned by every modality adapter."""
    modality: ModalityType
    p_critical: float                                        # Calibrated emergency/surgical risk [0, 1]
    confidence: float                                        # Decision certainty metric [0, 1]
    psi_score: float                                         # Saliency perturbation stability index [0, 1]
    saliency_map: np.ndarray                                 # 2D Grad-CAM heatmap (0 to 1 normalized)
    salient_slice_index: Optional[int] = None                # Slice index for 3D/multi-slice studies
    slice_probabilities: Optional[np.ndarray] = None         # Fine-grained sequence probabilities
    subtypes: Dict[str, float] = field(default_factory=dict) # Multi-label clinical finding breakdown
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTriageAdapter(ABC):
    """Abstract interface that all modality engines must implement."""

    def __init__(self, modality: ModalityType, device: str = "cpu"):
        self.modality = modality
        self.device = torch.device(device)

    @abstractmethod
    def predict(self, input_tensor: torch.Tensor, compute_psi: bool = True) -> ModalityTriageOutput:
        """Executes inference, explainability extraction, and stability auditing."""
        pass
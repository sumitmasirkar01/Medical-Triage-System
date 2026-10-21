# core/base.py
"""
STAT-Triage AI: Core Data Models, Base Interfaces & Dynamic Queue Structures.
Defines foundational abstractions, polymorphic outputs, priority enums, and worklist items.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import time
import numpy as np
import torch


class ModalityType(str, Enum):
    """Supported imaging modalities across foundation adapters."""
    HEAD_CT = "HEAD_CT"
    CHEST_XRAY = "CHEST_XRAY"
    KNEE_MRI = "KNEE_MRI"


class TriagePriorityZone(str, Enum):
    """Clinical triage priority urgency bins."""
    RED = "RED"         # Immediate STAT Review (Critical Pathology / SPRT Halt)
    YELLOW = "YELLOW"   # Urgent Review (Elevated Risk / Moderate Uncertainty)
    GREEN = "GREEN"     # Routine Review (Normal / Low Risk)


# Backward-compatible alias
PriorityZone = TriagePriorityZone


@dataclass
class ModalityTriageOutput:
    """Standardized polymorphic output payload produced by all modality adapters."""
    patient_id: str
    modality: ModalityType
    p_critical: float
    uncertainty_score: float
    conformal_set: List[str]
    conformal_guarantee: float
    saliency_map: Optional[np.ndarray] = None
    sprt_halted: bool = False
    sprt_exit_slice: int = 0
    sprt_compute_saved_pct: float = 0.0
    psi_score: float = 1.0
    diagnostic_logits: Dict[str, float] = field(default_factory=dict)


# Backward-compatible alias
TriageOutput = ModalityTriageOutput


@dataclass(order=True)
class DynamicQueueItem:
    """
    Worklist item managed by the Dynamic Prioritization Engine.

    The gateway explicitly sorts scores in descending order. The gateway-facing
    aliases below share storage with the original score and zone fields.
    """
    priority_score: float
    patient_id: str = field(compare=False)
    study_uid: str = field(default="", compare=False)
    modality: ModalityType = field(default=ModalityType.HEAD_CT, compare=False)
    urgency_zone: TriagePriorityZone = field(default=TriagePriorityZone.GREEN, compare=False)
    arrival_time: float = field(default_factory=time.time, compare=False)
    triage_output: Optional[ModalityTriageOutput] = field(default=None, compare=False)
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)
    clinician_overridden: bool = field(default=False, compare=False)
    override_reason: str = field(default="", compare=False)

    @property
    def dynamic_score(self) -> float:
        return self.priority_score

    @dynamic_score.setter
    def dynamic_score(self, value: float) -> None:
        self.priority_score = value

    @property
    def priority_zone(self) -> TriagePriorityZone:
        return self.urgency_zone

    @priority_zone.setter
    def priority_zone(self, value: TriagePriorityZone) -> None:
        self.urgency_zone = value

    @property
    def t_wait_minutes(self) -> float:
        return self.wait_time_minutes

    @property
    def wait_time_minutes(self) -> float:
        """Returns elapsed wait time in minutes."""
        return (time.time() - self.arrival_time) / 60.0


class BaseTriageAdapter(ABC):
    """Abstract base class for all polymorphic clinical triage models."""

    @abstractmethod
    def predict_triage(
        self,
        tensor: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
        explain: bool = False,
    ) -> ModalityTriageOutput:
        """
        Executes foundation model inference and statistical calibration.

        Args:
            tensor: Input tensor (3D volume, 2D radiograph, or multi-sequence array).
            metadata: Optional dictionary containing clinical or DICOM tags.
            explain: If True, computes saliency / perturbation stability maps.

        Returns:
            ModalityTriageOutput containing p_critical, conformal sets, and SPRT stopping stats.
        """
        pass

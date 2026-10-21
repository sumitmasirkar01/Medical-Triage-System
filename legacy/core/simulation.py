# core/simulation.py
"""
High-Throughput PACS Feed Simulator.
Simulates clinical DICOM arrival streams from hospital PACS routers,
generates deterministic synthetic 3D CT volumes for integration testing,
and models Poisson arrival queues for dynamic aging evaluation.
"""

import os
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Generator, List, Dict, Optional, Tuple, Any
import numpy as np
import torch


@dataclass
class SimulatedPACSStudy:
    """Represents an incoming volumetric study transmitted from PACS."""
    study_uid: str
    patient_id: str
    modality: str = "CT"
    body_part: str = "HEAD"
    arrival_timestamp: float = field(default_factory=time.time)
    slice_count: int = 32
    volume_tensor: Optional[torch.Tensor] = None
    is_ground_truth_positive: bool = False


class PACSFeedSimulator:
    """
    Simulates real-time hospital PACS streams for stress-testing triage gateways,
    throughput latencies, and starvation-free queue dynamics.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate_synthetic_volume(
        self,
        slice_count: int = 32,
        is_positive: bool = False,
        img_size: Tuple[int, int] = (256, 256)
    ) -> torch.Tensor:
        """
        Generates a synthetic 3-channel composite CT volume tensor of shape (S, 3, H, W).
        If is_positive=True, injects simulated hyperdense acute bleeding signatures.
        """
        H, W = img_size
        volume = np.zeros((slice_count, 3, H, W), dtype=np.float32)

        # Baseline parenchymal attenuation
        volume[:, 0, :, :] = self.rng.uniform(0.15, 0.35, size=(slice_count, H, W))  # Brain window
        volume[:, 1, :, :] = self.rng.uniform(0.05, 0.20, size=(slice_count, H, W))  # Subdural window
        volume[:, 2, :, :] = self.rng.uniform(0.00, 0.10, size=(slice_count, H, W))  # Bone window

        # Calvarial skull rim artifact on bone channel
        y, x = np.ogrid[:H, :W]
        center_y, center_x = H // 2, W // 2
        radius = H // 2 - 10
        skull_mask = (np.abs((x - center_x)**2 + (y - center_y)**2 - radius**2) < 150)
        volume[:, 2, skull_mask] = 0.95

        # Inject hyperdense acute bleed lesions across mid-cranial slices
        if is_positive:
            bleed_slices = slice_count // 3
            start_slice = slice_count // 3
            lesion_mask = ((x - (center_x + 30))**2 + (y - (center_y - 20))**2 < 400)
            for s in range(start_slice, start_slice + bleed_slices):
                volume[s, 0, lesion_mask] = 0.85  # Acute hyperdensity in Brain window
                volume[s, 1, lesion_mask] = 0.90  # Acute hyperdensity in Subdural window

        return torch.from_numpy(volume)

    def create_simulated_study(
        self,
        patient_id: Optional[str] = None,
        is_positive: Optional[bool] = None,
        slice_count: int = 32
    ) -> SimulatedPACSStudy:
        """Constructs a single standardized PACS study with synthetic volumetric imaging."""
        if patient_id is None:
            patient_id = f"PT-{self.rng.randint(10000, 99999)}"

        if is_positive is None:
            is_positive = bool(self.rng.rand() < 0.30)  # 30% baseline acute emergency rate

        study_uid = f"1.2.840.113619.2.55.{uuid.uuid4().hex[:12]}"
        volume = self.generate_synthetic_volume(slice_count=slice_count, is_positive=is_positive)

        return SimulatedPACSStudy(
            study_uid=study_uid,
            patient_id=patient_id,
            arrival_timestamp=time.time(),
            slice_count=slice_count,
            volume_tensor=volume,
            is_ground_truth_positive=is_positive
        )

    def stream_feed(
        self,
        total_studies: int = 10,
        interval_seconds: float = 0.1,
        positive_ratio: float = 0.35
    ) -> Generator[SimulatedPACSStudy, None, None]:
        """
        Yields a real-time stream of incoming volumetric PACS studies with simulated arrival deltas.
        """
        for _ in range(total_studies):
            is_pos = bool(self.rng.rand() < positive_ratio)
            study = self.create_simulated_study(is_positive=is_pos)
            yield study
            if interval_seconds > 0:
                time.sleep(interval_seconds)
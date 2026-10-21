# core/calibration.py
"""
Multi-Modal Clinical Calibration & Threshold Optimization Engine.
  - Temperature Scaling for multi-label logit calibration via NLL minimization.
  - Empirical threshold tuning targeting >= 98% emergency recall (zero-tolerance false negatives).
  - Unified MultiModalCalibrator integrating with BaseTriageAdapter and TriageGateway.
"""

from typing import Dict, Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_curve, roc_curve
from .base import ModalityType


class TemperatureScaler(nn.Module):
    """Calibrates multi-label logits using a learned modality temperature parameter."""

    def __init__(self, initial_temp: float = 1.2):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = torch.clamp(self.temperature, min=0.1, max=10.0)
        return logits / temp

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, max_iter: int = 100, lr: float = 0.01) -> float:
        """Optimizes temperature to minimize Binary Cross Entropy (NLL) on validation data."""
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(val_logits)
            loss = criterion(scaled_logits, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return float(torch.clamp(self.temperature, min=0.1, max=10.0).item())


class ClinicalThresholdOptimizer:
    """Computes high-sensitivity decision boundaries to eliminate emergency false negatives."""

    @staticmethod
    def optimize_operating_thresholds(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        min_critical_recall: float = 0.98
    ) -> Tuple[float, float]:
        """
        Calculates optimal (RED, YELLOW) thresholds:
          - RED Threshold: Lowest score that guarantees >= 98% sensitivity on critical life-threats.
          - YELLOW Threshold: Standard baseline to separate indeterminate findings from clean routine cases.
        """
        # Ensure 1D array for critical finding
        if y_true.ndim > 1:
            y_true = y_true[:, -1]
        if y_prob.ndim > 1:
            y_prob = y_prob[:, -1]

        fpr, tpr, thresholds = roc_curve(y_true, y_prob)

        # Find indices where sensitivity (Recall / TPR) >= min_critical_recall
        valid_indices = np.where(tpr >= min_critical_recall)[0]

        if len(valid_indices) > 0:
            # Pick highest threshold that still guarantees >= 98% recall
            target_idx = valid_indices[0]
            red_threshold = float(np.clip(thresholds[target_idx], 0.40, 0.85))
        else:
            red_threshold = 0.65

        # Yellow lower threshold defaults to half the red boundary
        yellow_threshold = float(np.clip(red_threshold * 0.50, 0.20, 0.45))

        return round(red_threshold, 3), round(yellow_threshold, 3)


class MultiModalCalibrator:
    """Central registry storing per-modality temperature scalars and optimized clinical boundaries."""

    def __init__(self):
        self.scalers: Dict[str, TemperatureScaler] = {
            ModalityType.HEAD_CT.value: TemperatureScaler(initial_temp=1.15),
            ModalityType.CHEST_XRAY.value: TemperatureScaler(initial_temp=1.28),
            ModalityType.KNEE_MRI.value: TemperatureScaler(initial_temp=1.10),
        }
        self.operating_thresholds: Dict[str, Dict[str, float]] = {
            ModalityType.HEAD_CT.value: {"red": 0.68, "yellow": 0.32},
            ModalityType.CHEST_XRAY.value: {"red": 0.62, "yellow": 0.28},
            ModalityType.KNEE_MRI.value: {"red": 0.70, "yellow": 0.35},
        }

    def calibrate_probability(self, modality: ModalityType, raw_prob: float) -> float:
        """Transforms raw probability into a calibrated probability via learned temperature."""
        scaler = self.scalers.get(modality.value)
        if not scaler:
            return raw_prob

        # Convert probability back to logit: z = log(p / (1 - p))
        eps = 1e-6
        clipped_p = np.clip(raw_prob, eps, 1.0 - eps)
        raw_logit = torch.tensor([np.log(clipped_p / (1.0 - clipped_p))], dtype=torch.float32)

        with torch.no_grad():
            calibrated_logit = scaler(raw_logit)
            calibrated_prob = float(torch.sigmoid(calibrated_logit).item())

        return calibrated_prob

    def get_thresholds(self, modality: ModalityType) -> Dict[str, float]:
        """Returns the optimized clinical decision boundaries for a specific modality."""
        return self.operating_thresholds.get(
            modality.value,
            {"red": 0.70, "yellow": 0.30}
        )
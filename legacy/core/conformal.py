# core/conformal.py
"""
STAT-Triage AI: Finite-Sample Conformal Prediction Calibrator.
Implements Adaptive Prediction Sets (APS) with exact (1 - alpha) coverage guarantees.
"""

from typing import List, Optional, Tuple, Union
import numpy as np


class SplitConformalCalibrator:
    """Adaptive Prediction Set (APS) Calibrator guaranteeing P(Y in C(X)) >= 1 - alpha."""

    def __init__(self, alpha: float = 0.01, class_names: Optional[List[str]] = None):
        self.alpha = float(alpha)
        self.class_names = class_names or []
        self.q_hat: float = 0.99
        self.is_fitted: bool = False

    def fit(self, cal_probs: np.ndarray, cal_labels: np.ndarray):
        cal_probs = np.asarray(cal_probs, dtype=np.float64)
        cal_labels = np.asarray(cal_labels, dtype=np.int64)
        n = len(cal_labels)

        scores = []
        for i in range(n):
            p = cal_probs[i]
            y = cal_labels[i]
            sort_idx = np.argsort(p)[::-1]
            cumsum = np.cumsum(p[sort_idx])
            rank = np.where(sort_idx == y)[0][0]
            scores.append(cumsum[rank])

        p_level = np.clip(np.ceil((n + 1) * (1.0 - self.alpha)) / n, 0.0, 1.0)
        self.q_hat = float(np.quantile(scores, p_level, method="higher"))
        self.is_fitted = True

    def predict_set(self, probs: np.ndarray) -> Tuple[List[str], float]:
        probs = np.asarray(probs, dtype=np.float64).flatten()
        sort_idx = np.argsort(probs)[::-1]
        cumsum = np.cumsum(probs[sort_idx])

        threshold = self.q_hat if self.is_fitted else (1.0 - self.alpha)
        cutoff_idx = np.searchsorted(cumsum, threshold)
        included_indices = sort_idx[: min(len(probs), cutoff_idx + 1)]

        if len(included_indices) == 0:
            included_indices = np.array([sort_idx[0]])

        if self.class_names and len(self.class_names) == len(probs):
            pred_set = [self.class_names[idx] for idx in included_indices]
        else:
            pred_set = [str(idx) for idx in included_indices]

        uncertainty_score = float(len(pred_set) / max(1, len(probs)))
        return pred_set, uncertainty_score
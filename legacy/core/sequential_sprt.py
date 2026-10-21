# core/sequential_sprt.py
"""
STAT-Triage AI: Sequential Probability Ratio Test (SPRT) Engine.
Implements Wald's sequential hypothesis testing with CUSUM drift accumulation
for early critical-pathology termination on volumetric CT slice streams.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np


class SPRTDecisionStatus(str, Enum):
    CONTINUE = "CONTINUE"
    CRITICAL_HALT = "CRITICAL_HALT"
    ACCEPT_NULL = "ACCEPT_NULL"


@dataclass
class SPRTStreamResult:
    status: SPRTDecisionStatus
    exit_slice_idx: int
    log_likelihood_ratio: float
    compute_savings_pct: float
    slice_llr_history: List[float] = field(default_factory=list)


# Backward-compatible alias
SPRTResult = SPRTStreamResult


class SequentialProbabilityRatioTest:
    """Wald's SPRT with CUSUM for sequential streaming detection."""

    def __init__(
        self,
        alpha: float = 0.03,
        beta: float = 0.001,
        baseline_prob: float = 0.10,
        target_prob: float = 0.85,
    ):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.baseline_prob = float(baseline_prob)
        self.target_prob = float(target_prob)

        self.upper_bound = float(np.log((1.0 - self.beta) / self.alpha))
        self.lower_bound = float(np.log(self.beta / (1.0 - self.alpha)))

    def calculate_step_llr(self, p: float) -> float:
        p = float(np.clip(p, 1e-4, 1.0 - 1e-4))
        pos_term = p * np.log(self.target_prob / self.baseline_prob)
        neg_term = (1.0 - p) * np.log((1.0 - self.target_prob) / (1.0 - self.baseline_prob))
        return float(pos_term + neg_term)

    def evaluate_full_stream(self, slice_probs: np.ndarray) -> SPRTStreamResult:
        slice_probs = np.asarray(slice_probs, dtype=np.float64).flatten()
        n_slices = len(slice_probs)

        cum_llr = 0.0
        history = []

        for idx, p in enumerate(slice_probs):
            step = self.calculate_step_llr(p)
            cum_llr = max(0.0, cum_llr + step)  # CUSUM non-negative clamping
            history.append(cum_llr)

            if cum_llr >= self.upper_bound:
                savings = float(((n_slices - 1 - idx) / max(1, n_slices)) * 100.0)
                return SPRTStreamResult(
                    status=SPRTDecisionStatus.CRITICAL_HALT,
                    exit_slice_idx=idx,
                    log_likelihood_ratio=cum_llr,
                    compute_savings_pct=savings,
                    slice_llr_history=history,
                )

        return SPRTStreamResult(
            status=SPRTDecisionStatus.CONTINUE,
            exit_slice_idx=n_slices - 1,
            log_likelihood_ratio=cum_llr,
            compute_savings_pct=0.0,
            slice_llr_history=history,
        )
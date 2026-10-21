"""Locked operating-point selection for binary clinical endpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd

from .evaluation import binary_metrics


@dataclass(frozen=True)
class OperatingPoint:
    """A threshold selected from calibration data under a sensitivity constraint."""

    threshold: float
    target_sensitivity: float
    sensitivity: float
    specificity: float
    tp: int
    tn: int
    fp: int
    fn: int
    positive_examples: int
    negative_examples: int
    selection_rule: str = "largest_threshold_with_sensitivity_at_least_target"

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def select_sensitivity_threshold(
    targets: np.ndarray | pd.Series,
    scores: np.ndarray | pd.Series,
    target_sensitivity: float,
) -> OperatingPoint:
    """Select the largest observed threshold meeting a sensitivity target.

    The threshold is the score of the lowest-ranked positive needed to attain the
    requested sensitivity. With the decision rule ``score >= threshold``, this is
    maximal even when positive examples have tied scores. No arbitrary clipping or
    test-set information is used.
    """

    if isinstance(target_sensitivity, bool) or not isinstance(
        target_sensitivity, (int, float)
    ):
        raise ValueError("target_sensitivity must be numeric")
    target = float(target_sensitivity)
    if not math.isfinite(target) or not 0.0 < target <= 1.0:
        raise ValueError("target_sensitivity must be finite and in (0, 1]")

    y = np.asarray(targets)
    p = np.asarray(scores, dtype=np.float64)
    # Reuse the evaluator's strict shape, label, score, and two-class checks.
    binary_metrics(y, p, threshold=0.5)

    positive_scores = np.sort(p[y == 1])[::-1]
    required_true_positives = int(math.ceil(target * len(positive_scores)))
    threshold = float(positive_scores[required_true_positives - 1])
    metrics = binary_metrics(y, p, threshold)
    if metrics.sensitivity + np.finfo(float).eps < target:
        raise RuntimeError("selected threshold failed its sensitivity constraint")

    return OperatingPoint(
        threshold=threshold,
        target_sensitivity=target,
        sensitivity=metrics.sensitivity,
        specificity=metrics.specificity,
        tp=metrics.tp,
        tn=metrics.tn,
        fp=metrics.fp,
        fn=metrics.fn,
        positive_examples=metrics.positive_images,
        negative_examples=metrics.negative_images,
    )


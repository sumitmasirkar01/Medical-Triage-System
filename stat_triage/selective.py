"""Locked three-way decisions for selective clinical prediction.

The model either emits its usual positive/negative decision or refers a case for
human review.  The abstention band is selected on development data and must be
applied unchanged to a held-out evaluation cohort.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from .evaluation import binary_metrics
from .bootstrap import ConfidenceInterval


NEGATIVE = -1
ABSTAIN = 0
POSITIVE = 1


@dataclass(frozen=True)
class SelectiveMetrics:
    total: int
    automatic: int
    abstained: int
    coverage: float
    abstention_rate: float
    automatic_positive: int
    automatic_negative: int
    tp: int
    tn: int
    fp: int
    fn: int
    selective_sensitivity: float
    selective_specificity: float
    selective_ppv: float
    selective_npv: float
    selective_accuracy: float
    positive_abstention_rate: float
    negative_abstention_rate: float
    automatic_false_negative_rate_all_positives: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class AbstentionOperatingPoint:
    threshold: float
    margin_fraction: float
    negative_boundary: float
    positive_boundary: float
    minimum_coverage: float
    maximum_automatic_false_negative_rate: float
    candidates_evaluated: int
    metrics: SelectiveMetrics
    selection_rule: str = (
        "smallest_margin_meeting_coverage_and_automatic_false_negative_constraints"
    )

    def to_dict(self) -> dict[str, object]:
        output = asdict(self)
        output["decision_rule"] = {
            "negative": "score < negative_boundary",
            "abstain": "negative_boundary <= score < positive_boundary",
            "positive": "score >= positive_boundary",
        }
        return output


SELECTIVE_BOOTSTRAP_METRICS = (
    "coverage", "abstention_rate", "selective_sensitivity",
    "selective_specificity", "selective_ppv", "selective_npv",
    "selective_accuracy", "positive_abstention_rate",
    "negative_abstention_rate", "automatic_false_negative_rate_all_positives",
)


@dataclass(frozen=True)
class GroupedSelectiveBootstrapResult:
    confidence_level: float
    seed: int
    requested_replicates: int
    sampling_unit: str
    groups: int
    rows: int
    intervals: Mapping[str, ConfidenceInterval]

    def __post_init__(self) -> None:
        object.__setattr__(self, "intervals", MappingProxyType(dict(self.intervals)))

    def to_dict(self) -> dict[str, object]:
        return {
            "confidence_level": self.confidence_level, "seed": self.seed,
            "requested_replicates": self.requested_replicates,
            "sampling_unit": self.sampling_unit, "groups": self.groups,
            "rows": self.rows,
            "intervals": {name: interval.to_dict() for name, interval in self.intervals.items()},
        }


def _validated_arrays(
    targets: np.ndarray | pd.Series, scores: np.ndarray | pd.Series, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(targets)
    p = np.asarray(scores, dtype=np.float64)
    # Reuse the established endpoint validation, including the two-class check.
    binary_metrics(y, p, threshold)
    return y.astype(np.int8, copy=False), p


def abstention_boundaries(threshold: float, margin_fraction: float) -> tuple[float, float]:
    """Map a normalized margin to valid lower and upper score boundaries.

    The margin is scaled independently on each side of the operating threshold.
    This keeps it meaningful for low clinical thresholds where a fixed absolute
    probability margin would immediately run below zero.
    """

    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ValueError("threshold must be numeric")
    if not math.isfinite(float(threshold)) or not 0 <= float(threshold) <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    if not isinstance(margin_fraction, (int, float)) or isinstance(margin_fraction, bool):
        raise ValueError("margin_fraction must be numeric")
    margin = float(margin_fraction)
    if not math.isfinite(margin) or not 0 <= margin < 1:
        raise ValueError("margin_fraction must be finite and in [0, 1)")
    threshold = float(threshold)
    return threshold * (1.0 - margin), threshold + (1.0 - threshold) * margin


def three_way_decisions(
    scores: np.ndarray | pd.Series, threshold: float, margin_fraction: float
) -> np.ndarray:
    """Return -1 for negative, 0 for abstain, and 1 for positive."""

    p = np.asarray(scores, dtype=np.float64)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("scores must be a non-empty one-dimensional array")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("scores must be finite and in [0, 1]")
    lower, upper = abstention_boundaries(threshold, margin_fraction)
    decisions = np.full(len(p), ABSTAIN, dtype=np.int8)
    # ``nextafter`` prevents decimal values exactly on a mathematically defined
    # boundary (for example 0.08) being pushed below it by floating-point noise.
    decisions[p < np.nextafter(lower, -np.inf)] = NEGATIVE
    decisions[p >= upper] = POSITIVE
    return decisions


def selective_metrics(
    targets: np.ndarray | pd.Series,
    scores: np.ndarray | pd.Series,
    threshold: float,
    margin_fraction: float,
) -> SelectiveMetrics:
    """Measure coverage, referral burden, and errors left to automation."""

    y, p = _validated_arrays(targets, scores, threshold)
    decisions = three_way_decisions(p, threshold, margin_fraction)
    abstained = decisions == ABSTAIN
    positive = y == 1
    negative = ~positive
    predicted_positive = decisions == POSITIVE
    predicted_negative = decisions == NEGATIVE
    tp = int(np.sum(predicted_positive & positive))
    tn = int(np.sum(predicted_negative & negative))
    fp = int(np.sum(predicted_positive & negative))
    fn = int(np.sum(predicted_negative & positive))
    automatic = int(np.sum(~abstained))
    automatic_positives = tp + fn
    automatic_negatives = tn + fp

    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else float("nan")

    return SelectiveMetrics(
        total=len(y), automatic=automatic, abstained=int(np.sum(abstained)),
        coverage=automatic / len(y), abstention_rate=float(np.mean(abstained)),
        automatic_positive=int(np.sum(predicted_positive)),
        automatic_negative=int(np.sum(predicted_negative)),
        tp=tp, tn=tn, fp=fp, fn=fn,
        selective_sensitivity=ratio(tp, automatic_positives),
        selective_specificity=ratio(tn, automatic_negatives),
        selective_ppv=ratio(tp, tp + fp), selective_npv=ratio(tn, tn + fn),
        selective_accuracy=ratio(tp + tn, automatic),
        positive_abstention_rate=ratio(int(np.sum(abstained & positive)), int(np.sum(positive))),
        negative_abstention_rate=ratio(int(np.sum(abstained & negative)), int(np.sum(negative))),
        automatic_false_negative_rate_all_positives=ratio(fn, int(np.sum(positive))),
    )


def select_abstention_operating_point(
    targets: np.ndarray | pd.Series,
    scores: np.ndarray | pd.Series,
    threshold: float,
    *,
    minimum_coverage: float = 0.8,
    maximum_automatic_false_negative_rate: float = 0.05,
    candidate_margins: np.ndarray | None = None,
) -> AbstentionOperatingPoint:
    """Lock the narrowest feasible abstention band on development data."""

    y, p = _validated_arrays(targets, scores, threshold)
    for name, value in (
        ("minimum_coverage", minimum_coverage),
        ("maximum_automatic_false_negative_rate", maximum_automatic_false_negative_rate),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
        if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    margins = (
        np.linspace(0.0, 0.95, 191, dtype=np.float64)
        if candidate_margins is None else np.asarray(candidate_margins, dtype=np.float64)
    )
    if margins.ndim != 1 or len(margins) == 0 or not np.isfinite(margins).all():
        raise ValueError("candidate_margins must be a non-empty finite one-dimensional array")
    if np.any((margins < 0) | (margins >= 1)):
        raise ValueError("candidate_margins must be in [0, 1)")
    margins = np.unique(margins)
    selected: tuple[float, SelectiveMetrics] | None = None
    for margin in margins:
        metrics = selective_metrics(y, p, threshold, float(margin))
        if (
            metrics.coverage >= float(minimum_coverage)
            and metrics.automatic_false_negative_rate_all_positives
            <= float(maximum_automatic_false_negative_rate) + np.finfo(float).eps
        ):
            selected = (float(margin), metrics)
            break
    if selected is None:
        raise ValueError("no candidate abstention margin satisfies the locked constraints")
    margin, metrics = selected
    lower, upper = abstention_boundaries(threshold, margin)
    return AbstentionOperatingPoint(
        threshold=float(threshold), margin_fraction=margin,
        negative_boundary=lower, positive_boundary=upper,
        minimum_coverage=float(minimum_coverage),
        maximum_automatic_false_negative_rate=float(maximum_automatic_false_negative_rate),
        candidates_evaluated=len(margins), metrics=metrics,
    )


def grouped_selective_bootstrap_confidence_intervals(
    targets: np.ndarray | pd.Series,
    scores: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series,
    threshold: float,
    margin_fraction: float,
    *,
    replicates: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
    sampling_unit: str = "patient",
    minimum_valid_fraction: float = 0.9,
) -> GroupedSelectiveBootstrapResult:
    """Bootstrap complete patients/exams under an already locked three-way rule."""

    y, p = _validated_arrays(targets, scores, threshold)
    abstention_boundaries(threshold, margin_fraction)
    group_values = np.asarray(groups, dtype=object)
    if group_values.ndim != 1 or len(group_values) != len(y):
        raise ValueError("groups must be a one-dimensional array matching targets")
    if pd.isna(group_values).any() or any(str(value).strip() == "" for value in group_values):
        raise ValueError("groups cannot contain missing or empty identifiers")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    if not math.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    if not math.isfinite(minimum_valid_fraction) or not 0 < minimum_valid_fraction <= 1:
        raise ValueError("minimum_valid_fraction must be in (0, 1]")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(sampling_unit, str) or not sampling_unit.strip():
        raise ValueError("sampling_unit must be a non-empty string")

    codes, unique_groups = pd.factorize(group_values, sort=False)
    members = tuple(np.flatnonzero(codes == code) for code in range(len(unique_groups)))
    rng = np.random.default_rng(seed)
    distributions: dict[str, list[float]] = {
        name: [] for name in SELECTIVE_BOOTSTRAP_METRICS
    }
    for _ in range(replicates):
        sampled = rng.integers(0, len(members), size=len(members))
        indices = np.concatenate([members[index] for index in sampled])
        if len(np.unique(y[indices])) != 2:
            continue
        measured = selective_metrics(
            y[indices], p[indices], threshold, margin_fraction
        )
        for name in SELECTIVE_BOOTSTRAP_METRICS:
            value = float(getattr(measured, name))
            if math.isfinite(value):
                distributions[name].append(value)

    required = math.ceil(replicates * minimum_valid_fraction)
    tail = (1.0 - confidence_level) / 2.0
    intervals: dict[str, ConfidenceInterval] = {}
    for name, values in distributions.items():
        if len(values) < required:
            raise ValueError(
                f"too few finite bootstrap values for {name}: {len(values)}/{replicates}"
            )
        lower, upper = np.quantile(values, [tail, 1.0 - tail])
        intervals[name] = ConfidenceInterval(
            lower=float(lower), upper=float(upper), valid_replicates=len(values)
        )
    return GroupedSelectiveBootstrapResult(
        confidence_level=confidence_level, seed=seed,
        requested_replicates=replicates, sampling_unit=sampling_unit.strip(),
        groups=len(members), rows=len(y), intervals=intervals,
    )

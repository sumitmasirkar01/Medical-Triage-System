"""Patient/exam-grouped bootstrap confidence intervals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from .evaluation import binary_metrics


BOOTSTRAP_METRICS = (
    "roc_auc",
    "average_precision",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "accuracy",
)


@dataclass(frozen=True)
class ConfidenceInterval:
    lower: float
    upper: float
    valid_replicates: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class GroupedBootstrapResult:
    confidence_level: float
    seed: int
    requested_replicates: int
    valid_class_replicates: int
    sampling_unit: str
    groups: int
    rows: int
    intervals: Mapping[str, ConfidenceInterval]

    def __post_init__(self) -> None:
        object.__setattr__(self, "intervals", MappingProxyType(dict(self.intervals)))

    def to_dict(self) -> dict[str, object]:
        return {
            "confidence_level": self.confidence_level,
            "seed": self.seed,
            "requested_replicates": self.requested_replicates,
            "valid_class_replicates": self.valid_class_replicates,
            "sampling_unit": self.sampling_unit,
            "groups": self.groups,
            "rows": self.rows,
            "intervals": {
                name: interval.to_dict() for name, interval in self.intervals.items()
            },
        }


def grouped_bootstrap_confidence_intervals(
    targets: np.ndarray | pd.Series,
    scores: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series,
    threshold: float,
    *,
    replicates: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
    sampling_unit: str = "patient",
    minimum_valid_fraction: float = 0.9,
) -> GroupedBootstrapResult:
    """Bootstrap complete groups and return percentile confidence intervals.

    A sampled patient/exam contributes all of its rows. If the same group is drawn
    more than once, all of its rows are repeated for each draw. Replicates missing
    either target class are excluded rather than assigned invented metric values.
    """

    y = np.asarray(targets)
    p = np.asarray(scores, dtype=np.float64)
    group_values = np.asarray(groups, dtype=object)
    # Validate endpoint arrays and the fixed operating threshold.
    binary_metrics(y, p, threshold)
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
    group_count = len(members)
    rng = np.random.default_rng(seed)
    distributions: dict[str, list[float]] = {name: [] for name in BOOTSTRAP_METRICS}
    valid_class_replicates = 0

    for _ in range(replicates):
        sampled_groups = rng.integers(0, group_count, size=group_count)
        indices = np.concatenate([members[index] for index in sampled_groups])
        sampled_y = y[indices]
        if len(np.unique(sampled_y)) != 2:
            continue
        measured = binary_metrics(sampled_y, p[indices], threshold)
        valid_class_replicates += 1
        for name in BOOTSTRAP_METRICS:
            value = float(getattr(measured, name))
            if math.isfinite(value):
                distributions[name].append(value)

    required_valid = math.ceil(replicates * minimum_valid_fraction)
    if valid_class_replicates < required_valid:
        raise ValueError(
            "too few valid bootstrap replicates contained both target classes: "
            f"{valid_class_replicates}/{replicates}"
        )

    tail = (1.0 - confidence_level) / 2.0
    intervals: dict[str, ConfidenceInterval] = {}
    for name, values in distributions.items():
        if len(values) < required_valid:
            raise ValueError(
                f"too few finite bootstrap values for {name}: {len(values)}/{replicates}"
            )
        lower, upper = np.quantile(values, [tail, 1.0 - tail])
        intervals[name] = ConfidenceInterval(
            lower=float(lower), upper=float(upper), valid_replicates=len(values)
        )

    return GroupedBootstrapResult(
        confidence_level=confidence_level,
        seed=seed,
        requested_replicates=replicates,
        valid_class_replicates=valid_class_replicates,
        sampling_unit=sampling_unit.strip(),
        groups=group_count,
        rows=len(y),
        intervals=intervals,
    )


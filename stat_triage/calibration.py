"""Calibration metrics and held-out scalar temperature scaling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class CalibrationMetrics:
    negative_log_likelihood: float
    brier_score: float
    expected_calibration_error: float
    examples: int
    positives: int
    bins: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class TemperatureCalibration:
    """A fitted scalar temperature and the rule used to accept it."""

    temperature: float
    selected_method: str
    before: CalibrationMetrics
    after: CalibrationMetrics
    minimum_nll_improvement: float

    @property
    def nll_improvement(self) -> float:
        return self.before.negative_log_likelihood - self.after.negative_log_likelihood

    def transform_logits(self, logits: np.ndarray | pd.Series) -> np.ndarray:
        values = _validate_logits(logits)
        return sigmoid(values / self.temperature)

    def to_dict(self) -> dict[str, object]:
        return {
            "temperature": self.temperature,
            "selected_method": self.selected_method,
            "minimum_nll_improvement": self.minimum_nll_improvement,
            "nll_improvement": self.nll_improvement,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
        }


@dataclass(frozen=True)
class AffineLogisticCalibration:
    """Platt scaling: sigmoid(slope * logit + intercept)."""

    slope: float
    intercept: float
    selected_method: str
    before: CalibrationMetrics
    after: CalibrationMetrics
    minimum_nll_improvement: float

    @property
    def nll_improvement(self) -> float:
        return self.before.negative_log_likelihood - self.after.negative_log_likelihood

    def transform_logits(self, logits: np.ndarray | pd.Series) -> np.ndarray:
        values = _validate_logits(logits)
        return sigmoid(self.slope * values + self.intercept)

    def to_dict(self) -> dict[str, object]:
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "selected_method": self.selected_method,
            "minimum_nll_improvement": self.minimum_nll_improvement,
            "nll_improvement": self.nll_improvement,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
        }


def _validate_targets(targets: np.ndarray | pd.Series, length: int) -> np.ndarray:
    y = np.asarray(targets)
    if y.ndim != 1 or len(y) != length or length == 0:
        raise ValueError("targets and predictions must be non-empty one-dimensional arrays")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("targets must contain only 0 and 1")
    if len(np.unique(y)) != 2:
        raise ValueError("both target classes are required for calibration")
    return y.astype(np.float64)


def _validate_logits(logits: np.ndarray | pd.Series) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("logits must be a non-empty one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("logits must be finite")
    return values


def _validate_probabilities(probabilities: np.ndarray | pd.Series) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional array")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    return values


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable elementwise sigmoid."""

    values = np.asarray(logits, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def calibration_metrics(
    targets: np.ndarray | pd.Series,
    probabilities: np.ndarray | pd.Series,
    *,
    bins: int = 10,
) -> CalibrationMetrics:
    """Compute NLL, Brier score, and equal-width ECE for a binary endpoint."""

    p = _validate_probabilities(probabilities)
    y = _validate_targets(targets, len(p))
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer of at least 2")

    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(p, epsilon, 1.0 - epsilon)
    nll = -np.mean(y * np.log(clipped) + (1.0 - y) * np.log1p(-clipped))
    brier = np.mean((p - y) ** 2)

    # Values exactly equal to 1 belong to the final equal-width bin.
    bin_indices = np.minimum((p * bins).astype(int), bins - 1)
    ece = 0.0
    for index in range(bins):
        selected = bin_indices == index
        if np.any(selected):
            weight = np.mean(selected)
            ece += weight * abs(float(np.mean(p[selected])) - float(np.mean(y[selected])))

    return CalibrationMetrics(
        negative_log_likelihood=float(nll),
        brier_score=float(brier),
        expected_calibration_error=float(ece),
        examples=len(y),
        positives=int(np.sum(y)),
        bins=bins,
    )


def _nll_for_log_temperature(log_temperature: float, logits: np.ndarray, y: np.ndarray) -> float:
    temperature = math.exp(log_temperature)
    probabilities = sigmoid(logits / temperature)
    epsilon = np.finfo(np.float64).eps
    probabilities = np.clip(probabilities, epsilon, 1.0 - epsilon)
    return float(
        -np.mean(y * np.log(probabilities) + (1.0 - y) * np.log1p(-probabilities))
    )


def _golden_section_temperature(
    logits: np.ndarray,
    y: np.ndarray,
    minimum_temperature: float,
    maximum_temperature: float,
) -> float:
    """Bounded deterministic minimization in log-temperature space."""

    lower = math.log(minimum_temperature)
    upper = math.log(maximum_temperature)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    left_value = _nll_for_log_temperature(left, logits, y)
    right_value = _nll_for_log_temperature(right, logits, y)
    for _ in range(128):
        if left_value <= right_value:
            upper, right, right_value = right, left, left_value
            left = upper - ratio * (upper - lower)
            left_value = _nll_for_log_temperature(left, logits, y)
        else:
            lower, left, left_value = left, right, right_value
            right = lower + ratio * (upper - lower)
            right_value = _nll_for_log_temperature(right, logits, y)
    return math.exp((lower + upper) / 2.0)


def fit_temperature_scaling(
    targets: np.ndarray | pd.Series,
    logits: np.ndarray | pd.Series,
    *,
    bins: int = 10,
    minimum_temperature: float = 0.05,
    maximum_temperature: float = 20.0,
    minimum_nll_improvement: float = 1e-6,
) -> TemperatureCalibration:
    """Fit on calibration logits and retain identity unless NLL improves enough."""

    z = _validate_logits(logits)
    y = _validate_targets(targets, len(z))
    if (
        not math.isfinite(minimum_temperature)
        or not math.isfinite(maximum_temperature)
        or minimum_temperature <= 0
        or maximum_temperature <= minimum_temperature
        or not minimum_temperature <= 1.0 <= maximum_temperature
    ):
        raise ValueError("temperature bounds must be finite, positive, and contain 1")
    if not math.isfinite(minimum_nll_improvement) or minimum_nll_improvement < 0:
        raise ValueError("minimum_nll_improvement must be finite and non-negative")

    before = calibration_metrics(y, sigmoid(z), bins=bins)
    candidate_temperature = _golden_section_temperature(
        z, y, minimum_temperature, maximum_temperature
    )
    candidate = calibration_metrics(
        y, sigmoid(z / candidate_temperature), bins=bins
    )
    improvement = before.negative_log_likelihood - candidate.negative_log_likelihood
    if improvement > minimum_nll_improvement:
        return TemperatureCalibration(
            temperature=float(candidate_temperature),
            selected_method="temperature_scaling",
            before=before,
            after=candidate,
            minimum_nll_improvement=minimum_nll_improvement,
        )
    return TemperatureCalibration(
        temperature=1.0,
        selected_method="identity",
        before=before,
        after=before,
        minimum_nll_improvement=minimum_nll_improvement,
    )


def fit_affine_logistic_scaling(
    targets: np.ndarray | pd.Series,
    logits: np.ndarray | pd.Series,
    *,
    bins: int = 10,
    minimum_nll_improvement: float = 1e-6,
) -> AffineLogisticCalibration:
    """Fit monotonic affine logistic calibration and retain identity if unhelpful.

    The intercept is necessary for logits learned with positive-class weighting,
    which changes the effective class prior and cannot be corrected by temperature
    scaling alone.
    """

    z = _validate_logits(logits)
    y = _validate_targets(targets, len(z))
    if not math.isfinite(minimum_nll_improvement) or minimum_nll_improvement < 0:
        raise ValueError("minimum_nll_improvement must be finite and non-negative")
    before = calibration_metrics(y, sigmoid(z), bins=bins)

    def objective(parameters: np.ndarray) -> float:
        transformed = parameters[0] * z + parameters[1]
        return float(np.mean(np.logaddexp(0.0, transformed) - y * transformed))

    fitted = minimize(
        objective,
        x0=np.array([1.0, 0.0]),
        method="L-BFGS-B",
        bounds=((1e-6, 100.0), (-100.0, 100.0)),
    )
    if not fitted.success or not np.isfinite(fitted.fun):
        raise RuntimeError(f"Affine logistic calibration failed: {fitted.message}")
    slope, intercept = (float(value) for value in fitted.x)
    candidate = calibration_metrics(y, sigmoid(slope * z + intercept), bins=bins)
    if before.negative_log_likelihood - candidate.negative_log_likelihood > minimum_nll_improvement:
        return AffineLogisticCalibration(
            slope, intercept, "affine_logistic_scaling", before, candidate,
            minimum_nll_improvement,
        )
    return AffineLogisticCalibration(
        1.0, 0.0, "identity", before, before, minimum_nll_improvement
    )


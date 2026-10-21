"""Masked multi-label objectives shared by CT, CXR, and MRI training."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _validated_inputs(logits: Tensor, targets: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor]:
    if not torch.is_floating_point(logits):
        raise ValueError("logits must be floating point")
    if logits.ndim < 2 or logits.numel() == 0:
        raise ValueError("logits must have shape [batch, ..., targets]")
    if targets.shape != logits.shape:
        raise ValueError("targets must have the same shape as logits")
    if mask is None:
        effective_mask = torch.ones_like(logits, dtype=torch.bool)
    else:
        if mask.shape != logits.shape:
            raise ValueError("mask must have the same shape as logits")
        if mask.dtype == torch.bool:
            effective_mask = mask
        else:
            if not torch.isfinite(mask).all() or not torch.all((mask == 0) | (mask == 1)):
                raise ValueError("mask must contain only 0 and 1")
            effective_mask = mask.bool()
    if not effective_mask.any():
        raise ValueError("at least one target must be annotated")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    annotated_targets = targets[effective_mask]
    if not torch.isfinite(annotated_targets).all() or not torch.all(
        (annotated_targets == 0) | (annotated_targets == 1)
    ):
        raise ValueError("annotated targets must contain only finite 0 and 1 values")
    # NaN or sentinel values are allowed only where the mask is zero. Replacing
    # them prevents IEEE NaN propagation before the masked reduction.
    safe_targets = torch.where(effective_mask, targets, torch.zeros_like(targets))
    return safe_targets.to(dtype=logits.dtype), effective_mask


def _validated_positive_weights(
    pos_weight: Tensor | Sequence[float] | None,
    logits: Tensor,
) -> Tensor | None:
    if pos_weight is None:
        return None
    weights = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    if weights.ndim != 1 or weights.shape[0] != logits.shape[-1]:
        raise ValueError("pos_weight must contain one value per target")
    if not torch.isfinite(weights).all() or torch.any(weights <= 0):
        raise ValueError("pos_weight values must be finite and positive")
    return weights


def masked_weighted_bce_with_logits(
    logits: Tensor,
    targets: Tensor,
    *,
    mask: Tensor | None = None,
    pos_weight: Tensor | Sequence[float] | None = None,
) -> Tensor:
    """Mean BCE over annotated cells, with training-only positive weights."""

    safe_targets, effective_mask = _validated_inputs(logits, targets, mask)
    weights = _validated_positive_weights(pos_weight, logits)
    elementwise = F.binary_cross_entropy_with_logits(
        logits, safe_targets, pos_weight=weights, reduction="none"
    )
    return elementwise.masked_select(effective_mask).mean()


def masked_asymmetric_loss_with_logits(
    logits: Tensor,
    targets: Tensor,
    *,
    mask: Tensor | None = None,
    gamma_negative: float = 4.0,
    gamma_positive: float = 1.0,
    negative_probability_clip: float = 0.05,
    epsilon: float = 1e-8,
) -> Tensor:
    """Complete clipped Asymmetric Loss, averaged over annotated cells.

    Negative probability clipping and separate positive/negative focusing powers
    follow the published multi-label ASL formulation. This is an ablation; masked
    weighted BCE remains the primary project objective.
    """

    for name, value in (
        ("gamma_negative", gamma_negative),
        ("gamma_positive", gamma_positive),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if (
        isinstance(negative_probability_clip, bool)
        or not isinstance(negative_probability_clip, (int, float))
        or not math.isfinite(float(negative_probability_clip))
        or not 0 <= negative_probability_clip < 1
    ):
        raise ValueError("negative_probability_clip must be finite and in [0, 1)")
    if not math.isfinite(epsilon) or not 0 < epsilon < 0.5:
        raise ValueError("epsilon must be finite and in (0, 0.5)")

    safe_targets, effective_mask = _validated_inputs(logits, targets, mask)
    positive_probability = torch.sigmoid(logits)
    negative_probability = 1.0 - positive_probability
    if negative_probability_clip > 0:
        negative_probability = (negative_probability + negative_probability_clip).clamp(max=1.0)

    log_likelihood = safe_targets * torch.log(
        positive_probability.clamp(min=epsilon)
    ) + (1.0 - safe_targets) * torch.log(negative_probability.clamp(min=epsilon))

    if gamma_negative > 0 or gamma_positive > 0:
        positive_term = positive_probability * safe_targets
        negative_term = negative_probability * (1.0 - safe_targets)
        base = (1.0 - positive_term - negative_term).clamp(min=0.0)
        exponent = gamma_positive * safe_targets + gamma_negative * (1.0 - safe_targets)
        log_likelihood = log_likelihood * torch.pow(base, exponent)

    return -log_likelihood.masked_select(effective_mask).mean()


class MaskedWeightedBCEWithLogitsLoss(nn.Module):
    """Module wrapper for the primary multi-label objective."""

    def __init__(self, pos_weight: Tensor | Sequence[float] | None = None) -> None:
        super().__init__()
        if pos_weight is None:
            self.register_buffer("pos_weight", None)
        else:
            self.register_buffer("pos_weight", torch.as_tensor(pos_weight, dtype=torch.float32))

    def forward(self, logits: Tensor, targets: Tensor, mask: Tensor | None = None) -> Tensor:
        return masked_weighted_bce_with_logits(
            logits, targets, mask=mask, pos_weight=self.pos_weight
        )


class MaskedAsymmetricLoss(nn.Module):
    """Module wrapper for the registered ASL ablation."""

    def __init__(
        self,
        *,
        gamma_negative: float = 4.0,
        gamma_positive: float = 1.0,
        negative_probability_clip: float = 0.05,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.gamma_negative = gamma_negative
        self.gamma_positive = gamma_positive
        self.negative_probability_clip = negative_probability_clip
        self.epsilon = epsilon

    def forward(self, logits: Tensor, targets: Tensor, mask: Tensor | None = None) -> Tensor:
        return masked_asymmetric_loss_with_logits(
            logits,
            targets,
            mask=mask,
            gamma_negative=self.gamma_negative,
            gamma_positive=self.gamma_positive,
            negative_probability_clip=self.negative_probability_clip,
            epsilon=self.epsilon,
        )


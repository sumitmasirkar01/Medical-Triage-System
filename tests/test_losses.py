"""Masked objectives must exclude unavailable labels from loss and gradient."""

import math

import pytest
import torch

from stat_triage.losses import (
    MaskedAsymmetricLoss,
    MaskedWeightedBCEWithLogitsLoss,
    masked_asymmetric_loss_with_logits,
    masked_weighted_bce_with_logits,
)


def test_masked_weighted_bce_matches_manual_annotated_mean():
    logits = torch.zeros((2, 2), dtype=torch.float64)
    targets = torch.tensor([[1.0, 0.0], [1.0, float("nan")]], dtype=torch.float64)
    mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.bool)

    loss = masked_weighted_bce_with_logits(
        logits, targets, mask=mask, pos_weight=[2.0, 3.0]
    )

    # Two positives weighted by 2 and one negative weighted by 1.
    assert loss.item() == pytest.approx((2 * math.log(2) + math.log(2) + 2 * math.log(2)) / 3)


def test_masked_cells_have_zero_gradient():
    logits = torch.tensor([[0.2, -0.3]], requires_grad=True)
    targets = torch.tensor([[1.0, float("nan")]])
    mask = torch.tensor([[True, False]])

    masked_weighted_bce_with_logits(logits, targets, mask=mask).backward()

    assert logits.grad[0, 0] != 0
    assert logits.grad[0, 1] == 0


def test_asymmetric_loss_reduces_to_bce_without_focusing_or_clipping():
    logits = torch.tensor([[0.3, -0.7], [1.2, -1.5]], dtype=torch.float64)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)

    actual = masked_asymmetric_loss_with_logits(
        logits,
        targets,
        gamma_negative=0,
        gamma_positive=0,
        negative_probability_clip=0,
    )
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)

    assert actual.item() == pytest.approx(expected.item())


def test_asymmetric_loss_is_finite_for_extreme_logits_and_backpropagates():
    logits = torch.tensor([[100.0, -100.0], [-100.0, 100.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    loss = MaskedAsymmetricLoss()(logits, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_module_wrapper_uses_registered_positive_weights():
    objective = MaskedWeightedBCEWithLogitsLoss([2.0, 1.0])
    logits = torch.zeros((1, 2))
    targets = torch.tensor([[1.0, 0.0]])

    assert objective(logits, targets).item() == pytest.approx(1.5 * math.log(2))
    assert "pos_weight" in dict(objective.named_buffers())


@pytest.mark.parametrize(
    ("targets", "mask", "message"),
    [
        (torch.tensor([[0.0, 2.0]]), None, "only finite 0 and 1"),
        (torch.tensor([[0.0, 1.0]]), torch.zeros((1, 2)), "at least one"),
        (torch.tensor([[0.0, 1.0]]), torch.tensor([[1.0, 0.5]]), "only 0 and 1"),
    ],
)
def test_invalid_targets_and_masks_are_rejected(targets, mask, message):
    with pytest.raises(ValueError, match=message):
        masked_weighted_bce_with_logits(torch.zeros((1, 2)), targets, mask=mask)


def test_positive_weights_must_match_target_count():
    with pytest.raises(ValueError, match="one value per target"):
        masked_weighted_bce_with_logits(
            torch.zeros((1, 2)), torch.zeros((1, 2)), pos_weight=[1.0]
        )


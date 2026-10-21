"""DenseNet121 baseline for independent Chest X-ray findings."""

from __future__ import annotations

from torch import nn
from torchvision.models import DenseNet121_Weights, densenet121


def build_densenet121_multilabel(
    number_of_targets: int,
    *,
    pretrained: bool = True,
    freeze_features: bool = False,
) -> nn.Module:
    if isinstance(number_of_targets, bool) or not isinstance(number_of_targets, int):
        raise ValueError("number_of_targets must be an integer")
    if number_of_targets < 1:
        raise ValueError("number_of_targets must be positive")
    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = densenet121(weights=weights)
    if freeze_features:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    model.classifier = nn.Linear(model.classifier.in_features, number_of_targets)
    return model


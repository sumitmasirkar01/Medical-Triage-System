"""CXR training selection and staged unfreezing remain explicit."""

import numpy as np
import pytest

from stat_triage.cxr_training import (
    configure_densenet_stage,
    multilabel_validation_metrics,
)
from stat_triage.models import build_densenet121_multilabel


def test_multilabel_validation_metrics_are_per_target_and_macro():
    targets = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
    scores = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])

    metrics = multilabel_validation_metrics(targets, scores, ("a", "b"))

    assert metrics["macro_roc_auc"] == 1.0
    assert metrics["macro_average_precision"] == 1.0
    assert set(metrics["per_target"]) == {"a", "b"}


def test_training_stages_freeze_expected_parameters():
    model = build_densenet121_multilabel(4, pretrained=False)
    classifier_parameters = configure_densenet_stage(model, "classifier")

    assert sum(parameter.numel() for parameter in classifier_parameters) == 4100
    assert not any(parameter.requires_grad for parameter in model.features.parameters())

    configure_densenet_stage(model, "upper_backbone")
    assert any(
        parameter.requires_grad
        for name, parameter in model.features.named_parameters()
        if name.startswith("denseblock4")
    )
    assert not any(
        parameter.requires_grad
        for name, parameter in model.features.named_parameters()
        if name.startswith("denseblock1")
    )


def test_validation_rejects_single_class_target():
    with pytest.raises(ValueError, match="both classes"):
        multilabel_validation_metrics(
            np.array([[0], [0]]), np.array([[0.1], [0.2]]), ("target",)
        )


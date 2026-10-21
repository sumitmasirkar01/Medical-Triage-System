"""Training utilities for the NIH DenseNet121 multi-label baseline."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn


def set_reproducible_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def multilabel_validation_metrics(
    targets: np.ndarray,
    scores: np.ndarray,
    labels: tuple[str, ...],
) -> dict[str, object]:
    y = np.asarray(targets)
    p = np.asarray(scores, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != len(labels):
        raise ValueError("targets and scores must have shape [examples, labels]")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("targets must be binary")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("scores must be finite and in [0, 1]")
    per_target: dict[str, dict[str, float | int]] = {}
    for index, label in enumerate(labels):
        values = y[:, index]
        if len(np.unique(values)) != 2:
            raise ValueError(f"both classes are required for validation target {label}")
        per_target[label] = {
            "roc_auc": float(roc_auc_score(values, p[:, index])),
            "average_precision": float(average_precision_score(values, p[:, index])),
            "positives": int(values.sum()),
            "negatives": int(len(values) - values.sum()),
        }
    return {
        "macro_roc_auc": float(np.mean([value["roc_auc"] for value in per_target.values()])),
        "macro_average_precision": float(
            np.mean([value["average_precision"] for value in per_target.values()])
        ),
        "per_target": per_target,
    }


def configure_densenet_stage(model: nn.Module, stage: str) -> list[nn.Parameter]:
    """Freeze all features or unfreeze only DenseNet's final feature block."""

    if stage not in {"classifier", "upper_backbone"}:
        raise ValueError(f"Unsupported training stage: {stage}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    if stage == "upper_backbone":
        for name, parameter in model.features.named_parameters():
            if name.startswith("denseblock4") or name.startswith("norm5"):
                parameter.requires_grad = True
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def set_densenet_training_modes(model: nn.Module, stage: str) -> None:
    """Keep frozen BatchNorm statistics fixed while training selected modules."""

    model.eval()
    model.classifier.train()
    if stage == "upper_backbone":
        model.features.denseblock4.train()
        model.features.norm5.train()


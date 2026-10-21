"""NIH dataset batches and DenseNet output obey the four-target contract."""

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from stat_triage.datasets.nih_training import (
    CachedNIHCXRDataset,
    NIHCXRDataset,
    compute_positive_weights,
)
from stat_triage.losses import MaskedWeightedBCEWithLogitsLoss
from stat_triage.models import build_densenet121_multilabel


TARGETS = ("pneumothorax", "pleural_effusion", "consolidation", "atelectasis")


def manifest(tmp_path):
    records = []
    labels = ([1, 0, 0, 1], [0, 1, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1])
    for index, values in enumerate(labels):
        path = tmp_path / f"{index}.png"
        Image.fromarray(np.full((24, 32), 50 + index, dtype=np.uint8)).save(path)
        record = {
            "image_id": path.name, "image_path": str(path),
            "patient_id": str(index), "split": "train",
        }
        for target, value in zip(TARGETS, values):
            record[f"target_{target}"] = value
            record[f"mask_{target}"] = 1
        records.append(record)
    return pd.DataFrame(records)


def test_dataset_returns_four_target_tensors(tmp_path):
    dataset = NIHCXRDataset(manifest(tmp_path), size=64)
    row = dataset[0]

    assert row["image"].shape == (3, 64, 64)
    assert row["targets"].shape == (4,)
    assert row["mask"].tolist() == [1.0] * 4


def test_positive_weights_use_annotated_negative_positive_ratio(tmp_path):
    weights = compute_positive_weights(manifest(tmp_path))

    assert weights.tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_densenet_forward_backward_contract(tmp_path):
    rows = manifest(tmp_path)
    dataset = NIHCXRDataset(rows, size=64)
    images = torch.stack([dataset[index]["image"] for index in range(2)])
    targets = torch.stack([dataset[index]["targets"] for index in range(2)])
    masks = torch.stack([dataset[index]["mask"] for index in range(2)])
    model = build_densenet121_multilabel(4, pretrained=False, freeze_features=True)

    logits = model(images)
    loss = MaskedWeightedBCEWithLogitsLoss(compute_positive_weights(rows))(
        logits, targets, masks
    )
    loss.backward()

    assert logits.shape == (2, 4)
    assert torch.isfinite(loss)
    assert model.classifier.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.features.parameters())


def test_cached_dataset_matches_direct_preprocessing(tmp_path):
    rows = manifest(tmp_path)
    cache = np.empty((len(rows), 64, 64), dtype=np.uint8)
    direct = NIHCXRDataset(rows, size=64)
    for index in range(len(rows)):
        # Reverse ImageNet normalization using channel zero, then round to uint8.
        normalized = direct[index]["image"][0]
        cache[index] = torch.round((normalized * 0.229 + 0.485) * 255).byte().numpy()
    cache_path = tmp_path / "cache.npy"
    np.save(cache_path, cache)
    cached_rows = rows.copy()
    cached_rows.insert(0, "cache_index", range(len(rows)))
    cached = CachedNIHCXRDataset(cached_rows, cache_path)

    assert torch.allclose(cached[0]["image"], direct[0]["image"], atol=1e-6)
    assert cached[0]["targets"].tolist() == direct[0]["targets"].tolist()


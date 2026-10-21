"""Training dataset and collation utilities for Stanford MRNet exams."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .mrnet import MRNET_PLANES, MRNET_TARGETS
from .mrnet_preprocessing import preprocess_mrnet_volume


def compute_mrnet_positive_weights(manifest: pd.DataFrame) -> torch.Tensor:
    """Return per-target negative/positive ratios for the training partition."""
    training = manifest.loc[manifest["split"] == "train"]
    if training.empty:
        raise ValueError("manifest contains no training exams")
    positives = training.loc[:, MRNET_TARGETS].sum(axis=0).to_numpy(dtype=np.float64)
    if np.any(positives <= 0):
        raise ValueError("every MRNet target must have at least one training positive")
    negatives = len(training) - positives
    return torch.tensor(negatives / positives, dtype=torch.float32)


class MRNetExamDataset(Dataset):
    """Load all three variable-length MRI volumes for one MRNet exam."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        *,
        split: str,
        output_size: tuple[int, int] = (224, 224),
    ) -> None:
        rows = manifest.loc[manifest["split"] == split].reset_index(drop=True)
        if rows.empty:
            raise ValueError(f"manifest contains no {split!r} exams")
        required = {"exam_id", *MRNET_TARGETS, *(f"{p}_path" for p in MRNET_PLANES)}
        missing = sorted(required.difference(rows.columns))
        if missing:
            raise ValueError(f"manifest is missing columns: {missing}")
        self.rows = rows
        self.output_size = output_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        volumes = {
            plane: preprocess_mrnet_volume(
                Path(row[f"{plane}_path"]), size=self.output_size[0]
            )[0]
            for plane in MRNET_PLANES
        }
        targets = torch.tensor(
            [float(row[target]) for target in MRNET_TARGETS], dtype=torch.float32
        )
        return {
            "exam_id": str(row["exam_id"]),
            "volumes": volumes,
            "targets": targets,
            "label_mask": torch.ones_like(targets),
        }


def collate_mrnet_exams(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad each anatomical plane to the longest volume in the batch."""
    if not items:
        raise ValueError("cannot collate an empty MRNet batch")
    volumes: dict[str, torch.Tensor] = {}
    slice_masks: dict[str, torch.Tensor] = {}
    for plane in MRNET_PLANES:
        plane_volumes = [item["volumes"][plane] for item in items]
        max_slices = max(volume.shape[0] for volume in plane_volumes)
        shape = (len(items), max_slices, *plane_volumes[0].shape[1:])
        padded = torch.zeros(shape, dtype=plane_volumes[0].dtype)
        mask = torch.zeros((len(items), max_slices), dtype=torch.bool)
        for batch_index, volume in enumerate(plane_volumes):
            slices = volume.shape[0]
            padded[batch_index, :slices] = volume
            mask[batch_index, :slices] = True
        volumes[plane] = padded
        slice_masks[plane] = mask
    return {
        "exam_id": [item["exam_id"] for item in items],
        "volumes": volumes,
        "slice_masks": slice_masks,
        "targets": torch.stack([item["targets"] for item in items]),
        "label_mask": torch.stack([item["label_mask"] for item in items]),
    }

"""Training dataset and label statistics for the NIH four-target contract."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .nih_cxr14 import NIH_PROJECT_TARGETS
from .nih_preprocessing import preprocess_nih_radiograph


NIH_TARGET_ORDER = tuple(NIH_PROJECT_TARGETS)


def compute_positive_weights(
    manifest: pd.DataFrame,
    *,
    targets: Sequence[str] = NIH_TARGET_ORDER,
) -> Tensor:
    """Compute negative/positive ratios using annotated training cells only."""

    weights = []
    for target in targets:
        target_column = f"target_{target}"
        mask_column = f"mask_{target}"
        missing = {target_column, mask_column}.difference(manifest.columns)
        if missing:
            raise ValueError(f"Missing label columns for {target}: {sorted(missing)}")
        annotated = manifest[manifest[mask_column] == 1]
        if annotated.empty:
            raise ValueError(f"No annotated examples for {target}")
        values = annotated[target_column]
        if not values.isin([0, 1]).all():
            raise ValueError(f"Annotated values for {target} must be binary")
        positives = int(values.sum())
        negatives = len(values) - positives
        if positives == 0 or negatives == 0:
            raise ValueError(f"Both classes are required for {target}")
        weights.append(negatives / positives)
    return torch.tensor(weights, dtype=torch.float32)


class NIHCXRDataset(Dataset):
    """Load manifest rows under the deterministic evaluation preprocessing."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        *,
        size: int = 224,
        targets: Sequence[str] = NIH_TARGET_ORDER,
    ) -> None:
        self.rows = manifest.reset_index(drop=True).copy()
        self.size = size
        self.targets = tuple(targets)
        required = {"image_id", "image_path", "patient_id", "split"}
        for target in self.targets:
            required.update({f"target_{target}", f"mask_{target}"})
        missing = sorted(required.difference(self.rows.columns))
        if missing:
            raise ValueError("Manifest is missing columns: " + ", ".join(missing))
        if self.rows.empty:
            raise ValueError("Dataset manifest cannot be empty")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows.iloc[index]
        image, _, _ = preprocess_nih_radiograph(Path(row["image_path"]), size=self.size)
        targets = torch.tensor(
            [float(row[f"target_{target}"]) for target in self.targets],
            dtype=torch.float32,
        )
        mask = torch.tensor(
            [float(row[f"mask_{target}"]) for target in self.targets],
            dtype=torch.float32,
        )
        return {
            "image": image,
            "targets": targets,
            "mask": mask,
            "image_id": str(row["image_id"]),
            "patient_id": str(row["patient_id"]),
        }


class NIHCacheSourceDataset(Dataset):
    """Decode manifest images to padded uint8 arrays for one-time cache creation."""

    def __init__(self, manifest: pd.DataFrame, *, size: int = 224) -> None:
        self.rows = manifest.reset_index(drop=True).copy()
        self.size = size
        if self.rows.empty or not {"image_id", "image_path"}.issubset(self.rows.columns):
            raise ValueError("Cache source requires non-empty image_id and image_path columns")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        _, preview, _ = preprocess_nih_radiograph(
            Path(self.rows.iloc[index]["image_path"]), size=self.size
        )
        return {
            "cache_index": index,
            "image": torch.from_numpy(preview.copy()),
        }


class CachedNIHCXRDataset(Dataset):
    """Read pre-resized grayscale images from a NumPy memory-mapped cache."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        cache_path: str | Path,
        *,
        targets: Sequence[str] = NIH_TARGET_ORDER,
    ) -> None:
        self.rows = manifest.reset_index(drop=True).copy()
        self.targets = tuple(targets)
        required = {"cache_index", "image_id", "patient_id", "split"}
        for target in self.targets:
            required.update({f"target_{target}", f"mask_{target}"})
        missing = sorted(required.difference(self.rows.columns))
        if missing:
            raise ValueError("Cached manifest is missing columns: " + ", ".join(missing))
        self.cache_path = Path(cache_path)
        if not self.cache_path.is_file():
            raise FileNotFoundError(self.cache_path)
        self.images = np.load(self.cache_path, mmap_mode="r")
        if self.images.ndim != 3 or self.images.dtype != np.uint8:
            raise ValueError("Image cache must have uint8 shape [images, height, width]")
        indices = self.rows["cache_index"].to_numpy(dtype=np.int64)
        if len(indices) == 0 or indices.min() < 0 or indices.max() >= len(self.images):
            raise ValueError("Cached manifest contains invalid cache indices")
        self.indices = indices
        self.mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows.iloc[index]
        grayscale = torch.from_numpy(np.array(self.images[self.indices[index]], copy=True))
        image = grayscale.to(torch.float32).div_(255.0).unsqueeze(0).repeat(3, 1, 1)
        image = (image - self.mean) / self.std
        return {
            "image": image,
            "targets": torch.tensor(
                [float(row[f"target_{target}"]) for target in self.targets],
                dtype=torch.float32,
            ),
            "mask": torch.tensor(
                [float(row[f"mask_{target}"]) for target in self.targets],
                dtype=torch.float32,
            ),
            "image_id": str(row["image_id"]),
            "patient_id": str(row["patient_id"]),
        }


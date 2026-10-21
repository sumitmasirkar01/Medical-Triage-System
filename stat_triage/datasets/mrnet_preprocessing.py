"""Deterministic volume validation and preprocessing for Stanford MRNet."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from .nih_preprocessing import IMAGENET_MEAN, IMAGENET_STD


class MRNetVolumeError(ValueError):
    """An MRNet array cannot satisfy the frozen volume contract."""


@dataclass(frozen=True)
class MRNetVolumeInspection:
    path: str
    slices: int
    height: int
    width: int
    dtype: str
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


def load_mrnet_volume(path: str | Path) -> tuple[np.ndarray, MRNetVolumeInspection]:
    source = Path(path)
    if not source.is_file():
        raise MRNetVolumeError(f"Volume does not exist: {source}")
    try:
        volume = np.load(source, allow_pickle=False)
    except Exception as exc:
        raise MRNetVolumeError(f"Volume could not be decoded: {source}: {exc}") from exc
    if volume.ndim != 3 or min(volume.shape) < 1:
        raise MRNetVolumeError(
            f"Expected [slices,height,width], received shape {volume.shape}"
        )
    if volume.dtype != np.uint8:
        raise MRNetVolumeError(f"Expected uint8 MRNet volume, received {volume.dtype}")
    if not np.isfinite(volume).all():
        raise MRNetVolumeError("Volume contains non-finite intensities")
    inspection = MRNetVolumeInspection(
        path=str(source), slices=int(volume.shape[0]), height=int(volume.shape[1]),
        width=int(volume.shape[2]), dtype=str(volume.dtype),
        minimum=float(volume.min()), maximum=float(volume.max()),
        mean=float(volume.mean()), standard_deviation=float(volume.std()),
    )
    return volume, inspection


def preprocess_mrnet_volume(
    path: str | Path,
    *,
    size: int = 224,
) -> tuple[Tensor, np.ndarray, MRNetVolumeInspection]:
    """Resize every slice, repeat grayscale to RGB, and ImageNet-normalize."""

    if isinstance(size, bool) or not isinstance(size, int) or size < 8:
        raise ValueError("size must be an integer of at least 8")
    volume, inspection = load_mrnet_volume(path)
    tensor = torch.from_numpy(volume.copy()).to(torch.float32).div_(255.0).unsqueeze(1)
    if tuple(tensor.shape[-2:]) != (size, size):
        tensor = F.interpolate(
            tensor, size=(size, size), mode="bilinear", align_corners=False,
            antialias=True,
        )
    preview = tensor[:, 0].mul(255).round().clamp(0, 255).to(torch.uint8).numpy()
    tensor = tensor.repeat(1, 3, 1, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    if tensor.shape != (volume.shape[0], 3, size, size) or not torch.isfinite(tensor).all():
        raise MRNetVolumeError("Preprocessing produced an invalid tensor")
    return tensor, preview, inspection

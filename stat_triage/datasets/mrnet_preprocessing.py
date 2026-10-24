"""Deterministic volume validation and preprocessing for Stanford MRNet."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from .nih_preprocessing import IMAGENET_MEAN, IMAGENET_STD


MAX_MRNET_VOLUME_FILE_BYTES = 80 * 1024 * 1024
MAX_MRNET_SLICES = 96
MAX_MRNET_HEIGHT = 1024
MAX_MRNET_WIDTH = 1024
MAX_MRNET_DECODED_BYTES = 64 * 1024 * 1024
MAX_MRNET_PREPROCESSED_BYTES = 64 * 1024 * 1024
MAX_MRNET_EXAM_FILE_BYTES = 240 * 1024 * 1024
MAX_MRNET_EXAM_DECODED_BYTES = 192 * 1024 * 1024
MAX_MRNET_EXAM_PREPROCESSED_BYTES = 192 * 1024 * 1024


class MRNetVolumeError(ValueError):
    """An MRNet array cannot satisfy the frozen volume contract."""

    def __init__(self, message: str, *, code: str = "input_contract_violation") -> None:
        super().__init__(message)
        self.code = code


def _resource_error(message: str) -> MRNetVolumeError:
    return MRNetVolumeError(message, code="input_resource_limit")


@dataclass(frozen=True)
class MRNetVolumeHeader:
    path: str
    slices: int
    height: int
    width: int
    dtype: str
    file_bytes: int
    decoded_bytes: int
    preprocessed_bytes: int


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
    file_bytes: int
    decoded_bytes: int
    preprocessed_bytes: int

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


def inspect_mrnet_volume_header(path: str | Path, *, size: int = 224) -> MRNetVolumeHeader:
    """Inspect a NumPy header through a memory map before allocating volume data."""
    if isinstance(size, bool) or not isinstance(size, int) or size < 8:
        raise ValueError("size must be an integer of at least 8")
    source = Path(path)
    if not source.is_file():
        raise MRNetVolumeError(f"Volume does not exist: {source}")
    file_bytes = source.stat().st_size
    if file_bytes > MAX_MRNET_VOLUME_FILE_BYTES:
        raise _resource_error(
            f"Volume file exceeds {MAX_MRNET_VOLUME_FILE_BYTES} bytes"
        )
    try:
        volume = np.load(source, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        raise MRNetVolumeError(f"Volume could not be decoded: {source}: {exc}") from exc
    if volume.ndim != 3 or min(volume.shape) < 1:
        raise MRNetVolumeError(
            f"Expected [slices,height,width], received shape {volume.shape}"
        )
    if volume.dtype != np.uint8:
        raise MRNetVolumeError(f"Expected uint8 MRNet volume, received {volume.dtype}")
    slices, height, width = (int(value) for value in volume.shape)
    if slices > MAX_MRNET_SLICES:
        raise _resource_error(f"Volume exceeds the {MAX_MRNET_SLICES}-slice limit")
    if height > MAX_MRNET_HEIGHT or width > MAX_MRNET_WIDTH:
        raise _resource_error(
            "Volume dimensions exceed the supported maximum of "
            f"{MAX_MRNET_HEIGHT} by {MAX_MRNET_WIDTH}"
        )
    decoded_bytes = int(volume.size * volume.dtype.itemsize)
    if decoded_bytes > MAX_MRNET_DECODED_BYTES:
        raise _resource_error(
            f"Volume decoded representation exceeds {MAX_MRNET_DECODED_BYTES} bytes"
        )
    preprocessed_bytes = slices * 3 * size * size * np.dtype(np.float32).itemsize
    if preprocessed_bytes > MAX_MRNET_PREPROCESSED_BYTES:
        raise _resource_error(
            "Volume preprocessed tensor would exceed "
            f"{MAX_MRNET_PREPROCESSED_BYTES} bytes"
        )
    return MRNetVolumeHeader(
        path=str(source), slices=slices, height=height, width=width,
        dtype=str(volume.dtype), file_bytes=file_bytes, decoded_bytes=decoded_bytes,
        preprocessed_bytes=preprocessed_bytes,
    )


def validate_mrnet_exam_resources(
    paths: dict[str, str | Path], *, size: int = 224
) -> dict[str, MRNetVolumeHeader]:
    """Validate all three planes and their combined resource budget."""
    expected = {"axial", "coronal", "sagittal"}
    if set(paths) != expected:
        raise MRNetVolumeError("MRI exam must contain axial, coronal, and sagittal paths")
    headers = {}
    for plane in sorted(expected):
        try:
            headers[plane] = inspect_mrnet_volume_header(paths[plane], size=size)
        except MRNetVolumeError as exc:
            raise MRNetVolumeError(
                f"Invalid {plane} volume: {exc}", code=exc.code
            ) from exc
    totals = {
        "file": sum(header.file_bytes for header in headers.values()),
        "decoded": sum(header.decoded_bytes for header in headers.values()),
        "preprocessed": sum(header.preprocessed_bytes for header in headers.values()),
    }
    limits = {
        "file": MAX_MRNET_EXAM_FILE_BYTES,
        "decoded": MAX_MRNET_EXAM_DECODED_BYTES,
        "preprocessed": MAX_MRNET_EXAM_PREPROCESSED_BYTES,
    }
    for resource, total in totals.items():
        if total > limits[resource]:
            raise _resource_error(
                f"MRI exam {resource} size exceeds {limits[resource]} bytes"
            )
    return headers


def load_mrnet_volume(
    path: str | Path, *, size: int = 224
) -> tuple[np.ndarray, MRNetVolumeInspection]:
    header = inspect_mrnet_volume_header(path, size=size)
    try:
        mapped = np.load(Path(path), mmap_mode="r", allow_pickle=False)
        volume = np.array(mapped, dtype=np.uint8, order="C", copy=True)
    except Exception as exc:
        raise MRNetVolumeError(f"Volume could not be decoded: {path}: {exc}") from exc
    if not np.isfinite(volume).all():
        raise MRNetVolumeError("Volume contains non-finite intensities")
    inspection = MRNetVolumeInspection(
        path=header.path, slices=header.slices, height=header.height,
        width=header.width, dtype=header.dtype,
        minimum=float(volume.min()), maximum=float(volume.max()),
        mean=float(volume.mean()), standard_deviation=float(volume.std()),
        file_bytes=header.file_bytes, decoded_bytes=header.decoded_bytes,
        preprocessed_bytes=header.preprocessed_bytes,
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
    volume, inspection = load_mrnet_volume(path, size=size)
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

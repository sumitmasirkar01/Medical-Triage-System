"""Validated image decoding and preprocessing for NIH ChestX-ray14."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torch import Tensor


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class NIHImageError(ValueError):
    """A radiograph cannot satisfy the locked image input contract."""


@dataclass(frozen=True)
class NIHImageInspection:
    path: str
    format: str
    original_mode: str
    original_width: int
    original_height: int
    grayscale_min: int
    grayscale_max: int

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


def decode_nih_radiograph(path: str | Path) -> tuple[np.ndarray, NIHImageInspection]:
    """Decode one image to uint8 grayscale while preserving audit metadata."""

    source = Path(path)
    if not source.is_file():
        raise NIHImageError(f"Image does not exist: {source}")
    try:
        with Image.open(source) as image:
            image_format = image.format or "UNKNOWN"
            original_mode = image.mode
            original_width, original_height = image.size
            if original_width <= 0 or original_height <= 0:
                raise NIHImageError(f"Image has invalid dimensions: {image.size}")
            grayscale = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    except NIHImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise NIHImageError(f"Image could not be decoded: {source}: {exc}") from exc
    if grayscale.ndim != 2 or grayscale.size == 0:
        raise NIHImageError(f"Decoded image is not two-dimensional: {source}")
    inspection = NIHImageInspection(
        path=str(source),
        format=image_format,
        original_mode=original_mode,
        original_width=original_width,
        original_height=original_height,
        grayscale_min=int(grayscale.min()),
        grayscale_max=int(grayscale.max()),
    )
    return grayscale, inspection


def resize_and_pad_grayscale(image: np.ndarray, size: int = 224) -> np.ndarray:
    """Resize without distortion and center-pad to a square uint8 image."""

    if not isinstance(size, int) or isinstance(size, bool) or size < 8:
        raise ValueError("size must be an integer of at least 8")
    array = np.asarray(image)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("image must be a non-empty two-dimensional array")
    if array.dtype != np.uint8:
        raise ValueError("image must use uint8 intensities")
    height, width = array.shape
    scale = min(size / width, size / height)
    resized_width = max(1, min(size, int(round(width * scale))))
    resized_height = max(1, min(size, int(round(height * scale))))
    pil = Image.fromarray(array, mode="L").resize(
        (resized_width, resized_height), resample=Image.Resampling.BILINEAR
    )
    canvas = np.zeros((size, size), dtype=np.uint8)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    canvas[top:top + resized_height, left:left + resized_width] = np.asarray(pil)
    return canvas


def preprocess_nih_radiograph(
    path: str | Path,
    *,
    size: int = 224,
) -> tuple[Tensor, np.ndarray, NIHImageInspection]:
    """Return ImageNet-normalized three-channel tensor plus preview image."""

    grayscale, inspection = decode_nih_radiograph(path)
    preview = resize_and_pad_grayscale(grayscale, size=size)
    tensor = torch.from_numpy(preview.copy()).to(torch.float32).div_(255.0)
    tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    tensor = (tensor - mean) / std
    if tensor.shape != (3, size, size) or not torch.isfinite(tensor).all():
        raise NIHImageError("Preprocessing produced an invalid tensor")
    return tensor, preview, inspection


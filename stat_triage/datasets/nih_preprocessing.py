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
MAX_NIH_IMAGE_FILE_BYTES = 32 * 1024 * 1024
MAX_NIH_IMAGE_WIDTH = 8192
MAX_NIH_IMAGE_HEIGHT = 8192
MAX_NIH_IMAGE_PIXELS = 32 * 1024 * 1024
MAX_NIH_DECODED_BYTES = 128 * 1024 * 1024


class NIHImageError(ValueError):
    """A radiograph cannot satisfy the locked image input contract."""

    def __init__(self, message: str, *, code: str = "image_decode_failed") -> None:
        super().__init__(message)
        self.code = code


def _resource_error(message: str) -> NIHImageError:
    return NIHImageError(message, code="input_resource_limit")


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
    file_bytes = source.stat().st_size
    if file_bytes > MAX_NIH_IMAGE_FILE_BYTES:
        raise _resource_error(
            f"Image file exceeds {MAX_NIH_IMAGE_FILE_BYTES} bytes"
        )
    try:
        with Image.open(source) as image:
            image_format = image.format or "UNKNOWN"
            original_mode = image.mode
            original_width, original_height = image.size
            if original_width <= 0 or original_height <= 0:
                raise NIHImageError(f"Image has invalid dimensions: {image.size}")
            if (
                original_width > MAX_NIH_IMAGE_WIDTH
                or original_height > MAX_NIH_IMAGE_HEIGHT
            ):
                raise _resource_error(
                    "Image dimensions exceed the supported maximum of "
                    f"{MAX_NIH_IMAGE_WIDTH} by {MAX_NIH_IMAGE_HEIGHT}"
                )
            pixels = original_width * original_height
            if pixels > MAX_NIH_IMAGE_PIXELS:
                raise _resource_error(
                    f"Image exceeds the {MAX_NIH_IMAGE_PIXELS}-pixel limit"
                )
            # Conservatively cover the source raster, grayscale conversion, and
            # copied NumPy output without trusting compressed file size.
            estimated_decoded_bytes = pixels * (max(4, len(image.getbands())) + 2)
            if estimated_decoded_bytes > MAX_NIH_DECODED_BYTES:
                raise _resource_error(
                    "Image decoded representation exceeds "
                    f"{MAX_NIH_DECODED_BYTES} bytes"
                )
            grayscale = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    except NIHImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise NIHImageError(f"Image could not be decoded: {source}: {exc}") from exc
    if grayscale.ndim != 2 or grayscale.size == 0:
        raise NIHImageError(f"Decoded image is not two-dimensional: {source}")
    if grayscale.nbytes > MAX_NIH_DECODED_BYTES:
        raise _resource_error("Decoded grayscale image exceeds the memory limit")
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


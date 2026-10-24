"""NIH radiographs decode and normalize without geometric distortion."""

import numpy as np
from PIL import Image
import pytest
import torch

from stat_triage.datasets import nih_preprocessing
from stat_triage.datasets.nih_preprocessing import (
    NIHImageError,
    decode_nih_radiograph,
    preprocess_nih_radiograph,
    resize_and_pad_grayscale,
)


def test_decode_and_preprocess_grayscale_image(tmp_path):
    source = tmp_path / "xray.png"
    pixels = np.tile(np.arange(80, dtype=np.uint8), (40, 1))
    Image.fromarray(pixels, mode="L").save(source)

    tensor, preview, inspection = preprocess_nih_radiograph(source, size=224)

    assert tensor.shape == (3, 224, 224)
    assert torch.isfinite(tensor).all()
    assert preview.shape == (224, 224)
    assert inspection.original_width == 80
    assert inspection.original_height == 40
    assert inspection.original_mode == "L"
    assert torch.equal(tensor[0], tensor[1]) is False  # channel normalization differs


def test_resize_preserves_aspect_and_centers_padding():
    image = np.full((20, 40), 255, dtype=np.uint8)
    result = resize_and_pad_grayscale(image, size=100)

    assert result.shape == (100, 100)
    assert np.all(result[:25] == 0)
    assert np.all(result[25:75] == 255)
    assert np.all(result[75:] == 0)


def test_rgb_input_is_audited_and_converted_to_grayscale(tmp_path):
    source = tmp_path / "rgb.png"
    Image.new("RGB", (16, 12), color=(255, 0, 0)).save(source)

    grayscale, inspection = decode_nih_radiograph(source)

    assert grayscale.shape == (12, 16)
    assert inspection.original_mode == "RGB"


def test_corrupt_image_is_rejected(tmp_path):
    source = tmp_path / "broken.png"
    source.write_bytes(b"not an image")

    with pytest.raises(NIHImageError, match="could not be decoded"):
        decode_nih_radiograph(source)


def test_file_size_limit_is_checked_before_image_decode(tmp_path, monkeypatch):
    source = tmp_path / "large.png"
    source.write_bytes(b"not-decoded")
    monkeypatch.setattr(nih_preprocessing, "MAX_NIH_IMAGE_FILE_BYTES", 4)

    with pytest.raises(NIHImageError, match="file exceeds") as error:
        decode_nih_radiograph(source)
    assert error.value.code == "input_resource_limit"


def test_dimensions_and_pixels_are_bounded_before_conversion(tmp_path, monkeypatch):
    source = tmp_path / "bounded.png"
    Image.new("L", (20, 10), color=1).save(source)
    monkeypatch.setattr(nih_preprocessing, "MAX_NIH_IMAGE_WIDTH", 10)
    with pytest.raises(NIHImageError, match="dimensions exceed"):
        decode_nih_radiograph(source)

    monkeypatch.setattr(nih_preprocessing, "MAX_NIH_IMAGE_WIDTH", 100)
    monkeypatch.setattr(nih_preprocessing, "MAX_NIH_IMAGE_PIXELS", 100)
    with pytest.raises(NIHImageError, match="pixel limit"):
        decode_nih_radiograph(source)


def test_multichannel_decoded_memory_is_bounded(tmp_path, monkeypatch):
    source = tmp_path / "rgb.png"
    Image.new("RGB", (10, 10), color=(1, 2, 3)).save(source)
    monkeypatch.setattr(nih_preprocessing, "MAX_NIH_DECODED_BYTES", 299)
    with pytest.raises(NIHImageError, match="decoded representation") as error:
        decode_nih_radiograph(source)
    assert error.value.code == "input_resource_limit"


"""MRNet preprocessing preserves slices and rejects incompatible arrays."""

import numpy as np
import pytest
import torch

from stat_triage.datasets import mrnet_preprocessing
from stat_triage.datasets.mrnet_preprocessing import (
    MRNetVolumeError,
    inspect_mrnet_volume_header,
    load_mrnet_volume,
    preprocess_mrnet_volume,
    validate_mrnet_exam_resources,
)


def test_volume_preprocessing_returns_all_normalized_slices(tmp_path):
    source = tmp_path / "volume.npy"
    np.save(source, np.arange(5 * 16 * 12, dtype=np.uint8).reshape(5, 16, 12))

    tensor, preview, inspection = preprocess_mrnet_volume(source, size=32)

    assert tensor.shape == (5, 3, 32, 32)
    assert preview.shape == (5, 32, 32)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()
    assert inspection.slices == 5


def test_volume_preprocessing_rejects_wrong_shape_and_dtype(tmp_path):
    wrong_shape = tmp_path / "shape.npy"
    np.save(wrong_shape, np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(MRNetVolumeError, match="Expected"):
        preprocess_mrnet_volume(wrong_shape)

    wrong_dtype = tmp_path / "dtype.npy"
    np.save(wrong_dtype, np.zeros((2, 8, 8), dtype=np.float32))
    with pytest.raises(MRNetVolumeError, match="uint8"):
        preprocess_mrnet_volume(wrong_dtype)


def test_volume_header_limits_are_enforced_before_copy(tmp_path, monkeypatch):
    source = tmp_path / "volume.npy"
    np.save(source, np.zeros((4, 16, 12), dtype=np.uint8))

    monkeypatch.setattr(mrnet_preprocessing, "MAX_MRNET_SLICES", 3)
    with pytest.raises(MRNetVolumeError, match="slice limit") as error:
        inspect_mrnet_volume_header(source)
    assert error.value.code == "input_resource_limit"

    monkeypatch.setattr(mrnet_preprocessing, "MAX_MRNET_SLICES", 10)
    monkeypatch.setattr(mrnet_preprocessing, "MAX_MRNET_WIDTH", 8)
    with pytest.raises(MRNetVolumeError, match="dimensions exceed"):
        inspect_mrnet_volume_header(source)


def test_volume_file_and_decoded_memory_limits_are_enforced(tmp_path, monkeypatch):
    source = tmp_path / "volume.npy"
    np.save(source, np.zeros((4, 16, 12), dtype=np.uint8))
    monkeypatch.setattr(mrnet_preprocessing, "MAX_MRNET_VOLUME_FILE_BYTES", 4)
    with pytest.raises(MRNetVolumeError, match="file exceeds"):
        inspect_mrnet_volume_header(source)

    monkeypatch.setattr(mrnet_preprocessing, "MAX_MRNET_VOLUME_FILE_BYTES", 10_000)
    monkeypatch.setattr(mrnet_preprocessing, "MAX_MRNET_DECODED_BYTES", 100)
    with pytest.raises(MRNetVolumeError, match="decoded representation"):
        inspect_mrnet_volume_header(source)


def test_volume_load_uses_memory_mapping_before_bounded_copy(tmp_path, monkeypatch):
    source = tmp_path / "volume.npy"
    np.save(source, np.zeros((4, 16, 12), dtype=np.uint8))
    real_load = np.load
    modes = []

    def tracked_load(*args, **kwargs):
        modes.append(kwargs.get("mmap_mode"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(mrnet_preprocessing.np, "load", tracked_load)
    volume, _ = load_mrnet_volume(source, size=32)
    assert volume.shape == (4, 16, 12)
    assert modes and all(mode == "r" for mode in modes)


def test_three_plane_exam_has_a_combined_memory_budget(tmp_path, monkeypatch):
    paths = {}
    for plane in ("axial", "coronal", "sagittal"):
        path = tmp_path / f"{plane}.npy"
        np.save(path, np.zeros((4, 16, 12), dtype=np.uint8))
        paths[plane] = path
    monkeypatch.setattr(mrnet_preprocessing, "MAX_MRNET_EXAM_DECODED_BYTES", 2000)

    with pytest.raises(MRNetVolumeError, match="exam decoded size") as error:
        validate_mrnet_exam_resources(paths, size=32)
    assert error.value.code == "input_resource_limit"

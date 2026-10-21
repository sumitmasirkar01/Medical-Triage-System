"""MRNet preprocessing preserves slices and rejects incompatible arrays."""

import numpy as np
import pytest
import torch

from stat_triage.datasets.mrnet_preprocessing import (
    MRNetVolumeError,
    preprocess_mrnet_volume,
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

"""The MRNet adapter returns calibrated exam-level common evidence."""

import numpy as np
import pytest

from stat_triage.contracts import Modality
from stat_triage.datasets.mrnet import MRNET_TARGETS
from stat_triage.mrnet import (
    DEFAULT_MRNET_BUNDLE_DIR,
    MRNetInferenceAdapter,
    MRNetInputError,
)


@pytest.fixture(scope="module")
def adapter():
    return MRNetInferenceAdapter(DEFAULT_MRNET_BUNDLE_DIR, device="cpu")


def test_adapter_loads_locked_mrnet_policy(adapter):
    assert adapter.model_epoch == 8
    assert tuple(adapter.thresholds) == MRNET_TARGETS
    assert adapter.thresholds["acl"] == pytest.approx(0.056712984052332584)
    assert all(slope > 0 for slope, _ in adapter.calibrators.values())


def test_real_checkpoint_three_plane_inference_returns_common_evidence(adapter, tmp_path):
    paths = {}
    for index, plane in enumerate(("axial", "coronal", "sagittal")):
        volume = np.stack(
            [
                np.linspace(0, 255, 32 * 32, dtype=np.uint8).reshape(32, 32)
                for _ in range(3 + index)
            ]
        )
        paths[plane] = tmp_path / f"{plane}.npy"
        np.save(paths[plane], volume)

    evidence = adapter.predict_exam(
        **paths, exam_id="MR-EXAM-1", patient_id="MR-PATIENT-1"
    )

    assert evidence.modality is Modality.KNEE_MRI
    assert evidence.exam_id == "MR-EXAM-1"
    assert tuple(evidence.raw_scores) == MRNET_TARGETS
    assert tuple(evidence.calibrated_scores) == MRNET_TARGETS
    assert tuple(evidence.thresholds) == MRNET_TARGETS
    assert set(evidence.input_hashes) == {
        "axial/axial.npy", "coronal/coronal.npy", "sagittal/sagittal.npy"
    }
    assert all(
        evidence.decisions[target]
        == (evidence.calibrated_scores[target] >= evidence.thresholds[target])
        for target in MRNET_TARGETS
    )


def test_adapter_rejects_invalid_plane_volume(adapter, tmp_path):
    valid = np.zeros((3, 32, 32), dtype=np.uint8)
    axial = tmp_path / "axial.npy"
    coronal = tmp_path / "coronal.npy"
    sagittal = tmp_path / "sagittal.npy"
    np.save(axial, valid.astype(np.float32))
    np.save(coronal, valid)
    np.save(sagittal, valid)

    with pytest.raises(MRNetInputError, match="Invalid axial volume"):
        adapter.predict_exam(
            axial=axial, coronal=coronal, sagittal=sagittal, exam_id="BAD-EXAM"
        )

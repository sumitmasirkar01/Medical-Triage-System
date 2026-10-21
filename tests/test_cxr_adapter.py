"""The CXR adapter verifies evidence and returns calibrated named decisions."""

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from stat_triage.contracts import Modality
from stat_triage.cxr import (
    DEFAULT_CXR_BUNDLE_DIR,
    NIHChestXrayAdapter,
    validate_cxr_evidence_bundle,
)
from stat_triage.datasets.nih_training import NIH_TARGET_ORDER


@pytest.fixture(scope="module")
def adapter():
    return NIHChestXrayAdapter(DEFAULT_CXR_BUNDLE_DIR, device="cpu")


def test_transferred_bundle_hashes_and_metrics_reproduce():
    summary = validate_cxr_evidence_bundle(
        DEFAULT_CXR_BUNDLE_DIR, recompute_metrics=True
    )

    assert summary["status"] == "validated_pilot"
    assert summary["artifact_count"] == 7
    assert summary["test_images"] == 25_596
    assert summary["test_patients"] == 2_797
    assert summary["checkpoint_sha256"] == (
        "609a353cf2d640ca7471e3952797b25ea4f5841d7f11fe7dfc8821dbfd9298bb"
    )


def test_adapter_loads_frozen_calibrators_and_thresholds(adapter):
    assert adapter.model_epoch == 4
    assert tuple(adapter.thresholds) == NIH_TARGET_ORDER
    assert adapter.thresholds["pneumothorax"] == pytest.approx(0.014949340588526329)
    assert all(slope > 0 for slope, _ in adapter.calibrators.values())


def test_real_checkpoint_inference_returns_common_evidence(adapter, tmp_path):
    image = np.linspace(0, 255, 96 * 128, dtype=np.uint8).reshape(96, 128)
    source = tmp_path / "synthetic_frontal.png"
    Image.fromarray(image, mode="L").save(source)

    evidence = adapter.predict_file(
        source, patient_id="PATIENT-1", exam_id="EXAM-1"
    )

    assert evidence.modality is Modality.CHEST_XRAY
    assert evidence.exam_id == "EXAM-1"
    assert tuple(evidence.raw_scores) == NIH_TARGET_ORDER
    assert tuple(evidence.calibrated_scores) == NIH_TARGET_ORDER
    assert tuple(evidence.thresholds) == NIH_TARGET_ORDER
    assert all(
        evidence.decisions[target]
        == (evidence.calibrated_scores[target] >= evidence.thresholds[target])
        for target in NIH_TARGET_ORDER
    )
    assert evidence.input_hashes[source.name]

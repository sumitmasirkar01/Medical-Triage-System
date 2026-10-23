import pandas as pd
import pytest

from stat_triage.cxr_robustness import (
    CXRRobustnessError,
    evaluate_locked_cxr_subgroups,
    lock_cxr_subgroups,
)
from stat_triage.datasets.nih_training import NIH_TARGET_ORDER


def _manifest(split: str, size: int = 120) -> pd.DataFrame:
    return pd.DataFrame({
        "image_id": [f"{split}-{index}" for index in range(size)],
        "patient_id": [f"patient-{split}-{index // 2}" for index in range(size)],
        "split": split,
        "view": ["AP" if index % 3 else "PA" for index in range(size)],
        "gender": ["F" if index % 4 < 2 else "M" for index in range(size)],
        "age": [20 + index % 60 for index in range(size)],
    })


def _predictions(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = manifest[["image_id", "patient_id"]].copy()
    for target in NIH_TARGET_ORDER:
        rows[f"target_{target}"] = [index % 2 for index in range(len(rows))]
        rows[f"calibrated_score_{target}"] = [
            0.8 if index % 2 else 0.2 for index in range(len(rows))
        ]
    return rows


def _locked_targets() -> dict[str, object]:
    return {
        target: {"operating_point": {"threshold": 0.5}}
        for target in NIH_TARGET_ORDER
    }


def test_cxr_subgroups_lock_on_calibration_and_apply_to_test():
    lock = lock_cxr_subgroups(_manifest("calibration"))
    test = _manifest("test")
    result = evaluate_locked_cxr_subgroups(
        test,
        _predictions(test),
        lock,
        _locked_targets(),
        bootstrap_replicates=20,
        minimum_images=10,
        minimum_positive_images=4,
        minimum_negative_images=4,
    )
    assert lock["test_rows_present"] == 0
    assert set(result["dimensions"]) == {"view", "recorded_sex", "age"}
    statuses = [
        values["status"]
        for dimension in result["dimensions"].values()
        for group in dimension.values()
        for values in group["targets"].values()
    ]
    assert set(statuses) == {"reported_exploratory"}


def test_cxr_subgroups_reject_wrong_partition_and_image_mismatch():
    with pytest.raises(CXRRobustnessError, match="calibration rows only"):
        lock_cxr_subgroups(_manifest("test"))
    lock = lock_cxr_subgroups(_manifest("calibration"))
    test = _manifest("test")
    with pytest.raises(CXRRobustnessError, match="different images"):
        evaluate_locked_cxr_subgroups(
            test,
            _predictions(test).iloc[:-1],
            lock,
            _locked_targets(),
        )

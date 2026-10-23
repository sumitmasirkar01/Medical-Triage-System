import pandas as pd
import pytest

from stat_triage.datasets.mrnet import MRNET_TARGETS
from stat_triage.mrnet_robustness import (
    MRNetRobustnessError,
    evaluate_threshold_robustness,
    lock_threshold_perturbations,
)


def _calibration() -> dict[str, object]:
    return {
        "status": "locked",
        "test_partition_accessed": False,
        "targets": {
            target: {"operating_point": {"threshold": 0.5}}
            for target in MRNET_TARGETS
        },
    }


def _predictions() -> pd.DataFrame:
    rows = pd.DataFrame({"exam_id": [f"exam-{index}" for index in range(40)]})
    for target in MRNET_TARGETS:
        rows[f"target_{target}"] = [index % 2 for index in range(40)]
        rows[f"calibrated_score_{target}"] = [0.2 + index / 60 for index in range(40)]
    return rows


def test_mrnet_threshold_grid_is_locked_before_test_evaluation():
    lock = lock_threshold_perturbations(_calibration())
    report = evaluate_threshold_robustness(
        _predictions(), lock, bootstrap_replicates=20
    )
    assert lock["test_rows_present"] == 0
    assert lock["analysis_status"] == "post_test_exploratory"
    assert lock["multipliers"] == [0.8, 0.9, 1.0, 1.1, 1.2]
    assert set(report["targets"]) == set(MRNET_TARGETS)
    assert report["analysis_status"] == "post_test_exploratory"
    assert all(len(result["profiles"]) == 5 for result in report["targets"].values())
    assert all(
        next(profile for profile in result["profiles"] if profile["multiplier"] == 1.0)[
            "changed_decisions_from_locked"
        ] == 0
        for result in report["targets"].values()
    )


def test_mrnet_threshold_grid_rejects_invalid_contracts():
    calibration = _calibration()
    calibration["test_partition_accessed"] = True
    with pytest.raises(MRNetRobustnessError, match="accessed the test"):
        lock_threshold_perturbations(calibration)
    with pytest.raises(MRNetRobustnessError, match="include 1.0"):
        lock_threshold_perturbations(_calibration(), multipliers=(0.8, 0.9))

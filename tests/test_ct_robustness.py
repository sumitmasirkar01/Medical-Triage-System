import pandas as pd
import pytest

from stat_triage.ct_robustness import (
    CTRobustnessError, evaluate_locked_slice_count_strata, lock_slice_count_strata,
)


def _rows(split: str) -> pd.DataFrame:
    counts = list(range(20, 80))
    return pd.DataFrame({
        "patient_id": [f"p-{split}-{i}" for i in range(60)],
        "study_uid": [f"s-{split}-{i}" for i in range(60)],
        "split": split, "slice_count": counts,
        "target_any": [i % 2 for i in range(60)],
        "score_any": [0.8 if i % 2 else 0.2 for i in range(60)],
    })


def test_slice_count_strata_lock_on_calibration_and_apply_to_test():
    lock = lock_slice_count_strata(_rows("calibration"))
    result = evaluate_locked_slice_count_strata(
        _rows("test"), lock, 0.5, bootstrap_replicates=20,
        minimum_group_size=10, minimum_class_size=4,
    )
    assert lock["test_rows_present"] == 0
    assert len(result["groups"]) == 3
    assert all(group["status"] == "reported_exploratory" for group in result["groups"].values())


def test_slice_count_strata_reject_wrong_partitions():
    with pytest.raises(CTRobustnessError, match="calibration studies only"):
        lock_slice_count_strata(_rows("test"))
    lock = lock_slice_count_strata(_rows("calibration"))
    with pytest.raises(CTRobustnessError, match="test studies only"):
        evaluate_locked_slice_count_strata(_rows("validation"), lock, 0.5)

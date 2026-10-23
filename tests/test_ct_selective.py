import pandas as pd
import pytest

from stat_triage.ct_selective import (
    CTSelectiveError, evaluate_locked_ct_selective_policy, lock_ct_selective_policy,
)


def _rows(split: str) -> pd.DataFrame:
    targets = [0] * 10 + [1] * 10
    scores = [0.05, 0.1, 0.15, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9, 0.95,
              0.15, 0.25, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 0.95]
    return pd.DataFrame({
        "patient_id": [f"p-{split}-{i}" for i in range(20)],
        "study_uid": [f"s-{split}-{i}" for i in range(20)],
        "split": split, "target_any": targets, "score_any": scores,
    })


def _protocol() -> dict[str, object]:
    return {"status": "locked", "test_evaluated": False,
            "operating_point": {"threshold": 0.5}}


def test_ct_selective_policy_locks_on_calibration_then_evaluates_test():
    policy = lock_ct_selective_policy(
        _rows("calibration"), _protocol(), minimum_coverage=0.5,
        maximum_automatic_false_negative_rate=0.1,
    )
    assert policy["test_rows_present"] == 0
    result = evaluate_locked_ct_selective_policy(
        _rows("test"), policy, bootstrap_replicates=20,
    )
    assert result["margin_reselected"] is False
    assert result["threshold_reselected"] is False
    assert result["patient_grouped_bootstrap"]["groups"] == 20


def test_ct_selective_policy_rejects_wrong_partitions():
    with pytest.raises(CTSelectiveError, match="calibration rows only"):
        lock_ct_selective_policy(_rows("test"), _protocol())
    policy = lock_ct_selective_policy(
        _rows("calibration"), _protocol(), minimum_coverage=0.5,
        maximum_automatic_false_negative_rate=0.1,
    )
    with pytest.raises(CTSelectiveError, match="test rows only"):
        evaluate_locked_ct_selective_policy(_rows("validation"), policy)

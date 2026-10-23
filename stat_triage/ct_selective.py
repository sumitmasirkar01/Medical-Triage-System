"""Locked selective-prediction policy for the RSNA Phase 2 CT endpoint."""

from __future__ import annotations

import math

import pandas as pd

from .selective import (
    grouped_selective_bootstrap_confidence_intervals,
    select_abstention_operating_point,
    selective_metrics,
)


class CTSelectiveError(ValueError):
    """Selective CT inputs or a locked policy are invalid."""


def lock_ct_selective_policy(
    calibration: pd.DataFrame,
    study_protocol: dict[str, object],
    *,
    minimum_coverage: float = 0.80,
    maximum_automatic_false_negative_rate: float = 0.05,
) -> dict[str, object]:
    """Choose the narrowest feasible abstention band on calibration studies."""

    required = {"patient_id", "study_uid", "split", "target_any", "score_any"}
    if missing := required.difference(calibration.columns):
        raise CTSelectiveError(f"Calibration rows missing columns: {sorted(missing)}")
    if set(calibration["split"]) != {"calibration"}:
        raise CTSelectiveError("Selective-policy locking requires calibration rows only")
    if calibration["patient_id"].duplicated().any() or calibration["study_uid"].duplicated().any():
        raise CTSelectiveError("Calibration must contain one study per patient")
    if study_protocol.get("status") != "locked" or study_protocol.get("test_evaluated") is not False:
        raise CTSelectiveError("Study protocol must be locked before selective-policy selection")
    operating = study_protocol.get("operating_point")
    if not isinstance(operating, dict) or "threshold" not in operating:
        raise CTSelectiveError("Study protocol has no locked threshold")
    threshold = float(operating["threshold"])
    selected = select_abstention_operating_point(
        calibration["target_any"], calibration["score_any"], threshold,
        minimum_coverage=minimum_coverage,
        maximum_automatic_false_negative_rate=maximum_automatic_false_negative_rate,
    )
    return {
        "status": "locked", "endpoint": "study_level_any_hemorrhage",
        "selection_partition": "calibration", "evaluation_partition": "test_not_loaded",
        "threshold": threshold, "margin_fraction": selected.margin_fraction,
        "negative_boundary": selected.negative_boundary,
        "positive_boundary": selected.positive_boundary,
        "minimum_coverage": minimum_coverage,
        "maximum_automatic_false_negative_rate": maximum_automatic_false_negative_rate,
        "selection_rule": selected.selection_rule,
        "calibration_metrics": selected.metrics.to_dict(),
        "calibration_studies": len(calibration), "test_rows_present": 0,
        "test_evaluated": False,
    }


def evaluate_locked_ct_selective_policy(
    test: pd.DataFrame,
    policy: dict[str, object],
    *,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 42,
) -> dict[str, object]:
    """Apply a locked abstention band once to test-only study predictions."""

    required = {"patient_id", "study_uid", "split", "target_any", "score_any"}
    if missing := required.difference(test.columns):
        raise CTSelectiveError(f"Test rows missing columns: {sorted(missing)}")
    if set(test["split"]) != {"test"}:
        raise CTSelectiveError("Selective evaluation requires test rows only")
    if test["patient_id"].duplicated().any() or test["study_uid"].duplicated().any():
        raise CTSelectiveError("Test data must contain one study per patient")
    if policy.get("status") != "locked" or policy.get("test_evaluated") is not False:
        raise CTSelectiveError("Selective policy is not cleanly locked")
    if policy.get("test_rows_present") != 0:
        raise CTSelectiveError("Selective policy was contaminated by test rows")
    threshold = float(policy.get("threshold", math.nan))
    margin = float(policy.get("margin_fraction", math.nan))
    measured = selective_metrics(
        test["target_any"], test["score_any"], threshold, margin
    )
    bootstrap = grouped_selective_bootstrap_confidence_intervals(
        test["target_any"], test["score_any"], test["patient_id"], threshold, margin,
        replicates=bootstrap_replicates, seed=bootstrap_seed,
        sampling_unit="patient_id",
    )
    return {
        "status": "complete", "endpoint": policy.get("endpoint"),
        "evaluation_partition": "test", "threshold": threshold,
        "margin_fraction": margin,
        "negative_boundary": policy.get("negative_boundary"),
        "positive_boundary": policy.get("positive_boundary"),
        "test_metrics": measured.to_dict(),
        "patient_grouped_bootstrap": bootstrap.to_dict(),
        "margin_reselected": False, "threshold_reselected": False,
        "test_evaluated_once": True,
    }

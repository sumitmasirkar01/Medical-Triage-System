"""Full-study RSNA cohort construction for Phase 2 evaluation."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..rsna import RSNA_LABELS


class RSNAStudyCohortError(ValueError):
    pass


_ID = re.compile(r"^(ID_[0-9a-f]+)_(epidural|intraparenchymal|intraventricular|subarachnoid|subdural|any)$")


def load_rsna_label_matrix(path: str | Path) -> pd.DataFrame:
    rows = pd.read_csv(path)
    if set(rows.columns) != {"ID", "Label"}:
        raise RSNAStudyCohortError("RSNA labels must contain ID and Label columns")
    parsed = rows["ID"].str.extract(_ID)
    if parsed.isna().any().any() or not rows["Label"].isin((0, 1)).all():
        raise RSNAStudyCohortError("Malformed RSNA label rows")
    normalized = pd.DataFrame({"image_id": parsed[0], "target": parsed[1], "label": rows["Label"]})
    if normalized.duplicated(["image_id", "target"]).any():
        duplicates = normalized.groupby(["image_id", "target"])["label"].nunique()
        if (duplicates > 1).any():
            raise RSNAStudyCohortError("Conflicting duplicate label rows")
        normalized = normalized.drop_duplicates(["image_id", "target"])
    matrix = normalized.pivot(index="image_id", columns="target", values="label")
    if set(matrix.columns) != set(RSNA_LABELS) or matrix.isna().any().any():
        raise RSNAStudyCohortError("Every image requires all six RSNA labels")
    return matrix.loc[:, RSNA_LABELS].reset_index()


def build_phase2_study_cohort(
    headers: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    patients_per_split: int = 240,
    seed: int = 42,
    excluded_patients: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {"image_id", "patient_id", "study_uid", "series_uid", "z_position", "dicom_path"}
    if missing := required.difference(headers.columns):
        raise RSNAStudyCohortError(f"Header index missing columns: {sorted(missing)}")
    if headers["image_id"].duplicated().any() or labels["image_id"].duplicated().any():
        raise RSNAStudyCohortError("image IDs must be unique")
    joined = headers.merge(labels, on="image_id", how="inner", validate="one_to_one")
    if len(joined) != len(headers):
        raise RSNAStudyCohortError("Some indexed DICOM slices lack labels")
    excluded = excluded_patients or set()
    joined = joined.loc[~joined["patient_id"].isin(excluded)].copy()
    studies = joined.groupby(["patient_id", "study_uid"], as_index=False).agg(
        target_any=("any", "max"), slice_count=("image_id", "size")
    )
    # One deterministic complete study per patient prevents patient weighting by exam count.
    studies = studies.sort_values(
        ["patient_id", "target_any", "slice_count", "study_uid"],
        ascending=[True, False, False, True], kind="stable",
    ).drop_duplicates("patient_id")
    if patients_per_split < 4:
        raise RSNAStudyCohortError("patients_per_split must be at least four")
    required_total = 3 * patients_per_split
    if len(studies) < required_total:
        raise RSNAStudyCohortError("Not enough eligible patients for the requested cohort")
    selected_parts = []
    remaining = studies.copy()
    for offset, split in enumerate(("test", "calibration", "validation")):
        chosen, remaining = train_test_split(
            remaining, train_size=patients_per_split, random_state=seed + offset,
            stratify=remaining["target_any"],
        )
        selected_parts.append(chosen.assign(split=split))
    selected = pd.concat(selected_parts, ignore_index=True)
    cohort = joined.merge(selected[["patient_id", "study_uid", "split"]],
                          on=["patient_id", "study_uid"], how="inner", validate="many_to_one")
    cohort = cohort.sort_values(["split", "study_uid", "z_position", "image_id"], kind="stable")
    overlap = cohort.groupby("patient_id")["split"].nunique().max()
    if overlap != 1:
        raise RSNAStudyCohortError("Patient leakage detected across Phase 2 splits")
    summary = {
        "seed": seed, "excluded_phase1_patients": len(excluded),
        "patients_per_split": patients_per_split, "slices": len(cohort),
        "studies": int(cohort["study_uid"].nunique()),
        "patients": int(cohort["patient_id"].nunique()),
        "splits": {
            split: {"slices": len(group), "studies": int(group["study_uid"].nunique()),
                    "patients": int(group["patient_id"].nunique()),
                    "positive_studies": int(group.groupby("study_uid")["any"].max().sum()),
                    "median_slices_per_study": float(group.groupby("study_uid").size().median())}
            for split, group in cohort.groupby("split")
        },
        "patient_leakage": 0,
    }
    return cohort.reset_index(drop=True), summary

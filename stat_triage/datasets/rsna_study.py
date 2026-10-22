"""Full-study RSNA cohort construction for Phase 2 evaluation."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..rsna import RSNA_LABELS
from ..evaluation import sha256


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


def validate_phase2_study_cohort(
    manifest_path: str | Path,
    summary_path: str | Path,
    *,
    phase1_manifest_path: str | Path | None = None,
    check_dicom_files: bool = True,
    minimum_median_slices_per_study: int = 10,
) -> dict[str, object]:
    """Independently validate a saved Phase 2 cohort before inference."""

    manifest_path = Path(manifest_path)
    summary_path = Path(summary_path)
    try:
        rows = pd.read_csv(manifest_path, dtype={"patient_id": str, "study_uid": str})
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RSNAStudyCohortError(f"Cannot read Phase 2 cohort artifacts: {exc}") from exc
    if not isinstance(summary, dict):
        raise RSNAStudyCohortError("Cohort summary must contain a JSON object")

    required = {
        "image_id", "patient_id", "study_uid", "series_uid", "z_position",
        "dicom_path", "split", *RSNA_LABELS,
    }
    if missing := required.difference(rows.columns):
        raise RSNAStudyCohortError(f"Saved manifest missing columns: {sorted(missing)}")
    if rows.empty or rows["image_id"].duplicated().any():
        raise RSNAStudyCohortError("Saved manifest must contain unique image IDs")
    if rows[list(required)].isna().any().any():
        raise RSNAStudyCohortError("Saved manifest contains missing required values")
    if set(rows["split"]) != {"validation", "calibration", "test"}:
        raise RSNAStudyCohortError("Saved manifest must contain exactly three Phase 2 splits")
    if not rows[list(RSNA_LABELS)].isin((0, 1)).all().all():
        raise RSNAStudyCohortError("Saved manifest labels must be binary")
    if not rows["z_position"].map(np.isfinite).all():
        raise RSNAStudyCohortError("Saved manifest has non-finite slice positions")
    subtype_max = rows[list(RSNA_LABELS[:-1])].max(axis=1)
    if not np.array_equal(subtype_max.to_numpy(), rows["any"].to_numpy()):
        raise RSNAStudyCohortError("The any label disagrees with the subtype union")
    if rows.groupby("patient_id")["split"].nunique().max() != 1:
        raise RSNAStudyCohortError("Patient leakage detected across Phase 2 splits")
    if rows.groupby("patient_id")["study_uid"].nunique().max() != 1:
        raise RSNAStudyCohortError("More than one study was selected for a patient")
    if rows.groupby("study_uid")["patient_id"].nunique().max() != 1:
        raise RSNAStudyCohortError("A study maps to more than one patient")
    sort_columns = ["split", "study_uid", "z_position", "image_id"]
    expected_order = rows.sort_values(sort_columns, kind="stable").index.to_numpy()
    if not np.array_equal(expected_order, np.arange(len(rows))):
        raise RSNAStudyCohortError("Manifest slices are not in deterministic study order")

    declaration = summary.get("manifest")
    if not isinstance(declaration, dict):
        raise RSNAStudyCohortError("Cohort summary has no manifest declaration")
    if declaration.get("bytes") != manifest_path.stat().st_size:
        raise RSNAStudyCohortError("Manifest byte count disagrees with cohort summary")
    if declaration.get("sha256") != sha256(manifest_path):
        raise RSNAStudyCohortError("Manifest SHA-256 disagrees with cohort summary")

    split_results: dict[str, dict[str, int | float]] = {}
    for split, group in rows.groupby("split", sort=True):
        slice_counts = group.groupby("study_uid").size()
        measured = {
            "slices": len(group), "studies": int(group["study_uid"].nunique()),
            "patients": int(group["patient_id"].nunique()),
            "positive_studies": int(group.groupby("study_uid")["any"].max().sum()),
            "median_slices_per_study": float(slice_counts.median()),
            "minimum_slices_per_study": int(slice_counts.min()),
            "maximum_slices_per_study": int(slice_counts.max()),
        }
        declared = summary.get("splits", {}).get(split)
        if not isinstance(declared, dict) or any(
            measured[key] != declared.get(key)
            for key in ("slices", "studies", "patients", "positive_studies",
                        "median_slices_per_study")
        ):
            raise RSNAStudyCohortError(f"Summary counts disagree for {split}")
        if measured["median_slices_per_study"] < minimum_median_slices_per_study:
            raise RSNAStudyCohortError(
                f"{split} median slices per study is below the protocol minimum"
            )
        split_results[split] = measured
    if summary.get("slices") != len(rows):
        raise RSNAStudyCohortError("Summary total slice count disagrees with manifest")
    if summary.get("studies") != rows["study_uid"].nunique():
        raise RSNAStudyCohortError("Summary total study count disagrees with manifest")
    if summary.get("patients") != rows["patient_id"].nunique():
        raise RSNAStudyCohortError("Summary total patient count disagrees with manifest")

    phase1_overlap = 0
    if phase1_manifest_path is not None:
        phase1 = pd.read_csv(phase1_manifest_path, dtype={"patient_id": str})
        if "patient_id" not in phase1:
            raise RSNAStudyCohortError("Phase 1 manifest has no patient_id column")
        phase1_overlap = len(set(rows["patient_id"]) & set(phase1["patient_id"]))
        if phase1_overlap:
            raise RSNAStudyCohortError("Phase 1 patients leaked into the Phase 2 cohort")

    missing_dicoms = 0
    if check_dicom_files:
        missing_dicoms = int(sum(not Path(path).is_file() for path in rows["dicom_path"]))
        if missing_dicoms:
            raise RSNAStudyCohortError(f"Phase 2 manifest references {missing_dicoms} missing DICOMs")
    return {
        "status": "passed", "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path), "slices": len(rows),
        "studies": int(rows["study_uid"].nunique()),
        "patients": int(rows["patient_id"].nunique()), "splits": split_results,
        "patient_leakage": 0, "phase1_patient_overlap": phase1_overlap,
        "missing_dicom_files": missing_dicoms,
        "minimum_median_slices_per_study": minimum_median_slices_per_study,
    }

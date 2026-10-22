import json
from pathlib import Path

import pandas as pd
import pytest

from stat_triage.datasets.rsna_study import (
    RSNAStudyCohortError, build_phase2_study_cohort, load_rsna_label_matrix,
    validate_phase2_study_cohort,
)
from stat_triage.evaluation import sha256
from stat_triage.rsna import RSNA_LABELS


def test_label_matrix_and_patient_disjoint_full_study_selection(tmp_path: Path):
    label_rows, headers = [], []
    for patient in range(18):
        for slice_index in range(3):
            image = f"ID_{patient:08x}{slice_index:x}"
            positive = int(patient % 2 == 0 and slice_index == 2)
            values = {label: 0 for label in RSNA_LABELS}
            values["any"] = positive
            values["subdural"] = positive
            for label, value in values.items():
                label_rows.append({"ID": f"{image}_{label}", "Label": value})
            headers.append({"image_id": image, "patient_id": f"p{patient}",
                            "study_uid": f"s{patient}", "series_uid": f"r{patient}",
                            "z_position": slice_index, "dicom_path": f"{image}.dcm"})
    label_path = tmp_path / "labels.csv"
    pd.DataFrame(label_rows).to_csv(label_path, index=False)
    labels = load_rsna_label_matrix(label_path)
    cohort, summary = build_phase2_study_cohort(
        pd.DataFrame(headers), labels, patients_per_split=4, seed=7,
        excluded_patients={"p0"},
    )
    assert summary["patient_leakage"] == 0
    assert summary["studies"] == 12
    assert cohort.groupby("study_uid").size().eq(3).all()
    assert cohort.groupby("patient_id")["split"].nunique().max() == 1


def _saved_cohort(tmp_path: Path):
    label_rows, headers = [], []
    for patient in range(18):
        for slice_index in range(12):
            image = f"ID_{patient:08x}{slice_index:02x}"
            positive = int(patient % 2 == 0 and slice_index == 11)
            values = {label: 0 for label in RSNA_LABELS}
            values["any"] = values["subdural"] = positive
            for label, value in values.items():
                label_rows.append({"ID": f"{image}_{label}", "Label": value})
            headers.append({"image_id": image, "patient_id": f"p{patient}",
                            "study_uid": f"s{patient:02d}", "series_uid": f"r{patient}",
                            "z_position": slice_index, "dicom_path": f"{image}.dcm"})
    labels_path = tmp_path / "labels.csv"
    pd.DataFrame(label_rows).to_csv(labels_path, index=False)
    cohort, summary = build_phase2_study_cohort(
        pd.DataFrame(headers), load_rsna_label_matrix(labels_path),
        patients_per_split=4, seed=7, excluded_patients={"p0"},
    )
    manifest = tmp_path / "manifest.csv"
    cohort.to_csv(manifest, index=False)
    summary["manifest"] = {"path": str(manifest), "bytes": manifest.stat().st_size,
                           "sha256": sha256(manifest)}
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    phase1 = tmp_path / "phase1.csv"
    pd.DataFrame({"patient_id": ["outside"]}).to_csv(phase1, index=False)
    return manifest, summary_path, phase1


def test_saved_phase2_cohort_validator_recomputes_integrity_checks(tmp_path: Path):
    manifest, summary, phase1 = _saved_cohort(tmp_path)
    report = validate_phase2_study_cohort(
        manifest, summary, phase1_manifest_path=phase1, check_dicom_files=False,
    )
    assert report["status"] == "passed"
    assert report["patients"] == 12
    assert report["phase1_patient_overlap"] == 0


def test_saved_phase2_cohort_validator_rejects_patient_leakage(tmp_path: Path):
    manifest, summary, phase1 = _saved_cohort(tmp_path)
    rows = pd.read_csv(manifest)
    leaked_patient = rows.loc[0, "patient_id"]
    pd.DataFrame({"patient_id": [leaked_patient]}).to_csv(phase1, index=False)
    with pytest.raises(RSNAStudyCohortError, match="Phase 1 patients leaked"):
        validate_phase2_study_cohort(
            manifest, summary, phase1_manifest_path=phase1, check_dicom_files=False,
        )

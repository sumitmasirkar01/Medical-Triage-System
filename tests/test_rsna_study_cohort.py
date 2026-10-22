from pathlib import Path

import pandas as pd

from stat_triage.datasets.rsna_study import build_phase2_study_cohort, load_rsna_label_matrix
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

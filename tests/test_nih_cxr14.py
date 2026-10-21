"""NIH discovery validates labels, image presence, and patient split safety."""

from pathlib import Path

import pandas as pd
import pytest

from stat_triage.datasets.nih_cxr14 import (
    NIH_METADATA_COLUMNS,
    NIHDatasetContractError,
    build_nih_cxr14_manifest,
    inventory_nih_cxr14,
)


def write_dataset(root: Path, *, patient_overlap: bool = False, missing_image: bool = False):
    image_names = ["a.png", "b.png", "c.png", "d.png", "e.png"]
    patients = [1, 1 if patient_overlap else 2, 3, 4, 5]
    labels = [
        "Pneumothorax|Effusion", "No Finding", "Consolidation",
        "Atelectasis", "Effusion|Atelectasis",
    ]
    rows = []
    for index, (image, patient, finding) in enumerate(zip(image_names, patients, labels)):
        rows.append({
            "Image Index": image, "Finding Labels": finding, "Follow-up #": index,
            "Patient ID": patient, "Patient Age": 50 + index,
            "Patient Gender": "M" if index % 2 else "F",
            "View Position": "PA" if index % 2 else "AP",
            "OriginalImage[Width": 1024, "Height]": 1024,
            "OriginalImagePixelSpacing[x": 0.14, "y]": 0.14,
        })
    pd.DataFrame(rows).to_csv(root / "Data_Entry_2017.csv", index=False)
    image_dir = root / "images_001" / "images"
    image_dir.mkdir(parents=True)
    for image in image_names[:-1] if missing_image else image_names:
        (image_dir / image).write_bytes(b"not-decoded-during-inventory")
    (root / "train_val_list.txt").write_text("a.png\nc.png\nd.png\ne.png\n", encoding="utf-8")
    (root / "test_list.txt").write_text("b.png\n", encoding="utf-8")


def test_inventory_maps_project_targets_and_checks_images(tmp_path):
    write_dataset(tmp_path)
    result = inventory_nih_cxr14(tmp_path)

    assert result["cohort"]["metadata_rows"] == 5
    assert result["cohort"]["unique_patients"] == 5
    assert result["targets"]["pleural_effusion"] == {
        "source_label": "Effusion", "positive_images": 2, "positive_patients": 2,
    }
    assert result["targets"]["pneumothorax"]["positive_images"] == 1
    assert result["supplied_split"]["patient_overlap"] == 0
    assert result["gate"]["ready_for_manifest"] is True
    assert result["gate"]["requires_new_patient_split"] is False


def test_inventory_detects_patient_leakage_in_supplied_lists(tmp_path):
    write_dataset(tmp_path, patient_overlap=True)
    result = inventory_nih_cxr14(tmp_path)

    assert result["supplied_split"]["image_overlap"] == 0
    assert result["supplied_split"]["patient_overlap"] == 1
    assert result["gate"]["requires_new_patient_split"] is True


def test_missing_images_block_manifest_gate(tmp_path):
    write_dataset(tmp_path, missing_image=True)
    result = inventory_nih_cxr14(tmp_path)

    assert result["cohort"]["metadata_images_missing_on_disk"] == 1
    assert result["examples"]["missing_images"] == ["e.png"]
    assert result["gate"]["ready_for_manifest"] is False


def test_missing_required_metadata_column_is_rejected(tmp_path):
    write_dataset(tmp_path)
    path = tmp_path / "Data_Entry_2017.csv"
    rows = pd.read_csv(path).drop(columns=[NIH_METADATA_COLUMNS[-1]])
    rows.to_csv(path, index=False)

    with pytest.raises(NIHDatasetContractError, match="missing required columns"):
        inventory_nih_cxr14(tmp_path)


def test_manifest_preserves_official_test_and_is_patient_disjoint(tmp_path):
    write_dataset(tmp_path)

    manifest, summary = build_nih_cxr14_manifest(tmp_path, candidate_seeds=16)

    assert len(manifest) == 5
    assert set(manifest["split"]) == {"train", "validation", "calibration", "test"}
    assert manifest.loc[manifest["split"] == "test", "image_id"].tolist() == ["b.png"]
    assert summary["checks"] == {
        "rows": 5,
        "unique_images": 5,
        "unique_patients": 5,
        "patient_leakage": 0,
        "image_leakage": 0,
        "unassigned_rows": 0,
        "official_test_preserved": True,
    }
    assert all(manifest[f"mask_{name}"].eq(1).all() for name in (
        "pneumothorax", "pleural_effusion", "consolidation", "atelectasis"
    ))


def test_manifest_assignment_is_deterministic(tmp_path):
    write_dataset(tmp_path)

    first, first_summary = build_nih_cxr14_manifest(tmp_path, candidate_seeds=16)
    second, second_summary = build_nih_cxr14_manifest(tmp_path, candidate_seeds=16)

    assert first[["image_id", "split"]].equals(second[["image_id", "split"]])
    assert first_summary["selected_seed"] == second_summary["selected_seed"]
    assert first_summary["balance_score"] == second_summary["balance_score"]


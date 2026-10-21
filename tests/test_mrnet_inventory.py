"""MRNet discovery joins labels and all three planes deterministically."""

from pathlib import Path

import numpy as np

from stat_triage.datasets.mrnet import (
    audit_mrnet,
    build_mrnet_manifest,
    discover_mrnet_layout,
)


def build_fixture(root: Path):
    for partition, count in (("train", 3), ("valid", 2)):
        for target_index, target in enumerate(("abnormal", "acl", "meniscus")):
            rows = "".join(f"{index:04d},{int((index + target_index) % 2)}\n" for index in range(count))
            (root / f"{partition}-{target}.csv").write_text(rows, encoding="utf-8")
        for plane_index, plane in enumerate(("axial", "coronal", "sagittal")):
            directory = root / partition / plane
            directory.mkdir(parents=True)
            for index in range(count):
                np.save(directory / f"{index:04d}.npy", np.full((2 + plane_index, 8, 9), index, dtype=np.uint8))


def test_inventory_validates_complete_three_plane_exams(tmp_path):
    build_fixture(tmp_path)

    layout = discover_mrnet_layout(tmp_path)
    audit, summary = audit_mrnet(tmp_path, progress_every=0)

    assert layout.plane_directories["train"]["axial"].name == "axial"
    assert len(audit) == 15
    assert summary["checks"] == {
        "audit_rows": 15,
        "valid_arrays": 15,
        "invalid_arrays": 0,
        "missing_arrays": 0,
        "complete_exams": 5,
    }
    assert summary["patient_identifiers_available"] is False


def test_redivis_underscore_header_export_restores_first_real_record(tmp_path):
    build_fixture(tmp_path)
    # Reproduce the Redivis representation observed in the official download.
    for partition, first_id in (("train", "0000"), ("valid", "0000")):
        for target in ("abnormal", "acl", "meniscus"):
            path = tmp_path / f"{partition}-{target}.csv"
            lines = path.read_text(encoding="utf-8").splitlines()
            first_exam, first_label = lines[0].split(",")
            path.write_text(
                f'"_{first_exam}","_{first_label}"\n' + "\n".join(lines[1:]) + "\n",
                encoding="utf-8",
            )

    audit, summary = audit_mrnet(tmp_path, progress_every=0)

    assert len(audit) == 15
    assert summary["checks"]["complete_exams"] == 5


def test_manifest_preserves_official_validation_as_test(tmp_path):
    # A larger balanced fixture is needed for all three development partitions.
    for partition, count in (("train", 30), ("valid", 9)):
        offset = 0 if partition == "train" else 30
        for target_index, target in enumerate(("abnormal", "acl", "meniscus")):
            rows = "".join(
                f"{index + offset:04d},{int((index + target_index) % 3 == 0)}\n"
                for index in range(count)
            )
            (tmp_path / f"{partition}-{target}.csv").write_text(rows, encoding="utf-8")
        for plane in ("axial", "coronal", "sagittal"):
            directory = tmp_path / partition / plane
            directory.mkdir(parents=True)
            for index in range(count):
                np.save(directory / f"{index + offset:04d}.npy", np.zeros((2, 8, 8), dtype=np.uint8))

    manifest, summary = build_mrnet_manifest(
        tmp_path, validation_exams=6, calibration_exams=6, seed_candidates=50
    )

    assert manifest["exam_id"].is_unique
    assert manifest["split"].value_counts().to_dict() == {
        "train": 18, "test": 9, "validation": 6, "calibration": 6
    }
    assert set(manifest.loc[manifest["split"] == "test", "source_partition"]) == {"valid"}
    assert summary["official_valid_preserved_as_locked_test"] is True
    assert summary["checks"]["all_plane_paths_exist"] is True

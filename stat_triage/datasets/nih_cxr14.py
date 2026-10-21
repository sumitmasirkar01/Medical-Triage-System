"""Read-only discovery and validation for the NIH ChestX-ray14 release."""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


NIH_METADATA_COLUMNS = (
    "Image Index", "Finding Labels", "Follow-up #", "Patient ID",
    "Patient Age", "Patient Gender", "View Position", "OriginalImage[Width",
    "Height]", "OriginalImagePixelSpacing[x", "y]",
)

# Project name -> exact source label. Preserve both names in every artifact.
NIH_PROJECT_TARGETS = {
    "pneumothorax": "Pneumothorax",
    "pleural_effusion": "Effusion",
    "consolidation": "Consolidation",
    "atelectasis": "Atelectasis",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


class NIHDatasetContractError(ValueError):
    """The mounted dataset does not satisfy the NIH metadata contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _single_candidate(root: Path, pattern: str, *, required: bool) -> Path | None:
    candidates = sorted(path for path in root.rglob(pattern) if path.is_file())
    if not candidates:
        if required:
            raise NIHDatasetContractError(f"No file matching {pattern!r} under {root}")
        return None
    if len(candidates) > 1:
        joined = ", ".join(str(path.relative_to(root)) for path in candidates[:5])
        raise NIHDatasetContractError(
            f"Multiple files match {pattern!r}; select one dataset root: {joined}"
        )
    return candidates[0]


def _read_split(path: Path | None) -> list[str]:
    if path is None:
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _relative(path: Path | None, root: Path) -> str | None:
    return None if path is None else str(path.relative_to(root)).replace("\\", "/")


def inventory_nih_cxr14(root: str | Path) -> dict[str, Any]:
    """Inspect metadata, image presence, labels, views, and supplied split safety."""

    dataset_root = Path(root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise NIHDatasetContractError(f"Dataset root is not a directory: {dataset_root}")
    metadata_path = _single_candidate(dataset_root, "Data_Entry_2017*.csv", required=True)
    assert metadata_path is not None
    train_list_path = _single_candidate(dataset_root, "train_val_list.txt", required=False)
    test_list_path = _single_candidate(dataset_root, "test_list.txt", required=False)

    rows = pd.read_csv(metadata_path)
    missing_columns = sorted(set(NIH_METADATA_COLUMNS).difference(rows.columns))
    if missing_columns:
        raise NIHDatasetContractError(
            "Metadata is missing required columns: " + ", ".join(missing_columns)
        )
    if rows.empty:
        raise NIHDatasetContractError("Metadata contains no image rows")
    if rows["Image Index"].isna().any() or rows["Patient ID"].isna().any():
        raise NIHDatasetContractError("Image Index and Patient ID cannot be missing")

    image_names = rows["Image Index"].astype(str)
    label_sets = rows["Finding Labels"].fillna("").astype(str).map(
        lambda value: frozenset(part.strip() for part in value.split("|") if part.strip())
    )
    label_counts: Counter[str] = Counter(label for labels in label_sets for label in labels)
    target_summary: dict[str, dict[str, int | str]] = {}
    for project_name, source_name in NIH_PROJECT_TARGETS.items():
        positive = label_sets.map(lambda labels: source_name in labels)
        target_summary[project_name] = {
            "source_label": source_name,
            "positive_images": int(positive.sum()),
            "positive_patients": int(rows.loc[positive, "Patient ID"].nunique()),
        }

    discovered_images: dict[str, Path] = {}
    duplicate_filenames: set[str] = set()
    image_files_total = 0
    for path in dataset_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        image_files_total += 1
        if path.name in discovered_images:
            duplicate_filenames.add(path.name)
        else:
            discovered_images[path.name] = path
    metadata_names = set(image_names)
    discovered_names = set(discovered_images)
    missing_images = sorted(metadata_names - discovered_names)
    orphan_images = sorted(discovered_names - metadata_names)

    train_names = _read_split(train_list_path)
    test_names = _read_split(test_list_path)
    train_set, test_set = set(train_names), set(test_names)
    metadata_by_image = rows.drop_duplicates("Image Index").set_index("Image Index")
    known_train = train_set.intersection(metadata_by_image.index)
    known_test = test_set.intersection(metadata_by_image.index)
    train_patients = set(metadata_by_image.loc[list(known_train), "Patient ID"]) if known_train else set()
    test_patients = set(metadata_by_image.loc[list(known_test), "Patient ID"]) if known_test else set()
    split_status = "present" if train_list_path and test_list_path else "absent_or_incomplete"

    no_finding_conflicts = int(
        label_sets.map(lambda labels: "No Finding" in labels and len(labels) > 1).sum()
    )
    age_values = pd.to_numeric(rows["Patient Age"], errors="coerce")
    gender_values = rows["Patient Gender"].fillna("MISSING").astype(str)
    view_values = rows["View Position"].fillna("MISSING").astype(str)

    return {
        "schema_version": "1.0",
        "dataset": "NIH ChestX-ray14",
        "status": "inventory_complete",
        "root": str(dataset_root),
        "files": {
            "metadata_csv": _relative(metadata_path, dataset_root),
            "metadata_sha256": _sha256(metadata_path),
            "train_val_list": _relative(train_list_path, dataset_root),
            "train_val_list_sha256": _sha256(train_list_path) if train_list_path else None,
            "test_list": _relative(test_list_path, dataset_root),
            "test_list_sha256": _sha256(test_list_path) if test_list_path else None,
        },
        "cohort": {
            "metadata_rows": len(rows),
            "unique_metadata_images": len(metadata_names),
            "unique_patients": int(rows["Patient ID"].nunique()),
            "duplicate_metadata_rows": int(image_names.duplicated().sum()),
            "image_files_discovered": image_files_total,
            "unique_image_filenames": len(discovered_names),
            "duplicate_image_filenames": len(duplicate_filenames),
            "metadata_images_missing_on_disk": len(missing_images),
            "images_without_metadata": len(orphan_images),
        },
        "targets": target_summary,
        "all_source_label_counts": dict(sorted(label_counts.items())),
        "metadata_quality": {
            "view_counts": {str(k): int(v) for k, v in view_values.value_counts().items()},
            "gender_counts": {str(k): int(v) for k, v in gender_values.value_counts().items()},
            "invalid_age_values": int(age_values.isna().sum()),
            "ages_over_120": int((age_values > 120).sum()),
            "no_finding_with_additional_labels": no_finding_conflicts,
        },
        "supplied_split": {
            "status": split_status,
            "train_entries": len(train_names),
            "test_entries": len(test_names),
            "duplicate_train_entries": len(train_names) - len(train_set),
            "duplicate_test_entries": len(test_names) - len(test_set),
            "image_overlap": len(train_set & test_set),
            "patient_overlap": len(train_patients & test_patients),
            "train_entries_missing_metadata": len(train_set - metadata_names),
            "test_entries_missing_metadata": len(test_set - metadata_names),
            "metadata_images_unassigned": len(metadata_names - (train_set | test_set)),
        },
        "examples": {
            "missing_images": missing_images[:10],
            "orphan_images": orphan_images[:10],
            "duplicate_image_filenames": sorted(duplicate_filenames)[:10],
        },
        "gate": {
            "ready_for_manifest": bool(
                image_names.duplicated().sum() == 0
                and not duplicate_filenames and not missing_images
                and no_finding_conflicts == 0
            ),
            "requires_new_patient_split": bool(
                split_status != "present" or train_patients & test_patients
            ),
            "training_started": False,
        },
    }


def build_nih_cxr14_manifest(
    root: str | Path,
    *,
    candidate_seeds: int = 256,
    validation_fraction: float = 0.10,
    calibration_fraction: float = 0.10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build deterministic patient-disjoint development splits plus official test."""

    if isinstance(candidate_seeds, bool) or not isinstance(candidate_seeds, int) or candidate_seeds < 1:
        raise ValueError("candidate_seeds must be a positive integer")
    for name, value in (
        ("validation_fraction", validation_fraction),
        ("calibration_fraction", calibration_fraction),
    ):
        if not isinstance(value, (int, float)) or not 0 < float(value) < 1:
            raise ValueError(f"{name} must be in (0, 1)")
    if validation_fraction + calibration_fraction >= 1:
        raise ValueError("validation and calibration fractions must sum to less than 1")

    dataset_root = Path(root).expanduser().resolve()
    inventory = inventory_nih_cxr14(dataset_root)
    if not inventory["gate"]["ready_for_manifest"]:
        raise NIHDatasetContractError("Dataset inventory is not ready for manifest creation")
    if inventory["supplied_split"]["status"] != "present":
        raise NIHDatasetContractError("Both official split-list files are required")
    if inventory["supplied_split"]["patient_overlap"]:
        raise NIHDatasetContractError("Official train and test lists have patient overlap")
    metadata_path = dataset_root / inventory["files"]["metadata_csv"]
    train_path = dataset_root / inventory["files"]["train_val_list"]
    test_path = dataset_root / inventory["files"]["test_list"]
    rows = pd.read_csv(metadata_path).copy()
    rows["Image Index"] = rows["Image Index"].astype(str)
    rows["patient_id"] = rows["Patient ID"].astype(str)
    train_names, test_names = set(_read_split(train_path)), set(_read_split(test_path))
    rows["source_split"] = np.where(
        rows["Image Index"].isin(test_names), "official_test", "official_train_val"
    )
    if not set(rows.loc[rows["source_split"] == "official_train_val", "Image Index"]) == train_names:
        raise NIHDatasetContractError("Official train list does not match metadata assignment")

    label_sets = rows["Finding Labels"].fillna("").astype(str).map(
        lambda value: frozenset(part.strip() for part in value.split("|") if part.strip())
    )
    for project_name, source_name in NIH_PROJECT_TARGETS.items():
        rows[f"target_{project_name}"] = label_sets.map(
            lambda labels: int(source_name in labels)
        )
        rows[f"mask_{project_name}"] = 1

    development = rows[rows["source_split"] == "official_train_val"].copy()
    target_columns = [f"target_{name}" for name in NIH_PROJECT_TARGETS]
    development["view_ap"] = (development["View Position"] == "AP").astype(int)
    patient_table = development.groupby("patient_id", sort=True).agg(
        images=("Image Index", "size"),
        view_ap=("view_ap", "sum"),
        **{column: (column, "sum") for column in target_columns},
    )
    patient_ids = patient_table.index.to_numpy()
    if len(patient_ids) < 3:
        raise NIHDatasetContractError("At least three development patients are required")
    n_validation = max(1, int(round(len(patient_ids) * validation_fraction)))
    n_calibration = max(1, int(round(len(patient_ids) * calibration_fraction)))
    n_train = len(patient_ids) - n_validation - n_calibration
    if n_train < 1:
        raise NIHDatasetContractError("Split fractions leave no training patients")

    feature_columns = ["images", *target_columns, "view_ap"]
    matrix = patient_table[feature_columns].to_numpy(dtype=np.float64)
    total = matrix.sum(axis=0)
    overall_prevalence = total[1:] / total[0]
    desired_patient_fractions = np.array(
        [n_train, n_validation, n_calibration], dtype=np.float64
    ) / len(patient_ids)

    def assignment_score(parts: tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
        score = 0.0
        for index, indices in enumerate(parts):
            subtotal = matrix[indices].sum(axis=0)
            image_fraction = subtotal[0] / total[0]
            prevalence = subtotal[1:] / subtotal[0]
            prevalence_scale = np.maximum(overall_prevalence, 0.01)
            score += float(np.mean(((prevalence - overall_prevalence) / prevalence_scale) ** 2))
            score += float((image_fraction - desired_patient_fractions[index]) ** 2)
        return score

    best: tuple[float, int, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None
    for seed in range(candidate_seeds):
        order = np.random.default_rng(seed).permutation(len(patient_ids))
        parts = (
            order[:n_train],
            order[n_train:n_train + n_validation],
            order[n_train + n_validation:],
        )
        score = assignment_score(parts)
        if best is None or score < best[0]:
            best = (score, seed, parts)
    assert best is not None
    balance_score, selected_seed, selected_parts = best
    patient_split: dict[str, str] = {}
    for split, indices in zip(("train", "validation", "calibration"), selected_parts):
        patient_split.update({str(patient_ids[index]): split for index in indices})

    rows["split"] = rows["patient_id"].map(patient_split)
    rows.loc[rows["source_split"] == "official_test", "split"] = "test"
    if rows["split"].isna().any():
        raise NIHDatasetContractError("Some metadata rows were not assigned to a split")

    image_paths: dict[str, str] = {}
    for path in dataset_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            image_paths[path.name] = str(path)
    rows["image_path"] = rows["Image Index"].map(image_paths)
    if rows["image_path"].isna().any():
        raise NIHDatasetContractError("Some manifest images have no resolved path")

    manifest = pd.DataFrame({
        "image_id": rows["Image Index"],
        "image_path": rows["image_path"],
        "patient_id": rows["patient_id"],
        "view": rows["View Position"].astype(str),
        "gender": rows["Patient Gender"].astype(str),
        "age": rows["Patient Age"],
        "source_split": rows["source_split"],
        "split": rows["split"],
        **{column: rows[column].astype(int) for column in target_columns},
        **{
            f"mask_{name}": rows[f"mask_{name}"].astype(int)
            for name in NIH_PROJECT_TARGETS
        },
        "source_labels": rows["Finding Labels"].astype(str),
    })

    patient_leakage = int((manifest.groupby("patient_id")["split"].nunique() > 1).sum())
    image_leakage = int(manifest.groupby("image_id")["split"].nunique().gt(1).sum())
    if patient_leakage or image_leakage or manifest["image_id"].duplicated().any():
        raise NIHDatasetContractError("Generated manifest failed leakage or uniqueness checks")

    split_summary: dict[str, Any] = {}
    for split in ("train", "validation", "calibration", "test"):
        selected = manifest[manifest["split"] == split]
        split_summary[split] = {
            "images": len(selected),
            "patients": int(selected["patient_id"].nunique()),
            "view_counts": {
                str(key): int(value) for key, value in selected["view"].value_counts().items()
            },
            "targets": {
                name: {
                    "positive_images": int(selected[f"target_{name}"].sum()),
                    "positive_patients": int(
                        selected.loc[selected[f"target_{name}"] == 1, "patient_id"].nunique()
                    ),
                    "prevalence": float(selected[f"target_{name}"].mean()),
                }
                for name in NIH_PROJECT_TARGETS
            },
        }
    summary = {
        "schema_version": "1.0",
        "dataset": "NIH ChestX-ray14",
        "selection_scope": "official_train_val_patients_only",
        "selected_seed": selected_seed,
        "candidate_seeds": candidate_seeds,
        "balance_score": balance_score,
        "development_patient_fractions": {
            "train": n_train / len(patient_ids),
            "validation": n_validation / len(patient_ids),
            "calibration": n_calibration / len(patient_ids),
        },
        "splits": split_summary,
        "checks": {
            "rows": len(manifest),
            "unique_images": int(manifest["image_id"].nunique()),
            "unique_patients": int(manifest["patient_id"].nunique()),
            "patient_leakage": patient_leakage,
            "image_leakage": image_leakage,
            "unassigned_rows": int(manifest["split"].isna().sum()),
            "official_test_preserved": bool(
                set(manifest.loc[manifest["split"] == "test", "image_id"]) == test_names
            ),
        },
    }
    return manifest, summary


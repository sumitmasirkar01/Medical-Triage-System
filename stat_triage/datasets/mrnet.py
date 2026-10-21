"""Discovery and integrity inventory for the Stanford MRNet release."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MRNET_TARGETS = ("abnormal", "acl", "meniscus")
MRNET_PLANES = ("axial", "coronal", "sagittal")
MRNET_SOURCE_PARTITIONS = ("train", "valid")


class MRNetIntegrityError(ValueError):
    """The attached MRNet copy does not satisfy the expected data contract."""


@dataclass(frozen=True)
class MRNetLayout:
    label_files: dict[str, dict[str, Path]]
    plane_directories: dict[str, dict[str, Path]]


def _unique_match(root: Path, name: str, *, directory: bool = False) -> Path:
    matches = sorted(
        path for path in root.rglob(name)
        if path.is_dir() if directory
    ) if directory else sorted(path for path in root.rglob(name) if path.is_file())
    if len(matches) != 1:
        raise MRNetIntegrityError(
            f"Expected exactly one {'directory' if directory else 'file'} named "
            f"{name!r}; found {len(matches)}: {[str(path) for path in matches]}"
        )
    return matches[0]


def discover_mrnet_layout(root: str | Path) -> MRNetLayout:
    """Locate the canonical six label files and six plane directories."""

    source = Path(root).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    labels: dict[str, dict[str, Path]] = {}
    planes: dict[str, dict[str, Path]] = {}
    for partition in MRNET_SOURCE_PARTITIONS:
        labels[partition] = {
            target: _unique_match(source, f"{partition}-{target}.csv")
            for target in MRNET_TARGETS
        }
        partition_directories = [
            path for path in source.rglob(partition) if path.is_dir()
            and all((path / plane).is_dir() for plane in MRNET_PLANES)
        ]
        if len(partition_directories) != 1:
            raise MRNetIntegrityError(
                f"Expected one {partition!r} directory containing all three planes; "
                f"found {len(partition_directories)}"
            )
        planes[partition] = {
            plane: partition_directories[0] / plane for plane in MRNET_PLANES
        }
    return MRNetLayout(labels, planes)


def _read_label_file(path: Path, target: str) -> pd.DataFrame:
    rows = pd.read_csv(
        path,
        header=None,
        names=["exam_id", target],
        dtype=str,
        keep_default_na=False,
    )
    # Redivis CSV exports can promote the first real headerless MRNet row to
    # column names and serialize it back as values prefixed by underscores,
    # e.g. `"_0000","_1"`. Treat that representation as the original row.
    rows["exam_id"] = (
        rows["exam_id"].astype(str).str.strip().str.removeprefix("_").str.zfill(4)
    )
    normalized_labels = rows[target].astype(str).str.strip().str.removeprefix("_")
    if not normalized_labels.isin(["0", "1", "0.0", "1.0"]).all():
        unexpected = sorted(normalized_labels[~normalized_labels.isin(["0", "1", "0.0", "1.0"])].unique())
        raise MRNetIntegrityError(
            f"{path.name} contains non-binary labels: {unexpected}"
        )
    rows[target] = normalized_labels.astype(float).astype(int)
    if rows.empty or rows["exam_id"].duplicated().any():
        raise MRNetIntegrityError(f"{path.name} is empty or has duplicate exam IDs")
    return rows


def load_mrnet_labels(layout: MRNetLayout, partition: str) -> pd.DataFrame:
    if partition not in MRNET_SOURCE_PARTITIONS:
        raise ValueError(f"Unsupported MRNet source partition: {partition}")
    tables = [
        _read_label_file(layout.label_files[partition][target], target)
        for target in MRNET_TARGETS
    ]
    expected = set(tables[0]["exam_id"])
    for table, target in zip(tables[1:], MRNET_TARGETS[1:]):
        if set(table["exam_id"]) != expected:
            raise MRNetIntegrityError(
                f"{partition} label IDs disagree for target {target}"
            )
    joined = tables[0]
    for table in tables[1:]:
        joined = joined.merge(table, on="exam_id", how="inner", validate="one_to_one")
    joined.insert(0, "source_partition", partition)
    return joined.sort_values("exam_id").reset_index(drop=True)


def _array_files(directory: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(directory.glob("*.npy")):
        exam_id = path.stem.strip().zfill(4)
        if exam_id in files:
            raise MRNetIntegrityError(f"Duplicate exam array for {exam_id} in {directory}")
        files[exam_id] = path
    return files


def audit_mrnet(
    root: str | Path,
    *,
    progress_every: int = 250,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Inspect every labelled exam-plane array and return row-level evidence."""

    layout = discover_mrnet_layout(root)
    audit_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "dataset": "Stanford MRNet",
        "targets": list(MRNET_TARGETS),
        "planes": list(MRNET_PLANES),
        "partitions": {},
        "patient_identifiers_available": False,
    }
    examined = 0
    for partition in MRNET_SOURCE_PARTITIONS:
        labels = load_mrnet_labels(layout, partition)
        label_ids = set(labels["exam_id"])
        plane_files = {
            plane: _array_files(layout.plane_directories[partition][plane])
            for plane in MRNET_PLANES
        }
        missing = {
            plane: sorted(label_ids.difference(files))
            for plane, files in plane_files.items()
        }
        extra = {
            plane: sorted(set(files).difference(label_ids))
            for plane, files in plane_files.items()
        }
        for label in labels.itertuples(index=False):
            for plane in MRNET_PLANES:
                path = plane_files[plane].get(label.exam_id)
                row: dict[str, object] = {
                    "source_partition": partition,
                    "exam_id": label.exam_id,
                    "plane": plane,
                    "path": str(path) if path else "",
                    **{target: int(getattr(label, target)) for target in MRNET_TARGETS},
                }
                if path is None:
                    row.update({"status": "missing", "error": "missing_plane_array"})
                else:
                    try:
                        array = np.load(path, mmap_mode="r", allow_pickle=False)
                        if array.ndim != 3 or array.shape[0] < 1:
                            raise MRNetIntegrityError(
                                f"expected [slices,height,width], received {array.shape}"
                            )
                        finite = bool(np.isfinite(array).all())
                        row.update({
                            "status": "valid" if finite else "invalid",
                            "error": "" if finite else "non_finite_values",
                            "slices": int(array.shape[0]),
                            "height": int(array.shape[1]),
                            "width": int(array.shape[2]),
                            "dtype": str(array.dtype),
                            "minimum": float(np.min(array)),
                            "maximum": float(np.max(array)),
                            "mean": float(np.mean(array)),
                            "standard_deviation": float(np.std(array)),
                            "bytes": int(path.stat().st_size),
                        })
                    except Exception as exc:
                        row.update({"status": "invalid", "error": str(exc)})
                audit_rows.append(row)
                examined += 1
                if progress_every and examined % progress_every == 0:
                    print(f"Audited {examined:,} exam-plane arrays", flush=True)
        summary["partitions"][partition] = {
            "labelled_exams": len(labels),
            "positive_exams": {
                target: int(labels[target].sum()) for target in MRNET_TARGETS
            },
            "plane_array_counts": {
                plane: len(files) for plane, files in plane_files.items()
            },
            "missing_labelled_arrays": {
                plane: len(values) for plane, values in missing.items()
            },
            "unlabelled_arrays": {
                plane: len(values) for plane, values in extra.items()
            },
        }

    audit = pd.DataFrame(audit_rows)
    summary["checks"] = {
        "audit_rows": len(audit),
        "valid_arrays": int((audit["status"] == "valid").sum()),
        "invalid_arrays": int((audit["status"] == "invalid").sum()),
        "missing_arrays": int((audit["status"] == "missing").sum()),
        "complete_exams": int(
            (audit.groupby(["source_partition", "exam_id"])["status"]
             .apply(lambda values: len(values) == 3 and (values == "valid").all())).sum()
        ),
    }
    return audit, summary


def build_mrnet_manifest(
    root: str | Path,
    *,
    validation_exams: int = 113,
    calibration_exams: int = 113,
    seed_candidates: int = 1000,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Create deterministic exam-disjoint development splits and locked test rows."""

    if min(validation_exams, calibration_exams, seed_candidates) < 1:
        raise ValueError("split sizes and seed_candidates must be positive")
    layout = discover_mrnet_layout(root)
    train = load_mrnet_labels(layout, "train")
    official_valid = load_mrnet_labels(layout, "valid")
    train_count = len(train) - validation_exams - calibration_exams
    if train_count < 1:
        raise ValueError("requested validation/calibration cohorts exhaust training exams")

    labels = train[list(MRNET_TARGETS)].to_numpy(dtype=np.float64)
    overall_prevalence = labels.mean(axis=0)
    combination_codes = labels.dot(1 << np.arange(len(MRNET_TARGETS))).astype(int)
    overall_combinations = np.bincount(combination_codes, minlength=8) / len(train)
    sizes = (train_count, validation_exams, calibration_exams)

    best_seed: int | None = None
    best_score = float("inf")
    best_assignments: np.ndarray | None = None
    for seed in range(seed_candidates):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(train))
        assignments = np.empty(len(train), dtype=object)
        boundaries = np.cumsum((0,) + sizes)
        for split, start, stop in zip(
            ("train", "validation", "calibration"), boundaries[:-1], boundaries[1:]
        ):
            assignments[order[start:stop]] = split
        score = 0.0
        valid = True
        for split in ("train", "validation", "calibration"):
            selected = assignments == split
            split_labels = labels[selected]
            positives = split_labels.sum(axis=0)
            negatives = len(split_labels) - positives
            if np.any(positives == 0) or np.any(negatives == 0):
                valid = False
                break
            prevalence = split_labels.mean(axis=0)
            score += float(
                np.sum(np.abs(prevalence - overall_prevalence) / np.maximum(overall_prevalence, 0.05))
            )
            combo = np.bincount(combination_codes[selected], minlength=8) / selected.sum()
            score += 0.25 * float(np.sum(np.abs(combo - overall_combinations)))
        if valid and score < best_score:
            best_seed, best_score, best_assignments = seed, score, assignments.copy()
    if best_assignments is None or best_seed is None:
        raise MRNetIntegrityError("No valid multi-label split assignment was found")

    train = train.copy()
    train["split"] = best_assignments
    official_valid = official_valid.copy()
    official_valid["split"] = "test"
    manifest = pd.concat([train, official_valid], ignore_index=True)
    manifest["source_partition"] = manifest["source_partition"].astype(str)
    for plane in MRNET_PLANES:
        manifest[f"{plane}_path"] = [
            str(layout.plane_directories[source][plane] / f"{exam_id}.npy")
            for source, exam_id in zip(manifest["source_partition"], manifest["exam_id"])
        ]

    split_summary: dict[str, object] = {}
    for split in ("train", "validation", "calibration", "test"):
        selected = manifest[manifest["split"] == split]
        split_summary[split] = {
            "exams": len(selected),
            "positive_exams": {
                target: int(selected[target].sum()) for target in MRNET_TARGETS
            },
            "prevalence": {
                target: float(selected[target].mean()) for target in MRNET_TARGETS
            },
            "source_partition": sorted(selected["source_partition"].unique().tolist()),
        }
    summary = {
        "dataset": "Stanford MRNet",
        "selected_seed": best_seed,
        "balance_score": best_score,
        "seed_candidates": seed_candidates,
        "patient_identifiers_available": False,
        "split_unit": "exam_id",
        "official_valid_preserved_as_locked_test": True,
        "splits": split_summary,
        "checks": {
            "rows": len(manifest),
            "unique_exams": int(manifest["exam_id"].nunique()),
            "duplicate_exam_ids": int(manifest["exam_id"].duplicated().sum()),
            "unassigned_rows": int(manifest["split"].isna().sum()),
            "all_plane_paths_exist": bool(
                all(Path(path).is_file() for plane in MRNET_PLANES for path in manifest[f"{plane}_path"])
            ),
        },
    }
    return manifest.sort_values(["split", "exam_id"]).reset_index(drop=True), summary

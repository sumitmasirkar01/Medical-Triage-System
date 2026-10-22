"""Scan RSNA DICOM headers and build a complete patient-disjoint study cohort."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import pandas as pd
import pydicom

from stat_triage.datasets.rsna_study import build_phase2_study_cohort, load_rsna_label_matrix
from stat_triage.evaluation import sha256


def read_header(path: str) -> dict[str, object]:
    source = Path(path)
    ds = pydicom.dcmread(source, stop_before_pixels=True, specific_tags=[
        "PatientID", "StudyInstanceUID", "SeriesInstanceUID", "ImagePositionPatient",
        "InstanceNumber", "SOPInstanceUID",
    ])
    position = getattr(ds, "ImagePositionPatient", None)
    return {"image_id": source.stem, "patient_id": str(ds.PatientID),
            "study_uid": str(ds.StudyInstanceUID), "series_uid": str(ds.SeriesInstanceUID),
            "z_position": float(position[2]) if position is not None else float(ds.InstanceNumber),
            "dicom_path": str(source)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--phase1-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patients-per-split", type=int, default=240)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    paths = sorted(str(path) for path in args.images.glob("*.dcm"))
    if not paths:
        raise FileNotFoundError(f"No DICOM files under {args.images}")
    print(f"Scanning {len(paths):,} DICOM headers with {args.workers} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        indexed = []
        for number, header in enumerate(pool.map(read_header, paths, chunksize=128), 1):
            indexed.append(header)
            if number % 10000 == 0 or number == len(paths):
                print(f"Headers: {number:,}/{len(paths):,}", flush=True)
        headers = pd.DataFrame(indexed)
    labels = load_rsna_label_matrix(args.labels)
    phase1 = pd.read_csv(args.phase1_manifest)
    cohort, summary = build_phase2_study_cohort(
        headers, labels, patients_per_split=args.patients_per_split, seed=args.seed,
        excluded_patients=set(phase1["patient_id"].astype(str)),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "rsna_phase2_study_manifest.csv"
    cohort.to_csv(manifest, index=False)
    summary["manifest"] = {"path": str(manifest), "bytes": manifest.stat().st_size,
                           "sha256": sha256(manifest)}
    (args.output_dir / "rsna_phase2_cohort_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

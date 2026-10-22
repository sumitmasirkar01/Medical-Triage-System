"""Validate the saved RSNA Phase 2 study cohort before slice inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stat_triage.datasets.rsna_study import validate_phase2_study_cohort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--phase1-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-median-slices", type=int, default=10)
    parser.add_argument("--skip-file-check", action="store_true")
    args = parser.parse_args()
    report = validate_phase2_study_cohort(
        args.manifest, args.summary, phase1_manifest_path=args.phase1_manifest,
        check_dicom_files=not args.skip_file_check,
        minimum_median_slices_per_study=args.minimum_median_slices,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

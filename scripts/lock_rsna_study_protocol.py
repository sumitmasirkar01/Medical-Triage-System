"""Lock RSNA study pooling and threshold without loading the test partition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stat_triage.ct_study import POOLING_METHODS, lock_study_pooling_protocol
from stat_triage.evaluation import sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--target-sensitivity", type=float, default=0.9)
    parser.add_argument("--minimum-median-slices", type=int, default=10)
    args = parser.parse_args()
    report, calibration = lock_study_pooling_protocol(
        pd.read_csv(args.predictions), pd.read_csv(args.manifest),
        methods=POOLING_METHODS, top_k=args.top_k,
        target_sensitivity=args.target_sensitivity,
        minimum_median_slices=args.minimum_median_slices,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = args.output_dir / "study_calibration_predictions.csv"
    calibration.to_csv(calibration_path, index=False)
    report["provenance"] = {
        "slice_predictions_sha256": sha256(args.predictions),
        "development_manifest_sha256": sha256(args.manifest),
        "calibration_predictions_sha256": sha256(calibration_path),
    }
    report_path = args.output_dir / "locked_study_protocol.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()

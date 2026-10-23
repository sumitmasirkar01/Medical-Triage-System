"""Run calibration-defined CT study-size robustness analysis locally."""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import zipfile

import pandas as pd

from stat_triage.ct_robustness import (
    evaluate_locked_slice_count_strata,
    lock_slice_count_strata,
)


ROOT = Path(__file__).resolve().parents[1]


def read_member(path: Path, name: str) -> bytes:
    with zipfile.ZipFile(path) as archive:
        return archive.read(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir", type=Path, default=ROOT / "artifacts" / "rsna_phase2"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts" / "rsna_phase2" / "robustness_analysis",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    locked_bundle = args.artifact_dir / "rsna_phase2_locked_protocol_bundle.zip"
    final_bundle = args.artifact_dir / "rsna_phase2_final_evidence_bundle.zip"
    calibration_bytes = read_member(locked_bundle, "study_calibration_predictions.csv")
    protocol_bytes = read_member(locked_bundle, "locked_study_protocol.json")
    calibration = pd.read_csv(BytesIO(calibration_bytes))
    protocol = json.loads(protocol_bytes)
    strata = lock_slice_count_strata(calibration)
    strata["provenance"] = {
        "calibration_predictions_sha256": hashlib.sha256(calibration_bytes).hexdigest(),
        "study_protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
    }
    # Load test studies only after the stratum definitions are fixed.
    test_bytes = read_member(final_bundle, "study_test_predictions.csv")
    report = evaluate_locked_slice_count_strata(
        pd.read_csv(BytesIO(test_bytes)), strata,
        float(protocol["operating_point"]["threshold"]),
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    report["provenance"] = {
        "study_test_predictions_sha256": hashlib.sha256(test_bytes).hexdigest()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "locked_slice_count_strata.json").write_text(
        json.dumps(strata, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "slice_count_robustness.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

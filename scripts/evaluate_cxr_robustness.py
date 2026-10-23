"""Run calibration-defined NIH ChestX-ray14 subgroup robustness analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from stat_triage.cxr_robustness import (
    evaluate_locked_cxr_subgroups,
    lock_cxr_subgroups,
)


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts" / "chest_xray" / "nih_cxr14_locked_test_bundle"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts" / "chest_xray" / "robustness_analysis",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()

    manifest_path = args.bundle / "nih_cxr14_manifest_split.csv"
    calibration_path = (
        args.bundle / "nih_densenet_calibration" / "locked_calibration.json"
    )
    predictions_path = args.bundle / "nih_densenet_test" / "test_predictions.csv"
    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
    calibration_manifest = manifest[manifest["split"] == "calibration"].copy()
    subgroup_lock = lock_cxr_subgroups(calibration_manifest)
    subgroup_lock["provenance"] = {
        "manifest_sha256": sha256(manifest_path),
        "locked_calibration_sha256": sha256(calibration_path),
    }

    # Read test predictions only after every subgroup definition is frozen.
    locked_calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    test_manifest = manifest[manifest["split"] == "test"].copy()
    test_predictions = pd.read_csv(predictions_path, dtype={"patient_id": str})
    report = evaluate_locked_cxr_subgroups(
        test_manifest,
        test_predictions,
        subgroup_lock,
        locked_calibration["targets"],
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    report["provenance"] = {
        "manifest_sha256": sha256(manifest_path),
        "locked_calibration_sha256": sha256(calibration_path),
        "test_predictions_sha256": sha256(predictions_path),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "locked_subgroups.json").write_text(
        json.dumps(subgroup_lock, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "subgroup_robustness.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

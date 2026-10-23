"""Run post-test exploratory MRNet operating-point robustness analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from stat_triage.mrnet_robustness import (
    evaluate_threshold_robustness,
    lock_threshold_perturbations,
)


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts" / "mrnet_baseline"


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
        default=ROOT / "artifacts" / "mrnet_robustness",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()

    calibration_path = args.bundle / "mrnet_calibration" / "locked_calibration.json"
    predictions_path = args.bundle / "mrnet_test" / "test_predictions.csv"
    locked_calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    perturbation_lock = lock_threshold_perturbations(locked_calibration)
    perturbation_lock["provenance"] = {
        "locked_calibration_sha256": sha256(calibration_path)
    }

    # Test predictions are loaded only after the perturbation grid is frozen.
    predictions = pd.read_csv(predictions_path, dtype={"exam_id": str})
    report = evaluate_threshold_robustness(
        predictions,
        perturbation_lock,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    report["provenance"] = {
        "locked_calibration_sha256": sha256(calibration_path),
        "test_predictions_sha256": sha256(predictions_path),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "locked_threshold_perturbations.json").write_text(
        json.dumps(perturbation_lock, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "threshold_robustness.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

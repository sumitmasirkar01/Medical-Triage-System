"""Evaluate the locked RSNA study protocol once on test-only predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stat_triage.ct_study import evaluate_locked_study_protocol
from stat_triage.evaluation import sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--locked-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-median-slices", type=int, default=10)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    locked = json.loads(args.locked_protocol.read_text(encoding="utf-8"))
    report, studies = evaluate_locked_study_protocol(
        pd.read_csv(args.predictions), pd.read_csv(args.manifest), locked,
        minimum_median_slices=args.minimum_median_slices,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "study_test_predictions.csv"
    studies.to_csv(predictions_path, index=False)
    report["provenance"] = {
        "slice_predictions_sha256": sha256(args.predictions),
        "test_manifest_sha256": sha256(args.manifest),
        "locked_protocol_sha256": sha256(args.locked_protocol),
        "study_test_predictions_sha256": sha256(predictions_path),
    }
    report_path = args.output_dir / "study_test_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()

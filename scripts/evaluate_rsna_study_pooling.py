"""Run the locked study-level CT pooling protocol on full-study predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stat_triage.ct_study import POOLING_METHODS, evaluate_study_pooling_protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--minimum-median-slices", type=int, default=10)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    report, test = evaluate_study_pooling_protocol(
        pd.read_csv(args.predictions), pd.read_csv(args.manifest), methods=POOLING_METHODS,
        top_k=args.top_k, minimum_median_slices=args.minimum_median_slices,
        bootstrap_replicates=args.bootstrap_replicates, bootstrap_seed=args.bootstrap_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "study_pooling_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    test.to_csv(args.output_dir / "study_test_predictions.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

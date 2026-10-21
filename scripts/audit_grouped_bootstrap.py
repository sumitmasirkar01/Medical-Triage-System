"""Recompute RSNA patient-grouped intervals as a versioned audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stat_triage.bootstrap import grouped_bootstrap_confidence_intervals
from stat_triage.evaluation import sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "artifacts" / "rsna_baseline" / "rsna_baseline_bundle"
DEFAULT_OUTPUT = ROOT / "artifacts" / "rsna_baseline" / "grouped_bootstrap_audit.json"


def build_audit(bundle: Path, *, replicates: int = 2000, seed: int = 42) -> dict[str, object]:
    prediction_path = bundle / "test_predictions.csv"
    operating_path = bundle / "operating_point.json"
    rows = pd.read_csv(prediction_path)
    operating = json.loads(operating_path.read_text(encoding="utf-8"))
    result = grouped_bootstrap_confidence_intervals(
        rows["target_any"],
        rows["score_any"],
        rows["patient_id"],
        operating["threshold"],
        replicates=replicates,
        seed=seed,
        sampling_unit="patient_id",
    )
    return {
        "status": "independent_audit_does_not_replace_locked_release_intervals",
        "method": "grouped_percentile_bootstrap",
        "prediction_source": {
            "path": str(prediction_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(prediction_path),
        },
        "threshold_source": {
            "path": str(operating_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(operating_path),
            "threshold": operating["threshold"],
        },
        "result": result.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    audit = build_audit(args.bundle.resolve(), replicates=args.replicates, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()


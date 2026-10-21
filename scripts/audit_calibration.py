"""Fit per-target RSNA temperature scalers without touching locked test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stat_triage.calibration import fit_temperature_scaling
from stat_triage.evaluation import sha256
from stat_triage.rsna import RSNA_LABELS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "artifacts" / "rsna_baseline" / "rsna_baseline_bundle"
DEFAULT_OUTPUT = ROOT / "artifacts" / "rsna_baseline" / "calibration_audit.json"


def build_audit(bundle: Path) -> dict[str, object]:
    source = bundle / "calibration_predictions.csv"
    rows = pd.read_csv(source)
    targets: dict[str, object] = {}
    for label in RSNA_LABELS:
        fitted = fit_temperature_scaling(
            rows[f"target_{label}"], rows[f"logit_{label}"]
        )
        targets[label] = fitted.to_dict()
    return {
        "status": "calibration_only_audit_not_applied_to_locked_release",
        "method": "per_target_scalar_temperature_scaling_with_identity_fallback",
        "selection_objective": "binary_negative_log_likelihood",
        "source": {
            "path": str(source.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(source),
            "rows": len(rows),
        },
        "targets": targets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    audit = build_audit(args.bundle.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()


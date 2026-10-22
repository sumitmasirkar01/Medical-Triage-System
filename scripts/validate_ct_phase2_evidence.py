"""Validate the complete locally archived RSNA Phase 2 evidence chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stat_triage.ct_phase2_evidence import validate_ct_phase2_evidence


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts" / "rsna_phase2")
    parser.add_argument(
        "--registry", type=Path,
        default=ROOT / "artifacts" / "rsna_phase2" / "evidence_registry.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts" / "rsna_phase2" / "validation_report.json",
    )
    args = parser.parse_args()
    report = validate_ct_phase2_evidence(args.artifact_dir, args.registry)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

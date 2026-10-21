"""Create a read-only NIH ChestX-ray14 dataset inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stat_triage.datasets.nih_cxr14 import inventory_nih_cxr14


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("/kaggle/working/nih_cxr14_inventory.json")
    )
    args = parser.parse_args()
    inventory = inventory_nih_cxr14(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": inventory["status"], "cohort": inventory["cohort"],
        "targets": inventory["targets"],
        "views": inventory["metadata_quality"]["view_counts"],
        "supplied_split": inventory["supplied_split"],
        "gate": inventory["gate"], "saved": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()


"""Audit labels and all three MRI planes in an attached MRNet release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stat_triage.datasets.mrnet import audit_mrnet, discover_mrnet_layout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/mrnet_inventory"))
    args = parser.parse_args()
    layout = discover_mrnet_layout(args.root)
    print("Discovered label files:")
    for partition, targets in layout.label_files.items():
        for target, path in targets.items():
            print(f"  {partition}/{target}: {path}")
    print("Discovered plane directories:")
    for partition, planes in layout.plane_directories.items():
        for plane, path in planes.items():
            print(f"  {partition}/{plane}: {path}")

    audit, summary = audit_mrnet(args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "mrnet_exam_plane_audit.csv"
    summary_path = args.output_dir / "mrnet_inventory.json"
    audit.to_csv(audit_path, index=False)
    summary["artifacts"] = {
        "audit": str(audit_path),
        "summary": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

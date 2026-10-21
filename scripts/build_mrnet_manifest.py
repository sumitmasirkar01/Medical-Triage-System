"""Build the frozen exam-disjoint MRNet development and locked-test manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from stat_triage.datasets.mrnet import build_mrnet_manifest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/mrnet_manifest"))
    parser.add_argument("--seed-candidates", type=int, default=1000)
    args = parser.parse_args()
    manifest, summary = build_mrnet_manifest(
        args.root, seed_candidates=args.seed_candidates
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "mrnet_manifest_split.csv"
    summary_path = args.output_dir / "mrnet_split_summary.json"
    manifest.to_csv(manifest_path, index=False)
    summary["manifest"] = {
        "path": str(manifest_path),
        "bytes": manifest_path.stat().st_size,
        "sha256": sha256(manifest_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {manifest_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()

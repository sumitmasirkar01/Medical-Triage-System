"""Build the locked NIH patient-disjoint development manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from stat_triage.datasets.nih_cxr14 import build_nih_cxr14_manifest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("/kaggle/working/nih_cxr14_manifest_split.csv"),
    )
    parser.add_argument(
        "--summary", type=Path,
        default=Path("/kaggle/working/nih_cxr14_split_summary.json"),
    )
    parser.add_argument("--candidate-seeds", type=int, default=256)
    args = parser.parse_args()
    manifest, summary = build_nih_cxr14_manifest(
        args.root, candidate_seeds=args.candidate_seeds
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest, index=False)
    summary["manifest"] = {
        "path": str(args.manifest.resolve()),
        "sha256": sha256(args.manifest),
        "bytes": args.manifest.stat().st_size,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected_seed": summary["selected_seed"],
        "balance_score": summary["balance_score"],
        "splits": summary["splits"],
        "checks": summary["checks"],
        "manifest": summary["manifest"],
        "summary": str(args.summary.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()

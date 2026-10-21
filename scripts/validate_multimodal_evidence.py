"""Validate all locked model bundles through the unified artifact registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stat_triage.artifacts import (
    DEFAULT_MULTIMODAL_REGISTRY,
    PROJECT_ROOT,
    validate_multimodal_evidence_registry,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_MULTIMODAL_REGISTRY)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--hashes-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_multimodal_evidence_registry(
        args.registry,
        project_root=args.project_root,
        recompute_metrics=not args.hashes_only,
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print("Saved:", args.output.resolve())
    print(rendered, end="")


if __name__ == "__main__":
    main()

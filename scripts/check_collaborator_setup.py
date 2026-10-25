"""Verify that a fresh collaborator checkout is complete and runnable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

import torch

from stat_triage.artifacts import (
    DEFAULT_MULTIMODAL_REGISTRY,
    PROJECT_ROOT,
    validate_multimodal_evidence_registry,
)


def build_report(*, project_root: Path = PROJECT_ROOT) -> dict[str, object]:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("STAT-Triage requires Python 3.12")

    registry = project_root / DEFAULT_MULTIMODAL_REGISTRY.relative_to(PROJECT_ROOT)
    evidence = validate_multimodal_evidence_registry(
        registry,
        project_root=project_root,
        recompute_metrics=False,
    )
    return {
        "status": "passed",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_for_first_run": "cuda" if torch.cuda.is_available() else "cpu",
        "evidence_registry": registry.relative_to(project_root).as_posix(),
        "evidence_status": evidence["status"],
        "modalities_verified": sorted(evidence["modalities"]),
        "verification_scope": "imports_and_locked_artifact_hashes",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = build_report(project_root=args.project_root.resolve())
        exit_code = 0
    except Exception:
        report = {
            "status": "failed",
            "reason": "collaborator_environment_validation_failed",
        }
        exit_code = 2
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

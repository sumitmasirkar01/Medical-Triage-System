"""Atomically refresh and validate deterministic tracked release metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from scripts.generate_release_manifest import (
    ROOT,
    build_manifest,
    sha256,
    write_json_atomic,
)
from scripts.validate_release import build_validation_report


DEFAULT_MANIFEST = ROOT / "release" / "release_manifest.json"
DEFAULT_REPORT = ROOT / "release" / "validation_report.json"


def pytest_collected_count() -> int:
    """Ask pytest for its authoritative collection count without running tests."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = f"{result.stdout}\n{result.stderr}"
    match = re.search(r"(\d+) tests? collected", output)
    if result.returncode != 0 or match is None:
        raise RuntimeError("pytest collection failed during release refresh")
    return int(match.group(1))


def refresh_release(
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
    *,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 42,
) -> dict[str, object]:
    """Generate the manifest first, validate it, then write its derived report."""
    manifest = build_manifest()
    collected = pytest_collected_count()
    if manifest["verification"]["expected_tests"] != collected:
        raise RuntimeError(
            "static test inventory does not match pytest collection: "
            f"{manifest['verification']['expected_tests']} != {collected}"
        )
    write_json_atomic(manifest_path, manifest)
    manifest_hash = sha256(manifest_path)

    report = build_validation_report(
        release_manifest=manifest_path,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    if report.get("release_manifest_sha256") != manifest_hash:
        raise RuntimeError("validation report does not identify the generated manifest")
    write_json_atomic(report_path, report)

    regenerated = build_manifest()
    if regenerated != json.loads(manifest_path.read_text(encoding="utf-8")):
        raise RuntimeError("release manifest changed during validation")
    return {
        "status": "passed",
        "manifest": manifest_path.resolve(),
        "manifest_sha256": manifest_hash,
        "validation_report": report_path.resolve(),
        "validation_report_sha256": sha256(report_path),
        "supported_source_files": len(
            manifest["deliverables"]["supported_source_sha256"]
        ),
        "expected_tests": manifest["verification"]["expected_tests"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    result = refresh_release(
        args.manifest,
        args.report,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

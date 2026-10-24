"""Generate one machine-readable report from all supported evidence gates."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from stat_triage.artifacts import (
    DEFAULT_MULTIMODAL_REGISTRY,
    validate_multimodal_evidence_registry,
)
from stat_triage.orchestration import evaluate_rsna_release
from stat_triage.experiments import validate_experiment_registry
from scripts.generate_release_manifest import sha256, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "artifacts" / "rsna_baseline" / "rsna_baseline_bundle"
DEFAULT_RELEASE = ROOT / "release" / "release_manifest.json"
DEFAULT_OUTPUT = ROOT / "release" / "validation_report.json"
DEFAULT_EXPERIMENT_REGISTRY = ROOT / "research" / "experiment_registry.json"


def _portable_multimodal_summary(summary: dict[str, object]) -> dict[str, object]:
    """Remove machine-specific absolute paths from a tracked validation artifact."""
    portable = copy.deepcopy(summary)
    modalities = portable.get("modalities")
    if not isinstance(modalities, dict):
        return portable
    for values in modalities.values():
        if not isinstance(values, dict) or "bundle_path" not in values:
            continue
        path = Path(str(values["bundle_path"])).resolve()
        try:
            values["bundle_path"] = path.relative_to(ROOT).as_posix()
        except ValueError as exc:
            raise ValueError(f"Validated bundle escapes project root: {path}") from exc
    return portable


def build_validation_report(
    *,
    bundle: Path = DEFAULT_BUNDLE,
    release_manifest: Path = DEFAULT_RELEASE,
    registry: Path = DEFAULT_MULTIMODAL_REGISTRY,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 42,
) -> dict[str, object]:
    """Validate the complete release and retain the deeper CT statistical audit."""

    head_ct = evaluate_rsna_release(
        bundle,
        release_manifest_path=release_manifest,
        project_root=ROOT,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    multimodal = _portable_multimodal_summary(
        validate_multimodal_evidence_registry(
            registry,
            project_root=ROOT,
            recompute_metrics=True,
        )
    )
    experiments = validate_experiment_registry(DEFAULT_EXPERIMENT_REGISTRY, ROOT)
    head_ct_detail = {
        key: value
        for key, value in head_ct.items()
        if key not in {"schema_version", "status", "generated_at", "release_manifest_validation"}
    }
    return {
        "schema_version": "2.1",
        "status": "passed",
        "release_manifest_sha256": sha256(release_manifest),
        "scope": {
            "system_type": "routed_multi_model_multimodal_imaging_triage",
            "modalities": ["head_ct", "chest_xray", "knee_mri"],
            "modality_count": 3,
            "metrics_recomputed": True,
            "clinical_use_authorized": False,
        },
        "release_manifest_validation": head_ct["release_manifest_validation"],
        "multimodal_validation": multimodal,
        "experiment_registry_validation": {
            "status": "passed",
            "registered_runs": len(experiments["runs"]),
            "run_ids": [run["run_id"] for run in experiments["runs"]],
        },
        "head_ct_deep_validation": head_ct_detail,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--release-manifest", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--registry", type=Path, default=DEFAULT_MULTIMODAL_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    report = build_validation_report(
        bundle=args.bundle,
        release_manifest=args.release_manifest,
        registry=args.registry,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_json_atomic(args.output, report)
    print(f"{report['status']}: {args.output.resolve()}")


if __name__ == "__main__":
    main()


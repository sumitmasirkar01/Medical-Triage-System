"""Generate a deterministic release summary from locked evidence and source files."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
import tomllib


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts" / "rsna_baseline" / "rsna_baseline_bundle"
CXR_BUNDLE = ROOT / "artifacts" / "chest_xray" / "nih_cxr14_locked_test_bundle"
MRNET_BUNDLE = ROOT / "artifacts" / "mrnet_baseline"
_TOP_LEVEL_SOURCE = (
    "app.py",
    "generate_pilot_report.py",
    "PROJECT_PLAN.md",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
)
_EVIDENCE_DECLARATIONS = (
    "artifacts/multimodal_registry.json",
    "artifacts/rsna_phase2/evidence_registry.json",
)


def supported_source_paths(root: Path = ROOT) -> tuple[str, ...]:
    """Discover the complete release-controlled source inventory deterministically."""
    candidates = {root / relative for relative in _TOP_LEVEL_SOURCE}
    candidates.update((root / "stat_triage").rglob("*.py"))
    candidates.update((root / "scripts").glob("*.py"))
    candidates.update((root / "tests").glob("test_*.py"))
    candidates.update((root / "docs").glob("*.md"))
    candidates.update((root / "research").glob("*.json"))
    candidates.update(root / relative for relative in _EVIDENCE_DECLARATIONS)
    missing = [path for path in candidates if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing release-controlled source: {missing[0]}")
    return tuple(
        sorted(path.relative_to(root).as_posix() for path in candidates)
    )


def _parametrize_cases(decorator: ast.expr, path: Path) -> int:
    if not isinstance(decorator, ast.Call):
        return 1
    function = decorator.func
    if not isinstance(function, ast.Attribute) or function.attr != "parametrize":
        return 1
    if len(decorator.args) < 2 or not isinstance(
        decorator.args[1], (ast.List, ast.Tuple, ast.Set)
    ):
        raise ValueError(f"Cannot statically count pytest parametrization in {path}")
    return len(decorator.args[1].elts)


def collected_test_count(root: Path = ROOT) -> int:
    """Count this suite's statically declared pytest cases without invoking pytest."""
    total = 0
    for relative in supported_source_paths(root):
        if not relative.startswith("tests/") or not relative.endswith(".py"):
            continue
        path = root / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module_tests = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        }
        nodes = list(module_tests.values())
        for item in tree.body:
            if isinstance(item, ast.ClassDef):
                class_tests = {
                    node.name: node
                    for node in item.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name.startswith("test_")
                }
                nodes.extend(class_tests.values())
        for node in nodes:
            cases = 1
            for decorator in node.decorator_list:
                cases *= _parametrize_cases(decorator, path)
            total += cases
    return total


def write_json_atomic(path: Path, value: object) -> None:
    """Replace a JSON artifact only after its complete deterministic serialization."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def declaration(path: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def build_manifest() -> dict:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    config = load_json(BUNDLE / "run_config.json")
    operating = load_json(BUNDLE / "operating_point.json")
    metrics = load_json(BUNDLE / "test_metrics.json")
    evidence_manifest = load_json(BUNDLE / "bundle_manifest.json")
    registry_path = ROOT / "artifacts" / "multimodal_registry.json"
    validation_path = ROOT / "artifacts" / "multimodal_validation.json"
    smoke_path = ROOT / "artifacts" / "multimodal_smoke_report.json"
    results_dir = ROOT / "artifacts" / "consolidated_results"
    registry = load_json(registry_path)
    cxr_locked = load_json(
        CXR_BUNDLE / "nih_densenet_calibration" / "locked_calibration.json"
    )
    cxr_metrics = load_json(CXR_BUNDLE / "nih_densenet_test" / "test_metrics.json")
    cxr_robustness_dir = ROOT / "artifacts" / "chest_xray" / "robustness_analysis"
    cxr_subgroups = load_json(cxr_robustness_dir / "locked_subgroups.json")
    cxr_robustness = load_json(cxr_robustness_dir / "subgroup_robustness.json")
    mrnet_config = load_json(MRNET_BUNDLE / "mrnet_baseline" / "run_config.json")
    mrnet_locked = load_json(MRNET_BUNDLE / "mrnet_calibration" / "locked_calibration.json")
    mrnet_metrics = load_json(MRNET_BUNDLE / "mrnet_test" / "test_metrics.json")
    mrnet_robustness_dir = ROOT / "artifacts" / "mrnet_robustness"
    mrnet_perturbations = load_json(
        mrnet_robustness_dir / "locked_threshold_perturbations.json"
    )
    mrnet_robustness = load_json(
        mrnet_robustness_dir / "threshold_robustness.json"
    )
    ct_phase2_dir = ROOT / "artifacts" / "rsna_phase2"
    ct_phase2 = load_json(ct_phase2_dir / "validation_report.json")
    ct_phase2_registry = load_json(ct_phase2_dir / "evidence_registry.json")
    ct_selective_dir = ct_phase2_dir / "selective_analysis"
    ct_selective_policy = load_json(ct_selective_dir / "locked_selective_policy.json")
    ct_selective = load_json(ct_selective_dir / "selective_test_metrics.json")
    ct_robustness_dir = ct_phase2_dir / "robustness_analysis"
    ct_strata = load_json(ct_robustness_dir / "locked_slice_count_strata.json")
    ct_robustness = load_json(ct_robustness_dir / "slice_count_robustness.json")
    report = ROOT / "docs" / "STAT_Triage_RSNA_Pilot_Report.docx"

    return {
        "schema_version": "2.0",
        "release": {
            "name": project["name"],
            "version": project["version"],
            "date": "2026-10-21",
            "status": "research_demo",
            "system_type": "routed_multi_model_multimodal_imaging_triage",
            "clinical_use_authorized": False,
        },
        "modalities": {
            "head_ct": {
                "dataset": "RSNA Intracranial Hemorrhage Detection 2019",
                "endpoint": metrics["endpoint"],
                "evaluation_level": "slice",
                "architecture": config["model"],
                "labels": config["labels"],
                "selected_epoch": operating["selected_model_epoch"],
                "checkpoint_sha256": operating["model_checkpoint_sha256"],
                "probability_calibration": "none_locked",
                "locked_thresholds": {"any": operating["threshold"]},
                "test_cohort": {
                    "images": metrics["test_images"],
                    "patients": metrics["test_patients"],
                },
                "test_results": {
                    "any": {
                        "point_estimates": metrics["primary_metrics"],
                        "patient_bootstrap_95_ci": metrics["patient_bootstrap_95_ci"],
                    }
                },
                "phase_2_study_level": {
                    "status": ct_phase2["status"],
                    "endpoint": ct_phase2["endpoint"],
                    "evaluation_level": "study",
                    "patients": ct_phase2["patients"],
                    "positive_studies": ct_phase2["positive_studies"],
                    "selected_pooling": ct_phase2["selected_method"],
                    "locked_threshold": ct_phase2["threshold"],
                    "test_results": ct_phase2["test_metrics"],
                    "bootstrap_replicates": ct_phase2["bootstrap_replicates"],
                    "evidence_registry": declaration(
                        ct_phase2_dir / "evidence_registry.json"
                    ),
                    "validation_report": declaration(
                        ct_phase2_dir / "validation_report.json"
                    ),
                    "bundles": {
                        role: declaration(ct_phase2_dir / values["file"])
                        for role, values in ct_phase2_registry["bundles"].items()
                    },
                    "test_evaluated_once": True,
                    "pooling_reselected": False,
                    "threshold_reselected": False,
                    "selective_prediction": {
                        "analysis_status": ct_selective["analysis_status"],
                        "prespecified_primary_policy": ct_selective[
                            "prespecified_primary_policy"
                        ],
                        "exploratory_policy": {
                            "minimum_coverage": ct_selective_policy["minimum_coverage"],
                            "maximum_automatic_false_negative_rate": ct_selective_policy[
                                "maximum_automatic_false_negative_rate"
                            ],
                            "margin_fraction": ct_selective["margin_fraction"],
                            "negative_boundary": ct_selective["negative_boundary"],
                            "positive_boundary": ct_selective["positive_boundary"],
                        },
                        "test_metrics": ct_selective["test_metrics"],
                        "patient_grouped_bootstrap": ct_selective[
                            "patient_grouped_bootstrap"
                        ],
                        "policy": declaration(
                            ct_selective_dir / "locked_selective_policy.json"
                        ),
                        "report": declaration(
                            ct_selective_dir / "selective_test_metrics.json"
                        ),
                    },
                    "slice_count_robustness": {
                        "analysis_status": ct_robustness["analysis_status"],
                        "stratification_method": ct_strata["method"],
                        "lower_cutpoint": ct_strata["lower_cutpoint"],
                        "upper_cutpoint": ct_strata["upper_cutpoint"],
                        "groups": ct_robustness["groups"],
                        "multiplicity_adjustment": ct_robustness[
                            "multiplicity_adjustment"
                        ],
                        "interpretation": ct_robustness["interpretation"],
                        "strata_lock": declaration(
                            ct_robustness_dir / "locked_slice_count_strata.json"
                        ),
                        "report": declaration(
                            ct_robustness_dir / "slice_count_robustness.json"
                        ),
                    },
                },
                "limitations": metrics["limitations"],
            },
            "chest_xray": {
                "dataset": "NIH ChestX-ray14",
                "endpoint": "nih_cxr14_four_target_triage",
                "evaluation_level": "image",
                "architecture": "torchvision_densenet121",
                "labels": list(cxr_locked["targets"]),
                "selected_epoch": cxr_locked["checkpoint"]["epoch"],
                "checkpoint_sha256": cxr_metrics["checkpoint_sha256"],
                "probability_calibration": "per_target_affine_logistic_scaling",
                "locked_thresholds": {
                    target: values["operating_point"]["threshold"]
                    for target, values in cxr_locked["targets"].items()
                },
                "test_cohort": {
                    "images": cxr_metrics["test_images"],
                    "patients": cxr_metrics["test_patients"],
                },
                "test_results": cxr_metrics["targets"],
                "subgroup_robustness": {
                    "analysis_status": cxr_robustness["analysis_status"],
                    "definitions": cxr_subgroups["definitions"],
                    "dimensions": cxr_robustness["dimensions"],
                    "bootstrap_replicates": cxr_robustness["bootstrap_replicates"],
                    "multiplicity_adjustment": cxr_robustness[
                        "multiplicity_adjustment"
                    ],
                    "interpretation": cxr_robustness["interpretation"],
                    "subgroup_lock": declaration(
                        cxr_robustness_dir / "locked_subgroups.json"
                    ),
                    "report": declaration(
                        cxr_robustness_dir / "subgroup_robustness.json"
                    ),
                },
                "limitations": [
                    "Labels are report-derived and may contain label noise.",
                    "Only four selected findings are modeled.",
                    "High-sensitivity operating points have low specificity.",
                    "External and prospective validation have not been performed.",
                ],
            },
            "knee_mri": {
                "dataset": "Stanford MRNet",
                "endpoint": "mrnet_three_target_exam_triage",
                "evaluation_level": "exam",
                "architecture": mrnet_config["model"],
                "labels": mrnet_config["targets"],
                "planes": mrnet_config["planes"],
                "selected_epoch": mrnet_locked["checkpoint"]["epoch"],
                "checkpoint_sha256": mrnet_metrics["checkpoint_sha256"],
                "probability_calibration": "per_target_affine_logistic_scaling",
                "locked_thresholds": {
                    target: values["operating_point"]["threshold"]
                    for target, values in mrnet_locked["targets"].items()
                },
                "test_cohort": {
                    "exams": mrnet_metrics["test_exams"],
                    "patient_identifiers_available": False,
                },
                "test_results": mrnet_metrics["targets"],
                "threshold_robustness": {
                    "analysis_status": mrnet_robustness["analysis_status"],
                    "method": mrnet_perturbations["method"],
                    "multipliers": mrnet_perturbations["multipliers"],
                    "targets": mrnet_robustness["targets"],
                    "bootstrap_replicates": mrnet_robustness[
                        "bootstrap_replicates"
                    ],
                    "multiplicity_adjustment": mrnet_robustness[
                        "multiplicity_adjustment"
                    ],
                    "interpretation": mrnet_robustness["interpretation"],
                    "perturbation_lock": declaration(
                        mrnet_robustness_dir / "locked_threshold_perturbations.json"
                    ),
                    "report": declaration(
                        mrnet_robustness_dir / "threshold_robustness.json"
                    ),
                },
                "limitations": [
                    "The official validation partition is locked as the test set.",
                    "Patient identifiers are unavailable, so patient independence cannot be verified.",
                    "Bootstrap sampling uses exam identifiers.",
                    "High-sensitivity operating points have low specificity.",
                    "External and prospective validation have not been performed.",
                ],
            },
        },
        "multimodal_evidence": {
            "registry": declaration(registry_path),
            "registry_status": registry["status"],
            "bundle_manifests": {
                "head_ct": declaration(BUNDLE / "bundle_manifest.json"),
                "chest_xray": declaration(CXR_BUNDLE / "bundle_manifest.json"),
                "knee_mri": declaration(MRNET_BUNDLE / "mrnet_bundle_manifest.json"),
            },
            "validation_report": declaration(validation_path),
            "smoke_report": declaration(smoke_path),
            "consolidated_results": {
                "csv": declaration(results_dir / "locked_multimodal_results.csv"),
                "json": declaration(results_dir / "locked_multimodal_results.json"),
                "markdown": declaration(results_dir / "locked_multimodal_results.md"),
            },
            "smoke_uses_synthetic_deidentified_inputs": True,
        },
        "system_limitations": [
            "This is a research demonstration and is not authorized for clinical use.",
            "The system routes to separate modality-specific models; it does not perform cross-modality neural fusion.",
            "Outputs and thresholds are not comparable across modalities.",
            "The worklist urgency policy has not been prospectively clinically validated.",
            "Synthetic smoke fixtures validate software behavior, not clinical performance.",
        ],
        # Compatibility fields retained for the established CT release validator.
        "supported_scope": {
            "modality": "Head CT",
            "endpoint": metrics["endpoint"],
            "evaluation_level": "slice",
            "labels": config["labels"],
            "model": config["model"],
            "selected_epoch": operating["selected_model_epoch"],
            "checkpoint_sha256": operating["model_checkpoint_sha256"],
            "locked_threshold": operating["threshold"],
        },
        "locked_test": {
            "images": metrics["test_images"],
            "patients": metrics["test_patients"],
            "primary_metrics": metrics["primary_metrics"],
            "patient_bootstrap_95_ci": metrics["patient_bootstrap_95_ci"],
            "limitations": metrics["limitations"],
        },
        "evidence_bundle": evidence_manifest,
        "deliverables": {
            "report": {
                "path": str(report.relative_to(ROOT)).replace("\\", "/"),
                "bytes": report.stat().st_size,
                "sha256": sha256(report),
            },
            "supported_source_sha256": {
                name: sha256(ROOT / name) for name in supported_source_paths()
            },
        },
        "verification": {
            "python": "3.12.7",
            "expected_tests": collected_test_count(),
            "commands": [
                ".\\.venv\\Scripts\\python.exe -m pip check",
                ".\\.venv\\Scripts\\python.exe -m scripts.validate_multimodal_evidence",
                ".\\.venv\\Scripts\\python.exe -m scripts.multimodal_smoke --device cpu",
                ".\\.venv\\Scripts\\python.exe -m pytest -q",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "release" / "release_manifest.json"
    )
    args = parser.parse_args()
    write_json_atomic(args.output, build_manifest())
    print(args.output.resolve())


if __name__ == "__main__":
    main()

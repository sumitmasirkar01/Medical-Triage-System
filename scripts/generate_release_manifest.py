"""Generate a deterministic release summary from locked evidence and source files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts" / "rsna_baseline" / "rsna_baseline_bundle"
CXR_BUNDLE = ROOT / "artifacts" / "chest_xray" / "nih_cxr14_locked_test_bundle"
MRNET_BUNDLE = ROOT / "artifacts" / "mrnet_baseline"
SUPPORTED_SOURCE = (
    "app.py",
    "generate_pilot_report.py",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "research/protocol.json",
    "research/protocol_amendment_phase2.json",
    "research/experiment_registry.json",
    "stat_triage/__init__.py",
    "stat_triage/contracts.py",
    "stat_triage/calibration.py",
    "stat_triage/bootstrap.py",
    "stat_triage/evaluation.py",
    "stat_triage/ingestion.py",
    "stat_triage/rsna.py",
    "stat_triage/ct_study.py",
    "stat_triage/ct_phase2_evidence.py",
    "stat_triage/ct_selective.py",
    "stat_triage/ct_robustness.py",
    "stat_triage/thresholds.py",
    "stat_triage/selective.py",
    "stat_triage/orchestration.py",
    "stat_triage/losses.py",
    "stat_triage/datasets/__init__.py",
    "stat_triage/datasets/rsna_study.py",
    "stat_triage/datasets/nih_cxr14.py",
    "scripts/inventory_nih_cxr14.py",
    "scripts/build_nih_cxr14_manifest.py",
    "stat_triage/datasets/nih_preprocessing.py",
    "scripts/audit_nih_preprocessing.py",
    "stat_triage/datasets/nih_training.py",
    "stat_triage/models/__init__.py",
    "stat_triage/models/densenet_multilabel.py",
    "scripts/preflight_nih_densenet.py",
    "stat_triage/cxr_training.py",
    "stat_triage/cxr_calibration.py",
    "stat_triage/cxr_evaluation.py",
    "stat_triage/cxr_robustness.py",
    "stat_triage/cxr.py",
    "stat_triage/cxr_policy.py",
    "scripts/train_nih_densenet.py",
    "scripts/calibrate_nih_densenet.py",
    "scripts/evaluate_nih_densenet.py",
    "scripts/evaluate_cxr_robustness.py",
    "scripts/cache_nih_images.py",
    "stat_triage/datasets/mrnet.py",
    "scripts/inventory_mrnet.py",
    "scripts/build_mrnet_manifest.py",
    "stat_triage/datasets/mrnet_preprocessing.py",
    "scripts/audit_mrnet_preprocessing.py",
    "stat_triage/datasets/mrnet_training.py",
    "stat_triage/models/mrnet_multiplane.py",
    "scripts/preflight_mrnet_model.py",
    "scripts/train_mrnet_multiplane.py",
    "stat_triage/mrnet_calibration.py",
    "scripts/calibrate_mrnet_multiplane.py",
    "stat_triage/mrnet_evaluation.py",
    "stat_triage/mrnet_robustness.py",
    "scripts/evaluate_mrnet_multiplane.py",
    "scripts/evaluate_mrnet_robustness.py",
    "stat_triage/mrnet.py",
    "stat_triage/multimodal.py",
    "stat_triage/multimodal_worklist.py",
    "stat_triage/triage_outcome.py",
    "stat_triage/ct_series.py",
    "stat_triage/triage_service.py",
    "stat_triage/persistent_cases.py",
    "stat_triage/artifacts.py",
    "stat_triage/experiments.py",
    "artifacts/multimodal_registry.json",
    "scripts/validate_multimodal_evidence.py",
    "scripts/validate_release.py",
    "scripts/experiment_registry.py",
    "scripts/rebuild_experiment_registry.py",
    "docs/PROTOCOL_AMENDMENT_PHASE2.md",
    "docs/PHASE3_TRIAGE_CONTRACT.md",
    "docs/PHASE3_CT_SERIES_INGESTION.md",
    "docs/PHASE3_UNIFIED_SERVICE.md",
    "docs/PHASE3_PERSISTENT_ORCHESTRATION.md",
    "scripts/multimodal_smoke.py",
    "stat_triage/results.py",
    "scripts/generate_multimodal_results.py",
    "stat_triage/cli.py",
    "scripts/infer_multimodal.py",
    "stat_triage/worklist.py",
    "scripts/create_demo_dicom.py",
    "scripts/demo_smoke.py",
    "scripts/evaluate_rsna_study_pooling.py",
    "scripts/lock_rsna_study_protocol.py",
    "scripts/evaluate_locked_rsna_studies.py",
    "scripts/build_rsna_study_cohort.py",
    "scripts/validate_rsna_study_cohort.py",
    "scripts/predict_rsna_study_slices.py",
    "scripts/validate_ct_phase2_evidence.py",
    "scripts/evaluate_ct_phase2_selective.py",
    "scripts/evaluate_ct_phase2_robustness.py",
    "artifacts/rsna_phase2/evidence_registry.json",
)


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
                name: sha256(ROOT / name) for name in SUPPORTED_SOURCE
            },
        },
        "verification": {
            "python": "3.12.7",
            "expected_tests": 197,
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(build_manifest(), indent=2) + "\n", encoding="utf-8"
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()

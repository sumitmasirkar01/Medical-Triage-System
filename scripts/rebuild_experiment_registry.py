"""Deterministically reconstruct provenance for completed supported experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from stat_triage.contracts import Modality
from stat_triage.experiments import (
    build_run_record,
    initialize_registry,
    register_run,
    validate_experiment_registry,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "research" / "experiment_registry.json"
PROTOCOL = ROOT / "research" / "protocol.json"
AMENDMENT = ROOT / "research" / "protocol_amendment_phase2.json"


RUNS = (
    {
        "run_id": "head-ct-phase1-locked-evaluation",
        "modality": Modality.HEAD_CT,
        "purpose": "locked slice-level baseline evaluation",
        "dataset_id": "rsna-ich-2019-phase1-pilot",
        "seed": 42,
        "split_manifest": "artifacts/rsna_baseline/rsna_baseline_bundle/rsna_pilot_manifest_split.csv",
        "config": "artifacts/rsna_baseline/rsna_baseline_bundle/run_config.json",
        "artifacts": (
            ("checkpoint", "artifacts/rsna_baseline/rsna_baseline_bundle/best_efficientnet_b0.pt"),
            ("operating_point", "artifacts/rsna_baseline/rsna_baseline_bundle/operating_point.json"),
            ("test_metrics", "artifacts/rsna_baseline/rsna_baseline_bundle/test_metrics.json"),
        ),
        "code_commit": "5ae9adfb8d33ea9cf48bf155bc0e497a4f4c19f4",
        "created_at": "2026-10-21T16:41:33+05:30",
    },
    {
        "run_id": "chest-xray-locked-evaluation",
        "modality": Modality.CHEST_XRAY,
        "purpose": "locked four-target image-level baseline evaluation",
        "dataset_id": "nih-chestxray14",
        "seed": 42,
        "split_manifest": "artifacts/chest_xray/nih_cxr14_locked_test_bundle/nih_cxr14_manifest_split.csv",
        "config": "artifacts/chest_xray/nih_cxr14_locked_test_bundle/nih_densenet_calibration/locked_calibration.json",
        "artifacts": (
            ("checkpoint", "artifacts/chest_xray/nih_cxr14_locked_test_bundle/nih_densenet/best_densenet121.pt"),
            ("test_metrics", "artifacts/chest_xray/nih_cxr14_locked_test_bundle/nih_densenet_test/test_metrics.json"),
        ),
        "code_commit": "5ae9adfb8d33ea9cf48bf155bc0e497a4f4c19f4",
        "created_at": "2026-10-21T16:41:33+05:30",
    },
    {
        "run_id": "knee-mri-locked-evaluation",
        "modality": Modality.KNEE_MRI,
        "purpose": "locked three-target exam-level baseline evaluation",
        "dataset_id": "stanford-mrnet-v1",
        "seed": 42,
        "split_manifest": "artifacts/mrnet_baseline/mrnet_manifest/mrnet_manifest_split.csv",
        "config": "artifacts/mrnet_baseline/mrnet_baseline/run_config.json",
        "artifacts": (
            ("checkpoint", "artifacts/mrnet_baseline/mrnet_baseline/best_mrnet_resnet18.pt"),
            ("locked_calibration", "artifacts/mrnet_baseline/mrnet_calibration/locked_calibration.json"),
            ("test_metrics", "artifacts/mrnet_baseline/mrnet_test/test_metrics.json"),
        ),
        "code_commit": "5ae9adfb8d33ea9cf48bf155bc0e497a4f4c19f4",
        "created_at": "2026-10-21T16:41:33+05:30",
    },
    {
        "run_id": "head-ct-phase2-study-evaluation",
        "modality": Modality.HEAD_CT,
        "purpose": "locked study-level pooling and test evaluation",
        "dataset_id": "rsna-ich-2019-phase2-study-cohort",
        "seed": 42,
        "split_manifest": "artifacts/rsna_phase2/rsna_phase2_cohort_bundle.zip",
        "config": "artifacts/rsna_phase2/rsna_phase2_locked_protocol_bundle.zip",
        "artifacts": (
            ("evidence_registry", "artifacts/rsna_phase2/evidence_registry.json"),
            ("validation_report", "artifacts/rsna_phase2/validation_report.json"),
            ("final_evaluation", "artifacts/rsna_phase2/rsna_phase2_final_evidence_bundle.zip"),
        ),
        "code_commit": "5e8830cc8bacc8d65d6d6e140495e42e9c3b9601",
        "created_at": "2026-10-23T03:36:33+05:30",
    },
    {
        "run_id": "head-ct-selective-post-test-exploratory",
        "modality": Modality.HEAD_CT,
        "purpose": "post-test exploratory selective-prediction analysis",
        "dataset_id": "rsna-ich-2019-phase2-study-cohort",
        "seed": 42,
        "split_manifest": "artifacts/rsna_phase2/rsna_phase2_cohort_bundle.zip",
        "config": "artifacts/rsna_phase2/selective_analysis/locked_selective_policy.json",
        "artifacts": (("report", "artifacts/rsna_phase2/selective_analysis/selective_test_metrics.json"),),
        "code_commit": "f65352fa4922f282041772fd3a2a9209bd329b9c",
        "created_at": "2026-10-23T16:11:30+05:30",
    },
    {
        "run_id": "head-ct-study-size-exploratory",
        "modality": Modality.HEAD_CT,
        "purpose": "exploratory study-size robustness analysis",
        "dataset_id": "rsna-ich-2019-phase2-study-cohort",
        "seed": 42,
        "split_manifest": "artifacts/rsna_phase2/rsna_phase2_cohort_bundle.zip",
        "config": "artifacts/rsna_phase2/robustness_analysis/locked_slice_count_strata.json",
        "artifacts": (("report", "artifacts/rsna_phase2/robustness_analysis/slice_count_robustness.json"),),
        "code_commit": "ac87335c3692971f62c371b25a3216c241c00664",
        "created_at": "2026-10-23T16:16:39+05:30",
    },
    {
        "run_id": "chest-xray-subgroup-exploratory",
        "modality": Modality.CHEST_XRAY,
        "purpose": "exploratory view, recorded-sex, and age robustness analysis",
        "dataset_id": "nih-chestxray14",
        "seed": 42,
        "split_manifest": "artifacts/chest_xray/nih_cxr14_locked_test_bundle/nih_cxr14_manifest_split.csv",
        "config": "artifacts/chest_xray/robustness_analysis/locked_subgroups.json",
        "artifacts": (("report", "artifacts/chest_xray/robustness_analysis/subgroup_robustness.json"),),
        "code_commit": "784e56082db849540b626482b03e0b89d26a013f",
        "created_at": "2026-10-23T16:28:46+05:30",
    },
    {
        "run_id": "knee-mri-threshold-post-test-exploratory",
        "modality": Modality.KNEE_MRI,
        "purpose": "post-test exploratory threshold robustness analysis",
        "dataset_id": "stanford-mrnet-v1",
        "seed": 42,
        "split_manifest": "artifacts/mrnet_baseline/mrnet_manifest/mrnet_manifest_split.csv",
        "config": "artifacts/mrnet_robustness/locked_threshold_perturbations.json",
        "artifacts": (("report", "artifacts/mrnet_robustness/threshold_robustness.json"),),
        "code_commit": "356da4b11eca2d80797067c075396e083c2173f9",
        "created_at": "2026-10-23T16:50:51+05:30",
    },
)


def rebuild(destination: Path = REGISTRY) -> dict[str, object]:
    initialize_registry(destination, PROTOCOL, ROOT, amendments=(AMENDMENT,))
    for specification in RUNS:
        register_run(
            destination,
            build_run_record(
                project_root=ROOT,
                protocol_path=PROTOCOL,
                **specification,
            ),
            ROOT,
        )
    return validate_experiment_registry(destination, ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REGISTRY)
    args = parser.parse_args()
    registry = rebuild(args.output)
    print(f"rebuilt: {args.output.resolve()} ({len(registry['runs'])} runs)")


if __name__ == "__main__":
    main()

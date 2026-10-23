"""The unified evaluator must fail closed and reproduce the locked release."""

from pathlib import Path

import scripts.validate_release as release_validator
from scripts.validate_release import build_validation_report
from stat_triage.orchestration import (
    evaluate_rsna_release,
    validate_project_release_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts" / "rsna_baseline" / "rsna_baseline_bundle"
RELEASE = ROOT / "release" / "release_manifest.json"


def test_release_builder_validates_experiment_registry_once(monkeypatch):
    calls = []

    monkeypatch.setattr(
        release_validator,
        "evaluate_rsna_release",
        lambda *args, **kwargs: {
            "schema_version": "2.0",
            "status": "passed",
            "generated_at": "2026-10-23T00:00:00+00:00",
            "release_manifest_validation": {"status": "passed"},
        },
    )
    monkeypatch.setattr(
        release_validator,
        "validate_multimodal_evidence_registry",
        lambda *args, **kwargs: {"status": "passed"},
    )

    def validate_once(*args, **kwargs):
        calls.append((args, kwargs))
        return {"runs": [{"run_id": "run-1"}]}

    monkeypatch.setattr(release_validator, "validate_experiment_registry", validate_once)
    report = release_validator.build_validation_report()

    assert len(calls) == 1
    assert report["experiment_registry_validation"] == {
        "status": "passed",
        "registered_runs": 1,
        "run_ids": ["run-1"],
    }


def test_release_manifest_sources_and_report_are_intact():
    result = validate_project_release_manifest(RELEASE, ROOT)

    assert result["version"] == "0.1.0"
    assert result["status"] == "research_demo"
    assert result["schema_version"] == "2.0"
    assert set(result["modality_checkpoint_sha256"]) == {
        "head_ct", "chest_xray", "knee_mri"
    }
    assert result["supported_source_files"] >= 10


def test_unified_evaluator_passes_every_rsna_gate():
    report = evaluate_rsna_release(
        BUNDLE,
        release_manifest_path=RELEASE,
        project_root=ROOT,
        bootstrap_replicates=100,
        bootstrap_seed=42,
    )

    assert report["status"] == "passed"
    assert report["cohorts"] == {
        "calibration_rows": 244,
        "calibration_patients": 222,
        "test_rows": 242,
        "test_patients": 222,
        "patient_overlap": 0,
    }
    assert report["locked_test_metrics"]["roc_auc"] == 0.8417158598976782
    assert report["threshold_audit"]["tp"] == 72
    assert report["policy"]["threshold_retuned_on_test"] is False

    unified = build_validation_report(
        bootstrap_replicates=100,
        bootstrap_seed=42,
    )
    assert unified["status"] == "passed"
    assert unified["scope"]["modalities"] == [
        "head_ct",
        "chest_xray",
        "knee_mri",
    ]
    assert unified["multimodal_validation"]["metrics_recomputed"] is True
    assert unified["head_ct_deep_validation"]["policy"][
        "threshold_retuned_on_test"
    ] is False


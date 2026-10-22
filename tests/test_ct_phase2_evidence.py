from pathlib import Path

from stat_triage.ct_phase2_evidence import validate_ct_phase2_evidence


ROOT = Path(__file__).resolve().parents[1]


def test_archived_ct_phase2_evidence_chain_recomputes():
    artifact_dir = ROOT / "artifacts" / "rsna_phase2"
    result = validate_ct_phase2_evidence(
        artifact_dir, artifact_dir / "evidence_registry.json"
    )
    assert result["status"] == "passed"
    assert result["patients"] == 240
    assert result["positive_studies"] == 92
    assert result["selected_method"] == "mean"
    assert result["bootstrap_replicates"] == 2000

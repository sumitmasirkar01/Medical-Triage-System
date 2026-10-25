from pathlib import Path

from scripts.check_collaborator_setup import build_report


ROOT = Path(__file__).resolve().parents[1]


def test_collaborator_setup_verifies_all_locked_modalities():
    report = build_report(project_root=ROOT)
    assert report["status"] == "passed"
    assert report["verification_scope"] == "imports_and_locked_artifact_hashes"
    assert report["modalities_verified"] == [
        "chest_xray",
        "head_ct",
        "knee_mri",
    ]

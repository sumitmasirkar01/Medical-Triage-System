import json
from pathlib import Path

from stat_triage.results import build_consolidated_results, render_results_markdown


ROOT = Path(__file__).resolve().parents[1]


def test_consolidated_results_preserve_locked_values_and_evaluation_units():
    frame, summary = build_consolidated_results(ROOT)
    assert len(frame) == 9
    assert set(frame["evaluation_level"]) == {"slice", "study", "image", "exam"}
    ct = frame[
        (frame["modality"] == "head_ct") & (frame["evaluation_level"] == "slice")
    ].iloc[0]
    ct_study = frame[
        (frame["modality"] == "head_ct") & (frame["evaluation_level"] == "study")
    ].iloc[0]
    acl = frame[(frame["modality"] == "knee_mri") & (frame["target"] == "acl")].iloc[0]
    assert ct["roc_auc"] == 0.8417158598976782
    assert ct_study["roc_auc"] == 0.9078290246768508
    assert ct_study["test_units"] == 240
    assert ct_study["evidence_classification"] == "confirmatory_prespecified_locked"
    assert ct_study["roc_auc_ci_lower"] == 0.8678533484241479
    assert acl["roc_auc"] == 0.8571829405162739
    assert summary["rows"] == 9
    assert "not direct" in summary["comparison_warning"]
    assert "confirmatory_prespecified_locked" in summary["evidence_hierarchy"]
    assert summary["source_artifacts"]["head_ct_study_bundle"]["sha256"] == (
        "1b94d27131496ee67dac07f397c85e9543d791674e118ed90bb7b0d14fd83e29"
    )


def test_results_markdown_labels_the_noncomparability_and_all_targets():
    frame, _ = build_consolidated_results(ROOT)
    rendered = render_results_markdown(frame)
    assert "not direct model comparisons" in rendered
    assert "| head_ct | slice | retrospective_locked_baseline | any |" in rendered
    assert "| head_ct | study | confirmatory_prespecified_locked | any |" in rendered
    assert "| chest_xray | image | retrospective_locked_baseline | pneumothorax |" in rendered
    assert "| knee_mri | exam | retrospective_locked_baseline | meniscus |" in rendered
    encoded = json.dumps(_, allow_nan=False)
    assert "NaN" not in encoded

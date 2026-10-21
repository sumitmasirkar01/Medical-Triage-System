import json
from pathlib import Path

from stat_triage.results import build_consolidated_results, render_results_markdown


ROOT = Path(__file__).resolve().parents[1]


def test_consolidated_results_preserve_locked_values_and_evaluation_units():
    frame, summary = build_consolidated_results(ROOT)
    assert len(frame) == 8
    assert set(frame["evaluation_level"]) == {"slice", "image", "exam"}
    ct = frame[(frame["modality"] == "head_ct") & (frame["target"] == "any")].iloc[0]
    acl = frame[(frame["modality"] == "knee_mri") & (frame["target"] == "acl")].iloc[0]
    assert ct["roc_auc"] == 0.8417158598976782
    assert acl["roc_auc"] == 0.8571829405162739
    assert summary["rows"] == 8
    assert "not direct" in summary["comparison_warning"]


def test_results_markdown_labels_the_noncomparability_and_all_targets():
    frame, _ = build_consolidated_results(ROOT)
    rendered = render_results_markdown(frame)
    assert "not direct model comparisons" in rendered
    assert "| head_ct | slice | any |" in rendered
    assert "| chest_xray | image | pneumothorax |" in rendered
    assert "| knee_mri | exam | meniscus |" in rendered
    encoded = json.dumps(_, allow_nan=False)
    assert "NaN" not in encoded

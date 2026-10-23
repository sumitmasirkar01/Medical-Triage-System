"""Release summary must remain traceable to the locked evidence bundle."""

from scripts.generate_release_manifest import build_manifest


def test_release_manifest_uses_locked_model_and_test_metrics():
    manifest = build_manifest()

    assert manifest["release"]["version"] == "0.1.0"
    assert manifest["schema_version"] == "2.0"
    assert set(manifest["modalities"]) == {"head_ct", "chest_xray", "knee_mri"}
    assert manifest["supported_scope"]["checkpoint_sha256"] == (
        "4a15c605bb1481de7f12a4881f9c9b5adb36d9386ecac0aa6eb3ae7c3ae805b1"
    )
    assert manifest["supported_scope"]["locked_threshold"] == 0.2538610100746155
    assert manifest["locked_test"]["images"] == 242
    assert manifest["locked_test"]["primary_metrics"]["roc_auc"] == 0.8417158598976782
    assert manifest["verification"]["expected_tests"] == 149
    phase2 = manifest["modalities"]["head_ct"]["phase_2_study_level"]
    assert phase2["status"] == "passed"
    assert phase2["selected_pooling"] == "mean"
    assert phase2["test_evaluated_once"] is True
    selective = phase2["selective_prediction"]
    assert selective["analysis_status"] == "post_test_exploratory"
    assert selective["prespecified_primary_policy"]["feasible"] is False
    robustness = phase2["slice_count_robustness"]
    assert robustness["analysis_status"] == "exploratory_subgroup_analysis"
    assert set(robustness["groups"]) == {
        "low_<=_32", "medium_33_to_36", "high_>_36"
    }
    assert manifest["modalities"]["chest_xray"]["checkpoint_sha256"] == (
        "609a353cf2d640ca7471e3952797b25ea4f5841d7f11fe7dfc8821dbfd9298bb"
    )
    cxr_robustness = manifest["modalities"]["chest_xray"]["subgroup_robustness"]
    assert cxr_robustness["analysis_status"] == "exploratory_subgroup_analysis"
    assert set(cxr_robustness["dimensions"]) == {"view", "recorded_sex", "age"}
    assert cxr_robustness["definitions"]["age"]["lower_cutpoint"] == 41
    assert cxr_robustness["definitions"]["age"]["upper_cutpoint"] == 56
    assert manifest["modalities"]["knee_mri"]["checkpoint_sha256"] == (
        "610847b2e4b9f6779528ab820c5744ef161d363f0e0748c3a0b2084881c350cc"
    )

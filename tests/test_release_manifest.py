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
    assert manifest["verification"]["expected_tests"] == 130
    assert manifest["modalities"]["chest_xray"]["checkpoint_sha256"] == (
        "609a353cf2d640ca7471e3952797b25ea4f5841d7f11fe7dfc8821dbfd9298bb"
    )
    assert manifest["modalities"]["knee_mri"]["checkpoint_sha256"] == (
        "610847b2e4b9f6779528ab820c5744ef161d363f0e0748c3a0b2084881c350cc"
    )

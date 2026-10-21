import numpy as np
from PIL import Image

from scripts.multimodal_smoke import create_cxr_fixture, create_mri_fixtures


def test_multimodal_smoke_fixtures_are_deterministic_and_contract_valid(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    create_cxr_fixture(first)
    create_cxr_fixture(second)
    assert first.read_bytes() == second.read_bytes()
    assert Image.open(first).mode == "L"

    paths = create_mri_fixtures(tmp_path / "mri")
    assert tuple(paths) == ("axial", "coronal", "sagittal")
    assert np.load(paths["axial"], allow_pickle=False).shape == (5, 96, 96)
    assert all(np.load(path, allow_pickle=False).dtype == np.uint8 for path in paths.values())

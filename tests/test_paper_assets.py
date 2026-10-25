import hashlib
from pathlib import Path

from scripts.generate_paper_assets import generate


ROOT = Path(__file__).resolve().parents[1]


def test_paper_asset_generator_emits_complete_reproducible_pack(tmp_path):
    generate(tmp_path, ROOT)
    expected = {
        "README.md",
        "methods_equations.md",
        "table_1_locked_results.csv",
        "table_1_locked_results.md",
        "table_2_primary_ct_result.md",
        "figure_1_system_architecture.svg",
        "figure_1_system_architecture.png",
        "figure_2_evaluation_protocol.svg",
        "figure_2_evaluation_protocol.png",
        "figure_3_locked_auc_forest.svg",
        "figure_3_locked_auc_forest.png",
        "figure_4_locked_operating_points.svg",
        "figure_4_locked_operating_points.png",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
    assert all((tmp_path / name).stat().st_size > 100 for name in expected)
    primary = (tmp_path / "table_2_primary_ct_result.md").read_text(encoding="utf-8")
    assert "0.908 (0.868–0.941)" in primary
    equations = (tmp_path / "methods_equations.md").read_text(encoding="utf-8")
    assert "s_0=0.90" in equations
    assert "ROC-AUC is threshold-independent" in equations

    first_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.iterdir()
    }
    generate(tmp_path, ROOT)
    second_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.iterdir()
    }
    assert second_hashes == first_hashes

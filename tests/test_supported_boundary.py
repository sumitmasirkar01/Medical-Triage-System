"""Supported code cannot silently depend on quarantined legacy modules."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_TOP_LEVEL_PATHS = ("core", "models", "weights")


def test_supported_package_has_no_legacy_core_imports():
    violations = []
    for path in (ROOT / "stat_triage").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == "core" or name.startswith("core.") for name in names):
                violations.append(str(path.relative_to(ROOT)))

    assert violations == []


def test_legacy_prototype_trees_are_quarantined():
    assert [name for name in LEGACY_TOP_LEVEL_PATHS if (ROOT / name).exists()] == []


def test_superseded_fixture_generator_is_quarantined():
    assert not (ROOT / "create_sample_dicoms.py").exists()
    assert (ROOT / "scripts" / "create_demo_dicom.py").is_file()


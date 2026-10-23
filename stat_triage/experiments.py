"""Fail-closed provenance registry for research experiments."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from typing import Iterable

from .contracts import Modality
from .evaluation import sha256


SCHEMA_VERSION = "1.0"
RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,79}$")


class ExperimentRegistryError(ValueError):
    """Experiment provenance is incomplete, unsafe, or modified."""


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentRegistryError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExperimentRegistryError(f"{path} must contain a JSON object")
    return value


def _project_file(root: Path, value: str | Path) -> Path:
    root = root.resolve()
    candidate = (root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ExperimentRegistryError(f"Path escapes project root: {value}") from exc
    if not candidate.is_file():
        raise ExperimentRegistryError(f"Missing experiment file: {candidate}")
    return candidate


def _declaration(root: Path, value: str | Path, role: str) -> dict[str, object]:
    path = _project_file(root, value)
    if not role or not role.strip():
        raise ExperimentRegistryError("Artifact role must be non-empty")
    return {
        "role": role.strip(),
        "path": path.relative_to(root.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def current_git_commit(root: str | Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(root), capture_output=True, text=True
    )
    commit = result.stdout.strip().lower()
    if result.returncode or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ExperimentRegistryError("Cannot resolve a full Git commit for this run")
    return commit


def build_run_record(
    *,
    project_root: str | Path,
    protocol_path: str | Path,
    run_id: str,
    modality: Modality,
    purpose: str,
    dataset_id: str,
    seed: int,
    split_manifest: str | Path,
    config: str | Path,
    artifacts: Iterable[tuple[str, str | Path]],
    code_commit: str | None = None,
    created_at: str | None = None,
) -> dict[str, object]:
    root = Path(project_root).resolve()
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ExperimentRegistryError("run_id must be 3-80 lowercase safe characters")
    if not purpose.strip() or not dataset_id.strip():
        raise ExperimentRegistryError("purpose and dataset_id must be non-empty")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ExperimentRegistryError("seed must be a non-negative integer")
    declarations = [_declaration(root, path, role) for role, path in artifacts]
    roles = [item["role"] for item in declarations]
    if len(roles) != len(set(roles)):
        raise ExperimentRegistryError("Artifact roles must be unique")
    protocol = _declaration(root, protocol_path, "research_protocol")
    resolved_commit = code_commit or current_git_commit(root)
    if not re.fullmatch(r"[0-9a-f]{40}", resolved_commit):
        raise ExperimentRegistryError("code_commit must be a full lowercase Git commit")
    resolved_created_at = created_at or datetime.now(timezone.utc).isoformat()
    try:
        timestamp = datetime.fromisoformat(resolved_created_at)
    except ValueError as exc:
        raise ExperimentRegistryError("created_at must be an ISO-8601 timestamp") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ExperimentRegistryError("created_at must include a timezone")
    return {
        "run_id": run_id,
        "status": "complete",
        "created_at": resolved_created_at,
        "modality": modality.value,
        "purpose": purpose.strip(),
        "dataset_id": dataset_id.strip(),
        "seed": seed,
        "code_commit": resolved_commit,
        "protocol": protocol,
        "split_manifest": _declaration(root, split_manifest, "split_manifest"),
        "config": _declaration(root, config, "run_config"),
        "artifacts": declarations,
    }


def initialize_registry(path: str | Path, protocol_path: str | Path, project_root: str | Path) -> None:
    destination = Path(path)
    protocol = _declaration(Path(project_root).resolve(), protocol_path, "research_protocol")
    payload = {"schema_version": SCHEMA_VERSION, "protocol": protocol, "runs": []}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def register_run(path: str | Path, record: dict[str, object], project_root: str | Path) -> None:
    registry_path = Path(path)
    registry = validate_experiment_registry(registry_path, project_root)
    runs = registry["runs"]
    if any(item["run_id"] == record.get("run_id") for item in runs):
        raise ExperimentRegistryError(f"Duplicate run_id: {record.get('run_id')}")
    runs.append(record)
    temporary = registry_path.with_suffix(registry_path.suffix + ".tmp")
    temporary.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    temporary.replace(registry_path)
    validate_experiment_registry(registry_path, project_root)


def validate_experiment_registry(path: str | Path, project_root: str | Path) -> dict[str, object]:
    root = Path(project_root).resolve()
    registry = _read_object(Path(path))
    if registry.get("schema_version") != SCHEMA_VERSION or not isinstance(registry.get("runs"), list):
        raise ExperimentRegistryError("Unsupported experiment registry schema")
    declarations = [registry.get("protocol")]
    identifiers: set[str] = set()
    for run in registry["runs"]:
        if not isinstance(run, dict) or not RUN_ID_PATTERN.fullmatch(str(run.get("run_id", ""))):
            raise ExperimentRegistryError("Invalid run record")
        if run["run_id"] in identifiers:
            raise ExperimentRegistryError(f"Duplicate run_id: {run['run_id']}")
        identifiers.add(run["run_id"])
        if run.get("modality") not in {item.value for item in Modality}:
            raise ExperimentRegistryError(f"Invalid modality for {run['run_id']}")
        if not re.fullmatch(r"[0-9a-f]{40}", str(run.get("code_commit", ""))):
            raise ExperimentRegistryError(f"Invalid Git commit for {run['run_id']}")
        try:
            created_at = datetime.fromisoformat(str(run.get("created_at", "")))
        except ValueError as exc:
            raise ExperimentRegistryError(
                f"Invalid created_at for {run['run_id']}"
            ) from exc
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ExperimentRegistryError(
                f"created_at lacks timezone for {run['run_id']}"
            )
        if run.get("protocol") != registry.get("protocol"):
            raise ExperimentRegistryError(
                f"Protocol declaration mismatch for {run['run_id']}"
            )
        declarations.extend([run.get("protocol"), run.get("split_manifest"), run.get("config")])
        declarations.extend(run.get("artifacts", []))
    for declaration in declarations:
        if not isinstance(declaration, dict):
            raise ExperimentRegistryError("Invalid file declaration")
        file_path = _project_file(root, str(declaration.get("path", "")))
        if file_path.stat().st_size != declaration.get("bytes") or sha256(file_path) != declaration.get("sha256"):
            raise ExperimentRegistryError(f"Experiment artifact changed: {file_path}")
    return registry

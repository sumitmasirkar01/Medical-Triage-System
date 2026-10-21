"""Create, append to, and verify the research experiment registry."""

from __future__ import annotations

import argparse
from pathlib import Path

from stat_triage.contracts import Modality
from stat_triage.experiments import (
    build_run_record,
    initialize_registry,
    register_run,
    validate_experiment_registry,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "research" / "experiment_registry.json"
DEFAULT_PROTOCOL = ROOT / "research" / "protocol.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init")
    commands.add_parser("validate")
    add = commands.add_parser("register")
    add.add_argument("--run-id", required=True)
    add.add_argument("--modality", choices=[item.value for item in Modality], required=True)
    add.add_argument("--purpose", required=True)
    add.add_argument("--dataset-id", required=True)
    add.add_argument("--seed", type=int, required=True)
    add.add_argument("--split-manifest", type=Path, required=True)
    add.add_argument("--config", type=Path, required=True)
    add.add_argument("--artifact", action="append", default=[], metavar="ROLE=PATH")
    args = parser.parse_args()
    if args.command == "init":
        initialize_registry(DEFAULT_REGISTRY, DEFAULT_PROTOCOL, ROOT)
    elif args.command == "validate":
        registry = validate_experiment_registry(DEFAULT_REGISTRY, ROOT)
        print(f"passed: {len(registry['runs'])} registered runs")
    else:
        artifacts = []
        for item in args.artifact:
            if "=" not in item:
                parser.error("--artifact must use ROLE=PATH")
            role, path = item.split("=", 1)
            artifacts.append((role, path))
        record = build_run_record(
            project_root=ROOT, protocol_path=DEFAULT_PROTOCOL,
            run_id=args.run_id, modality=Modality(args.modality), purpose=args.purpose,
            dataset_id=args.dataset_id, seed=args.seed, split_manifest=args.split_manifest,
            config=args.config, artifacts=artifacts,
        )
        register_run(DEFAULT_REGISTRY, record, ROOT)
        print(f"registered: {args.run_id}")


if __name__ == "__main__":
    main()

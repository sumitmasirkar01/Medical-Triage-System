"""Run unified inference from the command line and emit structured JSON."""

from __future__ import annotations

import json
import sys

from stat_triage.cli import parse_arguments, run_inference


def main() -> None:
    arguments = parse_arguments()
    try:
        result = run_inference(arguments)
        rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if arguments.output:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
    except (OSError, ValueError) as exc:
        error = {
            "schema_version": "1.0",
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        print(json.dumps(error, indent=2), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()

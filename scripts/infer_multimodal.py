"""Run unified inference from the command line and emit structured JSON."""

from __future__ import annotations

import json
from typing import Sequence

from stat_triage.cli import (
    exit_code_for_result,
    parse_arguments,
    run_inference,
    serialize_outcome,
)
from stat_triage.triage_outcome import TriageOutcome
from stat_triage.triage_service import UnifiedTriageService


def main(
    argv: Sequence[str] | None = None,
    *,
    service: UnifiedTriageService | None = None,
) -> int:
    arguments = parse_arguments(argv)
    try:
        result = run_inference(arguments, service=service)
    except Exception:
        result = serialize_outcome(TriageOutcome.failed("cli_execution_failed"))

    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if arguments.output:
        try:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(rendered, encoding="utf-8", newline="\n")
        except OSError:
            result = serialize_outcome(TriageOutcome.failed("output_write_failed"))
            rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    print(rendered, end="")
    return exit_code_for_result(result)


if __name__ == "__main__":
    raise SystemExit(main())

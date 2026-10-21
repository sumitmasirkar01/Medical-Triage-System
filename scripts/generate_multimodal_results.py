"""Generate locked multimodal result tables from immutable evidence bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stat_triage.results import build_consolidated_results, render_results_markdown


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts" / "consolidated_results",
    )
    args = parser.parse_args()
    frame, summary = build_consolidated_results(args.project_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "locked_multimodal_results.csv"
    json_path = args.output_dir / "locked_multimodal_results.json"
    markdown_path = args.output_dir / "locked_multimodal_results.md"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_results_markdown(frame), encoding="utf-8")
    print(frame.to_string(index=False))
    print("Saved:", csv_path.resolve())
    print("Saved:", json_path.resolve())
    print("Saved:", markdown_path.resolve())


if __name__ == "__main__":
    main()

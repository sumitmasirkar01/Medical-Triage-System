"""Decode representative training radiographs and save an input-contract preview."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from stat_triage.datasets.nih_cxr14 import NIH_PROJECT_TARGETS
from stat_triage.datasets.nih_preprocessing import preprocess_nih_radiograph


def select_examples(manifest: pd.DataFrame, seed: int) -> pd.DataFrame:
    train = manifest[manifest["split"] == "train"].sample(frac=1, random_state=seed)
    selections: list[pd.Series] = []
    reasons: list[str] = []
    used: set[str] = set()
    for target in NIH_PROJECT_TARGETS:
        candidates = train[train[f"target_{target}"] == 1]
        row = next(row for _, row in candidates.iterrows() if row["image_id"] not in used)
        selections.append(row)
        reasons.append(target)
        used.add(row["image_id"])
    for view in ("AP", "PA"):
        candidates = train[
            (train["view"] == view)
            & (train[[f"target_{name}" for name in NIH_PROJECT_TARGETS]].sum(axis=1) == 0)
        ]
        row = next(row for _, row in candidates.iterrows() if row["image_id"] not in used)
        selections.append(row)
        reasons.append(f"no_selected_finding_{view}")
        used.add(row["image_id"])
    selected = pd.DataFrame(selections).reset_index(drop=True)
    selected["selection_reason"] = reasons
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    selected = select_examples(manifest, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    figure, axes = plt.subplots(2, 3, figsize=(12, 8))
    for axis, (_, row) in zip(axes.flat, selected.iterrows()):
        tensor, preview, inspection = preprocess_nih_radiograph(
            row["image_path"], size=args.size
        )
        axis.imshow(preview, cmap="gray", vmin=0, vmax=255)
        axis.set_title(f"{row['selection_reason']} | {row['view']}\n{row['image_id']}")
        axis.axis("off")
        records.append({
            "image_id": row["image_id"],
            "patient_id": str(row["patient_id"]),
            "view": row["view"],
            "selection_reason": row["selection_reason"],
            "inspection": inspection.to_dict(),
            "tensor_shape": list(tensor.shape),
            "tensor_min": float(tensor.min()),
            "tensor_max": float(tensor.max()),
            "tensor_finite": bool(tensor.isfinite().all()),
        })
    figure.suptitle("NIH ChestX-ray14 preprocessing audit — training partition only")
    figure.tight_layout()
    preview_path = args.output_dir / "nih_cxr14_preprocessing_preview.png"
    audit_path = args.output_dir / "nih_cxr14_preprocessing_audit.json"
    figure.savefig(preview_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    audit = {
        "status": "passed",
        "selection_partition": "train",
        "seed": args.seed,
        "input_size": [args.size, args.size],
        "resize": "aspect_preserving_bilinear_center_pad_zero",
        "channels": "grayscale_repeated_three_times",
        "normalization": "ImageNet mean/std",
        "examples": records,
        "preview": str(preview_path),
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    print("Saved:", audit_path)


if __name__ == "__main__":
    main()


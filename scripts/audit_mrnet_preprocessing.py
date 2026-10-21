"""Create a training-only visual and numerical MRNet preprocessing audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from stat_triage.datasets.mrnet import MRNET_PLANES, MRNET_TARGETS
from stat_triage.datasets.mrnet_preprocessing import preprocess_mrnet_volume


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/mrnet_preprocessing"))
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"exam_id": str})
    rows = manifest[manifest["split"] == "train"].copy()
    if rows.empty:
        raise ValueError("Manifest has no training rows")
    selections = []
    definitions = (
        ("negative", (rows[list(MRNET_TARGETS)] == 0).all(axis=1)),
        ("abnormal", rows["abnormal"] == 1),
        ("acl", rows["acl"] == 1),
        ("meniscus", rows["meniscus"] == 1),
    )
    for reason, mask in definitions:
        candidates = rows[mask].sort_values("exam_id")
        if candidates.empty:
            raise ValueError(f"No training example available for {reason}")
        selections.append((reason, candidates.iloc[0]))

    figure, axes = plt.subplots(len(selections), len(MRNET_PLANES), figsize=(10, 12))
    audit_rows = []
    for row_index, (reason, row) in enumerate(selections):
        for column_index, plane in enumerate(MRNET_PLANES):
            tensor, preview, inspection = preprocess_mrnet_volume(row[f"{plane}_path"])
            middle = len(preview) // 2
            axes[row_index, column_index].imshow(preview[middle], cmap="gray", vmin=0, vmax=255)
            axes[row_index, column_index].set_title(
                f"{reason} | {plane}\nexam {row['exam_id']} | slice {middle + 1}/{len(preview)}"
            )
            axes[row_index, column_index].axis("off")
            audit_rows.append({
                "selection_reason": reason,
                "exam_id": str(row["exam_id"]),
                "plane": plane,
                **inspection.to_dict(),
                "tensor_shape": list(tensor.shape),
                "tensor_minimum": float(tensor.min()),
                "tensor_maximum": float(tensor.max()),
                "tensor_finite": bool(tensor.isfinite().all()),
            })
    figure.suptitle("Stanford MRNet training-only preprocessing audit", fontsize=15)
    figure.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = args.output_dir / "mrnet_preprocessing_preview.png"
    audit_path = args.output_dir / "mrnet_preprocessing_audit.json"
    figure.savefig(preview_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    report = {
        "status": "passed",
        "selection_partition": "train",
        "input_contract": "uint8_[slices,height,width]",
        "resize": "bilinear_224x224_each_slice",
        "channels": "grayscale_repeated_three_times",
        "normalization": "ImageNet mean/std",
        "examples": audit_rows,
        "preview": str(preview_path),
    }
    audit_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {audit_path}")


if __name__ == "__main__":
    main()

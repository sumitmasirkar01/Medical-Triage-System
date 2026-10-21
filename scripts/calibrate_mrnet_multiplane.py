"""Infer on MRNet calibration exams and lock calibration and thresholds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from stat_triage.datasets.mrnet import MRNET_PLANES, MRNET_TARGETS
from stat_triage.datasets.mrnet_training import MRNetExamDataset, collate_mrnet_exams
from stat_triage.models import MRNetMultiPlaneModel
from stat_triage.mrnet_calibration import (
    DEFAULT_MRNET_TARGET_SENSITIVITIES,
    calibrate_mrnet_predictions,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    logits_rows, target_rows, exam_ids = [], [], []
    for index, batch in enumerate(loader, start=1):
        volumes = {
            p: batch["volumes"][p].to(device, non_blocking=True) for p in MRNET_PLANES
        }
        masks = {
            p: batch["slice_masks"][p].to(device, non_blocking=True) for p in MRNET_PLANES
        }
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(volumes, masks)
        logits_rows.append(logits.float().cpu())
        target_rows.append(batch["targets"])
        exam_ids.extend(batch["exam_id"])
        if index % 25 == 0 or index == len(loader):
            print(f"Calibration inference: {index}/{len(loader)} batches", flush=True)
    return torch.cat(logits_rows).numpy(), torch.cat(target_rows).numpy(), exam_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/kaggle/working/mrnet_calibration"),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--slice-chunk-size", type=int, default=32)
    parser.add_argument("--input-predictions", type=Path)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, dtype={"exam_id": str})
    rows = manifest.loc[manifest["split"] == "calibration"].reset_index(drop=True)
    if rows.empty:
        raise ValueError("Manifest has no calibration exams")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if tuple(checkpoint.get("targets", ())) != MRNET_TARGETS:
        raise ValueError("Checkpoint target order does not match MRNet")
    if tuple(checkpoint.get("planes", ())) != MRNET_PLANES:
        raise ValueError("Checkpoint plane order does not match MRNet")

    if args.input_predictions:
        predictions = pd.read_csv(args.input_predictions, dtype={"exam_id": str})
        if predictions["exam_id"].tolist() != rows["exam_id"].tolist():
            raise ValueError("Saved predictions do not match calibration manifest order")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MRNetMultiPlaneModel(
            pretrained=False, slice_chunk_size=args.slice_chunk_size
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device)
        loader = DataLoader(
            MRNetExamDataset(manifest, split="calibration"),
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
            pin_memory=device.type == "cuda", collate_fn=collate_mrnet_exams,
            persistent_workers=args.workers > 0,
        )
        logits, targets, exam_ids = predict(model, loader, device)
        predictions = pd.DataFrame({"exam_id": exam_ids})
        for i, target in enumerate(MRNET_TARGETS):
            predictions[f"target_{target}"] = targets[:, i].astype(np.int8)
            predictions[f"logit_{target}"] = logits[:, i]
            predictions[f"raw_score_{target}"] = 1.0 / (1.0 + np.exp(-logits[:, i]))

    calibrated, result = calibrate_mrnet_predictions(predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "calibration_predictions.csv"
    lock_path = args.output_dir / "locked_calibration.json"
    calibrated.to_csv(predictions_path, index=False)
    report = {
        "status": "locked", "partition": "calibration",
        "test_partition_accessed": False,
        "checkpoint": {
            "path": str(args.checkpoint), "sha256": sha256(args.checkpoint),
            "epoch": int(checkpoint["epoch"]), "stage": checkpoint["stage"],
            "validation_macro_auc": float(checkpoint["validation_macro_auc"]),
        },
        "manifest_sha256": sha256(args.manifest),
        "target_sensitivity_policy": DEFAULT_MRNET_TARGET_SENSITIVITIES,
        "inference_reused": args.input_predictions is not None,
        **result.to_dict(),
        "predictions": {"path": str(predictions_path), "sha256": sha256(predictions_path)},
    }
    lock_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Saved:", lock_path)


if __name__ == "__main__":
    main()

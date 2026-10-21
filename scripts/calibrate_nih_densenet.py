"""Run calibration-only NIH inference, probability calibration, and threshold locking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from stat_triage.cxr_calibration import (
    DEFAULT_TARGET_SENSITIVITIES,
    calibrate_cxr_predictions,
)
from stat_triage.cxr_training import sha256
from stat_triage.datasets.nih_training import NIHCXRDataset, NIH_TARGET_ORDER
from stat_triage.models import build_densenet121_multilabel


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    logits_rows, target_rows, image_ids, patient_ids = [], [], [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
        logits_rows.append(logits.float().cpu())
        target_rows.append(batch["targets"])
        image_ids.extend(batch["image_id"])
        patient_ids.extend(batch["patient_id"])
    return (
        torch.cat(logits_rows).numpy(),
        torch.cat(target_rows).numpy(),
        image_ids,
        patient_ids,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/kaggle/working/nih_densenet_calibration")
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--input-predictions",
        type=Path,
        help="Reuse saved calibration logits after a calibration-method revision.",
    )
    args = parser.parse_args()

    manifest_hash = sha256(args.manifest)
    manifest = pd.read_csv(args.manifest)
    if "split" not in manifest or set(manifest["split"].unique()) - {
        "train", "validation", "calibration", "test"
    }:
        raise ValueError("Manifest contains an invalid split contract")
    rows = manifest[manifest["split"] == "calibration"].reset_index(drop=True)
    if rows.empty:
        raise ValueError("Manifest has no calibration rows")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("manifest_sha256") != manifest_hash:
        raise ValueError("Checkpoint and manifest SHA-256 do not match")
    if tuple(checkpoint.get("labels", ())) != NIH_TARGET_ORDER:
        raise ValueError("Checkpoint label order does not match the NIH contract")
    if checkpoint.get("architecture") != "torchvision_densenet121":
        raise ValueError("Checkpoint architecture is not torchvision_densenet121")

    if args.input_predictions:
        predictions = pd.read_csv(args.input_predictions)
        expected_ids = rows["image_id"].astype(str).tolist()
        if predictions["image_id"].astype(str).tolist() != expected_ids:
            raise ValueError("Saved predictions do not exactly match calibration manifest order")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_densenet121_multilabel(
            len(NIH_TARGET_ORDER), pretrained=False, freeze_features=False
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device)
        loader = DataLoader(
            NIHCXRDataset(rows, size=224),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        logits, targets, image_ids, patient_ids = predict(model, loader, device)
        predictions = pd.DataFrame({"image_id": image_ids, "patient_id": patient_ids})
        for index, target in enumerate(NIH_TARGET_ORDER):
            predictions[f"target_{target}"] = targets[:, index].astype(np.int8)
            predictions[f"logit_{target}"] = logits[:, index]
            predictions[f"raw_score_{target}"] = 1.0 / (1.0 + np.exp(-logits[:, index]))

    calibrated, result = calibrate_cxr_predictions(
        predictions, target_sensitivities=DEFAULT_TARGET_SENSITIVITIES
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "calibration_predictions.csv"
    config_path = args.output_dir / "locked_calibration.json"
    calibrated.to_csv(predictions_path, index=False)
    report = {
        "status": "locked",
        "partition": "calibration",
        "test_partition_accessed": False,
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": sha256(args.checkpoint),
            "epoch": int(checkpoint["epoch"]),
        },
        "manifest_sha256": manifest_hash,
        "target_sensitivity_policy": DEFAULT_TARGET_SENSITIVITIES,
        "inference_reused": args.input_predictions is not None,
        **result.to_dict(),
        "predictions": {
            "path": str(predictions_path),
            "sha256": sha256(predictions_path),
        },
    }
    config_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {config_path}")


if __name__ == "__main__":
    main()

"""One-time evaluation of the locked NIH DenseNet on the official test split."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from stat_triage.cxr_evaluation import evaluate_locked_cxr_predictions
from stat_triage.cxr_training import sha256
from stat_triage.datasets.nih_training import NIHCXRDataset, NIH_TARGET_ORDER
from stat_triage.models import build_densenet121_multilabel


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    logits_rows, target_rows, image_ids, patient_ids = [], [], [], []
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
        logits_rows.append(logits.float().cpu())
        target_rows.append(batch["targets"])
        image_ids.extend(batch["image_id"])
        patient_ids.extend(batch["patient_id"])
        if batch_index % 25 == 0 or batch_index == len(loader):
            print(f"Test inference: {len(image_ids):,}/{len(loader.dataset):,} images", flush=True)
    return (
        torch.cat(logits_rows).numpy(), torch.cat(target_rows).numpy(),
        image_ids, patient_ids, time.perf_counter() - started,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--locked-calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/nih_densenet_test"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()

    locked = json.loads(args.locked_calibration.read_text(encoding="utf-8"))
    manifest_hash = sha256(args.manifest)
    checkpoint_hash = sha256(args.checkpoint)
    if locked.get("status") != "locked" or locked.get("partition") != "calibration":
        raise ValueError("Calibration artifact is not locked")
    if locked.get("manifest_sha256") != manifest_hash:
        raise ValueError("Locked calibration and manifest SHA-256 do not match")
    if locked.get("checkpoint", {}).get("sha256") != checkpoint_hash:
        raise ValueError("Locked calibration and checkpoint SHA-256 do not match")

    manifest = pd.read_csv(args.manifest)
    rows = manifest[manifest["split"] == "test"].reset_index(drop=True)
    if rows.empty:
        raise ValueError("Manifest has no test rows")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("manifest_sha256") != manifest_hash:
        raise ValueError("Checkpoint and manifest SHA-256 do not match")
    if tuple(checkpoint.get("labels", ())) != NIH_TARGET_ORDER:
        raise ValueError("Checkpoint label order does not match")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_densenet121_multilabel(len(NIH_TARGET_ORDER), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    loader = DataLoader(
        NIHCXRDataset(rows, size=224), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    logits, targets, image_ids, patient_ids, inference_seconds = predict(model, loader, device)
    predictions = pd.DataFrame({"image_id": image_ids, "patient_id": patient_ids})
    for index, target in enumerate(NIH_TARGET_ORDER):
        predictions[f"target_{target}"] = targets[:, index].astype(np.int8)
        predictions[f"logit_{target}"] = logits[:, index]

    print("Computing patient-grouped bootstrap confidence intervals...", flush=True)
    evaluated, results = evaluate_locked_cxr_predictions(
        predictions, locked["targets"],
        bootstrap_replicates=args.bootstrap_replicates, bootstrap_seed=42,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "test_predictions.csv"
    metrics_path = args.output_dir / "test_metrics.json"
    evaluated.to_csv(predictions_path, index=False)
    report = {
        "status": "complete",
        "partition": "official_test",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "test_images": len(evaluated),
        "test_patients": int(evaluated["patient_id"].nunique()),
        "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": manifest_hash,
        "locked_calibration_sha256": sha256(args.locked_calibration),
        "threshold_retuned_on_test": False,
        "calibrator_refit_on_test": False,
        "inference_seconds": inference_seconds,
        "mean_inference_ms_per_image": 1000 * inference_seconds / len(evaluated),
        "bootstrap_replicates": args.bootstrap_replicates,
        "targets": results,
        "predictions": {"path": str(predictions_path), "sha256": sha256(predictions_path)},
    }
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()

"""Perform the single locked evaluation on the official MRNet valid partition."""

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

from stat_triage.datasets.mrnet import MRNET_PLANES, MRNET_TARGETS
from stat_triage.datasets.mrnet_training import MRNetExamDataset, collate_mrnet_exams
from stat_triage.evaluation import sha256
from stat_triage.models import MRNetMultiPlaneModel
from stat_triage.mrnet_evaluation import evaluate_locked_mrnet_predictions


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    logits_rows, target_rows, exam_ids = [], [], []
    started = time.perf_counter()
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
            print(f"Test inference: {index}/{len(loader)} batches", flush=True)
    return (
        torch.cat(logits_rows).numpy(), torch.cat(target_rows).numpy(), exam_ids,
        time.perf_counter() - started,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--locked-calibration", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/kaggle/working/mrnet_test")
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--slice-chunk-size", type=int, default=32)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()

    locked = json.loads(args.locked_calibration.read_text(encoding="utf-8"))
    manifest_hash = sha256(args.manifest)
    checkpoint_hash = sha256(args.checkpoint)
    if locked.get("status") != "locked" or locked.get("partition") != "calibration":
        raise ValueError("MRNet calibration artifact is not locked")
    if locked.get("manifest_sha256") != manifest_hash:
        raise ValueError("Calibration and manifest hashes do not match")
    if locked.get("checkpoint", {}).get("sha256") != checkpoint_hash:
        raise ValueError("Calibration and checkpoint hashes do not match")

    manifest = pd.read_csv(args.manifest, dtype={"exam_id": str})
    rows = manifest.loc[manifest["split"] == "test"].reset_index(drop=True)
    if rows.empty:
        raise ValueError("Manifest has no locked test exams")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if tuple(checkpoint.get("targets", ())) != MRNET_TARGETS:
        raise ValueError("Checkpoint target order does not match MRNet")
    if tuple(checkpoint.get("planes", ())) != MRNET_PLANES:
        raise ValueError("Checkpoint plane order does not match MRNet")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MRNetMultiPlaneModel(
        pretrained=False, slice_chunk_size=args.slice_chunk_size
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    loader = DataLoader(
        MRNetExamDataset(manifest, split="test"),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda", collate_fn=collate_mrnet_exams,
        persistent_workers=args.workers > 0,
    )
    logits, targets, exam_ids, seconds = predict(model, loader, device)
    predictions = pd.DataFrame({"exam_id": exam_ids})
    for index, target in enumerate(MRNET_TARGETS):
        predictions[f"target_{target}"] = targets[:, index].astype(np.int8)
        predictions[f"logit_{target}"] = logits[:, index]

    print("Computing exam-grouped bootstrap confidence intervals...", flush=True)
    evaluated, results = evaluate_locked_mrnet_predictions(
        predictions, locked["targets"],
        bootstrap_replicates=args.bootstrap_replicates, bootstrap_seed=42,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "test_predictions.csv"
    metrics_path = args.output_dir / "test_metrics.json"
    evaluated.to_csv(predictions_path, index=False)
    report = {
        "status": "complete", "partition": "official_valid_locked_as_test",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "test_exams": len(evaluated), "sampling_unit": "exam_id",
        "patient_identifiers_available": False,
        "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": manifest_hash,
        "locked_calibration_sha256": sha256(args.locked_calibration),
        "threshold_retuned_on_test": False, "calibrator_refit_on_test": False,
        "inference_seconds": seconds,
        "mean_inference_ms_per_exam": 1000 * seconds / len(evaluated),
        "bootstrap_replicates": args.bootstrap_replicates,
        "targets": results,
        "predictions": {"path": str(predictions_path), "sha256": sha256(predictions_path)},
    }
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Saved:", metrics_path)


if __name__ == "__main__":
    main()

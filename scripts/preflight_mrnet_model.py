"""Run an MRNet exam-level forward/backward preflight without optimization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

from stat_triage.datasets.mrnet import MRNET_PLANES, MRNET_TARGETS
from stat_triage.datasets.mrnet_training import (
    MRNetExamDataset,
    collate_mrnet_exams,
    compute_mrnet_positive_weights,
)
from stat_triage.losses import MaskedWeightedBCEWithLogitsLoss
from stat_triage.models import (
    MRNetMultiPlaneModel,
    configure_mrnet_stage,
    set_mrnet_training_modes,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/mrnet_model_preflight.json"),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--slice-chunk-size", type=int, default=32)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, dtype={"exam_id": str})
    weights = compute_mrnet_positive_weights(manifest)
    dataset = MRNetExamDataset(
        manifest, split="train", output_size=(args.size, args.size)
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_mrnet_exams,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    batch = next(iter(loader))
    load_seconds = time.perf_counter() - started
    model = MRNetMultiPlaneModel(
        pretrained=not args.no_pretrained,
        slice_chunk_size=args.slice_chunk_size,
    ).to(device)
    trainable = configure_mrnet_stage(model, "fusion")
    set_mrnet_training_modes(model, "fusion")
    objective = MaskedWeightedBCEWithLogitsLoss(weights).to(device)
    volumes = {
        plane: batch["volumes"][plane].to(device, non_blocking=True)
        for plane in MRNET_PLANES
    }
    masks = {
        plane: batch["slice_masks"][plane].to(device, non_blocking=True)
        for plane in MRNET_PLANES
    }
    targets = batch["targets"].to(device, non_blocking=True)
    label_mask = batch["label_mask"].to(device, non_blocking=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    compute_started = time.perf_counter()
    logits = model(volumes, masks)
    loss = objective(logits, targets, label_mask)
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    compute_seconds = time.perf_counter() - compute_started
    fusion_gradients = [
        p.grad for p in model.fusion.parameters() if p.requires_grad
    ]
    report = {
        "status": "passed",
        "scope": "training_partition_exam_forward_backward_no_optimizer_step",
        "device": str(device),
        "targets": list(MRNET_TARGETS),
        "planes": list(MRNET_PLANES),
        "training_exams": len(dataset),
        "batch_size": len(batch["exam_id"]),
        "exam_ids": batch["exam_id"],
        "volume_shapes": {p: list(volumes[p].shape) for p in MRNET_PLANES},
        "valid_slices": {
            p: masks[p].sum(dim=1).cpu().tolist() for p in MRNET_PLANES
        },
        "volumes_finite": bool(
            all(torch.isfinite(volumes[p]).all() for p in MRNET_PLANES)
        ),
        "normalized_range": {
            "minimum": min(float(volumes[p].min()) for p in MRNET_PLANES),
            "maximum": max(float(volumes[p].max()) for p in MRNET_PLANES),
        },
        "positive_weights": {
            target: float(weight) for target, weight in zip(MRNET_TARGETS, weights)
        },
        "model_output_shape": list(logits.shape),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "loss": float(loss.detach()),
        "loss_finite": bool(torch.isfinite(loss)),
        "fusion_gradients_present": bool(
            fusion_gradients and all(g is not None for g in fusion_gradients)
        ),
        "parameters": {
            "total": sum(p.numel() for p in model.parameters()),
            "trainable": trainable,
        },
        "timing_seconds": {
            "first_batch_load": load_seconds,
            "forward_backward": compute_seconds,
        },
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated() / (1024**2)
            if device.type == "cuda" else None
        ),
        "pretrained_imagenet": not args.no_pretrained,
        "training_stage": "fusion",
        "optimizer_step_performed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Saved:", args.output)


if __name__ == "__main__":
    main()

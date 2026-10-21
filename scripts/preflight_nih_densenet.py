"""Run a real-data DenseNet121 forward/backward preflight without optimization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

from stat_triage.datasets.nih_training import (
    CachedNIHCXRDataset,
    NIHCXRDataset,
    NIH_TARGET_ORDER,
    compute_positive_weights,
)
from stat_triage.losses import MaskedWeightedBCEWithLogitsLoss
from stat_triage.models import build_densenet121_multilabel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/nih_densenet_preflight.json"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--cache-images", type=Path)
    parser.add_argument("--cache-manifest", type=Path)
    args = parser.parse_args()

    if (args.cache_images is None) != (args.cache_manifest is None):
        raise ValueError("--cache-images and --cache-manifest must be provided together")
    manifest = pd.read_csv(args.cache_manifest or args.manifest)
    train = manifest[manifest["split"] == "train"].reset_index(drop=True)
    weights = compute_positive_weights(train)
    dataset = (
        CachedNIHCXRDataset(train, args.cache_images)
        if args.cache_images else NIHCXRDataset(train, size=args.size)
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    batch = next(iter(loader))
    load_seconds = time.perf_counter() - started
    model = build_densenet121_multilabel(
        len(NIH_TARGET_ORDER),
        pretrained=not args.no_pretrained,
        freeze_features=True,
    ).to(device)
    objective = MaskedWeightedBCEWithLogitsLoss(weights).to(device)
    images = batch["image"].to(device, non_blocking=True)
    targets = batch["targets"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    compute_started = time.perf_counter()
    logits = model(images)
    loss = objective(logits, targets, mask)
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    compute_seconds = time.perf_counter() - compute_started
    classifier_gradients = [
        parameter.grad for parameter in model.classifier.parameters()
        if parameter.requires_grad
    ]
    report = {
        "status": "passed",
        "scope": "training_partition_forward_backward_no_optimizer_step",
        "device": str(device),
        "targets": list(NIH_TARGET_ORDER),
        "training_rows": len(train),
        "batch_size": int(images.shape[0]),
        "image_shape": list(images.shape),
        "target_shape": list(targets.shape),
        "mask_shape": list(mask.shape),
        "image_dtype": str(images.dtype),
        "images_finite": bool(torch.isfinite(images).all()),
        "normalized_min": float(images.min()),
        "normalized_max": float(images.max()),
        "positive_weights": {
            target: float(weight) for target, weight in zip(NIH_TARGET_ORDER, weights)
        },
        "model_output_shape": list(logits.shape),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "loss": float(loss.detach()),
        "loss_finite": bool(torch.isfinite(loss)),
        "classifier_gradients_present": bool(
            classifier_gradients and all(gradient is not None for gradient in classifier_gradients)
        ),
        "parameters": {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
        },
        "timing_seconds": {
            "first_batch_load": load_seconds,
            "forward_backward": compute_seconds,
        },
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == "cuda" else None
        ),
        "pretrained_imagenet": not args.no_pretrained,
        "feature_extractor_frozen": True,
        "optimizer_step_performed": False,
        "cached_images": bool(args.cache_images),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Saved:", args.output)


if __name__ == "__main__":
    main()


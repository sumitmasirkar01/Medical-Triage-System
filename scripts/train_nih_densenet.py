"""Train the first NIH ChestX-ray14 four-target DenseNet121 baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from stat_triage.cxr_training import (
    configure_densenet_stage,
    multilabel_validation_metrics,
    set_densenet_training_modes,
    set_reproducible_seed,
    sha256,
)
from stat_triage.datasets.nih_training import (
    CachedNIHCXRDataset,
    NIHCXRDataset,
    NIH_TARGET_ORDER,
    compute_positive_weights,
)
from stat_triage.losses import MaskedWeightedBCEWithLogitsLoss
from stat_triage.models import build_densenet121_multilabel


@torch.inference_mode()
def evaluate(model, loader, objective, device):
    model.eval()
    losses, logits_rows, target_rows, image_ids, patient_ids = [], [], [], [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = objective(logits, targets, mask)
        losses.append(float(loss))
        logits_rows.append(logits.float().cpu())
        target_rows.append(targets.cpu())
        image_ids.extend(batch["image_id"])
        patient_ids.extend(batch["patient_id"])
    logits = torch.cat(logits_rows).numpy()
    targets = torch.cat(target_rows).numpy()
    scores = 1.0 / (1.0 + np.exp(-logits))
    metrics = multilabel_validation_metrics(targets, scores, NIH_TARGET_ORDER)
    metrics["loss"] = float(np.mean(losses))
    return metrics, logits, scores, targets, image_ids, patient_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/nih_densenet"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--classifier-epochs", type=int, default=2)
    parser.add_argument("--upper-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-images", type=Path)
    parser.add_argument("--cache-manifest", type=Path)
    args = parser.parse_args()
    set_reproducible_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    if (args.cache_images is None) != (args.cache_manifest is None):
        raise ValueError("--cache-images and --cache-manifest must be provided together")
    dataset_rows = pd.read_csv(args.cache_manifest) if args.cache_manifest else manifest
    train_rows = dataset_rows[dataset_rows["split"] == "train"].reset_index(drop=True)
    validation_rows = dataset_rows[dataset_rows["split"] == "validation"].reset_index(drop=True)
    weights = compute_positive_weights(train_rows)
    if args.cache_images:
        train_dataset = CachedNIHCXRDataset(train_rows, args.cache_images)
        validation_dataset = CachedNIHCXRDataset(validation_rows, args.cache_images)
    else:
        train_dataset = NIHCXRDataset(train_rows, size=224)
        validation_dataset = NIHCXRDataset(validation_rows, size=224)
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_options
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_densenet121_multilabel(
        len(NIH_TARGET_ORDER), pretrained=True, freeze_features=True
    ).to(device)
    objective = MaskedWeightedBCEWithLogitsLoss(weights).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    best_macro_ap = float("-inf")
    best_epoch = None
    global_epoch = 0
    training_started = time.perf_counter()
    plan = (
        ("classifier", args.classifier_epochs, 1e-3),
        ("upper_backbone", args.upper_epochs, 2e-4),
    )
    for stage, epochs, learning_rate in plan:
        parameters = configure_densenet_stage(model, stage)
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
        for _ in range(epochs):
            global_epoch += 1
            set_densenet_training_modes(model, stage)
            epoch_started = time.perf_counter()
            train_loss_sum = 0.0
            train_batches = 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                images = batch["image"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(images)
                    loss = objective(logits, targets, mask)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss_sum += float(loss.detach())
                train_batches += 1
            validation, _, _, _, _, _ = evaluate(
                model, validation_loader, objective, device
            )
            record = {
                "epoch": global_epoch,
                "stage": stage,
                "learning_rate": learning_rate,
                "train_loss": train_loss_sum / train_batches,
                "validation_loss": validation["loss"],
                "validation_macro_roc_auc": validation["macro_roc_auc"],
                "validation_macro_average_precision": validation["macro_average_precision"],
                "elapsed_seconds": time.perf_counter() - epoch_started,
            }
            for label, values in validation["per_target"].items():
                record[f"validation_{label}_roc_auc"] = values["roc_auc"]
                record[f"validation_{label}_average_precision"] = values["average_precision"]
            history.append(record)
            improved = validation["macro_average_precision"] > best_macro_ap
            if improved:
                best_macro_ap = validation["macro_average_precision"]
                best_epoch = global_epoch
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": global_epoch,
                    "stage": stage,
                    "labels": NIH_TARGET_ORDER,
                    "architecture": "torchvision_densenet121",
                    "validation_metrics": validation,
                    "manifest_sha256": sha256(args.manifest),
                    "positive_weights": weights.tolist(),
                }, args.output_dir / "best_densenet121.pt")
            print(
                f"Epoch {global_epoch:02d} [{stage}] train_loss={record['train_loss']:.4f} "
                f"val_loss={record['validation_loss']:.4f} "
                f"val_macro_AP={record['validation_macro_average_precision']:.4f} "
                f"val_macro_AUC={record['validation_macro_roc_auc']:.4f} "
                f"seconds={record['elapsed_seconds']:.1f} saved={improved}",
                flush=True,
            )

    checkpoint = torch.load(
        args.output_dir / "best_densenet121.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    validation, logits, scores, targets, image_ids, patient_ids = evaluate(
        model, validation_loader, objective, device
    )
    predictions = pd.DataFrame({"image_id": image_ids, "patient_id": patient_ids})
    for index, label in enumerate(NIH_TARGET_ORDER):
        predictions[f"target_{label}"] = targets[:, index].astype(int)
        predictions[f"logit_{label}"] = logits[:, index]
        predictions[f"score_{label}"] = scores[:, index]
    predictions.to_csv(args.output_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
    config = {
        "seed": args.seed,
        "architecture": "torchvision_densenet121",
        "initial_weights": "DenseNet121_Weights.IMAGENET1K_V1",
        "labels": list(NIH_TARGET_ORDER),
        "input_size": [224, 224],
        "preprocessing": "grayscale_repeat3_aspect_resize_center_pad_imagenet_normalization",
        "loss": "masked_weighted_bce_with_logits",
        "positive_weights": {
            label: float(weight) for label, weight in zip(NIH_TARGET_ORDER, weights)
        },
        "selection_metric": "validation_macro_average_precision",
        "train_images": len(train_rows),
        "validation_images": len(validation_rows),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "training_plan": [
            {"stage": stage, "epochs": epochs, "learning_rate": lr}
            for stage, epochs, lr in plan
        ],
        "best_epoch": best_epoch,
        "best_validation_macro_average_precision": best_macro_ap,
        "best_validation_metrics": validation,
        "manifest_sha256": sha256(args.manifest),
        "cached_images": bool(args.cache_images),
        "cache_images_sha256": sha256(args.cache_images) if args.cache_images else None,
        "cache_manifest_sha256": sha256(args.cache_manifest) if args.cache_manifest else None,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "total_training_seconds": time.perf_counter() - training_started,
        "test_partition_accessed": False,
        "calibration_partition_accessed": False,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "complete",
        "best_epoch": best_epoch,
        "best_validation_macro_average_precision": best_macro_ap,
        "output_dir": str(args.output_dir),
        "test_partition_accessed": False,
        "calibration_partition_accessed": False,
    }, indent=2))


if __name__ == "__main__":
    main()


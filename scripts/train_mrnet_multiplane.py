"""Train the shared-encoder, three-plane Stanford MRNet baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader, TensorDataset

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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loaders(manifest: pd.DataFrame, args: argparse.Namespace) -> dict[str, DataLoader]:
    return {
        split: DataLoader(
            MRNetExamDataset(
                manifest, split=split, output_size=(args.size, args.size)
            ),
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_mrnet_exams,
            persistent_workers=args.workers > 0,
        )
        for split in ("train", "validation")
    }


def move_exam(batch: dict, device: torch.device) -> tuple[dict, dict, torch.Tensor, torch.Tensor]:
    volumes = {
        p: batch["volumes"][p].to(device, non_blocking=True) for p in MRNET_PLANES
    }
    masks = {
        p: batch["slice_masks"][p].to(device, non_blocking=True) for p in MRNET_PLANES
    }
    targets = batch["targets"].to(device, non_blocking=True)
    label_mask = batch["label_mask"].to(device, non_blocking=True)
    return volumes, masks, targets, label_mask


@torch.no_grad()
def extract_frozen_features(
    model: MRNetMultiPlaneModel,
    loader: DataLoader,
    device: torch.device,
    split: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    features, targets = [], []
    started = time.perf_counter()
    for index, batch in enumerate(loader, start=1):
        volumes, masks, labels, _ = move_exam(batch, device)
        features.append(model.extract_exam_features(volumes, masks).cpu())
        targets.append(labels.cpu())
        if index % 100 == 0 or index == len(loader):
            print(
                f"Cached {split}: {index}/{len(loader)} batches "
                f"({(time.perf_counter() - started) / 60:.1f} min)",
                flush=True,
            )
    return torch.cat(features), torch.cat(targets)


def aucs(targets: np.ndarray, probabilities: np.ndarray) -> tuple[dict[str, float], float]:
    values = {
        target: float(roc_auc_score(targets[:, i], probabilities[:, i]))
        for i, target in enumerate(MRNET_TARGETS)
    }
    return values, float(np.mean(list(values.values())))


@torch.no_grad()
def evaluate_features(
    model: MRNetMultiPlaneModel,
    features: torch.Tensor,
    targets: torch.Tensor,
    objective: MaskedWeightedBCEWithLogitsLoss,
    device: torch.device,
) -> tuple[float, dict[str, float], float]:
    model.eval()
    logits = model.fusion(features.to(device))
    labels = targets.to(device)
    loss = objective(logits, labels, torch.ones_like(labels))
    target_aucs, macro = aucs(targets.numpy(), torch.sigmoid(logits).cpu().numpy())
    return float(loss), target_aucs, macro


@torch.no_grad()
def evaluate_exams(
    model: MRNetMultiPlaneModel,
    loader: DataLoader,
    objective: MaskedWeightedBCEWithLogitsLoss,
    device: torch.device,
) -> tuple[float, dict[str, float], float]:
    model.eval()
    losses, logits_all, targets_all = [], [], []
    for batch in loader:
        volumes, masks, targets, label_mask = move_exam(batch, device)
        logits = model(volumes, masks)
        losses.append(float(objective(logits, targets, label_mask)))
        logits_all.append(logits.cpu())
        targets_all.append(targets.cpu())
    logits = torch.cat(logits_all)
    targets = torch.cat(targets_all)
    target_aucs, macro = aucs(targets.numpy(), torch.sigmoid(logits).numpy())
    return float(np.mean(losses)), target_aucs, macro


def save_checkpoint(
    path: Path,
    model: MRNetMultiPlaneModel,
    epoch: int,
    stage: str,
    macro_auc: float,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "stage": stage,
            "validation_macro_auc": macro_auc,
            "targets": list(MRNET_TARGETS),
            "planes": list(MRNET_PLANES),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/mrnet_baseline"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--slice-chunk-size", type=int, default=32)
    parser.add_argument("--fusion-epochs", type=int, default=5)
    parser.add_argument("--upper-epochs", type=int, default=3)
    parser.add_argument("--fusion-lr", type=float, default=1e-3)
    parser.add_argument("--upper-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.fusion_epochs, args.upper_epochs, args.batch_size, args.accumulation_steps) < 1:
        raise ValueError("epochs, batch size, and accumulation steps must be positive")

    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest, dtype={"exam_id": str})
    data_loaders = loaders(manifest, args)
    weights = compute_mrnet_positive_weights(manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MRNetMultiPlaneModel(
        pretrained=True, slice_chunk_size=args.slice_chunk_size
    ).to(device)
    objective = MaskedWeightedBCEWithLogitsLoss(weights).to(device)
    checkpoint_path = args.output_dir / "best_mrnet_resnet18.pt"
    history: list[dict] = []
    best_auc = -np.inf
    global_epoch = 0

    print("Extracting frozen train and validation features...", flush=True)
    train_features, train_targets = extract_frozen_features(
        model, data_loaders["train"], device, "train"
    )
    validation_features, validation_targets = extract_frozen_features(
        model, data_loaders["validation"], device, "validation"
    )
    torch.save(
        {
            "train_features": train_features,
            "train_targets": train_targets,
            "validation_features": validation_features,
            "validation_targets": validation_targets,
        },
        args.output_dir / "frozen_exam_features.pt",
    )
    configure_mrnet_stage(model, "fusion")
    feature_loader = DataLoader(
        TensorDataset(train_features, train_targets),
        batch_size=32,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(
        model.fusion.parameters(), lr=args.fusion_lr, weight_decay=args.weight_decay
    )
    for _ in range(args.fusion_epochs):
        global_epoch += 1
        model.eval()
        model.fusion.train()
        losses = []
        for features, targets in feature_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.fusion(features)
            loss = objective(logits, targets, torch.ones_like(targets))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val_loss, target_aucs, macro_auc = evaluate_features(
            model, validation_features, validation_targets, objective, device
        )
        saved = macro_auc > best_auc
        if saved:
            best_auc = macro_auc
            save_checkpoint(checkpoint_path, model, global_epoch, "fusion", macro_auc)
        row = {
            "epoch": global_epoch, "stage": "fusion",
            "train_loss": float(np.mean(losses)), "validation_loss": val_loss,
            "validation_macro_auc": macro_auc, "saved": saved, **target_aucs,
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    configure_mrnet_stage(model, "upper_backbone")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.upper_lr,
        weight_decay=args.weight_decay,
    )
    for _ in range(args.upper_epochs):
        global_epoch += 1
        set_mrnet_training_modes(model, "upper_backbone")
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, batch in enumerate(data_loaders["train"], start=1):
            volumes, masks, targets, label_mask = move_exam(batch, device)
            logits = model(volumes, masks)
            raw_loss = objective(logits, targets, label_mask)
            (raw_loss / args.accumulation_steps).backward()
            if step % args.accumulation_steps == 0 or step == len(data_loaders["train"]):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(raw_loss.detach()))
            if step % 100 == 0:
                print(f"Epoch {global_epoch}: {step}/{len(data_loaders['train'])}", flush=True)
        val_loss, target_aucs, macro_auc = evaluate_exams(
            model, data_loaders["validation"], objective, device
        )
        saved = macro_auc > best_auc
        if saved:
            best_auc = macro_auc
            save_checkpoint(checkpoint_path, model, global_epoch, "upper_backbone", macro_auc)
        row = {
            "epoch": global_epoch, "stage": "upper_backbone",
            "train_loss": float(np.mean(losses)), "validation_loss": val_loss,
            "validation_macro_auc": macro_auc, "saved": saved, **target_aucs,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
        print(json.dumps(row), flush=True)

    pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    config = {
        "model": "shared_resnet18_max_pool_three_plane_fusion",
        "targets": list(MRNET_TARGETS), "planes": list(MRNET_PLANES),
        "manifest": str(args.manifest), "seed": args.seed,
        "positive_weights": dict(zip(MRNET_TARGETS, map(float, weights))),
        "best_validation_macro_auc": float(best_auc),
        "checkpoint_sha256": checkpoint_hash,
        "test_partition_used": False,
        "arguments": vars(args) | {"manifest": str(args.manifest), "output_dir": str(args.output_dir)},
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(config, indent=2), flush=True)


if __name__ == "__main__":
    main()

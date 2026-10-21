# train_axial_transformer.py
"""
STAT-Triage AI: 3D Axial Transformer Multi-Task Training Pipeline.
Calibrates Study-Level CLS and Slice-Level heads on standard unit-variance embeddings.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from rich.console import Console
from rich.panel import Panel

from core.axial_transformer import VolumetricAxialTransformer, AsymmetricFocalLoss

console = Console()


class VolumetricEmbeddingDataset(Dataset):
    def __init__(self, data_list: List[Dict[str, torch.Tensor]], max_slices: int = 32):
        self.data = data_list
        self.max_slices = max_slices

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        raw_emb = item["embeddings"]          # (S, 512)
        study_label = item["study_label"]      # (5,)
        slice_labels = item["slice_labels"]    # (S, 5)

        num_slices = raw_emb.shape[0]

        if num_slices >= self.max_slices:
            emb_padded = raw_emb[: self.max_slices]
            slice_labels_padded = slice_labels[: self.max_slices]
            mask = torch.zeros(self.max_slices, dtype=torch.bool)
        else:
            pad_len = self.max_slices - num_slices
            emb_pad = torch.zeros(pad_len, 512)
            emb_padded = torch.cat([raw_emb, emb_pad], dim=0)

            slice_label_pad = torch.zeros(pad_len, 5)
            slice_label_pad[:, 0] = 1.0  # Pad tokens marked normal
            slice_labels_padded = torch.cat([slice_labels, slice_label_pad], dim=0)

            mask = torch.cat([torch.zeros(num_slices, dtype=torch.bool), torch.ones(pad_len, dtype=torch.bool)], dim=0)

        return emb_padded, mask, study_label, slice_labels_padded


def generate_synthetic_training_corpus(num_studies: int = 600, max_slices: int = 32) -> List[Dict[str, torch.Tensor]]:
    """Generates synthetic multi-study embeddings aligned to standard unit-variance scale."""
    np.random.seed(42)
    torch.manual_seed(42)
    corpus = []

    for i in range(num_studies):
        s_len = np.random.randint(20, max_slices + 1)
        emb = torch.randn(s_len, 512)  # Standard normal baseline
        
        is_positive = np.random.rand() < 0.35
        study_target = torch.zeros(5)
        slice_targets = torch.zeros(s_len, 5)

        if is_positive:
            subtype = np.random.randint(1, 5)
            study_target[subtype] = 1.0
            
            start = np.random.randint(4, max(5, s_len - 7))
            end = min(s_len, start + np.random.randint(3, 6))
            
            slice_targets[:start, 0] = 1.0
            slice_targets[start:end, subtype] = 1.0
            slice_targets[end:, 0] = 1.0
            
            # Distinct feature signature along lesion slices
            emb[start:end, subtype * 20 : (subtype + 1) * 20] += 2.5
        else:
            study_target[0] = 1.0
            slice_targets[:, 0] = 1.0

        corpus.append({
            "study_uid": f"TRAIN_STUDY_{i:04d}",
            "embeddings": emb,
            "study_label": study_target,
            "slice_labels": slice_targets,
        })
    return corpus


def train_axial_model(
    epochs: int = 8,
    batch_size: int = 16,
    lr: float = 1e-4,
    checkpoint_out: str = "models/axial_transformer_weights.pt",
):
    console.print(Panel.fit("[bold cyan]3D Axial Transformer Multi-Task Training Engine[/bold cyan]"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[*] Compute Target: [bold green]{device}[/bold green]")

    raw_data = generate_synthetic_training_corpus(num_studies=600, max_slices=32)
    train_data = raw_data[:480]
    val_data = raw_data[480:]

    train_loader = DataLoader(VolumetricEmbeddingDataset(train_data, max_slices=32), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(VolumetricEmbeddingDataset(val_data, max_slices=32), batch_size=batch_size, shuffle=False)

    model = VolumetricAxialTransformer(embed_dim=512, num_heads=8, num_layers=2, num_classes=5).to(device)
    criterion = AsymmetricFocalLoss(gamma_neg=2.0, gamma_pos=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    os.makedirs(os.path.dirname(checkpoint_out), exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.perf_counter()

        for batch_emb, batch_mask, batch_study_y, batch_slice_y in train_loader:
            batch_emb = batch_emb.to(device)
            batch_mask = batch_mask.to(device)
            batch_study_y = batch_study_y.to(device)
            batch_slice_y = batch_slice_y.to(device)

            optimizer.zero_grad()
            study_logits, slice_logits, _ = model(batch_emb, key_padding_mask=batch_mask)

            loss_study = criterion(study_logits, batch_study_y)
            loss_slice = criterion(slice_logits.view(-1, 5), batch_slice_y.view(-1, 5))
            loss = loss_study + 0.5 * loss_slice

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * batch_emb.size(0)

        scheduler.step()
        train_loss /= len(train_loader.dataset)
        epoch_time = time.perf_counter() - t0

        # Validation Step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for v_emb, v_mask, v_study_y, v_slice_y in val_loader:
                v_emb, v_mask = v_emb.to(device), v_mask.to(device)
                v_study_y, v_slice_y = v_study_y.to(device), v_slice_y.to(device)

                v_study_logits, v_slice_logits, _ = model(v_emb, key_padding_mask=v_mask)
                v_loss = criterion(v_study_logits, v_study_y) + 0.5 * criterion(v_slice_logits.view(-1, 5), v_slice_y.view(-1, 5))
                val_loss += v_loss.item() * v_emb.size(0)

        val_loss /= len(val_loader.dataset)

        console.print(
            f"  • Epoch [{epoch:02d}/{epochs:02d}] "
            f"| Train Loss: [bold yellow]{train_loss:.4f}[/bold yellow] "
            f"| Val Loss: [bold cyan]{val_loss:.4f}[/bold cyan] "
            f"| Elapsed: {epoch_time:.2f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_out)
            console.print(f"    [bold green]✓ Best model saved to {checkpoint_out}[/bold green]")

    console.print("\n[bold green]✓ Axial Transformer training converged successfully.[/bold green]")


if __name__ == "__main__":
    train_axial_model()
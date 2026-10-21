# core/axial_transformer.py
"""
STAT-Triage AI: Volumetric Axial Cross-Attention Transformer.
Fuses intra-slice spatial embeddings across the Z-axis for study & slice-level triage.
Includes Asymmetric Focal Loss (ASL) and saliency attention extraction.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss (ASL) for multi-label classification with severe class imbalance.
    Decouples positive and negative focusing parameters (gamma_pos, gamma_neg).
    """

    def __init__(self, gamma_neg: float = 2.0, gamma_pos: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, self.eps, 1.0 - self.eps)
        pos_loss = targets * torch.pow(1.0 - probs, self.gamma_pos) * torch.log(probs)
        neg_loss = (1.0 - targets) * torch.pow(probs, self.gamma_neg) * torch.log(1.0 - probs)
        return -(pos_loss + neg_loss).mean()


class VolumetricAxialTransformer(nn.Module):
    """
    Sequence transformer operating over slice embeddings (S, 512).
    Generates study-level priority logits and slice-level localization logits.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        num_classes: int = 5,
        max_slices: int = 127,  # Sequence length with CLS token = 128
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Learnable CLS token representing the overall 3D study
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional embedding matching checkpoint shape (1, 128, 512)
        self.pos_embed = nn.Parameter(torch.zeros(1, 128, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=1024,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Multi-task diagnostic heads
        self.study_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self.slice_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention_map: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, S, embed_dim) sequence of slice embeddings.
            key_padding_mask: (B, S) boolean mask (True = padded/ignored).
            return_attention_map: If True, returns normalized slice attention saliency.
        Returns:
            study_logits: (B, num_classes)
            slice_logits: (B, S, num_classes)
            third_output: Saliency attention weights (B, S) if return_attention_map=True,
                          else study feature embeddings (B, embed_dim).
        """
        b, s, d = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embedding
        x = x + self.pos_embed[:, : s + 1, :]

        full_mask = None
        if key_padding_mask is not None:
            cls_mask = torch.zeros((b, 1), dtype=torch.bool, device=key_padding_mask.device)
            full_mask = torch.cat([cls_mask, key_padding_mask], dim=1)

        encoded = self.transformer_encoder(x, src_key_padding_mask=full_mask)

        study_features = encoded[:, 0, :]
        slice_features = encoded[:, 1:, :]

        study_logits = self.study_head(study_features)
        slice_logits = self.slice_head(slice_features)

        if return_attention_map:
            # Saliency vector: Softmax across slice embedding activations
            slice_saliency = torch.softmax(torch.norm(slice_features, dim=-1), dim=-1)
            return study_logits, slice_logits, slice_saliency

        return study_logits, slice_logits, study_features
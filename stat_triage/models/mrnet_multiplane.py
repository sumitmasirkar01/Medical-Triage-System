"""Shared-encoder, three-plane model for Stanford MRNet exams."""

from __future__ import annotations

import torch
from torch import nn

from stat_triage.datasets.mrnet import MRNET_PLANES


class MRNetMultiPlaneModel(nn.Module):
    """Encode slices with a shared ResNet and pool within each MRI plane."""

    def __init__(self, *, pretrained: bool = True, slice_chunk_size: int = 32) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * len(MRNET_PLANES), 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 3),
        )
        self.feature_dim = feature_dim
        self.slice_chunk_size = slice_chunk_size

    def _encode_plane(self, volume: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, slices, channels, height, width = volume.shape
        flattened = volume.reshape(batch * slices, channels, height, width)
        valid = mask.reshape(batch * slices)
        valid_images = flattened[valid]
        if valid_images.shape[0] == 0:
            raise ValueError("each plane batch must contain at least one valid slice")
        features = torch.cat(
            [self.backbone(chunk) for chunk in valid_images.split(self.slice_chunk_size)],
            dim=0,
        )
        exam_indices = (
            torch.arange(batch, device=mask.device)[:, None]
            .expand(batch, slices)
            .reshape(-1)[valid]
        )
        pooled = torch.full(
            (batch, self.feature_dim),
            -torch.inf,
            device=features.device,
            dtype=features.dtype,
        )
        pooled.scatter_reduce_(
            0,
            exam_indices[:, None].expand_as(features),
            features,
            reduce="amax",
            include_self=True,
        )
        return pooled

    def forward(
        self,
        volumes: dict[str, torch.Tensor],
        slice_masks: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.fusion(self.extract_exam_features(volumes, slice_masks))

    def extract_exam_features(
        self,
        volumes: dict[str, torch.Tensor],
        slice_masks: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return concatenated plane features before the fusion classifier."""
        plane_features = [
            self._encode_plane(volumes[plane], slice_masks[plane])
            for plane in MRNET_PLANES
        ]
        return torch.cat(plane_features, dim=1)


def configure_mrnet_stage(model: MRNetMultiPlaneModel, stage: str) -> int:
    """Configure the fusion-only or upper-backbone optimization stage."""
    if stage not in {"fusion", "upper_backbone"}:
        raise ValueError(f"unsupported MRNet training stage: {stage}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fusion.parameters():
        parameter.requires_grad = True
    if stage == "upper_backbone":
        for parameter in model.backbone.layer4.parameters():
            parameter.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_mrnet_training_modes(model: MRNetMultiPlaneModel, stage: str) -> None:
    """Keep frozen BatchNorm layers fixed while training selected modules."""
    model.eval()
    model.fusion.train()
    if stage == "upper_backbone":
        model.backbone.layer4.train()

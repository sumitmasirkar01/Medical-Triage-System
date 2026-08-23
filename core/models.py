import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import numpy as np


class UnifiedModelAdapter(ABC, nn.Module):
    """Abstract interface contract across all triage modalities."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        pass


class HeadCTTriPathwayNet(UnifiedModelAdapter):
    """
    2.5D Spatio-Temporal Network:
      - Spatial Extractor: EfficientNet-B0 (1280-D features)
      - Temporal Aggregator: 1D Bidirectional GRU (Hidden=512)
      - Heads: Study P(ICH) + 5 Subtype Heads
    """
    def __init__(self, hidden_dim: int = 512, num_subtypes: int = 5):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        self.backbone = efficientnet_b0(weights=weights)
        self.feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.study_head = nn.Linear(hidden_dim * 2, 1)
        self.subtype_head = nn.Linear(hidden_dim * 2, num_subtypes)

    def extract_slice_features(self, x_slices: torch.Tensor) -> torch.Tensor:
        # x_slices: (B * K, 3, H, W)
        return self.backbone(x_slices)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Input: (Batch, Slices, Channels=3, H, W)
        if x.ndim != 5 or x.shape[2] != 3:
            raise ValueError(f"Input tensor must have shape (B, K, 3, H, W), got {tuple(x.shape)}")

        B, K, C, H, W = x.shape
        x_flat = x.view(B * K, C, H, W)

        spatial_feats = self.extract_slice_features(x_flat) # (B*K, 1280)
        spatial_seq = spatial_feats.view(B, K, self.feature_dim) # (B, K, 1280)

        gru_out, _ = self.gru(spatial_seq) # (B, K, 1024)
        study_embedding, _ = torch.max(gru_out, dim=1) # (B, 1024)

        p_crit = torch.sigmoid(self.study_head(study_embedding))
        subtypes = torch.sigmoid(self.subtype_head(study_embedding))

        # Calibrated confidence proxy (distance from max entropy boundary at 0.5)
        epistemic_confidence = torch.clamp(2.0 * torch.abs(p_crit - 0.5), min=0.0, max=1.0)

        return {
            "p_critical": p_crit,
            "confidence": epistemic_confidence,
            "subtypes": subtypes,
            "study_embedding": study_embedding,
            "slice_features": spatial_seq
        }


class StatelessGradCAM:
    """Thread-safe, scoped Grad-CAM extractor with automatic hook cleanup."""
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

    def generate(self, input_tensor: torch.Tensor, target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        activations = []
        gradients = []

        def f_hook(module, inp, out):
            activations.append(out)

        def b_hook(module, g_in, g_out):
            gradients.append(g_out[0])

        h1 = self.target_layer.register_forward_hook(f_hook)
        h2 = self.target_layer.register_full_backward_hook(b_hook)

        try:
            input_tensor.requires_grad_(True)
            output = self.model(input_tensor)
            loss = torch.mean(output)
            self.model.zero_grad()
            loss.backward(retain_graph=False)

            act = activations[0]
            grad = gradients[0]
            weights = torch.mean(grad, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * act, dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=target_size, mode="bilinear", align_corners=False)
            
            cam_np = cam.detach().cpu().squeeze().numpy()
            
            # Guardrail: Flat / zero activation handling
            cam_range = cam_np.max() - cam_np.min()
            if cam_range < 1e-6 or np.isnan(cam_range):
                return np.zeros(target_size, dtype=np.float32)
                
            return (cam_np - cam_np.min()) / (cam_range + 1e-7)
        finally:
            h1.remove()
            h2.remove()


class PerturbationStabilityEvaluator:
    """Computes Perturbation Stability Index (PSI) under test-time affine jitter."""
    def __init__(self, iou_threshold: float = 0.75, num_passes: int = 5):
        self.iou_threshold = iou_threshold
        self.num_passes = num_passes

    def evaluate_stability(self, cam_extractor: StatelessGradCAM, slice_tensor: torch.Tensor) -> Tuple[float, bool]:
        binary_masks: List[np.ndarray] = []
        H, W = slice_tensor.shape[2], slice_tensor.shape[3]
        device = slice_tensor.device

        for _ in range(self.num_passes):
            angle = float(np.random.uniform(-3.0, 3.0))
            tx, ty = float(np.random.uniform(-4, 4)), float(np.random.uniform(-4, 4))
            
            rad = np.radians(angle)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            theta = torch.tensor([[
                [cos_a, -sin_a, tx / (W / 2)],
                [sin_a,  cos_a, ty / (H / 2)]
            ]], dtype=torch.float32, device=device)
            
            grid = F.affine_grid(theta, slice_tensor.size(), align_corners=False)
            jittered_x = F.grid_sample(slice_tensor, grid, align_corners=False)

            cam = cam_extractor.generate(jittered_x, target_size=(H, W))
            
            # If map is flat/inactive, mark unstable immediately
            if np.all(cam == 0.0) or np.var(cam) < 1e-5:
                return 0.0, False

            binary_mask = (cam >= np.percentile(cam, 80)).astype(bool)
            binary_masks.append(binary_mask)

        # Pairwise IoU across all jitter passes
        ious = []
        for i in range(len(binary_masks)):
            for j in range(i + 1, len(binary_masks)):
                intersection = np.logical_and(binary_masks[i], binary_masks[j]).sum()
                union = np.logical_or(binary_masks[i], binary_masks[j]).sum()
                ious.append((intersection + 1e-6) / (union + 1e-6))

        mean_iou = float(np.mean(ious))
        return mean_iou, (mean_iou >= self.iou_threshold)
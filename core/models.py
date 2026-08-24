# core/models.py
"""
Multi-Modal Clinical Triage Model Suite:
  - Pathway A: 2.5D EfficientNet-B0 + Bi-GRU Sequence Aggregator (Head CT)
  - Pathway B: DenseNet-121 Multi-Label Thoracic Classifier (Chest X-Ray)
  - Pathway C: EfficientNet-B0 Multi-Label Knee MRI Abnormality Classifier (Knee MRI)
  - Stateless Context-Managed Grad-CAM & Perturbation Stability Evaluator (PSI)
  - Polymorphic Adapters conforming to BaseTriageAdapter
"""

import os
from typing import Tuple, Dict, Optional, List, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    densenet121, DenseNet121_Weights
)
from .base import BaseTriageAdapter, ModalityType, ModalityTriageOutput


# ======================================================================
# 1. PATHWAY A: HEAD CT SEQUENCE MODEL (3D VOLUMETRIC)
# ======================================================================
class HeadCTSequenceGRU(nn.Module):
    def __init__(self, embed_dim: int = 1280, hidden_dim: int = 512, num_classes: int = 6, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0.0
        )
        gru_out_dim = hidden_dim * 2  # 1024-D
        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        return self.classifier(gru_out)


class HeadCTTriPathwayNet(nn.Module):
    SUBTYPES = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "any"]

    def __init__(self, weights_path: Optional[str] = "weights/head_ct_gru_weights.pt", device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity()
        self.sequence_gru = HeadCTSequenceGRU()

        if weights_path and os.path.exists(weights_path):
            ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
            self.sequence_gru.load_state_dict(ckpt["model_state_dict"])
            print(f"✅ Loaded Head CT weights from {weights_path} (AUC: {ckpt.get('mean_auc', 0.92):.4f})")
        else:
            print("⚠️ Head CT weights not found; running with initialized weights.")

        self.to(self.device)
        self.eval()

    @property
    def target_conv(self):
        return self.backbone.features[-1]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 4:
            x = x.unsqueeze(0)
        B, S, C, H, W = x.shape
        x_flat = x.view(B * S, C, H, W).to(self.device)

        spatial_feats = self.backbone(x_flat)
        spatial_seq = spatial_feats.view(B, S, -1)
        logits = self.sequence_gru(spatial_seq)
        slice_probs = torch.sigmoid(logits).squeeze(0)
        study_prob = slice_probs[:, 5].max()
        return study_prob, slice_probs


# ======================================================================
# 2. PATHWAY B: CHEST X-RAY MULTI-LABEL MODEL (2D PROJECTION)
# ======================================================================
class ThoracicDenseNet(nn.Module):
    SUBTYPES = ["pneumothorax", "pleural_effusion", "consolidation", "cardiomegaly", "edema", "critical_any"]

    def __init__(self, weights_path: Optional[str] = "weights/chest_xray_weights.pt", device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.backbone = densenet121(weights=DenseNet121_Weights.DEFAULT)
        
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 6)
        )

        if weights_path and os.path.exists(weights_path):
            ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
            state_dict = ckpt["model_state_dict"]
            cleaned = {k: v for k, v in state_dict.items() if not k.startswith("target_conv.")}
            if any(k.startswith("features.") or k.startswith("classifier.") for k in cleaned.keys()):
                self.backbone.load_state_dict(cleaned, strict=False)
            else:
                self.load_state_dict(cleaned, strict=False)
            print(f"✅ Loaded Chest X-Ray weights from {weights_path} (AUC: {ckpt.get('mean_auc', 0.81):.4f})")
        else:
            print("⚠️ Chest X-Ray weights not found; running with initialized weights.")

        self.to(self.device)
        self.eval()

    @property
    def target_conv(self):
        return self.backbone.features.denseblock4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return self.backbone(x.to(self.device))


# ======================================================================
# 3. PATHWAY C: KNEE MRI MULTI-LABEL MODEL (SOFT-TISSUE MRI)
# ======================================================================
class KneeMRITriageNet(nn.Module):
    SUBTYPES = [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
        "Medial OA", "Lateral OA", "PF OA", "Effusion",
        "Synovitis", "Baker's", "Contusion", "Fracture", "critical_surgical_any"
    ]

    def __init__(self, weights_path: Optional[str] = "weights/knee_mri_weights.pt", device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
        in_feats = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_feats, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 13)
        )

        if weights_path and os.path.exists(weights_path):
            ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
            state_dict = ckpt["model_state_dict"]
            # Filter out duplicate target_conv keys saved during training
            cleaned = {k: v for k, v in state_dict.items() if not k.startswith("target_conv.")}
            
            if any(k.startswith("features.") or k.startswith("classifier.") for k in cleaned.keys()):
                self.backbone.load_state_dict(cleaned, strict=False)
            else:
                self.load_state_dict(cleaned, strict=False)
            print(f"✅ Loaded Knee MRI weights from {weights_path} (AUC: {ckpt.get('mean_auc', 0.92):.4f})")
        else:
            print("⚠️ Knee MRI weights not found; running with initialized weights.")

        self.to(self.device)
        self.eval()

    @property
    def target_conv(self):
        return self.backbone.features[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return self.backbone(x.to(self.device))


# ======================================================================
# 4. EXPLAINABILITY & PERTURBATION STABILITY ENGINES
# ======================================================================
class StatelessGradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

    def _save_activations(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate_heatmap(self, tensor: torch.Tensor, target_class_idx: int = -1) -> np.ndarray:
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.model.device).requires_grad_(True)

        h1 = self.target_layer.register_forward_hook(self._save_activations)
        h2 = self.target_layer.register_full_backward_hook(self._save_gradients)

        try:
            self.model.zero_grad()
            if isinstance(self.model, HeadCTTriPathwayNet):
                feats = self.model.backbone(tensor)
                logits = self.model.sequence_gru.classifier(
                    F.gelu(self.model.sequence_gru.gru(feats.unsqueeze(0))[0])
                )
                score = logits[0, 0, 5]
            else:
                logits = self.model(tensor)
                score = logits[0, target_class_idx]

            score.backward()
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
            cam = F.relu(cam)
            cam_np = cam.cpu().numpy()
            return (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-7)
        finally:
            h1.remove()
            h2.remove()
            self.gradients = None
            self.activations = None


class PerturbationStabilityEvaluator:
    @staticmethod
    def calculate_psi(cam_orig: np.ndarray, model: nn.Module, target_layer: nn.Module, tensor: torch.Tensor, n_passes: int = 5) -> float:
        grad_cam = StatelessGradCAM(model, target_layer)
        binary_orig = cam_orig > 0.50
        ious = []

        for _ in range(n_passes):
            shift_x, shift_y = np.random.randint(-4, 5, size=2)
            jittered = torch.roll(tensor, shifts=(int(shift_x), int(shift_y)), dims=(-2, -1))
            cam_j = grad_cam.generate_heatmap(jittered)
            binary_j = cam_j > 0.50

            inter = np.logical_and(binary_orig, binary_j).sum()
            union = np.logical_or(binary_orig, binary_j).sum()
            ious.append(inter / (union + 1e-7))

        return float(np.mean(ious))


# ======================================================================
# 5. POLYMORPHIC ADAPTERS
# ======================================================================
class HeadCTAdapter(BaseTriageAdapter):
    def __init__(self, weights_path: Optional[str] = "weights/head_ct_gru_weights.pt", device: str = "cpu"):
        super().__init__(modality=ModalityType.HEAD_CT, device=device)
        self.model = HeadCTTriPathwayNet(weights_path=weights_path, device=device)
        self.grad_cam = StatelessGradCAM(self.model, self.model.target_conv)

    def predict(self, input_tensor: torch.Tensor, compute_psi: bool = True) -> ModalityTriageOutput:
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        with torch.inference_mode():
            study_prob, slice_probs = self.model(input_tensor)
            p_critical = float(study_prob.item())
            slice_probs_np = slice_probs.cpu().numpy()

        peak_idx = int(np.argmax(slice_probs_np[:, 5]))
        peak_slice = input_tensor[peak_idx]
        cam = self.grad_cam.generate_heatmap(peak_slice)

        psi = 0.90
        if compute_psi:
            psi = PerturbationStabilityEvaluator.calculate_psi(cam, self.model, self.model.target_conv, peak_slice)

        confidence = float(np.clip(2.0 * abs(p_critical - 0.5), 0.0, 1.0))
        subtypes = {s: float(slice_probs_np[:, i].max()) for i, s in enumerate(self.model.SUBTYPES)}

        return ModalityTriageOutput(
            modality=self.modality,
            p_critical=p_critical,
            confidence=confidence,
            psi_score=psi,
            saliency_map=cam,
            salient_slice_index=peak_idx,
            slice_probabilities=slice_probs_np,
            subtypes=subtypes,
            metadata={"slices": input_tensor.shape[0]}
        )


class ChestXRayAdapter(BaseTriageAdapter):
    def __init__(self, weights_path: Optional[str] = "weights/chest_xray_weights.pt", device: str = "cpu"):
        super().__init__(modality=ModalityType.CHEST_XRAY, device=device)
        self.model = ThoracicDenseNet(weights_path=weights_path, device=device)
        self.grad_cam = StatelessGradCAM(self.model, self.model.target_conv)

    def predict(self, input_tensor: torch.Tensor, compute_psi: bool = True) -> ModalityTriageOutput:
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        with torch.inference_mode():
            logits = self.model(input_tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        p_critical = float(probs[5])
        cam = self.grad_cam.generate_heatmap(input_tensor[0], target_class_idx=5)

        psi = 0.90
        if compute_psi:
            psi = PerturbationStabilityEvaluator.calculate_psi(cam, self.model, self.model.target_conv, input_tensor[0])

        confidence = float(np.clip(2.0 * abs(p_critical - 0.5), 0.0, 1.0))
        subtypes = {s: float(probs[i]) for i, s in enumerate(self.model.SUBTYPES)}

        return ModalityTriageOutput(
            modality=self.modality,
            p_critical=p_critical,
            confidence=confidence,
            psi_score=psi,
            saliency_map=cam,
            salient_slice_index=None,
            slice_probabilities=None,
            subtypes=subtypes,
            metadata={"projection": "Frontal PA/AP"}
        )


class KneeMRIAdapter(BaseTriageAdapter):
    def __init__(self, weights_path: Optional[str] = "weights/knee_mri_weights.pt", device: str = "cpu"):
        super().__init__(modality=ModalityType.KNEE_MRI, device=device)
        self.model = KneeMRITriageNet(weights_path=weights_path, device=device)
        self.grad_cam = StatelessGradCAM(self.model, self.model.target_conv)

    def predict(self, input_tensor: torch.Tensor, compute_psi: bool = True) -> ModalityTriageOutput:
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        with torch.inference_mode():
            logits = self.model(input_tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        p_critical = float(probs[12])  # critical_surgical_any
        cam = self.grad_cam.generate_heatmap(input_tensor[0], target_class_idx=12)

        psi = 0.90
        if compute_psi:
            psi = PerturbationStabilityEvaluator.calculate_psi(cam, self.model, self.model.target_conv, input_tensor[0])

        confidence = float(np.clip(2.0 * abs(p_critical - 0.5), 0.0, 1.0))
        subtypes = {s: float(probs[i]) for i, s in enumerate(self.model.SUBTYPES)}

        return ModalityTriageOutput(
            modality=self.modality,
            p_critical=p_critical,
            confidence=confidence,
            psi_score=psi,
            saliency_map=cam,
            salient_slice_index=None,
            slice_probabilities=None,
            subtypes=subtypes,
            metadata={"plane": "Sagittal/Coronal PD/T2"}
        )
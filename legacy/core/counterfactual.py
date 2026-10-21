# core/counterfactual.py
"""
STAT-Triage AI: Causal Counterfactual Inpainting Engine
Generates pixel-precise lesion masks by computing the minimal physical tissue
perturbation required to restore a pathological scan to a healthy equilibrium.
"""

from typing import Callable, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TotalVariationLoss(nn.Module):
    """Encourages spatial smoothness in the perturbation field."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (h_x - 1) * w_x
        count_w = h_x * (w_x - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class CausalCounterfactualEngine:
    """
    Causal Explanation Engine.
    Optimizes a bounded perturbation delta such that f(X + delta) predicts 'Healthy/Normal',
    isolating the exact anatomical lesion via subtraction: |X - X_counterfactual|.
    """

    def __init__(
        self,
        classifier_fn: Callable[[torch.Tensor], torch.Tensor],
        lambda_sparse: float = 0.02,
        lambda_tv: float = 0.01,
        lr: float = 0.08,
        max_iters: int = 40,
        device: Optional[torch.device] = None,
    ):
        self.classifier_fn = classifier_fn
        self.lambda_sparse = lambda_sparse
        self.lambda_tv = lambda_tv
        self.lr = lr
        self.max_iters = max_iters
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tv_loss = TotalVariationLoss()

    def generate_counterfactual(
        self,
        image_tensor: torch.Tensor,
        target_prob: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Computes the counterfactual scan and residual lesion attribution map.
        """
        x = image_tensor.clone().detach().to(self.device)
        if x.ndim == 3:
            x = x.unsqueeze(0)

        # Baseline initial probability
        with torch.no_grad():
            initial_p = float(self.classifier_fn(x).view(-1)[0].item())

        if initial_p <= target_prob:
            img_np = x.squeeze().cpu().numpy()
            if img_np.ndim == 3:
                img_np = img_np[0]
            return img_np, np.zeros_like(img_np), {
                "initial_p": initial_p,
                "final_p": initial_p,
                "iterations": 0,
                "l1_perturbation": 0.0,
            }

        # Initialize learnable perturbation parameter
        delta = nn.Parameter(torch.zeros_like(x, requires_grad=True))
        optimizer = torch.optim.Adam([delta], lr=self.lr)
        target_tensor = torch.tensor([target_prob], device=self.device, dtype=torch.float32)

        final_p = initial_p
        converged_iter = self.max_iters

        for step in range(self.max_iters):
            optimizer.zero_grad()
            
            # Physical pixel bounds [0, 1]
            x_cf = torch.clamp(x + delta, 0.0, 1.0)
            p_crit = self.classifier_fn(x_cf).view(-1)
            
            loss_task = F.mse_loss(p_crit, target_tensor)
            loss_l1 = torch.norm(delta, p=1) / delta.numel()
            loss_tv = self.tv_loss(delta)
            
            total_loss = loss_task + (self.lambda_sparse * loss_l1) + (self.lambda_tv * loss_tv)
            total_loss.backward()
            optimizer.step()

            current_p = float(p_crit[0].item())
            final_p = current_p

            if current_p <= target_prob * 1.05:
                converged_iter = step + 1
                break

        # Compute causal residual lesion map
        with torch.no_grad():
            optimal_cf = torch.clamp(x + delta, 0.0, 1.0).squeeze().cpu().numpy()
            if optimal_cf.ndim == 3:
                optimal_cf = optimal_cf[0]

            original_np = x.squeeze().cpu().numpy()
            if original_np.ndim == 3:
                original_np = original_np[0]

            residual = np.abs(original_np - optimal_cf)
            r_max = residual.max()
            residual_normalized = (residual / r_max) if r_max > 1e-6 else np.zeros_like(residual)

        metrics = {
            "initial_p": initial_p,
            "final_p": final_p,
            "iterations": converged_iter,
            "l1_perturbation": float(np.mean(np.abs(delta.detach().cpu().numpy()))),
        }

        return optimal_cf, residual_normalized, metrics
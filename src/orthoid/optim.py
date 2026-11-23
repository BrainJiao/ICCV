"""Conflict-free closed-form 2-task MGDA and gradient diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class MGDAStepStats:
    cos_sim: torch.Tensor
    conflict_mask: torch.Tensor
    gamma_star: torch.Tensor


def mgda_two_task(
    g_cnn: torch.Tensor,
    g_vit: torch.Tensor,
    normalize_scales: bool = True,
    cnn_weight: float = 0.62,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, MGDAStepStats]:
    """
    Conflict-free 2-task Pareto descent combining CNN-FR (g_cnn) and ViT-CLIP (g_vit):
      - When <u_cnn, u_vit> >= 0: combine directly with preference weights (cnn_weight, 1 - cnn_weight).
      - When <u_cnn, u_vit> < 0: project out the conflicting component from each unit gradient
        (u1_cf = u1 - c*u2, u2_cf = u2 - c*u1), guaranteeing strictly positive inner product
        <d*, g_cnn> > 0 AND <d*, g_vit> > 0 whenever cos(g_cnn, g_vit) > -1 (Proposition 2).
    """
    b = g_cnn.shape[0]
    f1 = g_cnn.reshape(b, -1)
    f2 = g_vit.reshape(b, -1)

    n1 = (f1.pow(2).sum(dim=1, keepdim=True) + eps).sqrt()
    n2 = (f2.pow(2).sum(dim=1, keepdim=True) + eps).sqrt()
    u1 = f1 / n1
    u2 = f2 / n2

    cos_sim = (u1 * u2).sum(dim=1)
    conflict_mask = cos_sim < 0.0

    if normalize_scales:
        c = cos_sim.unsqueeze(1)
        u1_cf = torch.where(c < 0.0, u1 - c * u2, u1)
        u2_cf = torch.where(c < 0.0, u2 - c * u1, u2)
        w1 = float(cnn_weight)
        w2 = 1.0 - w1
        d = w1 * u1_cf + w2 * u2_cf
        gamma = torch.full((b, 1), w1, device=g_cnn.device, dtype=g_cnn.dtype)
    else:
        diff = f2 - f1
        denom = diff.pow(2).sum(dim=1, keepdim=True) + eps
        num = (diff * f2).sum(dim=1, keepdim=True)
        gamma = torch.clamp(num / denom, 0.0, 1.0)
        d = gamma * f1 + (1.0 - gamma) * f2

    # Re-normalize combined direction to unit L2 norm for stable latent Adam / GD step sizes
    d = d / ((d.pow(2).sum(dim=1, keepdim=True) + eps).sqrt())

    stats = MGDAStepStats(
        cos_sim=cos_sim.detach(),
        conflict_mask=conflict_mask.detach(),
        gamma_star=gamma.squeeze(1).detach(),
    )
    return d.view_as(g_cnn), stats

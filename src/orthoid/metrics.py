"""Evaluation metrics aligned with OrthoID ICCV 2027 evaluation suite."""

from __future__ import annotations

import io
from typing import Callable, Dict, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_psnr(x01: torch.Tensor, y01: torch.Tensor) -> float:
    mse = (x01 - y01).pow(2).flatten(1).mean(dim=1).clamp(min=1e-12)
    return float((10.0 * torch.log10(1.0 / mse)).mean().item())


@torch.no_grad()
def compute_ssim(x01: torch.Tensor, y01: torch.Tensor) -> float:
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_x = F.avg_pool2d(x01, 11, 1, 5)
    mu_y = F.avg_pool2d(y01, 11, 1, 5)
    sigma_x = F.avg_pool2d(x01 * x01, 11, 1, 5) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y01 * y01, 11, 1, 5) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x01 * y01, 11, 1, 5) - mu_x * mu_y
    ssim_map = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + 1e-12
    )
    return float(ssim_map.mean().item())


@torch.no_grad()
def compute_budget_stats(x_prot: torch.Tensor, x_clean: torch.Tensor) -> Dict[str, float]:
    x_clean_q = torch.round(x_clean.clamp(0.0, 1.0) * 255.0) / 255.0
    delta = (x_prot - x_clean_q).abs()
    linf_255 = float(delta.max().item() * 255.0)
    rms_per_img = delta.flatten(1).pow(2).mean(dim=1).sqrt() * 255.0
    # Check 8-bit exactness
    quant_err = float((x_prot * 255.0 - torch.round(x_prot * 255.0)).abs().max().item())
    return {
        "max_linf_255": linf_255,
        "mean_rms_255": float(rms_per_img.mean().item()),
        "max_rms_255": float(rms_per_img.max().item()),
        "max_uint8_err": quant_err,
    }


@torch.no_grad()
def compute_fdr_mtcnn(
    x_prot: torch.Tensor,
    x_clean: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Evaluate real MTCNN Face Detection Rate (FDR %) and mean bounding-box IoU
    relative to clean detections.
    """
    from facenet_pytorch import MTCNN

    mtcnn = MTCNN(keep_all=False, device="cpu")
    b = x_prot.shape[0]
    detected = 0
    ious = []
    probs = []

    def _to_pil(t: torch.Tensor) -> Image.Image:
        arr = (t.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(arr)

    def _iou(b1, b2) -> float:
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
        a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
        return float(inter / (a1 + a2 - inter + 1e-8))

    for i in range(b):
        boxes_p, p_p = mtcnn.detect(_to_pil(x_prot[i]))
        if boxes_p is not None and len(boxes_p) > 0 and boxes_p[0] is not None:
            detected += 1
            if p_p is not None and p_p[0] is not None:
                probs.append(float(p_p[0]))
            if x_clean is not None:
                boxes_c, _ = mtcnn.detect(_to_pil(x_clean[i]))
                if boxes_c is not None and len(boxes_c) > 0 and boxes_c[0] is not None:
                    ious.append(_iou(boxes_p[0], boxes_c[0]))

    return {
        "fdr_pct": 100.0 * detected / max(b, 1),
        "mean_det_prob": float(np.mean(probs)) if probs else 0.0,
        "mean_bbox_iou": float(np.mean(ious)) if ious else 0.0,
    }


@torch.no_grad()
def purify_jpeg(x01: torch.Tensor, quality: int = 75) -> torch.Tensor:
    outs = []
    for i in range(x01.shape[0]):
        arr = (x01[i].detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        rec = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
        outs.append(torch.from_numpy(rec).permute(2, 0, 1))
    return torch.stack(outs, dim=0).to(x01.device)


@torch.no_grad()
def purify_resize_redetect(x01: torch.Tensor, down_size: int = 160) -> torch.Tensor:
    """Simulate social-media downscale + re-upload resize."""
    h, w = x01.shape[2], x01.shape[3]
    small = F.interpolate(x01, size=(down_size, down_size), mode="bilinear", align_corners=False)
    return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False).clamp(0.0, 1.0)


@torch.no_grad()
def compute_local_sharpness(
    x_prot: torch.Tensor,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    sigma: float = 4.0 / 255.0,
    num_samples: int = 8,
    seed: int = 0,
) -> float:
    """
    Empirical local loss sharpness probe:
        E_{v ~ N(0, sigma^2 I)} [ |L(clip(x + v)) - L(x)| ]
    Lower values indicate flatter optima (Proposition 2).
    """
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    base = loss_fn(x_prot)
    diffs = []
    for _ in range(num_samples):
        noise = torch.randn(x_prot.shape, generator=gen, dtype=x_prot.dtype).to(x_prot.device) * sigma
        pert = (x_prot + noise).clamp(0.0, 1.0)
        diffs.append((loss_fn(pert) - base).abs().mean().item())
    return float(np.mean(diffs))

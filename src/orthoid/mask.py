"""Soft face-localized mask builder for zero-background-distortion VAE latent residuals."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image


class FaceMaskBuilder:
    """
    Builds a smooth soft face mask M_face in [0, 1] of shape [B, 1, H, W].
    Uses MTCNN face bounding boxes when available, falling back cleanly to a
    portrait-centered elliptical mask so background pixels remain untouched.
    """

    def __init__(
        self,
        use_mtcnn: bool = True,
        margin_scale: float = 1.12,
        feather_sigma: float = 9.0,
    ):
        self.use_mtcnn = use_mtcnn
        self.margin_scale = float(margin_scale)
        self.feather_sigma = float(feather_sigma)
        self._mtcnn = None

    def _get_mtcnn(self):
        if self._mtcnn is None and self.use_mtcnn:
            from facenet_pytorch import MTCNN

            self._mtcnn = MTCNN(keep_all=False, device="cpu")
        return self._mtcnn

    @staticmethod
    def _elliptical_mask(
        h: int,
        w: int,
        cx: float,
        cy: float,
        rx: float,
        ry: float,
        feather: float = 0.22,
    ) -> torch.Tensor:
        ys = torch.linspace(0, h - 1, h).view(h, 1)
        xs = torch.linspace(0, w - 1, w).view(1, w)
        dist = torch.sqrt(((xs - cx) / max(rx, 1.0)) ** 2 + ((ys - cy) / max(ry, 1.0)) ** 2)
        # Smooth cosine taper from inner ellipse (1.0 - feather) to outer ellipse (1.0 + feather)
        inner = max(0.05, 1.0 - feather)
        outer = 1.0 + feather
        t = ((dist - inner) / max(outer - inner, 1e-6)).clamp(0.0, 1.0)
        mask = 0.5 * (1.0 + torch.cos(torch.pi * t))
        return mask.unsqueeze(0)  # [1, H, W]

    @torch.no_grad()
    def build(self, x01: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
        """Return soft face masks [B, 1, H, W] aligned to input batch x01 in [0, 1]."""
        target_device = device if device is not None else x01.device
        b, _, h, w = x01.shape
        mtcnn = self._get_mtcnn()
        masks = []

        for i in range(b):
            box = None
            if mtcnn is not None:
                arr = (
                    x01[i]
                    .detach()
                    .cpu()
                    .clamp(0.0, 1.0)
                    .permute(1, 2, 0)
                    .mul(255.0)
                    .round()
                    .to(torch.uint8)
                    .numpy()
                )
                boxes, _ = mtcnn.detect(Image.fromarray(arr))
                if boxes is not None and len(boxes) > 0 and boxes[0] is not None:
                    box = boxes[0]

            if box is not None:
                x1, y1, x2, y2 = [float(v) for v in box]
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                rx = 0.5 * (x2 - x1) * self.margin_scale
                ry = 0.5 * (y2 - y1) * self.margin_scale
            else:
                cx, cy = 0.5 * w, 0.5 * h
                rx, ry = 0.36 * w, 0.42 * h

            m = self._elliptical_mask(h, w, cx, cy, rx, ry, feather=0.22)
            masks.append(m)

        mask_batch = torch.stack(masks, dim=0).to(target_device)  # [B, 1, H, W]
        if self.feather_sigma > 0:
            ks = max(3, int(round(self.feather_sigma * 2)) | 1)
            ax = torch.arange(ks, dtype=torch.float32, device=target_device) - (ks - 1) / 2.0
            k1d = torch.exp(-(ax * ax) / (2.0 * (self.feather_sigma / 2.0) ** 2))
            k1d = k1d / k1d.sum()
            k2d = (k1d[:, None] * k1d[None, :]).view(1, 1, ks, ks)
            mask_batch = F.conv2d(mask_batch, k2d, padding=ks // 2).clamp(0.0, 1.0)
        return mask_batch

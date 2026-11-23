"""
Differentiable PyTorch IR-SE-50 Face Recognition encoder (standard TreB1eN / InsightFace_Pytorch).

Used strictly as an open-source zero-overlap surrogate CNN-FR encoder:
  - 100% disjoint from InsightFace buffalo_l (w600k_r50), antelopev2 (glintr100),
    PhotoMaker (CLIP-ViT-L/14), PuLID (EVA02-CLIP-L-14-336), and FaceNet-VGGFace2.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1)


class SEModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class BottleneckIRSE(nn.Module):
    def __init__(self, in_channel: int, depth: int, stride: int):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth),
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth),
            SEModule(depth, 16),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


_BottleneckSpec = namedtuple("Block", ["in_channel", "depth", "stride"])


def _get_block(in_channel: int, depth: int, num_units: int, stride: int = 2):
    return [_BottleneckSpec(in_channel, depth, stride)] + [
        _BottleneckSpec(depth, depth, 1) for _ in range(num_units - 1)
    ]


class BackboneIRSE50(nn.Module):
    def __init__(self, drop_ratio: float = 0.6):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(drop_ratio),
            Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512),
        )
        blocks = [
            _get_block(64, 64, 3),
            _get_block(64, 128, 4),
            _get_block(128, 256, 14),
            _get_block(256, 512, 3),
        ]
        modules = []
        for block in blocks:
            for spec in block:
                modules.append(BottleneckIRSE(spec.in_channel, spec.depth, spec.stride))
        self.body = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return F.normalize(x, dim=-1)


class IRSE50Encoder(nn.Module):
    """
    Differentiable IR-SE-50 wrapper taking [B, 3, H, W] RGB in [0, 1] and outputting
    L2-normalized 512-d identity embeddings.

    Verified against TreB1eN / InsightFace_Pytorch `model_ir_se50.pth`:
      - Expects BGR channel order in [-1, 1] (flip RGB -> BGR)
      - Expects 5-point ArcFace 112x112 canonical alignment (optional alignment-jitter EOT)
    """

    def __init__(
        self,
        device: str = "cpu",
        repo_id: str = "josephnguyencse/IRSE50",
        filename: str = "model_ir_se50.pth",
        local_files_only: bool = False,
        use_bgr: bool = True,
        crop_margin_frac: float = 0.08,
    ):
        super().__init__()
        self.name = "irse50"
        self.use_bgr = bool(use_bgr)
        self.crop_margin_frac = float(crop_margin_frac)
        self._active_thetas: Optional[torch.Tensor] = None
        self._eot_step: int = 0
        self._eot_views: int = 1
        ckpt_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_files_only=local_files_only,
        )
        self.backbone = BackboneIRSE50(drop_ratio=0.6)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.backbone.load_state_dict(state, strict=True)
        self.backbone.eval().to(device)
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def set_alignment(self, thetas: Optional[torch.Tensor], eot_views: int = 1) -> None:
        """Bind per-batch 5-point ArcFace alignment matrices [B, 2, 3] for differentiable warping."""
        self._active_thetas = thetas
        self._eot_views = max(1, int(eot_views))

    def _encode_aligned_112(self, x112_rgb: torch.Tensor) -> torch.Tensor:
        if self.use_bgr:
            x112 = x112_rgb.flip(1)
        else:
            x112 = x112_rgb
        return self.backbone(x112 * 2.0 - 1.0)

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        if self._active_thetas is not None and self._active_thetas.shape[0] == x01.shape[0]:
            from .align import DifferentiableArcFaceAligner

            if self._eot_views <= 1 or (not torch.is_grad_enabled()):
                x112 = DifferentiableArcFaceAligner.align(x01, self._active_thetas, jitter_seed=None)
                return self._encode_aligned_112(x112)

            # Batched alignment-jitter EOT across K views in a single forward/backward pass
            self._eot_step += 1
            views = [DifferentiableArcFaceAligner.align(x01, self._active_thetas, jitter_seed=None)]
            for k in range(1, self._eot_views):
                views.append(
                    DifferentiableArcFaceAligner.align(
                        x01,
                        self._active_thetas,
                        jitter_seed=10000 + self._eot_step * 31 + k,
                    )
                )
            stacked = torch.cat(views, dim=0)  # [K*B, 3, 112, 112]
            embs = self._encode_aligned_112(stacked).view(self._eot_views, x01.shape[0], -1)
            return F.normalize(embs.mean(dim=0), dim=-1)

        if self.crop_margin_frac > 0.0:
            _, _, h, w = x01.shape
            my = int(round(h * self.crop_margin_frac))
            mx = int(round(w * self.crop_margin_frac))
            if my > 0 and mx > 0 and (h - 2 * my) >= 32 and (w - 2 * mx) >= 32:
                x01 = x01[:, :, my : h - my, mx : w - mx]
        x112 = F.interpolate(x01, size=(112, 112), mode="bilinear", align_corners=False)
        return self._encode_aligned_112(x112)

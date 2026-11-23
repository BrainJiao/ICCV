"""Official baseline wrappers (e.g., IDProtector CVPR'25) under matched 8-bit budget Pi_x."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .align import ARCFACE_DST_112, DifferentiableArcFaceAligner, _estimate_similarity_2x3
from .budget import PublishableBudgetProjector


class IDProtectorCVPR25:
    """
    Official IDProtector (CVPR 2025, showlab/IDProtector) inference wrapper using
    its released `NoContextDiT` architecture and `protector_model.pt` weights.

    Constructs the exact 5-channel input expected by `protector_model.pt`:
      - Channels 0..2: RGB resized bicubically to 224x224 and normalized to [-1, 1]
      - Channel 3    : 224x224 ArcFace 112x112 canonical alignment region mask
      - Channel 4    : 224x224 CLIP center-crop prior mask (all ones for square images)
    and projects the output perturbation onto the unified publishable 8-bit budget Pi_x.
    """

    def __init__(
        self,
        repo_dir: str | Path = "/content/IDProtector",
        device: str = "cuda",
        eps: float = 8.0 / 255.0,
        rms_max: float = 5.5 / 255.0,
        native_epsilon: float = 0.07,
    ):
        self.repo_dir = Path(repo_dir)
        self.device = torch.device(device)
        self.native_epsilon = float(native_epsilon)
        self.projector = PublishableBudgetProjector(eps=eps, rms_max=rms_max)
        self.aligner = DifferentiableArcFaceAligner()

        if not (self.repo_dir / "protector_model.pt").exists():
            self.repo_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.check_call(
                ["git", "clone", "--depth", "1", "https://github.com/showlab/IDProtector.git", str(self.repo_dir)]
            )

        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))

        from modelling.NoContextDiT import NoContextDiT

        ckpt_path = self.repo_dir / "protector_model.pt"
        raw_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        clean_sd = {k.replace("module.", ""): v for k, v in raw_sd.items()}
        in_ch = int(clean_sd["x_embedder.proj.weight"].shape[1])
        self.in_channels = in_ch

        self.model = NoContextDiT(
            depth=12,
            hidden_size=384,
            patch_size=8,
            num_heads=6,
            input_size=224,
            learn_sigma=False,
            in_channels=in_ch,
            out_channels=3,
        )
        self.model.load_state_dict(clean_sd, strict=True)
        self.model = self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _build_affine_mask_224(h: int, w: int, landmarks_5: Optional[np.ndarray]) -> torch.Tensor:
        if landmarks_5 is None:
            # Reasonable fallback centered facial region
            m = torch.zeros((224, 224), dtype=torch.float32)
            m[28:196, 28:196] = 1.0
            return m
        mat = _estimate_similarity_2x3(landmarks_5, ARCFACE_DST_112)  # [2, 3] src(H,W) -> dst(112,112)
        ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        coords = torch.stack([xs.float(), ys.float(), torch.ones_like(xs.float())], dim=-1)  # [H, W, 3]
        mat_t = torch.from_numpy(mat).float()
        mapped = torch.einsum("ij,hwj->hwi", mat_t, coords)
        mask = ((mapped[..., 0] >= 0.0) & (mapped[..., 0] < 112.0) & (mapped[..., 1] >= 0.0) & (mapped[..., 1] < 112.0)).float()
        mask_224 = F.interpolate(mask[None, None], size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
        return mask_224[0, 0].clamp(0.0, 1.0)

    @staticmethod
    def _build_clip_prior_224(h: int, w: int) -> torch.Tensor:
        if h == w:
            return torch.ones((224, 224), dtype=torch.float32)
        m = torch.zeros((h, w), dtype=torch.float32)
        if h > w:
            off = (h - w) // 2
            m[off : off + w, :] = 1.0
        else:
            off = (w - h) // 2
            m[:, off : off + h] = 1.0
        return F.interpolate(m[None, None], size=(224, 224), mode="bicubic", align_corners=False, antialias=True)[0, 0].clamp(0.0, 1.0)

    @torch.no_grad()
    def protect(self, x_clean: torch.Tensor) -> torch.Tensor:
        """Protect batch x_clean [B, 3, H, W] in [0, 1] and return strict 8-bit Pi_x protected images."""
        from PIL import Image

        b, _, h, w = x_clean.shape
        mtcnn = self.aligner._get_mtcnn()

        x_224 = F.interpolate(x_clean, size=(224, 224), mode="bicubic", align_corners=False, antialias=True).clamp(0.0, 1.0)
        x_norm = x_224 * 2.0 - 1.0

        if self.in_channels == 5:
            aff_masks = []
            clip_priors = []
            for i in range(b):
                arr = (
                    x_clean[i]
                    .detach()
                    .cpu()
                    .clamp(0.0, 1.0)
                    .permute(1, 2, 0)
                    .mul(255.0)
                    .round()
                    .to(torch.uint8)
                    .numpy()
                )
                _, _, lms = mtcnn.detect(Image.fromarray(arr), landmarks=True)
                lm5 = np.asarray(lms[0], dtype=np.float32) if (lms is not None and len(lms) > 0 and lms[0] is not None) else None
                aff_masks.append(self._build_affine_mask_224(h, w, lm5))
                clip_priors.append(self._build_clip_prior_224(h, w))
            aff_t = torch.stack(aff_masks, dim=0).unsqueeze(1).to(self.device)
            clp_t = torch.stack(clip_priors, dim=0).unsqueeze(1).to(self.device)
            inp = torch.cat([x_norm.to(self.device), aff_t, clp_t], dim=1)
        elif self.in_channels == 3:
            inp = x_norm.to(self.device)
        else:
            raise ValueError(f"Unsupported IDProtector in_channels={self.in_channels}")

        pert_224 = self.model(inp).clamp(-1.0, 1.0) * (0.5 * self.native_epsilon)  # in [0, 1] pixel scale
        pert_hw = F.interpolate(pert_224, size=(h, w), mode="bicubic", align_corners=False, antialias=True)
        x_cand = (x_clean.to(self.device) + pert_hw).clamp(0.0, 1.0)
        return self.projector.publish(x_cand, x_clean.to(self.device))

"""
Differentiable 5-point similarity alignment to standard 112x112 ArcFace canonical frame
with random alignment-jitter EOT, implemented in pure PyTorch (F.affine_grid / F.grid_sample).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Standard ArcFace 112x112 reference 5-point landmarks:
# [left_eye, right_eye, nose, left_mouth, right_mouth]
ARCFACE_DST_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _estimate_similarity_2x3(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Closed-form Umeyama 2D similarity transform mapping src_pts [5, 2] -> dst_pts [5, 2].
    Returns 2x3 affine matrix M such that dst ~= M @ [src_x, src_y, 1]^T.
    """
    src = src_pts.astype(np.float64)
    dst = dst_pts.astype(np.float64)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    var_src = np.sum(src_c ** 2) / src.shape[0]
    cov = (dst_c.T @ src_c) / src.shape[0]
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(2, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[1] = -1.0
    rot = u @ np.diag(d) @ vt
    scale = np.sum(s * d) / max(var_src, 1e-12)
    t = dst_mean - scale * (rot @ src_mean)

    mat = np.zeros((2, 3), dtype=np.float32)
    mat[:, :2] = (scale * rot).astype(np.float32)
    mat[:, 2] = t.astype(np.float32)
    return mat


def _pixel_affine_to_theta(
    mat_src_to_dst: np.ndarray,
    src_h: int,
    src_w: int,
    dst_h: int = 112,
    dst_w: int = 112,
) -> torch.Tensor:
    """
    Convert a 2x3 pixel-space forward transform (src -> dst) into a PyTorch
    F.affine_grid theta [2, 3] (mapping normalized dst coords [-1, 1] -> normalized src coords [-1, 1]).
    """
    m_3x3 = np.eye(3, dtype=np.float64)
    m_3x3[:2, :] = mat_src_to_dst.astype(np.float64)
    m_inv = np.linalg.inv(m_3x3)

    # Pixel -> normalized [-1, 1] (align_corners=False convention)
    n_src = np.array(
        [
            [2.0 / src_w, 0.0, -1.0 + 1.0 / src_w],
            [0.0, 2.0 / src_h, -1.0 + 1.0 / src_h],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    n_dst_inv = np.array(
        [
            [dst_w / 2.0, 0.0, (dst_w - 1.0) / 2.0],
            [0.0, dst_h / 2.0, (dst_h - 1.0) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    theta_3x3 = n_src @ m_inv @ n_dst_inv
    return torch.from_numpy(theta_3x3[:2, :].astype(np.float32))


class DifferentiableArcFaceAligner:
    """
    Extracts clean-image 5-point landmarks via MTCNN once at t=0 and builds
    differentiable PyTorch grid-sampling matrices [B, 2, 3] for 112x112 ArcFace crops
    with optional small alignment-jitter EOT.
    """

    def __init__(self):
        self._mtcnn = None

    def _get_mtcnn(self):
        if self._mtcnn is None:
            from facenet_pytorch import MTCNN

            self._mtcnn = MTCNN(keep_all=False, device="cpu")
        return self._mtcnn

    @torch.no_grad()
    def compute_thetas(self, x01: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
        """Compute canonical 112x112 alignment theta [B, 2, 3] from clean batch x01 [B, 3, H, W]."""
        target_device = device if device is not None else x01.device
        b, _, h, w = x01.shape
        mtcnn = self._get_mtcnn()
        thetas = []
        for i in range(b):
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
            boxes, probs, landmarks = mtcnn.detect(Image.fromarray(arr), landmarks=True)
            if landmarks is not None and len(landmarks) > 0 and landmarks[0] is not None:
                src_pts = np.asarray(landmarks[0], dtype=np.float32)  # [5, 2]
                mat = _estimate_similarity_2x3(src_pts, ARCFACE_DST_112)
                theta = _pixel_affine_to_theta(mat, src_h=h, src_w=w, dst_h=112, dst_w=112)
            else:
                # Fallback: gentle 8% center crop
                theta = torch.tensor([[0.84, 0.0, 0.0], [0.0, 0.84, 0.0]], dtype=torch.float32)
            thetas.append(theta)
        return torch.stack(thetas, dim=0).to(target_device)

    @staticmethod
    def align(
        x01: torch.Tensor,
        thetas: torch.Tensor,
        jitter_seed: Optional[int] = None,
        scale_jitter: float = 0.04,
        rot_deg: float = 3.0,
        shift_frac: float = 0.03,
    ) -> torch.Tensor:
        """Differentiably warp x01 [B, 3, H, W] into 112x112 canonical ArcFace crops."""
        b = x01.shape[0]
        cur_theta = thetas.to(device=x01.device, dtype=x01.dtype)
        if jitter_seed is not None:
            g = torch.Generator(device="cpu").manual_seed(int(jitter_seed))
            s = 1.0 + (torch.rand((b,), generator=g) * 2.0 - 1.0) * scale_jitter
            ang = ((torch.rand((b,), generator=g) * 2.0 - 1.0) * rot_deg) * (np.pi / 180.0)
            tx = (torch.rand((b,), generator=g) * 2.0 - 1.0) * shift_frac
            ty = (torch.rand((b,), generator=g) * 2.0 - 1.0) * shift_frac

            cos_a = torch.cos(ang) * s
            sin_a = torch.sin(ang) * s
            j = torch.zeros((b, 3, 3), dtype=torch.float32)
            j[:, 0, 0] = cos_a
            j[:, 0, 1] = -sin_a
            j[:, 0, 2] = tx
            j[:, 1, 0] = sin_a
            j[:, 1, 1] = cos_a
            j[:, 1, 2] = ty
            j[:, 2, 2] = 1.0

            t3 = torch.zeros((b, 3, 3), device=x01.device, dtype=x01.dtype)
            t3[:, :2, :] = cur_theta
            t3[:, 2, 2] = 1.0
            cur_theta = torch.bmm(t3, j.to(device=x01.device, dtype=x01.dtype))[:, :2, :]

        grid = F.affine_grid(cur_theta, size=(b, x01.shape[1], 112, 112), align_corners=False)
        return F.grid_sample(x01, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

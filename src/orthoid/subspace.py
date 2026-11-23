"""Online semantic + photometric attribute subspace, latent pullback, and Pythagorean target."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

DEFAULT_ATTRIBUTE_PAIRS: List[Tuple[str, str, str]] = [
    (
        "makeup",
        "a portrait photo of a person wearing heavy makeup and lipstick",
        "a portrait photo of a person with no makeup and bare natural skin",
    ),
    (
        "gender",
        "a portrait photo of a masculine man's face",
        "a portrait photo of a feminine woman's face",
    ),
    (
        "expression",
        "a portrait photo of a smiling person",
        "a portrait photo of a person with a neutral facial expression",
    ),
    (
        "age",
        "a portrait photo of an elderly person",
        "a portrait photo of a young person",
    ),
    (
        "lighting",
        "a portrait photo with harsh dramatic shadows and bright highlights",
        "a portrait photo with soft even studio lighting",
    ),
]


def _orthonormalize_rows(vecs: torch.Tensor) -> torch.Tensor:
    """
    Orthonormalize a batch of row-stacked basis vectors [B, K, D] via reduced QR on CPU
    (robust across CUDA, MPS, and CPU). Returns [B, K_eff, D] on original device.
    """
    device = vecs.device
    dtype = vecs.dtype
    b = vecs.shape[0]
    qs = []
    for i in range(b):
        mat = vecs[i].detach().float().cpu().T  # [D, K]
        # Filter near-zero columns to keep QR numerically stable
        norms = mat.norm(dim=0)
        keep = norms > 1e-6
        if not bool(keep.any()):
            qs.append(torch.zeros((1, mat.shape[0]), dtype=torch.float32))
            continue
        q, _ = torch.linalg.qr(mat[:, keep], mode="reduced")  # [D, K_eff]
        qs.append(q.T)  # [K_eff, D]
    max_k = max(q.shape[0] for q in qs)
    padded = []
    for q in qs:
        if q.shape[0] < max_k:
            pad = torch.zeros((max_k - q.shape[0], q.shape[1]), dtype=torch.float32)
            q = torch.cat([q, pad], dim=0)
        padded.append(q)
    return torch.stack(padded, dim=0).to(device=device, dtype=dtype)


def project_onto_subspace(v: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project vectors v [B, D] onto row-orthonormal basis [B, K, D]."""
    coords = torch.einsum("bkd,bd->bk", basis, v)
    return torch.einsum("bk,bkd->bd", coords, basis)


def project_nullspace(v: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project vectors v [B, D] onto the orthogonal complement of row-orthonormal basis [B, K, D]."""
    return v - project_onto_subspace(v, basis)


def pythagorean_counterfactual_target(
    f_src: torch.Tensor,
    e_tar: Optional[torch.Tensor],
    q_attr: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Construct the exact unit-norm Pythagorean Counterfactual Target (Proposition 1):
        f_par  = Q_attr (Q_attr^T f_src)
        f_perp = f_src - f_par
        u_perp = P_attr^perp(e_tar) / ||P_attr^perp(e_tar)||_2   (or -f_perp / ||f_perp||_2 if target-free)
        e_tilde = f_par + ||f_perp||_2 * u_perp

    Guarantees:
      1. ||e_tilde||_2 == 1 identically (exact sphere feasibility)
      2. Q_attr^T e_tilde == Q_attr^T f_src (zero self-conflict with attribute preservation)
    """
    f_src = F.normalize(f_src, dim=-1)
    f_par = project_onto_subspace(f_src, q_attr)
    f_perp = f_src - f_par
    perp_norm = f_perp.norm(dim=-1, keepdim=True)

    if e_tar is None:
        u_perp = -f_perp / (perp_norm + eps)
    else:
        e_tar = F.normalize(e_tar, dim=-1)
        e_tar_perp = project_nullspace(e_tar, q_attr)
        u_perp = e_tar_perp / (e_tar_perp.norm(dim=-1, keepdim=True) + eps)

    e_tilde = f_par + perp_norm * u_perp
    return F.normalize(e_tilde, dim=-1)


@torch.no_grad()
def build_photometric_basis(
    encoder_fn: Callable[[torch.Tensor], torch.Tensor],
    x_clean: torch.Tensor,
) -> torch.Tensor:
    """
    Estimate per-image non-identity photometric/alignment attribute basis [B, K_p, D]
    via deterministic lighting, gamma, color-temperature, softness, and flip probes.
    """
    e0 = encoder_fn(x_clean)
    augs = [
        (x_clean * 1.08).clamp(0.0, 1.0),
        (x_clean * 0.92).clamp(0.0, 1.0),
        x_clean.clamp(1e-4, 1.0).pow(0.88),
        torch.stack(
            [x_clean[:, 0] * 1.05, x_clean[:, 1], x_clean[:, 2] * 0.95], dim=1
        ).clamp(0.0, 1.0),
        TF.gaussian_blur(x_clean, [5, 5], [0.9, 0.9]),
        torch.flip(x_clean, dims=[3]),
    ]
    diffs = [encoder_fn(xa) - e0 for xa in augs]
    stacked = torch.stack(diffs, dim=1)  # [B, K_p, D]
    return _orthonormalize_rows(stacked)


@dataclass
class AttributeSubspace:
    """Container for per-batch attribute bases and latent Jacobian pullback."""

    axis_names: List[str]
    text_axes_raw: torch.Tensor  # [K_text, D_vit] unit difference vectors (for 5-Axis Attr-Drift eval)
    q_vit: torch.Tensor  # [B, K_vit, D_vit] orthonormal semantic + photometric basis in CLIP space
    q_cnn_list: List[torch.Tensor]  # list of [B, K_cnn, D_cnn] per CNN-FR surrogate
    u_latent: Optional[torch.Tensor]  # [B, K_pull, D_z] orthonormal latent pullback basis

    @staticmethod
    @torch.no_grad()
    def build_text_axes(
        text_encoder_fn: Callable[[Sequence[str]], torch.Tensor],
        attribute_pairs: Sequence[Tuple[str, str, str]] = DEFAULT_ATTRIBUTE_PAIRS,
    ) -> Tuple[List[str], torch.Tensor]:
        names = [p[0] for p in attribute_pairs]
        pos_prompts = [p[1] for p in attribute_pairs]
        neg_prompts = [p[2] for p in attribute_pairs]
        e_pos = F.normalize(text_encoder_fn(pos_prompts), dim=-1)
        e_neg = F.normalize(text_encoder_fn(neg_prompts), dim=-1)
        axes = F.normalize(e_pos - e_neg, dim=-1)  # [K_text, D_vit]
        return names, axes

    @staticmethod
    def build_latent_pullback(
        decode_fn: Callable[[torch.Tensor], torch.Tensor],
        vit_encoder_fn: Callable[[torch.Tensor], torch.Tensor],
        dz_shape: Tuple[int, ...],
        q_pull: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute K-dim Jacobian pullback at dz = 0:
            J_k = grad_{dz} (q_k^T f_ViT(x'(dz)))|_{dz=0}
        and orthonormalize into U_attr [B, K_pull, D_z].
        """
        b, k_pull, _ = q_pull.shape
        dz = torch.zeros(dz_shape, device=device, dtype=torch.float32, requires_grad=True)
        x_prime = decode_fn(dz)
        f_vit = vit_encoder_fn(x_prime)  # [B, D_vit]

        jac_rows = []
        for k in range(k_pull):
            coord = (f_vit * q_pull[:, k, :]).sum(dim=-1)  # [B]
            g_k = torch.autograd.grad(
                coord.sum(),
                dz,
                retain_graph=(k < k_pull - 1),
                create_graph=False,
            )[0]
            jac_rows.append(g_k.detach().reshape(b, -1))
        stacked = torch.stack(jac_rows, dim=1)  # [B, K_pull, D_z]
        return _orthonormalize_rows(stacked)

    def project_latent_grad(self, g: torch.Tensor) -> torch.Tensor:
        """Apply P_latent^perp = I - U_attr U_attr^T to a latent/pixel gradient tensor."""
        if self.u_latent is None:
            return g
        b = g.shape[0]
        flat = g.reshape(b, -1)
        proj = project_nullspace(flat, self.u_latent)
        return proj.view_as(g)

    @torch.no_grad()
    def compute_semantic_drift(
        self,
        f_clean_vit: torch.Tensor,
        f_prot_vit: torch.Tensor,
    ) -> Dict[str, float]:
        """Measure per-axis and mean absolute shift along the 5 CLIP semantic attribute axes."""
        diff = F.normalize(f_prot_vit, dim=-1) - F.normalize(f_clean_vit, dim=-1)  # [B, D]
        proj = torch.einsum("bd,kd->bk", diff, self.text_axes_raw).abs()  # [B, K_text]
        per_axis = {name: float(proj[:, idx].mean().item()) for idx, name in enumerate(self.axis_names)}
        per_axis["mean_5axis_drift"] = float(proj.mean().item())
        return per_axis

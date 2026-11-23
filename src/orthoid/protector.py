"""Core OrthoID protection optimizer with clean ablation switches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .align import DifferentiableArcFaceAligner
from .budget import MidBandFilter, PublishableBudgetProjector, di_transform
from .mask import FaceMaskBuilder
from .models import LPIPSMetric, OpenCLIPViTEncoder, SDVAEResidualDecoder
from .optim import mgda_two_task
from .subspace import (
    DEFAULT_ATTRIBUTE_PAIRS,
    AttributeSubspace,
    _orthonormalize_rows,
    build_photometric_basis,
    project_nullspace,
    pythagorean_counterfactual_target,
)


@dataclass
class OrthoIDConfig:
    """Configuration & component ablation switches for OrthoID."""

    # Optimization steps & step sizes
    steps: int = 30
    lr_latent: float = 0.16
    lr_mid: float = 1.6 / 255.0
    lr_pixel: float = 1.6 / 255.0
    momentum: float = 0.85
    use_adam_latent: bool = True

    # Publishable 8-bit pixel budget Pi_x
    eps: float = 8.0 / 255.0
    rms_max: float = 5.5 / 255.0
    # Keep continuous projection inside optimization to avoid uint8 dead-zone at dz=0;
    # exact 8-bit quantization is always enforced on export via projector.publish().
    use_ste_budget: bool = False

    # Core OrthoID switches (toggle for Block 3 ablations)
    use_vae_residual: bool = True
    use_midband_residual: bool = True
    use_face_mask: bool = True
    use_pythagorean_target: bool = True
    use_latent_pullback: bool = True
    use_mgda: bool = True
    use_flatness_lookahead: bool = True
    use_di_eot: bool = True
    use_lpips_retraction: bool = True

    # Mid-frequency DoG band-pass parameters (4-12 px CNN-FR edge band, zero DC, zero Nyquist grain)
    mid_sigma_hi: float = 1.25
    mid_sigma_lo: float = 5.0
    mid_eps: float = 6.0 / 255.0

    # Mode: "targeted" or "target_free"
    target_mode: str = "targeted"

    # Loss & flatness hyperparameters (RMS-scaled for resolution invariance)
    flatness_rho_latent: float = 0.04
    flatness_rho_mid: float = 1.2 / 255.0
    flatness_rho_pixel: float = 1.5 / 255.0
    mgda_cnn_weight: float = 0.62
    dodge_weight: float = 0.75
    attr_lock_weight: float = 0.25
    lpips_tau: float = 0.125
    lpips_retract_step: float = 0.15
    di_prob: float = 0.7
    align_eot_views: int = 4
    seed: int = 20261121


class OrthoIDProtector:
    """
    Standalone OrthoID protector supporting both:
      - Zero-Init Face-Localized SD-VAE Latent-Residual optimization (default)
      - Pixel-space optimization fallback (for controlled ablations & PGD baselines)
    """

    def __init__(
        self,
        cnn_encoders: Sequence[Callable[[torch.Tensor], torch.Tensor]],
        clip_encoder: OpenCLIPViTEncoder,
        vae_decoder: Optional[SDVAEResidualDecoder] = None,
        lpips_metric: Optional[LPIPSMetric] = None,
        mask_builder: Optional[FaceMaskBuilder] = None,
        config: Optional[OrthoIDConfig] = None,
    ):
        self.cnn_encoders = list(cnn_encoders)
        self.clip_encoder = clip_encoder
        self.vae_decoder = vae_decoder
        self.lpips_metric = lpips_metric
        self.mask_builder = mask_builder if mask_builder is not None else FaceMaskBuilder(use_mtcnn=True)
        self.aligner = DifferentiableArcFaceAligner()
        self.cfg = config if config is not None else OrthoIDConfig()
        self.projector = PublishableBudgetProjector(eps=self.cfg.eps, rms_max=self.cfg.rms_max)
        self.mid_filter = MidBandFilter(
            sigma_hi=self.cfg.mid_sigma_hi,
            sigma_lo=self.cfg.mid_sigma_lo,
        )

        # Precompute 5 zero-shot CLIP semantic attribute axes once (cached on device)
        self.axis_names, self.text_axes_raw = AttributeSubspace.build_text_axes(
            self.clip_encoder.encode_text,
            DEFAULT_ATTRIBUTE_PAIRS,
        )

    def _build_subspace(
        self,
        x_clean: torch.Tensor,
        decode_fn: Callable,
        param_shape: tuple[int, ...],
        mid_shape: Optional[tuple[int, ...]] = None,
    ) -> tuple[AttributeSubspace, Optional[torch.Tensor]]:
        b = x_clean.shape[0]
        device = x_clean.device

        # 1. CLIP semantic text axes + photometric probes -> orthonormal Q_vit
        q_photo_vit = build_photometric_basis(self.clip_encoder.encode_image, x_clean)
        q_text_batch = self.text_axes_raw.unsqueeze(0).expand(b, -1, -1)  # [B, 5, D_vit]
        q_vit = _orthonormalize_rows(torch.cat([q_text_batch, q_photo_vit], dim=1))

        # 2. Photometric/alignment bases for each CNN-FR surrogate
        q_cnn_list = [build_photometric_basis(enc, x_clean) for enc in self.cnn_encoders]

        # 3. Single t=0 Jacobian pullback of the 5 semantic CLIP axes into optimization space(s)
        u_latent = None
        u_mid = None
        if self.cfg.use_latent_pullback:
            q_pull = _orthonormalize_rows(q_text_batch)
            k_pull = q_pull.shape[1]
            p0 = torch.zeros(param_shape, device=device, dtype=torch.float32, requires_grad=True)
            if mid_shape is not None:
                m0 = torch.zeros(mid_shape, device=device, dtype=torch.float32, requires_grad=True)
                x_prime = decode_fn(p0, m0)
                f_vit = self.clip_encoder.encode_image(x_prime)
                jac_z, jac_m = [], []
                for k in range(k_pull):
                    coord = (f_vit * q_pull[:, k, :]).sum(dim=-1)
                    gz, gm = torch.autograd.grad(
                        coord.sum(),
                        (p0, m0),
                        retain_graph=(k < k_pull - 1),
                        create_graph=False,
                    )
                    jac_z.append(gz.detach().reshape(b, -1))
                    jac_m.append(gm.detach().reshape(b, -1))
                u_latent = _orthonormalize_rows(torch.stack(jac_z, dim=1))
                u_mid = _orthonormalize_rows(torch.stack(jac_m, dim=1))
            else:
                x_prime = decode_fn(p0, None)
                f_vit = self.clip_encoder.encode_image(x_prime)
                jac_z = []
                for k in range(k_pull):
                    coord = (f_vit * q_pull[:, k, :]).sum(dim=-1)
                    gz = torch.autograd.grad(
                        coord.sum(),
                        p0,
                        retain_graph=(k < k_pull - 1),
                        create_graph=False,
                    )[0]
                    jac_z.append(gz.detach().reshape(b, -1))
                u_latent = _orthonormalize_rows(torch.stack(jac_z, dim=1))

        sub = AttributeSubspace(
            axis_names=self.axis_names,
            text_axes_raw=self.text_axes_raw,
            q_vit=q_vit,
            q_cnn_list=q_cnn_list,
            u_latent=u_latent,
        )
        return sub, u_mid

    def _compute_targets(
        self,
        x_clean: torch.Tensor,
        x_target: Optional[torch.Tensor],
        subspace: AttributeSubspace,
        thetas_src: Optional[torch.Tensor] = None,
        thetas_tar: Optional[torch.Tensor] = None,
    ) -> tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Compute clean source embeddings and counterfactual targets across CNN-FR and CLIP-ViT."""
        use_tar = (self.cfg.target_mode == "targeted") and (x_target is not None)
        with torch.no_grad():
            for enc in self.cnn_encoders:
                if hasattr(enc, "set_alignment") and thetas_src is not None:
                    enc.set_alignment(thetas_src, eot_views=1)
            f_src_cnns = [enc(x_clean) for enc in self.cnn_encoders]
            f_src_vit = self.clip_encoder.encode_image(x_clean)

            if use_tar:
                for enc in self.cnn_encoders:
                    if hasattr(enc, "set_alignment") and thetas_tar is not None:
                        enc.set_alignment(thetas_tar, eot_views=1)
                e_tar_cnns = [enc(x_target) for enc in self.cnn_encoders]
                e_tar_vit = self.clip_encoder.encode_image(x_target)
            else:
                e_tar_cnns = [None for _ in self.cnn_encoders]
                e_tar_vit = None

            for enc in self.cnn_encoders:
                if hasattr(enc, "set_alignment") and thetas_src is not None:
                    enc.set_alignment(thetas_src, eot_views=self.cfg.align_eot_views)

            if self.cfg.use_pythagorean_target:
                tar_cnns = [
                    pythagorean_counterfactual_target(f_s, e_t, q_c)
                    for f_s, e_t, q_c in zip(f_src_cnns, e_tar_cnns, subspace.q_cnn_list)
                ]
                tar_vit = pythagorean_counterfactual_target(f_src_vit, e_tar_vit, subspace.q_vit)
            else:
                # Baseline / ablation: naive raw target or antipodal dodging without Pythagorean grafting
                tar_cnns = [
                    F.normalize(e_t, dim=-1) if e_t is not None else -F.normalize(f_s, dim=-1)
                    for f_s, e_t in zip(f_src_cnns, e_tar_cnns)
                ]
                tar_vit = (
                    F.normalize(e_tar_vit, dim=-1)
                    if e_tar_vit is not None
                    else -F.normalize(f_src_vit, dim=-1)
                )

        return f_src_cnns, tar_cnns, f_src_vit, tar_vit

    def _cnn_loss(
        self,
        x_eval_raw: torch.Tensor,
        x_eval_eot: torch.Tensor,
        f_src_cnns: List[torch.Tensor],
        tar_cnns: List[torch.Tensor],
    ) -> torch.Tensor:
        losses = []
        for enc, f_s, t_c in zip(self.cnn_encoders, f_src_cnns, tar_cnns):
            x_in = x_eval_raw if hasattr(enc, "set_alignment") else x_eval_eot
            f_cur = enc(x_in)
            # Pull toward counterfactual target + push away from source identity
            l_pull = (1.0 - (f_cur * t_c).sum(dim=-1)).mean()
            l_dodge = (f_cur * f_s).sum(dim=-1).mean()
            losses.append(l_pull + self.cfg.dodge_weight * l_dodge)
        return sum(losses) / max(len(losses), 1)

    def _vit_loss(
        self,
        x_eval: torch.Tensor,
        f_src_vit: torch.Tensor,
        tar_vit: torch.Tensor,
        subspace: AttributeSubspace,
    ) -> torch.Tensor:
        f_cur = self.clip_encoder.encode_image(x_eval)
        l_pull = (1.0 - (f_cur * tar_vit).sum(dim=-1)).mean()
        l_dodge = (f_cur * f_src_vit).sum(dim=-1).mean()

        # Attribute preservation lock along the 5 zero-shot CLIP semantic axes
        diff = f_cur - f_src_vit
        attr_proj = torch.einsum("bd,kd->bk", diff, subspace.text_axes_raw)
        l_attr = attr_proj.pow(2).mean()
        return l_pull + self.cfg.dodge_weight * l_dodge + self.cfg.attr_lock_weight * l_attr

    def protect(
        self,
        x_clean: torch.Tensor,
        x_target: Optional[torch.Tensor] = None,
        face_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Optimize protected portraits for batch x_clean [B, 3, H, W] in [0, 1].
        Returns publishable 8-bit x_prot strictly inside Pi_x plus rich diagnostics.
        """
        t0 = time.time()
        device = x_clean.device
        b = x_clean.shape[0]
        x_clean = x_clean.clamp(0.0, 1.0).detach()

        if face_mask is None:
            if self.cfg.use_face_mask:
                face_mask = self.mask_builder.build(x_clean, device=device)
            else:
                face_mask = torch.ones((b, 1, x_clean.shape[2], x_clean.shape[3]), device=device)
        else:
            face_mask = face_mask.to(device=device, dtype=torch.float32)

        use_vae = self.cfg.use_vae_residual and (self.vae_decoder is not None)
        use_mid = use_vae and self.cfg.use_midband_residual
        self.mid_filter = self.mid_filter.to(device=device)

        if use_vae:
            z0, x_hat0 = self.vae_decoder.encode_clean(x_clean)
            param_shape = tuple(z0.shape)
            mid_shape = tuple(x_clean.shape) if use_mid else None
            lr = self.cfg.lr_latent
            rho_flat = self.cfg.flatness_rho_latent

            def decode_fn(p: torch.Tensor, m: Optional[torch.Tensor]) -> torch.Tensor:
                z_cur = (z0 + p).to(dtype=self.vae_decoder.dtype)
                x_hat = ((self.vae_decoder.vae.decode(z_cur).sample.float() + 1.0) * 0.5).clamp(0.0, 1.0)
                delta_total = x_hat - x_hat0
                if m is not None:
                    delta_total = delta_total + self.mid_filter(m)
                x_raw = (x_clean + face_mask * delta_total).clamp(0.0, 1.0)
                if self.cfg.use_ste_budget:
                    return self.projector.project_ste(x_raw, x_clean)
                return self.projector.project_continuous(x_raw, x_clean)

        else:
            param_shape = tuple(x_clean.shape)
            mid_shape = None
            lr = self.cfg.lr_pixel
            rho_flat = self.cfg.flatness_rho_pixel

            def decode_fn(p: torch.Tensor, m: Optional[torch.Tensor]) -> torch.Tensor:
                x_raw = (x_clean + face_mask * p).clamp(0.0, 1.0)
                if self.cfg.use_ste_budget:
                    return self.projector.project_ste(x_raw, x_clean)
                return self.projector.project_continuous(x_raw, x_clean)

        # Compute 5-point canonical ArcFace alignments once if any CNN surrogate supports it
        has_align_enc = any(hasattr(enc, "set_alignment") for enc in self.cnn_encoders)
        thetas_src = self.aligner.compute_thetas(x_clean, device=device) if has_align_enc else None
        thetas_tar = (
            self.aligner.compute_thetas(x_target, device=device)
            if (has_align_enc and x_target is not None and self.cfg.target_mode == "targeted")
            else None
        )
        if has_align_enc:
            for enc in self.cnn_encoders:
                if hasattr(enc, "set_alignment"):
                    enc.set_alignment(thetas_src, eot_views=1)

        # Build online attribute subspace + single t=0 pullback(s)
        t_sub0 = time.time()
        subspace, u_mid = self._build_subspace(x_clean, decode_fn, param_shape, mid_shape=mid_shape)
        subspace_time_s = time.time() - t_sub0

        f_src_cnns, tar_cnns, f_src_vit, tar_vit = self._compute_targets(
            x_clean, x_target, subspace, thetas_src=thetas_src, thetas_tar=thetas_tar
        )

        param = torch.zeros(param_shape, device=device, dtype=torch.float32)
        velocity = torch.zeros_like(param)
        adam_m = torch.zeros_like(param)
        adam_v = torch.zeros_like(param)

        mid_param = torch.zeros(mid_shape, device=device, dtype=torch.float32) if use_mid else None
        mid_adam_m = torch.zeros_like(mid_param) if use_mid else None
        mid_adam_v = torch.zeros_like(mid_param) if use_mid else None
        beta1, beta2 = 0.9, 0.99

        raw_cos_trace: List[float] = []
        proj_cos_trace: List[float] = []
        raw_conflict_hits: List[float] = []
        proj_conflict_hits: List[float] = []
        both_descent_hits: List[float] = []
        gamma_trace: List[float] = []
        lpips_retract_count = 0

        def _rms_norm(t: torch.Tensor) -> torch.Tensor:
            return (t.reshape(b, -1).pow(2).mean(dim=1) + 1e-12).sqrt().view(b, *([1] * (t.ndim - 1)))

        def _proj_mid(g: torch.Tensor) -> torch.Tensor:
            if u_mid is None:
                return g
            return project_nullspace(g.reshape(b, -1), u_mid).view_as(g)

        for step in range(self.cfg.steps):
            step_seed = self.cfg.seed + step * 17

            # Step A: Optional CWA-style sharpness lookahead probe
            if self.cfg.use_flatness_lookahead:
                p_probe = param.detach().requires_grad_(True)
                m_probe = mid_param.detach().requires_grad_(True) if use_mid else None
                x_probe = decode_fn(p_probe, m_probe)
                x_probe_eot = di_transform(x_probe, seed=step_seed, prob=self.cfg.di_prob) if self.cfg.use_di_eot else x_probe
                l_probe = self._cnn_loss(x_probe, x_probe_eot, f_src_cnns, tar_cnns) + self._vit_loss(
                    x_probe_eot, f_src_vit, tar_vit, subspace
                )
                if use_mid:
                    gp_z, gp_m = torch.autograd.grad(l_probe, (p_probe, m_probe))
                    p_eval = (param.detach() + rho_flat * (gp_z / _rms_norm(gp_z))).detach().requires_grad_(True)
                    m_eval = (
                        mid_param.detach() + self.cfg.flatness_rho_mid * (gp_m / _rms_norm(gp_m))
                    ).detach().requires_grad_(True)
                else:
                    gp_z = torch.autograd.grad(l_probe, p_probe)[0]
                    p_eval = (param.detach() + rho_flat * (gp_z / _rms_norm(gp_z))).detach().requires_grad_(True)
                    m_eval = None
            else:
                p_eval = param.detach().requires_grad_(True)
                m_eval = mid_param.detach().requires_grad_(True) if use_mid else None

            # Step B: Compute separate CNN-FR and ViT-CLIP gradients at (p_eval, m_eval)
            x_eval = decode_fn(p_eval, m_eval)
            x_eval_eot = (
                di_transform(x_eval, seed=step_seed + 1, prob=self.cfg.di_prob) if self.cfg.use_di_eot else x_eval
            )
            l_cnn = self._cnn_loss(x_eval, x_eval_eot, f_src_cnns, tar_cnns)
            l_vit = self._vit_loss(x_eval_eot, f_src_vit, tar_vit, subspace)

            if use_mid:
                g_cnn, g_cnn_m = torch.autograd.grad(l_cnn, (p_eval, m_eval), retain_graph=True)
                g_vit, g_vit_m = torch.autograd.grad(l_vit, (p_eval, m_eval), retain_graph=False)
                g_cnn, g_cnn_m = g_cnn.detach(), g_cnn_m.detach()
                g_vit, g_vit_m = g_vit.detach(), g_vit_m.detach()
            else:
                g_cnn = torch.autograd.grad(l_cnn, p_eval, retain_graph=True)[0].detach()
                g_vit = torch.autograd.grad(l_vit, p_eval, retain_graph=False)[0].detach()
                g_cnn_m = g_vit_m = None

            # Diagnostics BEFORE attribute-nullspace projection (on primary latent/pixel space)
            f1_raw = g_cnn.reshape(b, -1)
            f2_raw = g_vit.reshape(b, -1)
            raw_cos = F.cosine_similarity(f1_raw, f2_raw, dim=1)
            raw_cos_trace.append(float(raw_cos.mean().item()))
            raw_conflict_hits.append(float((raw_cos < 0.0).float().mean().item()))

            # Step C: Attribute-nullspace projection P^perp on each active band
            g_cnn_proj = subspace.project_latent_grad(g_cnn)
            g_vit_proj = subspace.project_latent_grad(g_vit)
            if use_mid:
                g_cnn_m_proj = _proj_mid(g_cnn_m)
                g_vit_m_proj = _proj_mid(g_vit_m)

            # Step D: Conflict-free 2-task MGDA (or naive raw sum for ablation)
            if self.cfg.use_mgda:
                direction, mgda_stats = mgda_two_task(
                    g_cnn_proj,
                    g_vit_proj,
                    normalize_scales=True,
                    cnn_weight=self.cfg.mgda_cnn_weight,
                )
                proj_cos_trace.append(float(mgda_stats.cos_sim.mean().item()))
                proj_conflict_hits.append(float(mgda_stats.conflict_mask.float().mean().item()))
                gamma_trace.append(float(mgda_stats.gamma_star.mean().item()))
                if use_mid:
                    # Mid-frequency edge band specializes more strongly on CNN-FR margins while still conflict-free w.r.t. ViT
                    dir_mid, _ = mgda_two_task(
                        g_cnn_m_proj,
                        g_vit_m_proj,
                        normalize_scales=True,
                        cnn_weight=min(0.80, self.cfg.mgda_cnn_weight + 0.13),
                    )
            else:
                f1 = g_cnn_proj.reshape(b, -1)
                f2 = g_vit_proj.reshape(b, -1)
                p_cos = F.cosine_similarity(f1, f2, dim=1)
                proj_cos_trace.append(float(p_cos.mean().item()))
                proj_conflict_hits.append(float((p_cos < 0.0).float().mean().item()))
                gamma_trace.append(0.5)
                comb = f1 + f2
                direction = (comb / ((comb.pow(2).sum(dim=1, keepdim=True) + 1e-12).sqrt())).view_as(param)
                if use_mid:
                    comb_m = g_cnn_m_proj.reshape(b, -1) + g_vit_m_proj.reshape(b, -1)
                    dir_mid = (comb_m / ((comb_m.pow(2).sum(dim=1, keepdim=True) + 1e-12).sqrt())).view_as(mid_param)

            # Verify simultaneous positive descent on both paradigms (Prop 2 check)
            d_flat = direction.reshape(b, -1)
            dot_cnn = (d_flat * g_cnn_proj.reshape(b, -1)).sum(dim=1)
            dot_vit = (d_flat * g_vit_proj.reshape(b, -1)).sum(dim=1)
            both_descent_hits.append(float(((dot_cnn > 0.0) & (dot_vit > 0.0)).float().mean().item()))

            # Continuous descent in latent (Adam) / mid-band (Adam) / pixel (momentum) — NO .sign() quantization!
            cos_sched = 0.5 * (1.0 + np.cos(np.pi * step / max(self.cfg.steps, 1))) * 0.7 + 0.3
            step_scale = lr * cos_sched
            if use_vae and self.cfg.use_adam_latent:
                t_idx = step + 1
                adam_m = beta1 * adam_m + (1.0 - beta1) * direction
                adam_v = beta2 * adam_v + (1.0 - beta2) * (direction * direction)
                m_hat = adam_m / (1.0 - beta1 ** t_idx)
                v_hat = adam_v / (1.0 - beta2 ** t_idx)
                upd = m_hat / (v_hat.sqrt() + 1e-8)
                param = (param - step_scale * (upd / _rms_norm(upd))).detach()
                if use_mid:
                    mid_adam_m = beta1 * mid_adam_m + (1.0 - beta1) * dir_mid
                    mid_adam_v = beta2 * mid_adam_v + (1.0 - beta2) * (dir_mid * dir_mid)
                    mm_hat = mid_adam_m / (1.0 - beta1 ** t_idx)
                    mv_hat = mid_adam_v / (1.0 - beta2 ** t_idx)
                    upd_m = mm_hat / (mv_hat.sqrt() + 1e-8)
                    mid_step = self.cfg.lr_mid * cos_sched
                    mid_param = (mid_param - mid_step * (upd_m / _rms_norm(upd_m))).clamp(
                        -self.cfg.mid_eps, self.cfg.mid_eps
                    ).detach()
            else:
                velocity = self.cfg.momentum * velocity + direction
                param = (param - step_scale * (velocity / _rms_norm(velocity))).detach()
                if use_mid:
                    mid_param = (mid_param - (self.cfg.lr_mid * cos_sched) * (dir_mid / _rms_norm(dir_mid))).clamp(
                        -self.cfg.mid_eps, self.cfg.mid_eps
                    ).detach()

            # Step E: Active-set LPIPS perceptual retraction (only backprop when tau is actually exceeded)
            if self.cfg.use_lpips_retraction and self.lpips_metric is not None and (step + 1) % 2 == 0:
                with torch.no_grad():
                    x_chk = decode_fn(param, mid_param)
                    lp_chk = self.lpips_metric(x_chk, x_clean)
                if float(lp_chk.max().item()) > self.cfg.lpips_tau:
                    lpips_retract_count += 1
                    p_lp = param.detach().requires_grad_(True)
                    m_lp = mid_param.detach().requires_grad_(True) if use_mid else None
                    x_lp = decode_fn(p_lp, m_lp)
                    lp_vals = self.lpips_metric(x_lp, x_clean)
                    excess = F.relu(lp_vals - self.cfg.lpips_tau)
                    active_z = (excess > 0.0).float().view(b, *([1] * (param.ndim - 1)))
                    if use_mid:
                        g_lp_z, g_lp_m = torch.autograd.grad(excess.sum(), (p_lp, m_lp))
                        param = (param - active_z * self.cfg.lpips_retract_step * lr * (g_lp_z.detach() / _rms_norm(g_lp_z.detach()))).detach()
                        active_m = (excess > 0.0).float().view(b, 1, 1, 1)
                        mid_param = (
                            mid_param - active_m * self.cfg.lpips_retract_step * self.cfg.lr_mid * (g_lp_m.detach() / _rms_norm(g_lp_m.detach()))
                        ).clamp(-self.cfg.mid_eps, self.cfg.mid_eps).detach()
                    else:
                        g_lp_z = torch.autograd.grad(excess.sum(), p_lp)[0].detach()
                        param = (param - active_z * self.cfg.lpips_retract_step * lr * (g_lp_z / _rms_norm(g_lp_z))).detach()

            # Keep pixel-space parameter within L_inf box
            if not use_vae:
                param = param.clamp(-self.cfg.eps, self.cfg.eps)

        with torch.no_grad():
            if use_vae:
                z_cur = (z0 + param).to(dtype=self.vae_decoder.dtype)
                x_hat = ((self.vae_decoder.vae.decode(z_cur).sample.float() + 1.0) * 0.5).clamp(0.0, 1.0)
                delta_total = x_hat - x_hat0
                if mid_param is not None:
                    delta_total = delta_total + self.mid_filter(mid_param)
                x_raw_final = (x_clean + face_mask * delta_total).clamp(0.0, 1.0)
            else:
                x_raw_final = (x_clean + face_mask * param).clamp(0.0, 1.0)
            x_prot_cont = self.projector.project_continuous(x_raw_final, x_clean)
            x_prot = self.projector.publish(x_raw_final, x_clean)

            f_prot_vit = self.clip_encoder.encode_image(x_prot)
            attr_drift = subspace.compute_semantic_drift(f_src_vit, f_prot_vit)
            lpips_final = (
                float(self.lpips_metric(x_prot, x_clean).mean().item())
                if self.lpips_metric is not None
                else None
            )

        if has_align_enc:
            for enc in self.cnn_encoders:
                if hasattr(enc, "set_alignment"):
                    enc.set_alignment(None)

        elapsed = time.time() - t0
        diagnostics = {
            "elapsed_total_s": float(elapsed),
            "elapsed_per_image_s": float(elapsed / max(b, 1)),
            "subspace_init_s": float(subspace_time_s),
            "raw_grad_cos_mean": float(np.mean(raw_cos_trace)),
            "proj_grad_cos_mean": float(np.mean(proj_cos_trace)),
            "raw_conflict_rate_pct": float(100.0 * np.mean(raw_conflict_hits)),
            "proj_conflict_rate_pct": float(100.0 * np.mean(proj_conflict_hits)),
            "both_paradigm_descent_pct": float(100.0 * np.mean(both_descent_hits)),
            "mean_gamma_star": float(np.mean(gamma_trace)),
            "lpips_retract_steps": int(lpips_retract_count),
            "lpips": lpips_final,
            "attr_drift": attr_drift,
            "config": asdict(self.cfg),
        }

        return {
            "x_prot": x_prot,
            "x_prot_cont": x_prot_cont,
            "face_mask": face_mask,
            "subspace": subspace,
            "diagnostics": diagnostics,
        }

    def protect_batched(
        self,
        x_clean: torch.Tensor,
        x_target: Optional[torch.Tensor] = None,
        face_mask: Optional[torch.Tensor] = None,
        microbatch: int = 2,
    ) -> Dict[str, Any]:
        """Memory-safe micro-batched wrapper around protect() for MPS and GPU efficiency."""
        n = x_clean.shape[0]
        if n <= microbatch:
            return self.protect(x_clean, x_target=x_target, face_mask=face_mask)

        x_prot_chunks = []
        x_cont_chunks = []
        mask_chunks = []
        diags = []
        last_subspace = None

        for i in range(0, n, microbatch):
            xc = x_clean[i : i + microbatch]
            xt = x_target[i : i + microbatch] if x_target is not None else None
            fm = face_mask[i : i + microbatch] if face_mask is not None else None
            out = self.protect(xc, x_target=xt, face_mask=fm)
            x_prot_chunks.append(out["x_prot"])
            x_cont_chunks.append(out["x_prot_cont"])
            mask_chunks.append(out["face_mask"])
            diags.append(out["diagnostics"])
            last_subspace = out["subspace"]
            gc.collect()
            if x_clean.device.type == "mps":
                torch.mps.empty_cache()
            elif x_clean.device.type == "cuda":
                torch.cuda.empty_cache()

        weights = [min(microbatch, n - i) for i in range(0, n, microbatch)]
        tot_w = float(sum(weights))

        def _wavg(key: str) -> float:
            return float(sum(w * d[key] for w, d in zip(weights, diags)) / tot_w)

        merged_attr = {}
        for ax in diags[0]["attr_drift"]:
            merged_attr[ax] = float(sum(w * d["attr_drift"][ax] for w, d in zip(weights, diags)) / tot_w)

        merged_diag = {
            "elapsed_total_s": float(sum(d["elapsed_total_s"] for d in diags)),
            "elapsed_per_image_s": float(sum(d["elapsed_total_s"] for d in diags) / tot_w),
            "subspace_init_s": float(sum(d["subspace_init_s"] for d in diags)),
            "raw_grad_cos_mean": _wavg("raw_grad_cos_mean"),
            "proj_grad_cos_mean": _wavg("proj_grad_cos_mean"),
            "raw_conflict_rate_pct": _wavg("raw_conflict_rate_pct"),
            "proj_conflict_rate_pct": _wavg("proj_conflict_rate_pct"),
            "both_paradigm_descent_pct": _wavg("both_paradigm_descent_pct"),
            "mean_gamma_star": _wavg("mean_gamma_star"),
            "lpips_retract_steps": int(sum(d["lpips_retract_steps"] for d in diags)),
            "lpips": _wavg("lpips") if diags[0]["lpips"] is not None else None,
            "attr_drift": merged_attr,
            "config": diags[0]["config"],
        }

        return {
            "x_prot": torch.cat(x_prot_chunks, dim=0),
            "x_prot_cont": torch.cat(x_cont_chunks, dim=0),
            "face_mask": torch.cat(mask_chunks, dim=0),
            "subspace": last_subspace,
            "diagnostics": merged_diag,
        }

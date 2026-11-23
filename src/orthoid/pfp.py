"""
Purification Fixed-Point Orthogonal Identity Protection (Scheme 1 Core Engine).

Single unified principle (zero bag-of-tricks):
  Optimize a face-localized perturbation delta inside the publishable 8-bit budget Pi_x
  (L_inf <= 8/255, RMS <= 5.5/255) that is a *paired local fixed point* of a frozen
  diffusion purifier P_t:
      L_FP(delta) = E_{t, eps} [ || (P_t(x + delta; t, eps) - P_t(x; t, eps)) - delta ||_2^2 ]
  while erasing attribute-orthogonal identity (Pythagorean counterfactual target +
  conflict-free 2-task CNN/ViT MGDA) on BOTH the direct image x' = x + delta AND
  the 1-step Tweedie-purified image P_t(x'; t, eps) with exact backprop through P_t.

Key mathematical properties:
  1. By 1st-order Taylor expansion, L_FP(delta) ~= ||(J_{P_t}(x) - I) delta||_2^2, driving
     delta into the unit-eigenvalue fixed-point subspace of P_t (surviving multi-step
     DiffPure / IMPRESS / JPEG while eliminating off-manifold visual noise naturally).
  2. Exact backprop through 1-step Tweedie P_t yields Jacobian
     d P_t / d x' = I - sqrt(1 - alpha_bar_t) J_{eps_theta}(x'_t), which automatically
     projects identity gradients onto the purification-surviving natural face manifold.
  3. Operating in full pixel resolution (no 8x VAE spatial bottleneck) preserves sharp
     transfer to ArcFace/InsightFace (buffalo_l, antelopev2) AND ViT (ViT-L/14, EVA02-L).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .align import DifferentiableArcFaceAligner
from .budget import PublishableBudgetProjector, di_transform
from .mask import FaceMaskBuilder
from .models import LPIPSMetric, OpenCLIPViTEncoder
from .optim import mgda_two_task
from .subspace import (
    DEFAULT_ATTRIBUTE_PAIRS,
    AttributeSubspace,
    _orthonormalize_rows,
    build_photometric_basis,
    pythagorean_counterfactual_target,
)


class TweediePurifierOperator(nn.Module):
    """
    Differentiable 1-step Tweedie Denoising Operator P_t(x; t, eps) backed by a frozen
    pixel-space diffusion UNet (google/ddpm-celebahq-256).

    Supports native bfloat16 on SM>=8 GPUs (A100/L4/H100/4090) for fast, underflow-free
    exact backpropagation through the UNet Jacobian.
    """

    def __init__(
        self,
        model_id: str = "google/ddpm-celebahq-256",
        device: str = "cpu",
        local_files_only: bool = False,
    ):
        super().__init__()
        from diffusers import DDIMScheduler, UNet2DModel

        self.device = torch.device(device)
        if str(device).startswith("cuda") and torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            self.dtype = torch.bfloat16 if (major >= 8 and torch.cuda.is_bf16_supported()) else torch.float32
        else:
            self.dtype = torch.float32

        self.unet = UNet2DModel.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            local_files_only=local_files_only,
        ).eval().to(self.device)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        sched = DDIMScheduler.from_pretrained(model_id, local_files_only=local_files_only)
        self.register_buffer("alphas_cumprod", sched.alphas_cumprod.float().to(self.device))

    def tweedie_denoise(
        self,
        x01: torch.Tensor,
        t_int: int,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Differentiable 1-step Tweedie posterior mean x_hat_0 in [0, 1] at timestep t_int
        using shared Gaussian noise `noise` (exact gradient flows through UNet).
        """
        a_bar = float(self.alphas_cumprod[t_int].item())
        s_a = math.sqrt(a_bar)
        s_1ma = math.sqrt(1.0 - a_bar)

        x_m11 = x01 * 2.0 - 1.0
        x_t = (s_a * x_m11 + s_1ma * noise).to(dtype=self.dtype)
        eps_pred = self.unet(x_t, int(t_int)).sample.float()
        x0_m11 = ((x_t.float() - s_1ma * eps_pred) / s_a).clamp(-1.0, 1.0)
        return (x0_m11 + 1.0) * 0.5


@dataclass
class PFPConfig:
    """Clean, minimal configuration for Purification Fixed-Point Orthogonal Protection."""

    # Optimization
    steps: int = 40
    lr: float = 1.4 / 255.0
    use_adam: bool = True

    # Publishable 8-bit pixel budget Pi_x
    eps: float = 8.0 / 255.0
    rms_max: float = 5.5 / 255.0
    use_face_mask: bool = True

    # Core Purification Fixed-Point switches (for clean ablations)
    use_purification_in_loop: bool = True
    use_fixed_point_loss: bool = True
    use_pythagorean_target: bool = True
    use_mgda: bool = True
    use_di_eot: bool = True

    # Mode: "target_free" (default for proactive privacy) or "targeted"
    target_mode: str = "target_free"

    # Core hyperparameters (clean & minimal)
    t_min: int = 35
    t_max: int = 110
    purified_id_weight: float = 1.0
    direct_id_weight: float = 0.65
    fixed_point_weight: float = 12.0
    attr_lock_weight: float = 0.5
    dodge_weight: float = 0.65
    mgda_cnn_weight: float = 0.50
    di_prob: float = 0.6
    align_eot_views: int = 4
    seed: int = 20261123


class PurificationFixedPointProtector:
    """
    Clean Purification Fixed-Point Orthogonal Identity Protector (Scheme 1).
    Supports balanced multi-CNN + multi-ViT zero-overlap surrogate ensembles.
    """

    def __init__(
        self,
        cnn_encoders: Sequence[Callable[[torch.Tensor], torch.Tensor]],
        vit_encoders: Sequence[OpenCLIPViTEncoder],
        tweedie_op: Optional[TweediePurifierOperator] = None,
        lpips_metric: Optional[LPIPSMetric] = None,
        mask_builder: Optional[FaceMaskBuilder] = None,
        config: Optional[PFPConfig] = None,
    ):
        assert len(cnn_encoders) >= 1 and len(vit_encoders) >= 1
        self.cnn_encoders = list(cnn_encoders)
        self.vit_encoders = list(vit_encoders)
        self.primary_clip = self.vit_encoders[0]  # Used for 5-axis zero-shot text attribute basis
        self.tweedie_op = tweedie_op
        self.lpips_metric = lpips_metric
        self.mask_builder = mask_builder if mask_builder is not None else FaceMaskBuilder(use_mtcnn=True)
        self.aligner = DifferentiableArcFaceAligner()
        self.cfg = config if config is not None else PFPConfig()
        self.projector = PublishableBudgetProjector(eps=self.cfg.eps, rms_max=self.cfg.rms_max)

        # Precompute 5 zero-shot CLIP semantic attribute axes once
        self.axis_names, self.text_axes_raw = AttributeSubspace.build_text_axes(
            self.primary_clip.encode_text,
            DEFAULT_ATTRIBUTE_PAIRS,
        )

    @torch.no_grad()
    def _build_targets_and_subspace(
        self,
        x_clean: torch.Tensor,
        x_target: Optional[torch.Tensor],
        thetas_src: Optional[torch.Tensor],
        thetas_tar: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        b = x_clean.shape[0]
        use_tar = (self.cfg.target_mode == "targeted") and (x_target is not None)

        # Bind clean alignments (eot_views=1) for deterministic anchor extraction
        for enc in self.cnn_encoders:
            if hasattr(enc, "set_alignment") and thetas_src is not None:
                enc.set_alignment(thetas_src, eot_views=1)

        # Photometric + semantic bases per encoder
        q_cnn_list = [build_photometric_basis(enc, x_clean) for enc in self.cnn_encoders]
        q_vit_list = []
        q_text_batch = self.text_axes_raw.unsqueeze(0).expand(b, -1, -1)
        for idx, venc in enumerate(self.vit_encoders):
            q_photo = build_photometric_basis(venc.encode_image, x_clean)
            if idx == 0:
                q_vit_list.append(_orthonormalize_rows(torch.cat([q_text_batch, q_photo], dim=1)))
            else:
                q_vit_list.append(q_photo)

        f_src_cnns = [enc(x_clean) for enc in self.cnn_encoders]
        f_src_vits = [venc.encode_image(x_clean) for venc in self.vit_encoders]

        if use_tar:
            for enc in self.cnn_encoders:
                if hasattr(enc, "set_alignment") and thetas_tar is not None:
                    enc.set_alignment(thetas_tar, eot_views=1)
            e_tar_cnns = [enc(x_target) for enc in self.cnn_encoders]
            e_tar_vits = [venc.encode_image(x_target) for venc in self.vit_encoders]
        else:
            e_tar_cnns = [None for _ in self.cnn_encoders]
            e_tar_vits = [None for _ in self.vit_encoders]

        # Re-bind source alignments with multi-view EOT for optimization
        for enc in self.cnn_encoders:
            if hasattr(enc, "set_alignment") and thetas_src is not None:
                enc.set_alignment(thetas_src, eot_views=self.cfg.align_eot_views)

        if self.cfg.use_pythagorean_target:
            tar_cnns = [
                pythagorean_counterfactual_target(fs, et, qc)
                for fs, et, qc in zip(f_src_cnns, e_tar_cnns, q_cnn_list)
            ]
            tar_vits = [
                pythagorean_counterfactual_target(fs, et, qv)
                for fs, et, qv in zip(f_src_vits, e_tar_vits, q_vit_list)
            ]
        else:
            tar_cnns = [
                F.normalize(et, dim=-1) if et is not None else -F.normalize(fs, dim=-1)
                for fs, et in zip(f_src_cnns, e_tar_cnns)
            ]
            tar_vits = [
                F.normalize(et, dim=-1) if et is not None else -F.normalize(fs, dim=-1)
                for fs, et in zip(f_src_vits, e_tar_vits)
            ]

        subspace = AttributeSubspace(
            axis_names=self.axis_names,
            text_axes_raw=self.text_axes_raw,
            q_vit=q_vit_list[0],
            q_cnn_list=q_cnn_list,
            u_latent=None,
        )
        return {
            "f_src_cnns": f_src_cnns,
            "tar_cnns": tar_cnns,
            "f_src_vits": f_src_vits,
            "tar_vits": tar_vits,
            "subspace": subspace,
        }

    def _cnn_id_loss(
        self,
        x_raw: torch.Tensor,
        x_eot: torch.Tensor,
        f_src_cnns: List[torch.Tensor],
        tar_cnns: List[torch.Tensor],
    ) -> torch.Tensor:
        losses = []
        for enc, fs, tc in zip(self.cnn_encoders, f_src_cnns, tar_cnns):
            x_in = x_raw if hasattr(enc, "set_alignment") else x_eot
            fc = enc(x_in)
            l_pull = (1.0 - (fc * tc).sum(dim=-1)).mean()
            l_dodge = (fc * fs).sum(dim=-1).mean()
            losses.append(l_pull + self.cfg.dodge_weight * l_dodge)
        return sum(losses) / len(losses)

    def _vit_id_loss(
        self,
        x_eot: torch.Tensor,
        f_src_vits: List[torch.Tensor],
        tar_vits: List[torch.Tensor],
        subspace: AttributeSubspace,
    ) -> torch.Tensor:
        losses = []
        for idx, (venc, fs, tv) in enumerate(zip(self.vit_encoders, f_src_vits, tar_vits)):
            fc = venc.encode_image(x_eot)
            l_pull = (1.0 - (fc * tv).sum(dim=-1)).mean()
            l_dodge = (fc * fs).sum(dim=-1).mean()
            l_tot = l_pull + self.cfg.dodge_weight * l_dodge
            if idx == 0 and self.cfg.attr_lock_weight > 0.0:
                diff = fc - fs
                attr_proj = torch.einsum("bd,kd->bk", diff, subspace.text_axes_raw)
                l_tot = l_tot + self.cfg.attr_lock_weight * attr_proj.pow(2).mean()
            losses.append(l_tot)
        return sum(losses) / len(losses)

    def protect(
        self,
        x_clean: torch.Tensor,
        x_target: Optional[torch.Tensor] = None,
        face_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
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

        has_align = any(hasattr(enc, "set_alignment") for enc in self.cnn_encoders)
        thetas_src = self.aligner.compute_thetas(x_clean, device=device) if has_align else None
        thetas_tar = (
            self.aligner.compute_thetas(x_target, device=device)
            if (has_align and x_target is not None and self.cfg.target_mode == "targeted")
            else None
        )

        pack = self._build_targets_and_subspace(x_clean, x_target, thetas_src, thetas_tar)
        f_src_cnns, tar_cnns = pack["f_src_cnns"], pack["tar_cnns"]
        f_src_vits, tar_vits = pack["f_src_vits"], pack["tar_vits"]
        subspace: AttributeSubspace = pack["subspace"]

        use_pfp = (self.cfg.use_purification_in_loop or self.cfg.use_fixed_point_loss) and (
            self.tweedie_op is not None
        )

        delta = torch.zeros_like(x_clean)
        adam_m = torch.zeros_like(delta)
        adam_v = torch.zeros_like(delta)
        beta1, beta2 = 0.9, 0.99

        raw_cos_trace: List[float] = []
        raw_conflict_hits: List[float] = []
        both_descent_hits: List[float] = []
        fp_residual_trace: List[float] = []

        def _rms_norm(t: torch.Tensor) -> torch.Tensor:
            return (t.reshape(b, -1).pow(2).mean(dim=1) + 1e-12).sqrt().view(b, 1, 1, 1)

        gen = torch.Generator(device="cpu").manual_seed(int(self.cfg.seed))

        for step in range(self.cfg.steps):
            d_req = delta.detach().requires_grad_(True)
            x_raw = (x_clean + face_mask * d_req).clamp(0.0, 1.0)
            x_cur = self.projector.project_continuous(x_raw, x_clean)
            eff_delta = x_cur - x_clean

            x_cur_eot = (
                di_transform(x_cur, seed=self.cfg.seed + step * 19, prob=self.cfg.di_prob)
                if self.cfg.use_di_eot
                else x_cur
            )

            # Direct paradigm losses
            l_cnn = self.cfg.direct_id_weight * self._cnn_id_loss(x_cur, x_cur_eot, f_src_cnns, tar_cnns)
            l_vit = self.cfg.direct_id_weight * self._vit_id_loss(x_cur_eot, f_src_vits, tar_vits, subspace)

            # Paired 1-step Tweedie Purification + Fixed-Point Eigenmode Self-Consistency
            if use_pfp:
                t_int = int(torch.randint(self.cfg.t_min, self.cfg.t_max + 1, (1,), generator=gen).item())
                eps_noise = torch.randn(x_clean.shape, generator=gen, dtype=torch.float32).to(device)
                with torch.no_grad():
                    p_clean = self.tweedie_op.tweedie_denoise(x_clean, t_int, eps_noise)
                p_cur = self.tweedie_op.tweedie_denoise(x_cur, t_int, eps_noise)

                # Paired Fixed-Point Residual: || (P_t(x + delta) - P_t(x)) - delta ||_2^2
                fp_err = (p_cur - p_clean) - eff_delta
                l_fp = fp_err.pow(2).mean()
                fp_residual_trace.append(float(fp_err.pow(2).mean().sqrt().item() * 255.0))

                if self.cfg.use_purification_in_loop:
                    p_cur_eot = (
                        di_transform(p_cur, seed=self.cfg.seed + step * 19 + 7, prob=self.cfg.di_prob)
                        if self.cfg.use_di_eot
                        else p_cur
                    )
                    l_cnn = l_cnn + self.cfg.purified_id_weight * self._cnn_id_loss(
                        p_cur, p_cur_eot, f_src_cnns, tar_cnns
                    )
                    l_vit = l_vit + self.cfg.purified_id_weight * self._vit_id_loss(
                        p_cur_eot, f_src_vits, tar_vits, subspace
                    )

                if self.cfg.use_fixed_point_loss and self.cfg.fixed_point_weight > 0.0:
                    fp_term = self.cfg.fixed_point_weight * l_fp
                    l_cnn = l_cnn + fp_term
                    l_vit = l_vit + fp_term

            g_cnn = torch.autograd.grad(l_cnn, d_req, retain_graph=True)[0].detach()
            g_vit = torch.autograd.grad(l_vit, d_req, retain_graph=False)[0].detach()

            f1 = g_cnn.reshape(b, -1)
            f2 = g_vit.reshape(b, -1)
            rcos = F.cosine_similarity(f1, f2, dim=1)
            raw_cos_trace.append(float(rcos.mean().item()))
            raw_conflict_hits.append(float((rcos < 0.0).float().mean().item()))

            if self.cfg.use_mgda:
                direction, _ = mgda_two_task(
                    g_cnn,
                    g_vit,
                    normalize_scales=True,
                    cnn_weight=self.cfg.mgda_cnn_weight,
                )
            else:
                comb = f1 + f2
                direction = (comb / ((comb.pow(2).sum(dim=1, keepdim=True) + 1e-12).sqrt())).view_as(delta)

            d_flat = direction.reshape(b, -1)
            both_descent_hits.append(
                float((((d_flat * f1).sum(dim=1) > 0.0) & ((d_flat * f2).sum(dim=1) > 0.0)).float().mean().item())
            )

            cos_sched = 0.5 * (1.0 + np.cos(np.pi * step / max(self.cfg.steps, 1))) * 0.75 + 0.25
            step_lr = self.cfg.lr * cos_sched
            if self.cfg.use_adam:
                t_idx = step + 1
                adam_m = beta1 * adam_m + (1.0 - beta1) * direction
                adam_v = beta2 * adam_v + (1.0 - beta2) * (direction * direction)
                m_hat = adam_m / (1.0 - beta1 ** t_idx)
                v_hat = adam_v / (1.0 - beta2 ** t_idx)
                upd = m_hat / (v_hat.sqrt() + 1e-8)
                delta = (delta - step_lr * (upd / _rms_norm(upd))).clamp(-self.cfg.eps, self.cfg.eps).detach()
            else:
                delta = (delta - step_lr * (direction / _rms_norm(direction))).clamp(-self.cfg.eps, self.cfg.eps).detach()

        with torch.no_grad():
            x_raw_final = (x_clean + face_mask * delta).clamp(0.0, 1.0)
            x_prot = self.projector.publish(x_raw_final, x_clean)
            f_prot_vit = self.primary_clip.encode_image(x_prot)
            attr_drift = subspace.compute_semantic_drift(f_src_vits[0], f_prot_vit)
            lpips_final = (
                float(self.lpips_metric(x_prot, x_clean).mean().item())
                if self.lpips_metric is not None
                else None
            )

        if has_align:
            for enc in self.cnn_encoders:
                if hasattr(enc, "set_alignment"):
                    enc.set_alignment(None)

        elapsed = time.time() - t0
        return {
            "x_prot": x_prot,
            "face_mask": face_mask,
            "subspace": subspace,
            "diagnostics": {
                "elapsed_total_s": float(elapsed),
                "elapsed_per_image_s": float(elapsed / max(b, 1)),
                "raw_grad_cos_mean": float(np.mean(raw_cos_trace)),
                "raw_conflict_rate_pct": float(100.0 * np.mean(raw_conflict_hits)),
                "both_paradigm_descent_pct": float(100.0 * np.mean(both_descent_hits)),
                "mean_fp_residual_255": float(np.mean(fp_residual_trace)) if fp_residual_trace else None,
                "final_fp_residual_255": float(fp_residual_trace[-1]) if fp_residual_trace else None,
                "lpips": lpips_final,
                "attr_drift": attr_drift,
                "config": asdict(self.cfg),
            },
        }

    def protect_batched(
        self,
        x_clean: torch.Tensor,
        x_target: Optional[torch.Tensor] = None,
        face_mask: Optional[torch.Tensor] = None,
        microbatch: int = 4,
    ) -> Dict[str, Any]:
        n = x_clean.shape[0]
        if n <= microbatch:
            return self.protect(x_clean, x_target=x_target, face_mask=face_mask)

        x_chunks, m_chunks, diags = [], [], []
        last_sub = None
        for i in range(0, n, microbatch):
            xc = x_clean[i : i + microbatch]
            xt = x_target[i : i + microbatch] if x_target is not None else None
            fm = face_mask[i : i + microbatch] if face_mask is not None else None
            out = self.protect(xc, x_target=xt, face_mask=fm)
            x_chunks.append(out["x_prot"])
            m_chunks.append(out["face_mask"])
            diags.append(out["diagnostics"])
            last_sub = out["subspace"]
            gc.collect()
            if x_clean.device.type == "cuda":
                torch.cuda.empty_cache()
            elif x_clean.device.type == "mps":
                torch.mps.empty_cache()

        weights = [min(microbatch, n - i) for i in range(0, n, microbatch)]
        tot_w = float(sum(weights))

        def _wavg(k: str) -> Optional[float]:
            if diags[0][k] is None:
                return None
            return float(sum(w * d[k] for w, d in zip(weights, diags)) / tot_w)

        merged_attr = {
            ax: float(sum(w * d["attr_drift"][ax] for w, d in zip(weights, diags)) / tot_w)
            for ax in diags[0]["attr_drift"]
        }
        return {
            "x_prot": torch.cat(x_chunks, dim=0),
            "face_mask": torch.cat(m_chunks, dim=0),
            "subspace": last_sub,
            "diagnostics": {
                "elapsed_total_s": float(sum(d["elapsed_total_s"] for d in diags)),
                "elapsed_per_image_s": float(sum(d["elapsed_total_s"] for d in diags) / tot_w),
                "raw_grad_cos_mean": _wavg("raw_grad_cos_mean"),
                "raw_conflict_rate_pct": _wavg("raw_conflict_rate_pct"),
                "both_paradigm_descent_pct": _wavg("both_paradigm_descent_pct"),
                "mean_fp_residual_255": _wavg("mean_fp_residual_255"),
                "final_fp_residual_255": _wavg("final_fp_residual_255"),
                "lpips": _wavg("lpips"),
                "attr_drift": merged_attr,
                "config": diags[0]["config"],
            },
        }

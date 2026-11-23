#!/usr/bin/env python3
"""
Scheme 1 Gate: Purification Fixed-Point Orthogonal Identity Protection (PFP-OrthoID).

Evaluates 100% zero-encoder-overlap protection under strict 8-bit Pi_x (L_inf <= 8/255, RMS <= 5.5/255):
  - Balanced Zero-Overlap Surrogates:
      CNN-FR : IRSE50 (5-pt ArcFace align + 4-view EOT) + FaceNet-CASIA-WebFace
      ViT    : OpenCLIP ViT-B/16 + OpenCLIP ViT-B/32 (openai)
  - Strictly Unseen Downstream Generator & FR Interfaces:
      1. PhotoMaker           : OpenCLIP ViT-L/14
      2. PuLID EVA-CLIP       : OpenCLIP EVA02-L-14-336
      3. IP-Adapter-FaceID    : InsightFace buffalo_l (native RetinaFace re-detect)
      4. InstantID / PuLID    : InsightFace antelopev2 (native RetinaFace re-detect)
      5. Independent FR       : FaceNet-VGGFace2
  - Purification Stress Suite (including UNSEEN cross-model SD-VAE / IMPRESS-style purifier!):
      1. Direct (no purification)
      2. JPEG (Q=75)
      3. Unseen Cross-Model Purifier: SD-VAE Re-Encoding (stabilityai/sd-vae-ft-mse, 100% disjoint from DDPM!)
      4. DiffPure (t=100, 5-step DDIM)
      5. DiffPure (t=150, 8-step DDIM)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from orthoid import (
    FaceMaskBuilder,
    IRSE50Encoder,
    PFPConfig,
    PurificationFixedPointProtector,
    TweediePurifierOperator,
    best_vae_dtype,
)
from orthoid.metrics import (
    compute_budget_stats,
    compute_psnr,
    compute_ssim,
    purify_jpeg,
)
from orthoid.models import (
    DiffPurePurifier,
    FaceNetEncoder,
    LPIPSMetric,
    OpenCLIPViTEncoder,
    SDVAEResidualDecoder,
)
from run_m1a_zero_overlap import (
    InsightFaceNativeEvaluator,
    UnseenViTEvaluator,
    load_lfw_cohort,
    save_png_batch,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clear_cache(device: str) -> None:
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()


@torch.no_grad()
def purify_unseen_sdvae(x01: torch.Tensor, vae_dec: SDVAEResidualDecoder, microbatch: int = 4) -> torch.Tensor:
    """
    Unseen cross-model purification (IMPRESS-style latent bottleneck reconstruction via
    frozen stabilityai/sd-vae-ft-mse, completely disjoint from ddpm-celebahq-256).
    """
    outs = []
    for i in range(0, x01.shape[0], microbatch):
        xb = x01[i : i + microbatch]
        _, x_rec = vae_dec.encode_clean(xb)
        outs.append(x_rec)
    return torch.cat(outs, dim=0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scheme 1 PFP-OrthoID Zero-Overlap & Cross-Purifier Gate")
    p.add_argument("--num-identities", type=int, default=12)
    p.add_argument("--src-per-id", type=int, default=2)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--microbatch", type=int, default=0)
    p.add_argument("--out-dir", type=str, default=str(REPO_ROOT / "results"))
    p.add_argument("--out-name", type=str, default="pfp_scheme1_gate.json")
    p.add_argument("--save-png-dir", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_all = time.time()
    torch.manual_seed(20261123)
    np.random.seed(20261123)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else device

    num_ids = args.num_identities
    steps = args.steps
    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        microbatch = args.microbatch if args.microbatch > 0 else (6 if vram_gb > 35 else 4)
        if "T4" in gpu_name:
            num_ids = min(num_ids, 6)
            steps = min(steps, 30)
            microbatch = min(microbatch, 2)
    else:
        microbatch = args.microbatch if args.microbatch > 0 else 2

    log(f"Scheme 1 PFP Gate on {gpu_name} | ids={num_ids}, src/id={args.src_per_id}, mb={microbatch}, steps={steps}")

    x_src, x_ref, x_tar, labels = load_lfw_cohort(
        num_identities=num_ids,
        photos_per_id=4,
        src_per_id=args.src_per_id,
        device=device,
    )
    n_imgs = x_src.shape[0]
    log(f"Loaded {n_imgs} real aligned 256x256 portraits across {len(set(labels))} identities")

    # 1. Balanced 100% Zero-Overlap Surrogates (2 CNN-FR + 2 OpenCLIP ViT) + Tweedie UNet
    log("Loading balanced zero-overlap surrogates (IRSE50 + FaceNet-CASIA | ViT-B/16 + ViT-B/32) + Tweedie UNet...")
    surr_irse50 = IRSE50Encoder(device=device, use_bgr=True)
    surr_casia = FaceNetEncoder(pretrained="casia-webface", device=device)
    surr_clip_b16 = OpenCLIPViTEncoder(model_name="ViT-B-16", pretrained="openai", device=device)
    surr_clip_b32 = OpenCLIPViTEncoder(model_name="ViT-B-32", pretrained="openai", device=device)
    tweedie_op = TweediePurifierOperator(model_id="google/ddpm-celebahq-256", device=device)
    lpips_fn = LPIPSMetric(net="alex", device=device)
    mask_builder = FaceMaskBuilder(use_mtcnn=True)
    face_masks = mask_builder.build(x_src, device=device)

    # 2. Strictly Unseen Evaluators + Unseen Cross-Model SD-VAE Purifier + Multi-Step DiffPure
    log("Loading strictly unseen evaluators & cross-model purifiers...")
    eval_facenet = FaceNetEncoder(pretrained="vggface2", device=device)
    eval_clip_l14 = UnseenViTEvaluator("ViT-L-14", "openai", img_size=224, device=device)
    eval_eva_l14 = UnseenViTEvaluator("EVA02-L-14-336", "merged2b_s6b_b61k", img_size=336, device=device)
    ctx_id = 0 if device == "cuda" else -1
    eval_buffalo = InsightFaceNativeEvaluator("buffalo_l", ctx_id=ctx_id)
    eval_antelope = InsightFaceNativeEvaluator("antelopev2", ctx_id=ctx_id)

    unseen_sdvae = SDVAEResidualDecoder(device=device, dtype=best_vae_dtype(device))
    diffpure = DiffPurePurifier(
        device=device,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        microbatch=min(4, microbatch),
    )

    with torch.no_grad():
        ref_fn = eval_facenet(x_ref)
        ref_clip_l = eval_clip_l14.encode(x_ref)
        ref_eva_l = eval_eva_l14.encode(x_ref)
        clean_src_clip_b16 = surr_clip_b16.encode_image(x_src)

    ref_buf, _ = eval_buffalo.extract_batch(x_ref)
    ref_ant, _ = eval_antelope.extract_batch(x_ref)

    def _calibrate_far01(src_embs: np.ndarray, ref_embs: np.ndarray) -> float:
        imps = []
        for i in range(n_imgs):
            for j in range(n_imgs):
                if labels[i] != labels[j]:
                    imps.append(float(np.dot(src_embs[i], ref_embs[j])))
        return float(np.quantile(imps, 0.99))

    src_fn_clean = eval_facenet(x_src).detach().cpu().numpy()
    src_buf_clean, _ = eval_buffalo.extract_batch(x_src)
    src_ant_clean, _ = eval_antelope.extract_batch(x_src)

    tau_fn = _calibrate_far01(src_fn_clean, ref_fn.detach().cpu().numpy())
    tau_buf = _calibrate_far01(src_buf_clean, ref_buf)
    tau_ant = _calibrate_far01(src_ant_clean, ref_ant)
    log(f"Calibrated unseen FAR=1% thresholds: FaceNet={tau_fn:.3f}, buffalo_l={tau_buf:.3f}, antelopev2={tau_ant:.3f}")

    @torch.no_grad()
    def score_all(x_eval: torch.Tensor) -> Dict[str, float]:
        e_fn = eval_facenet(x_eval)
        cos_fn = (e_fn * ref_fn).sum(dim=-1)
        e_cl = eval_clip_l14.encode(x_eval)
        cos_cl = (e_cl * ref_clip_l).sum(dim=-1)
        e_ev = eval_eva_l14.encode(x_eval)
        cos_ev = (e_ev * ref_eva_l).sum(dim=-1)

        e_buf, det_buf = eval_buffalo.extract_batch(x_eval)
        cos_buf = np.sum(e_buf * ref_buf, axis=-1)
        buf_valid = cos_buf[det_buf] if det_buf.any() else np.array([0.0])

        e_ant, det_ant = eval_antelope.extract_batch(x_eval)
        cos_ant = np.sum(e_ant * ref_ant, axis=-1)
        ant_valid = cos_ant[det_ant] if det_ant.any() else np.array([0.0])

        return {
            "pm_clip_l14_cos": float(cos_cl.mean().item()),
            "pulid_eva_cos": float(cos_ev.mean().item()),
            "faceid_buffalo_cos": float(buf_valid.mean()),
            "faceid_buffalo_dodge_pct": float(100.0 * np.mean(buf_valid < tau_buf)),
            "faceid_buffalo_fdr_pct": float(100.0 * np.mean(det_buf)),
            "instantid_antelope_cos": float(ant_valid.mean()),
            "instantid_antelope_dodge_pct": float(100.0 * np.mean(ant_valid < tau_ant)),
            "instantid_antelope_fdr_pct": float(100.0 * np.mean(det_ant)),
            "facenet_vgg_cos": float(cos_fn.mean().item()),
            "facenet_vgg_dodge_pct": float(100.0 * (cos_fn < tau_fn).float().mean().item()),
        }

    variants = [
        (
            "Baseline: Pixel PGD-40 (Balanced Naive)",
            PFPConfig(
                steps=steps,
                use_face_mask=False,
                use_purification_in_loop=False,
                use_fixed_point_loss=False,
                use_pythagorean_target=False,
                use_mgda=False,
                attr_lock_weight=0.0,
                direct_id_weight=1.0,
                purified_id_weight=0.0,
                fixed_point_weight=0.0,
                target_mode="target_free",
            ),
        ),
        (
            "Ablation: PFP w/o Fixed-Point Loss (EOT Only)",
            PFPConfig(
                steps=steps,
                use_face_mask=True,
                use_purification_in_loop=True,
                use_fixed_point_loss=False,
                use_pythagorean_target=True,
                use_mgda=True,
                fixed_point_weight=0.0,
                target_mode="target_free",
            ),
        ),
        (
            "Ours: PFP-OrthoID (Full Paired Fixed-Point)",
            PFPConfig(
                steps=steps,
                use_face_mask=True,
                use_purification_in_loop=True,
                use_fixed_point_loss=True,
                use_pythagorean_target=True,
                use_mgda=True,
                fixed_point_weight=12.0,
                target_mode="target_free",
            ),
        ),
    ]

    clean_scores = score_all(x_src)
    log(
        f"CLEAN | PM(ViT-L)={clean_scores['pm_clip_l14_cos']:.3f} | EVA={clean_scores['pulid_eva_cos']:.3f} | "
        f"Buf={clean_scores['faceid_buffalo_cos']:.3f} | Ant={clean_scores['instantid_antelope_cos']:.3f} | "
        f"FN={clean_scores['facenet_vgg_cos']:.3f}"
    )

    results = {
        "meta": {
            "device": device,
            "gpu_name": gpu_name,
            "num_portraits": int(n_imgs),
            "num_identities": int(len(set(labels))),
            "surrogates": [
                "IRSE50 (5-pt align + 4-view EOT)",
                "FaceNet-CASIA-WebFace",
                "OpenCLIP ViT-B/16",
                "OpenCLIP ViT-B/32",
            ],
            "tweedie_operator": "google/ddpm-celebahq-256 (1-step paired Tweedie)",
            "unseen_purifier": "stabilityai/sd-vae-ft-mse (cross-model)",
            "far01_thresholds": {"facenet": tau_fn, "buffalo_l": tau_buf, "antelopev2": tau_ant},
            "budget_spec": {"eps_255": 8.0, "rms_max_255": 5.5, "quantization": "uint8"},
            "steps": steps,
            "microbatch": microbatch,
        },
        "clean": clean_scores,
        "methods": [],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name

    for name, cfg in variants:
        log(f"--- Running {name} ---")
        protector = PurificationFixedPointProtector(
            cnn_encoders=[surr_irse50, surr_casia],
            vit_encoders=[surr_clip_b16, surr_clip_b32],
            tweedie_op=tweedie_op,
            lpips_metric=lpips_fn,
            mask_builder=mask_builder,
            config=cfg,
        )
        out = protector.protect_batched(x_src, x_target=x_tar, face_mask=face_masks, microbatch=microbatch)
        x_prot = out["x_prot"]

        if args.save_png_dir:
            slug = name.lower().replace(":", "").replace("(", "").replace(")", "").replace("/", "_").replace(" ", "_")
            save_png_batch(x_prot, Path(args.save_png_dir) / slug, labels)

        budget = compute_budget_stats(x_prot, x_src)
        psnr_val = compute_psnr(x_prot, x_src)
        ssim_val = compute_ssim(x_prot, x_src)
        with torch.no_grad():
            lpips_val = float(lpips_fn(x_prot, x_src).mean().item())
            attr_drift = out["subspace"].compute_semantic_drift(clean_src_clip_b16, surr_clip_b16.encode_image(x_prot))

        direct_s = score_all(x_prot)
        jpeg75_s = score_all(purify_jpeg(x_prot, quality=75))
        sdvae_s = score_all(purify_unseen_sdvae(x_prot, unseen_sdvae, microbatch=microbatch))
        dp100_s = score_all(diffpure.purify(x_prot, t_start=100, steps=5, seed=42))
        dp150_s = score_all(diffpure.purify(x_prot, t_start=150, steps=8, seed=42))

        row = {
            "method": name,
            "budget": budget,
            "visual": {"psnr_db": psnr_val, "ssim": ssim_val, "lpips": lpips_val},
            "attr_drift": attr_drift,
            "direct": direct_s,
            "jpeg75": jpeg75_s,
            "unseen_sdvae": sdvae_s,
            "diffpure_t100": dp100_s,
            "diffpure_t150": dp150_s,
            "mechanism": out["diagnostics"],
        }
        results["methods"].append(row)
        results["meta"]["total_elapsed_s"] = float(time.time() - t_all)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

        log(
            f"{name} ({out['diagnostics']['elapsed_per_image_s']:.2f}s/img) | "
            f"PSNR={psnr_val:.2f} LPIPS={lpips_val:.4f} 5Ax={attr_drift['mean_5axis_drift']:.4f} "
            f"FP_res={out['diagnostics']['final_fp_residual_255']} | "
            f"DIR[PM={direct_s['pm_clip_l14_cos']:.3f} EVA={direct_s['pulid_eva_cos']:.3f} "
            f"Buf={direct_s['faceid_buffalo_cos']:.3f} Ant={direct_s['instantid_antelope_cos']:.3f} "
            f"FN={direct_s['facenet_vgg_cos']:.3f}({direct_s['facenet_vgg_dodge_pct']:.0f}%)] | "
            f"SDVAE[Ant={sdvae_s['instantid_antelope_cos']:.3f} PM={sdvae_s['pm_clip_l14_cos']:.3f}] | "
            f"DP100[Ant={dp100_s['instantid_antelope_cos']:.3f}({dp100_s['instantid_antelope_dodge_pct']:.0f}%) "
            f"Buf={dp100_s['faceid_buffalo_cos']:.3f} PM={dp100_s['pm_clip_l14_cos']:.3f} "
            f"EVA={dp100_s['pulid_eva_cos']:.3f} FN={dp100_s['facenet_vgg_dodge_pct']:.0f}%] | "
            f"DP150[Ant={dp150_s['instantid_antelope_cos']:.3f} FN={dp150_s['facenet_vgg_dodge_pct']:.0f}%]"
        )
        clear_cache(device)

    log(f"Saved Scheme 1 PFP Gate results to {out_path}")


if __name__ == "__main__":
    main()

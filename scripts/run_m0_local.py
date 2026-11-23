#!/usr/bin/env python3
"""
M0-Local Sanity Gate for OrthoID (ICCV 2027) on Apple Silicon MPS / CUDA / CPU.

Verifies the standalone `src/orthoid/` package on real MTCNN-aligned 256x256 human portraits
before spending AutoDL RTX 4090 compute:
  1. Strict publishable 8-bit budget compliance Pi_x (L_inf <= 8/255, RMS <= 5.5/255, uint8 exact)
  2. Visual fidelity & non-destructive guardrails (PSNR > 34 dB, SSIM, LPIPS, MTCNN FDR ~100%)
  3. Mechanism verification (Gradient Conflict Rate, Simultaneous 2-Paradigm Descent, 5-Axis CLIP Attr-Drift)
  4. Strict zero-overlap unseen black-box FR transfer (VGGFace2 + OpenCLIP -> unseen CASIA-WebFace)
  5. Channel & purification survival (JPEG Q=75/50, Resize 160->256, real DDPM DiffPure t=50/100)
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from facenet_pytorch import MTCNN
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from orthoid import FaceMaskBuilder, OrthoIDConfig, OrthoIDProtector, best_vae_dtype
from orthoid.metrics import (
    compute_budget_stats,
    compute_fdr_mtcnn,
    compute_local_sharpness,
    compute_psnr,
    compute_ssim,
    purify_jpeg,
    purify_resize_redetect,
)
from orthoid.models import (
    DiffPurePurifier,
    FaceNetEncoder,
    LPIPSMetric,
    OpenCLIPViTEncoder,
    SDVAEResidualDecoder,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clear_cache(device: str) -> None:
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()


def load_lfw_portraits(
    identities: List[str],
    photos_per_id: int = 4,
    device: str = "cpu",
    local_files_only: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    Load real MTCNN-aligned 256x256 portraits from LFW dataset.
    Returns:
      x_src: [N, 3, 256, 256] source portraits to protect (2 per identity)
      x_ref: [N, 3, 256, 256] held-out clean gallery reference of the SAME identity
      x_tar: [N, 3, 256, 256] target portrait of a DIFFERENT identity (circular shift)
      labels: identity names per sample
    """
    zpath = hf_hub_download(
        repo_id="vilsonrodrigues/lfw",
        filename="lfw_multifaces-retrieval.zip",
        repo_type="dataset",
        local_files_only=local_files_only,
    )
    mtcnn = MTCNN(image_size=256, margin=24, post_process=False, device="cpu")

    per_id: Dict[str, List[torch.Tensor]] = {}
    with zipfile.ZipFile(zpath, "r") as zf:
        all_names = sorted(zf.namelist())
        for person in identities:
            matches = [
                n
                for n in all_names
                if f"/{person}_" in n
                and n.lower().endswith((".jpg", ".jpeg", ".png"))
                and "__MACOSX" not in n
            ]
            imgs: List[torch.Tensor] = []
            for m in matches:
                if len(imgs) >= photos_per_id:
                    break
                im = Image.open(io.BytesIO(zf.read(m))).convert("RGB")
                crop = mtcnn(im)
                if crop is None:
                    t = TF.to_tensor(TF.resize(TF.center_crop(im, 160), (256, 256)))
                else:
                    t = (crop / 255.0).clamp(0.0, 1.0)
                imgs.append(t)
            per_id[person] = imgs

    src_list, ref_list, tar_list, labels = [], [], [], []
    n_ids = len(identities)
    for idx, person in enumerate(identities):
        tar_person = identities[(idx + 1) % n_ids]
        # 2 source photos per identity -> 8 total portraits across 4 identities
        for k in range(2):
            src_list.append(per_id[person][k])
            ref_list.append(per_id[person][2 + (k % 2)])
            tar_list.append(per_id[tar_person][2])
            labels.append(person)

    return (
        torch.stack(src_list, dim=0).to(device),
        torch.stack(ref_list, dim=0).to(device),
        torch.stack(tar_list, dim=0).to(device),
        labels,
    )


@torch.no_grad()
def evaluate_protection(
    name: str,
    x_clean: torch.Tensor,
    x_ref: torch.Tensor,
    x_tar: torch.Tensor,
    x_prot: torch.Tensor,
    bb_fr: FaceNetEncoder,
    clip_enc: OpenCLIPViTEncoder,
    lpips_fn: LPIPSMetric,
    diffpure: DiffPurePurifier,
    subspace,
    far01_thresh: float,
    diagnostics: Dict,
) -> Dict:
    """Compute complete M0-Local evaluation metrics on publishable 8-bit x_prot."""
    budget = compute_budget_stats(x_prot, x_clean)
    psnr_val = compute_psnr(x_prot, x_clean)
    ssim_val = compute_ssim(x_prot, x_clean)
    lpips_val = float(lpips_fn(x_prot, x_clean).mean().item())
    fdr = compute_fdr_mtcnn(x_prot, x_clean)

    # Clean embeddings on unseen black-box FR & CLIP
    e_ref_bb = bb_fr(x_ref)
    e_tar_bb = bb_fr(x_tar)
    e_clean_clip = clip_enc.encode_image(x_clean)
    e_ref_clip = clip_enc.encode_image(x_ref)

    def _score_variant(x_eval: torch.Tensor) -> Dict[str, float]:
        e_bb = bb_fr(x_eval)
        e_clip = clip_enc.encode_image(x_eval)
        cos_same_bb = (e_bb * e_ref_bb).sum(dim=-1)
        cos_tar_bb = (e_bb * e_tar_bb).sum(dim=-1)
        cos_same_clip = (e_clip * e_ref_clip).sum(dim=-1)
        return {
            "bb_same_cos": float(cos_same_bb.mean().item()),
            "bb_dodge_psr_pct": float(100.0 * (cos_same_bb < far01_thresh).float().mean().item()),
            "bb_target_asr_pct": float(100.0 * (cos_tar_bb > far01_thresh).float().mean().item()),
            "clip_same_cos": float(cos_same_clip.mean().item()),
        }

    clean_eval = _score_variant(x_prot)
    jpeg75_eval = _score_variant(purify_jpeg(x_prot, quality=75))
    jpeg50_eval = _score_variant(purify_jpeg(x_prot, quality=50))
    resize_eval = _score_variant(purify_resize_redetect(x_prot, down_size=160))
    dp50_eval = _score_variant(diffpure.purify(x_prot, t_start=50, steps=5, seed=42))
    dp100_eval = _score_variant(diffpure.purify(x_prot, t_start=100, steps=5, seed=42))

    attr_drift = subspace.compute_semantic_drift(e_clean_clip, clip_enc.encode_image(x_prot))

    # Empirical local loss sharpness on unseen black-box FR similarity
    sharpness = compute_local_sharpness(
        x_prot,
        loss_fn=lambda x: (bb_fr(x) * e_ref_bb).sum(dim=-1),
        sigma=4.0 / 255.0,
        num_samples=6,
        seed=7,
    )

    return {
        "method": name,
        "budget": budget,
        "visual": {
            "psnr_db": psnr_val,
            "ssim": ssim_val,
            "lpips": lpips_val,
            "fdr_pct": fdr["fdr_pct"],
            "mean_det_prob": fdr["mean_det_prob"],
            "mean_bbox_iou": fdr["mean_bbox_iou"],
        },
        "attr_drift": attr_drift,
        "protection_direct": clean_eval,
        "purification": {
            "jpeg75": jpeg75_eval,
            "jpeg50": jpeg50_eval,
            "resize160": resize_eval,
            "diffpure_t50": dp50_eval,
            "diffpure_t100": dp100_eval,
        },
        "mechanism": {
            "local_bb_sharpness": sharpness,
            "raw_grad_cos_mean": diagnostics.get("raw_grad_cos_mean"),
            "proj_grad_cos_mean": diagnostics.get("proj_grad_cos_mean"),
            "raw_conflict_rate_pct": diagnostics.get("raw_conflict_rate_pct"),
            "proj_conflict_rate_pct": diagnostics.get("proj_conflict_rate_pct"),
            "both_paradigm_descent_pct": diagnostics.get("both_paradigm_descent_pct"),
            "subspace_init_s": diagnostics.get("subspace_init_s"),
            "elapsed_per_image_s": diagnostics.get("elapsed_per_image_s"),
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OrthoID M0 Sanity Gate (MPS / CUDA / CPU)")
    p.add_argument("--out-dir", type=str, default=str(REPO_ROOT / "results"))
    p.add_argument("--out-name", type=str, default="m0_sanity.json")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--microbatch", type=int, default=0, help="0 = auto (4 on CUDA, 2 on MPS/CPU)")
    p.add_argument("--local-only", action="store_true", help="Restrict HF downloads to local cache only")
    p.add_argument("--print-json", action="store_true", help="Print final JSON to stdout")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_all = time.time()
    torch.manual_seed(20261121)
    np.random.seed(20261121)

    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        microbatch = args.microbatch if args.microbatch > 0 else 4
    elif torch.backends.mps.is_available():
        device = "mps"
        gpu_name = "Apple Silicon MPS"
        microbatch = args.microbatch if args.microbatch > 0 else 2
    else:
        device = "cpu"
        gpu_name = "CPU"
        microbatch = args.microbatch if args.microbatch > 0 else 2

    log(f"Running M0 Sanity Gate on device={device} ({gpu_name}), microbatch={microbatch}, steps={args.steps}")

    identities = ["George_W_Bush", "Colin_Powell", "Tony_Blair", "Donald_Rumsfeld"]
    x_src, x_ref, x_tar, labels = load_lfw_portraits(
        identities,
        photos_per_id=4,
        device=device,
        local_files_only=args.local_only,
    )
    log(f"Loaded {x_src.shape[0]} real aligned 256x256 portraits ({len(identities)} identities)")

    log("Loading surrogate encoders (FaceNet-VGGFace2 + OpenCLIP ViT-B/16)...")
    surr_fr = FaceNetEncoder(pretrained="vggface2", device=device)
    clip_enc = OpenCLIPViTEncoder(model_name="ViT-B-16", pretrained="openai", device=device)

    log("Loading strictly unseen black-box FR evaluator (FaceNet-CASIA-WebFace)...")
    bb_fr = FaceNetEncoder(pretrained="casia-webface", device=device)

    vae_dtype = best_vae_dtype(device)
    log(f"Loading SD-VAE (stabilityai/sd-vae-ft-mse, dtype={vae_dtype})...")
    vae_dec = SDVAEResidualDecoder(
        model_id="stabilityai/sd-vae-ft-mse",
        device=device,
        dtype=vae_dtype,
        local_files_only=args.local_only,
    )

    log("Loading LPIPS & real DDPM DiffPure purifier (google/ddpm-celebahq-256)...")
    lpips_fn = LPIPSMetric(net="alex", device=device)
    diffpure = DiffPurePurifier(
        model_id="google/ddpm-celebahq-256",
        device=device,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        local_files_only=args.local_only,
        microbatch=microbatch,
    )
    mask_builder = FaceMaskBuilder(use_mtcnn=True)
    face_masks = mask_builder.build(x_src, device=device)

    # Calibrate empirical FAR=1% impostor threshold on unseen black-box FR
    with torch.no_grad():
        e_src_bb = bb_fr(x_src)
        e_ref_bb = bb_fr(x_ref)
        same_scores = (e_src_bb * e_ref_bb).sum(dim=-1)
        impostor_scores = []
        for i in range(x_src.shape[0]):
            for j in range(x_ref.shape[0]):
                if labels[i] != labels[j]:
                    impostor_scores.append(float((e_src_bb[i] * e_ref_bb[j]).sum().item()))
        far01_thresh = float(np.quantile(impostor_scores, 0.99))
        log(
            f"Clean unseen BB-FR: same_cos={same_scores.mean().item():.4f}, "
            f"impostor_99pct_thresh={far01_thresh:.4f}"
        )

    variants = [
        (
            "Baseline: Pixel PGD-Ensemble (Naive Sum)",
            OrthoIDConfig(
                steps=args.steps,
                use_vae_residual=False,
                use_midband_residual=False,
                use_face_mask=False,
                use_pythagorean_target=False,
                use_latent_pullback=False,
                use_mgda=False,
                use_flatness_lookahead=False,
                use_lpips_retraction=False,
                attr_lock_weight=0.0,
            ),
        ),
        (
            "Ablation: VAE-Residual Only (Naive Sum)",
            OrthoIDConfig(
                steps=args.steps,
                use_vae_residual=True,
                use_midband_residual=False,
                use_face_mask=True,
                use_pythagorean_target=False,
                use_latent_pullback=False,
                use_mgda=False,
                use_flatness_lookahead=False,
                use_lpips_retraction=False,
                attr_lock_weight=0.0,
            ),
        ),
        (
            "OrthoID-v2 (Dual-Band Targeted)",
            OrthoIDConfig(
                steps=args.steps,
                use_vae_residual=True,
                use_midband_residual=True,
                use_face_mask=True,
                use_pythagorean_target=True,
                use_latent_pullback=True,
                use_mgda=True,
                use_flatness_lookahead=True,
                use_lpips_retraction=True,
                target_mode="targeted",
            ),
        ),
        (
            "OrthoID-v2 (Dual-Band Target-Free)",
            OrthoIDConfig(
                steps=args.steps,
                use_vae_residual=True,
                use_midband_residual=True,
                use_face_mask=True,
                use_pythagorean_target=True,
                use_latent_pullback=True,
                use_mgda=True,
                use_flatness_lookahead=True,
                use_lpips_retraction=True,
                target_mode="target_free",
            ),
        ),
    ]

    results = {
        "meta": {
            "device": device,
            "gpu_name": gpu_name,
            "num_portraits": int(x_src.shape[0]),
            "identities": identities,
            "microbatch": microbatch,
            "steps": args.steps,
            "budget_spec": {"eps_255": 8.0, "rms_max_255": 5.5, "quantization": "uint8"},
            "clean_bb_same_cos": float(same_scores.mean().item()),
            "far01_thresh": far01_thresh,
        },
        "methods": [],
    }

    # Evaluate Clean baseline reference
    clean_fdr = compute_fdr_mtcnn(x_src, x_src)
    with torch.no_grad():
        clean_clip_same = float((clip_enc.encode_image(x_src) * clip_enc.encode_image(x_ref)).sum(dim=-1).mean().item())
    results["clean_reference"] = {
        "fdr_pct": clean_fdr["fdr_pct"],
        "bb_same_cos": float(same_scores.mean().item()),
        "clip_same_cos": clean_clip_same,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name

    for name, cfg in variants:
        log(f"--- Running {name} ---")
        protector = OrthoIDProtector(
            cnn_encoders=[surr_fr],
            clip_encoder=clip_enc,
            vae_decoder=vae_dec,
            lpips_metric=lpips_fn,
            mask_builder=mask_builder,
            config=cfg,
        )
        out = protector.protect_batched(x_src, x_target=x_tar, face_mask=face_masks, microbatch=microbatch)
        row = evaluate_protection(
            name=name,
            x_clean=x_src,
            x_ref=x_ref,
            x_tar=x_tar,
            x_prot=out["x_prot"],
            bb_fr=bb_fr,
            clip_enc=clip_enc,
            lpips_fn=lpips_fn,
            diffpure=diffpure,
            subspace=out["subspace"],
            far01_thresh=far01_thresh,
            diagnostics=out["diagnostics"],
        )
        results["methods"].append(row)
        results["meta"]["total_elapsed_s"] = float(time.time() - t_all)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        log(
            f"{name} ({row['mechanism']['elapsed_per_image_s']:.2f}s/img) | "
            f"PSNR={row['visual']['psnr_db']:.2f}dB, SSIM={row['visual']['ssim']:.3f}, "
            f"LPIPS={row['visual']['lpips']:.4f}, FDR={row['visual']['fdr_pct']:.1f}%, "
            f"Linf={row['budget']['max_linf_255']:.1f}/255, RMS={row['budget']['max_rms_255']:.2f}/255 | "
            f"BB_cos={row['protection_direct']['bb_same_cos']:.3f} (Dodge={row['protection_direct']['bb_dodge_psr_pct']:.1f}%), "
            f"CLIP_cos={row['protection_direct']['clip_same_cos']:.3f}, "
            f"5AxisDrift={row['attr_drift']['mean_5axis_drift']:.4f}, "
            f"DP100_cos={row['purification']['diffpure_t100']['bb_same_cos']:.3f} "
            f"(DP100_Dodge={row['purification']['diffpure_t100']['bb_dodge_psr_pct']:.1f}%)"
        )
        clear_cache(device)

    log(f"Saved M0 sanity results to {out_path}")
    if args.print_json:
        print("=== M0_JSON_START ===")
        print(json.dumps(results, ensure_ascii=False))
        print("=== M0_JSON_END ===")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
M1-A: Strict Zero-Encoder-Overlap Cross-Interface & Native Re-Detection Gate (ICCV 2027).

Surrogate (100% DISJOINT from all evaluators & downstream generators):
  - CNN-FR: IRSE50 (5-point ArcFace alignment + 4-view alignment-jitter EOT) + FaceNet-CASIA-WebFace
  - ViT-Semantic: OpenCLIP ViT-B/16 (openai)

Strictly Unseen Downstream Generator & FR Interfaces Evaluated:
  1. PhotoMaker interface      : OpenCLIP ViT-L/14 (openai)
  2. IP-Adapter-FaceID         : InsightFace buffalo_l (w600k_r50) w/ NATIVE RetinaFace re-detection
  3. InstantID / PuLID ArcFace : InsightFace antelopev2 (glintr100) w/ NATIVE RetinaFace re-detection
  4. PuLID EVA-CLIP            : OpenCLIP EVA02-L-14-336 (merged2b_s6b_b61k)
  5. Independent Black-Box FR  : FaceNet-VGGFace2 (calibrated FAR=1% threshold)

Also evaluates:
  - Strict publishable 8-bit Pi_x budget (L_inf <= 8/255, RMS <= 5.5/255)
  - PSNR, SSIM, LPIPS, 5-Axis CLIP Attr-Drift, Native RetinaFace FDR (%)
  - Post-JPEG (Q=75) and Post-DiffPure (t=100) survival across ALL 5 unseen interfaces
  - Exports exact 8-bit PNGs to `--save-png-dir` for immediate Stage M1-B SDXL 4-generator synthesis.
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
from typing import Dict, List, Optional, Tuple

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

from orthoid import FaceMaskBuilder, IRSE50Encoder, OrthoIDConfig, OrthoIDProtector, best_vae_dtype
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


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clear_cache(device: str) -> None:
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()


def load_lfw_cohort(
    num_identities: int = 16,
    photos_per_id: int = 4,
    src_per_id: int = 2,
    device: str = "cpu",
    local_files_only: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    Load N = num_identities * src_per_id real MTCNN-aligned 256x256 portraits from LFW.
    Selects the top `num_identities` people with >= photos_per_id clean photos.
    """
    zpath = hf_hub_download(
        repo_id="vilsonrodrigues/lfw",
        filename="lfw_multifaces-retrieval.zip",
        repo_type="dataset",
        local_files_only=local_files_only,
    )
    mtcnn = MTCNN(image_size=256, margin=28, post_process=False, device="cpu")

    by_person: Dict[str, List[str]] = {}
    with zipfile.ZipFile(zpath, "r") as zf:
        for n in sorted(zf.namelist()):
            if "__MACOSX" in n or not n.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            parts = n.strip("/").split("/")
            if len(parts) >= 2:
                person = parts[-2]
                by_person.setdefault(person, []).append(n)

        # Deterministic selection of identities with ample photos (sorted by count desc, then name)
        candidates = [p for p, files in by_person.items() if len(files) >= photos_per_id + 1]
        candidates.sort(key=lambda p: (-len(by_person[p]), p))
        chosen: List[str] = []
        per_id: Dict[str, List[torch.Tensor]] = {}

        for person in candidates:
            if len(chosen) >= num_identities:
                break
            imgs: List[torch.Tensor] = []
            for m in by_person[person]:
                if len(imgs) >= photos_per_id:
                    break
                im = Image.open(io.BytesIO(zf.read(m))).convert("RGB")
                crop = mtcnn(im)
                if crop is None:
                    continue
                t = (crop / 255.0).clamp(0.0, 1.0)
                imgs.append(t)
            if len(imgs) == photos_per_id:
                chosen.append(person)
                per_id[person] = imgs

    src_list, ref_list, tar_list, labels = [], [], [], []
    n_ids = len(chosen)
    for idx, person in enumerate(chosen):
        tar_person = chosen[(idx + 1) % n_ids]
        for k in range(src_per_id):
            src_list.append(per_id[person][k])
            ref_list.append(per_id[person][src_per_id + (k % (photos_per_id - src_per_id))])
            tar_list.append(per_id[tar_person][src_per_id])
            labels.append(person)

    return (
        torch.stack(src_list, dim=0).to(device),
        torch.stack(ref_list, dim=0).to(device),
        torch.stack(tar_list, dim=0).to(device),
        labels,
    )


class InsightFaceNativeEvaluator:
    """
    Runs NATIVE InsightFace RetinaFace detection + 5-point alignment + ArcFace recognition
    (zero-overlap downstream evaluator for buffalo_l / antelopev2).
    """

    @staticmethod
    def _ensure_antelopev2() -> None:
        from huggingface_hub import snapshot_download

        target_dir = Path.home() / ".insightface" / "models" / "antelopev2"
        if (target_dir / "glintr100.onnx").exists() and (target_dir / "scrfd_10g_bnkps.onnx").exists():
            return
        target_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id="DIAMONIK7777/antelopev2",
            local_dir=str(target_dir),
            allow_patterns=["*.onnx"],
        )

    def __init__(self, pack_name: str, ctx_id: int = 0):
        from insightface.app import FaceAnalysis

        self.pack_name = pack_name
        if pack_name == "antelopev2":
            self._ensure_antelopev2()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ctx_id >= 0 else ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name=pack_name, providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=(320, 320))

    @staticmethod
    def _tensor_to_bgr_uint8(t: torch.Tensor) -> np.ndarray:
        rgb = (t.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        return rgb[:, :, ::-1].copy()

    def extract_batch(self, x01: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (embeddings [B, 512] unit-normalized, detected_mask [B] bool)."""
        b = x01.shape[0]
        embs = np.zeros((b, 512), dtype=np.float32)
        det = np.zeros((b,), dtype=bool)
        for i in range(b):
            bgr = self._tensor_to_bgr_uint8(x01[i])
            faces = self.app.get(bgr)
            if faces:
                # Select largest detected face (standard downstream generator behavior)
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                emb = np.asarray(face.normed_embedding, dtype=np.float32)
                norm = float(np.linalg.norm(emb)) + 1e-12
                embs[i] = emb / norm
                det[i] = True
        return embs, det


class UnseenViTEvaluator:
    """Wrapper for strictly unseen ViT generator interfaces (CLIP-ViT-L/14 and EVA02-CLIP-L-14-336)."""

    def __init__(self, model_name: str, pretrained: str, img_size: int, device: str):
        import open_clip

        self.device = torch.device(device)
        self.img_size = int(img_size)
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode(self, x01: torch.Tensor, batch_size: int = 8) -> torch.Tensor:
        outs = []
        for i in range(0, x01.shape[0], batch_size):
            xb = F.interpolate(x01[i : i + batch_size], size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
            xb = (xb - self.mean) / self.std
            outs.append(F.normalize(self.model.encode_image(xb).float(), dim=-1))
        return torch.cat(outs, dim=0)


def save_png_batch(x01: torch.Tensor, out_dir: Path, labels: List[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(x01.shape[0]):
        arr = (x01[i].detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(arr).save(out_dir / f"{i:03d}_{labels[i]}.png")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M1-A Zero-Encoder-Overlap Cross-Interface Gate")
    p.add_argument("--num-identities", type=int, default=16)
    p.add_argument("--src-per-id", type=int, default=2)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--microbatch", type=int, default=0)
    p.add_argument("--out-dir", type=str, default=str(REPO_ROOT / "results"))
    p.add_argument("--out-name", type=str, default="m1a_zero_overlap.json")
    p.add_argument("--save-png-dir", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_all = time.time()
    torch.manual_seed(20261123)
    np.random.seed(20261123)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else device
    microbatch = args.microbatch if args.microbatch > 0 else (8 if device == "cuda" else 2)
    log(f"M1-A Zero-Overlap Gate on {gpu_name} | ids={args.num_identities}, src/id={args.src_per_id}, mb={microbatch}, steps={args.steps}")

    x_src, x_ref, x_tar, labels = load_lfw_cohort(
        num_identities=args.num_identities,
        photos_per_id=4,
        src_per_id=args.src_per_id,
        device=device,
    )
    n_imgs = x_src.shape[0]
    log(f"Loaded {n_imgs} real aligned 256x256 portraits across {len(set(labels))} identities")

    # 1. STRICT ZERO-OVERLAP SURROGATES (100% disjoint from ALL evaluators & downstream generators)
    log("Loading zero-overlap surrogates: IRSE50 + FaceNet-CASIA + OpenCLIP ViT-B/16 + SD-VAE...")
    surr_irse50 = IRSE50Encoder(device=device, use_bgr=True)
    surr_casia = FaceNetEncoder(pretrained="casia-webface", device=device)
    surr_clip_b16 = OpenCLIPViTEncoder(model_name="ViT-B-16", pretrained="openai", device=device)
    vae_dec = SDVAEResidualDecoder(device=device, dtype=best_vae_dtype(device))
    lpips_fn = LPIPSMetric(net="alex", device=device)
    mask_builder = FaceMaskBuilder(use_mtcnn=True)
    face_masks = mask_builder.build(x_src, device=device)

    # 2. STRICTLY UNSEEN DOWNSTREAM GENERATOR INTERFACES + INDEPENDENT FR + DIFFPURE
    log("Loading strictly unseen downstream evaluators: FaceNet-VGGFace2, CLIP-ViT-L/14, EVA02-L-14-336, buffalo_l, antelopev2, DiffPure...")
    eval_facenet = FaceNetEncoder(pretrained="vggface2", device=device)
    eval_clip_l14 = UnseenViTEvaluator("ViT-L-14", "openai", img_size=224, device=device)
    eval_eva_l14 = UnseenViTEvaluator("EVA02-L-14-336", "merged2b_s6b_b61k", img_size=336, device=device)
    ctx_id = 0 if device == "cuda" else -1
    eval_buffalo = InsightFaceNativeEvaluator("buffalo_l", ctx_id=ctx_id)
    eval_antelope = InsightFaceNativeEvaluator("antelopev2", ctx_id=ctx_id)
    diffpure = DiffPurePurifier(
        device=device,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        microbatch=min(4, microbatch),
    )

    # Precompute clean reference embeddings on all 5 unseen interfaces + calibrate FAR=1% thresholds
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

    # Helper to score any image tensor batch across ALL 5 strictly unseen interfaces
    @torch.no_grad()
    def score_all_interfaces(x_eval: torch.Tensor) -> Dict[str, float]:
        # 1. FaceNet-VGGFace2 (unseen FR)
        e_fn = eval_facenet(x_eval)
        cos_fn = (e_fn * ref_fn).sum(dim=-1)
        # 2. PhotoMaker interface (unseen CLIP-ViT-L/14)
        e_cl = eval_clip_l14.encode(x_eval)
        cos_cl = (e_cl * ref_clip_l).sum(dim=-1)
        # 3. PuLID EVA-CLIP interface (unseen EVA02-L-14-336)
        e_ev = eval_eva_l14.encode(x_eval)
        cos_ev = (e_ev * ref_eva_l).sum(dim=-1)
        # 4. IP-Adapter-FaceID interface (native RetinaFace re-detect + buffalo_l ArcFace)
        e_buf, det_buf = eval_buffalo.extract_batch(x_eval)
        cos_buf = np.sum(e_buf * ref_buf, axis=-1)
        buf_valid = cos_buf[det_buf] if det_buf.any() else np.array([0.0])
        # 5. PuLID / InstantID interface (native RetinaFace re-detect + antelopev2 ArcFace)
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

    # Methods to compare under exact same zero-overlap surrogates & 8-bit Pi_x budget
    variants = [
        (
            "Baseline: Pixel PGD-Ensemble (Naive)",
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
                target_mode="target_free",
            ),
        ),
        (
            "Ablation: VAE-Residual Only (Naive)",
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
                target_mode="target_free",
            ),
        ),
        (
            "OrthoID-v2 (Target-Free, Ours)",
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
        (
            "OrthoID-v2 (Targeted, Ours)",
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
    ]

    clean_scores = score_all_interfaces(x_src)
    log(
        f"CLEAN | PM(ViT-L)={clean_scores['pm_clip_l14_cos']:.3f} | "
        f"PuLID(EVA)={clean_scores['pulid_eva_cos']:.3f} | "
        f"FaceID(buf)={clean_scores['faceid_buffalo_cos']:.3f} (FDR={clean_scores['faceid_buffalo_fdr_pct']:.0f}%) | "
        f"InstantID(ant)={clean_scores['instantid_antelope_cos']:.3f} (FDR={clean_scores['instantid_antelope_fdr_pct']:.0f}%) | "
        f"FaceNet={clean_scores['facenet_vgg_cos']:.3f}"
    )

    if args.save_png_dir:
        save_png_batch(x_src, Path(args.save_png_dir) / "clean", labels)
        save_png_batch(x_ref, Path(args.save_png_dir) / "ref", labels)

    results = {
        "meta": {
            "device": device,
            "gpu_name": gpu_name,
            "num_portraits": int(n_imgs),
            "num_identities": int(len(set(labels))),
            "surrogates": ["IRSE50 (5-pt align + 4-view EOT)", "FaceNet-CASIA-WebFace", "OpenCLIP ViT-B/16"],
            "unseen_interfaces": [
                "PhotoMaker: OpenCLIP ViT-L/14",
                "IP-Adapter-FaceID: InsightFace buffalo_l (native re-detect)",
                "InstantID/PuLID: InsightFace antelopev2 (native re-detect)",
                "PuLID: OpenCLIP EVA02-L-14-336",
                "Independent FR: FaceNet-VGGFace2",
            ],
            "far01_thresholds": {"facenet": tau_fn, "buffalo_l": tau_buf, "antelopev2": tau_ant},
            "budget_spec": {"eps_255": 8.0, "rms_max_255": 5.5, "quantization": "uint8"},
            "steps": args.steps,
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
        protector = OrthoIDProtector(
            cnn_encoders=[surr_irse50, surr_casia],
            clip_encoder=surr_clip_b16,
            vae_decoder=vae_dec,
            lpips_metric=lpips_fn,
            mask_builder=mask_builder,
            config=cfg,
        )
        out = protector.protect_batched(x_src, x_target=x_tar, face_mask=face_masks, microbatch=microbatch)
        x_prot = out["x_prot"]

        if args.save_png_dir:
            slug = name.lower().replace(":", "").replace("(", "").replace(")", "").replace(",", "").replace(" ", "_")
            save_png_batch(x_prot, Path(args.save_png_dir) / slug, labels)

        budget = compute_budget_stats(x_prot, x_src)
        psnr_val = compute_psnr(x_prot, x_src)
        ssim_val = compute_ssim(x_prot, x_src)
        with torch.no_grad():
            lpips_val = float(lpips_fn(x_prot, x_src).mean().item())
            attr_drift = out["subspace"].compute_semantic_drift(clean_src_clip_b16, surr_clip_b16.encode_image(x_prot))

        direct_scores = score_all_interfaces(x_prot)
        jpeg75_scores = score_all_interfaces(purify_jpeg(x_prot, quality=75))
        dp100_scores = score_all_interfaces(diffpure.purify(x_prot, t_start=100, steps=5, seed=42))

        row = {
            "method": name,
            "budget": budget,
            "visual": {"psnr_db": psnr_val, "ssim": ssim_val, "lpips": lpips_val},
            "attr_drift": attr_drift,
            "direct": direct_scores,
            "jpeg75": jpeg75_scores,
            "diffpure_t100": dp100_scores,
            "mechanism": {
                "raw_conflict_rate_pct": out["diagnostics"]["raw_conflict_rate_pct"],
                "both_paradigm_descent_pct": out["diagnostics"]["both_paradigm_descent_pct"],
                "elapsed_per_image_s": out["diagnostics"]["elapsed_per_image_s"],
            },
        }
        results["methods"].append(row)
        results["meta"]["total_elapsed_s"] = float(time.time() - t_all)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

        log(
            f"{name} ({row['mechanism']['elapsed_per_image_s']:.2f}s/img) | "
            f"PSNR={psnr_val:.2f} LPIPS={lpips_val:.4f} 5Ax={attr_drift['mean_5axis_drift']:.4f} | "
            f"PM(ViT-L)={direct_scores['pm_clip_l14_cos']:.3f} | "
            f"EVA={direct_scores['pulid_eva_cos']:.3f} | "
            f"FaceID(buf)={direct_scores['faceid_buffalo_cos']:.3f} (Dodge={direct_scores['faceid_buffalo_dodge_pct']:.0f}%, FDR={direct_scores['faceid_buffalo_fdr_pct']:.0f}%) | "
            f"AntV2={direct_scores['instantid_antelope_cos']:.3f} (Dodge={direct_scores['instantid_antelope_dodge_pct']:.0f}%, FDR={direct_scores['instantid_antelope_fdr_pct']:.0f}%) | "
            f"FN={direct_scores['facenet_vgg_cos']:.3f} | "
            f"DP100(Ant={dp100_scores['instantid_antelope_cos']:.3f}, PM={dp100_scores['pm_clip_l14_cos']:.3f})"
        )
        clear_cache(device)

    log(f"Saved M1-A results to {out_path}")


if __name__ == "__main__":
    main()

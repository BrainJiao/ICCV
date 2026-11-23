#!/usr/bin/env python3
"""
Scheme 1 Real End-to-End Benchmark (ICCV 2027):
  Part 1: 5 Unseen Interfaces x 5 Purifiers (Direct, JPEG75, Unseen SD-VAE, DiffPure t=100, DiffPure t=150)
          comparing Clean, Official IDProtector (CVPR'25), Pixel PGD-40, PFP w/o L_FP, and Ours: PFP-OrthoID (Scale-Fixed L_FP).
  Part 2: REAL End-to-End SDXL Personalized Generation (PhotoMaker + InstantID)
          under BOTH Direct AND Post-DiffPure (t=100), scored by independent unseen
          FaceNet-VGGFace2 and InsightFace buffalo_l (ASR% ↓, ISM ↓, Gen-FDR% ↑).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

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
from orthoid.baselines import IDProtectorCVPR25
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
)
from run_pfp_gate import purify_unseen_sdvae


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def ensure_repo(url: str, dst: Path) -> Path:
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["git", "clone", "--depth", "1", url, str(dst)])
    return dst


def score_generated_pils(
    gen_pils: List[Image.Image],
    gt_id_indices: List[int],
    gallery_fn: torch.Tensor,
    gallery_buf: np.ndarray,
    eval_fn: FaceNetEncoder,
    eval_buf: InsightFaceNativeEvaluator,
    tau_fn: float,
    tau_buf: float,
    device: str,
) -> Dict[str, float]:
    """
    Score real generated portraits using TWO independent unseen recognizers
    (FaceNet-VGGFace2 and InsightFace buffalo_l):
      - Gen-FDR% : % of generated images with a valid detected face
      - ISM ↓    : mean cosine similarity to ground-truth victim gallery among detected faces
      - ASR% ↓   : % of planned attempts with a detected face that ranks victim #1 AND exceeds FAR=1% threshold
    """
    from facenet_pytorch import MTCNN

    mtcnn = MTCNN(image_size=160, margin=14, post_process=False, device="cpu")
    n = len(gen_pils)

    fn_sims, fn_asr_hits, fn_det = [], 0, 0
    buf_sims, buf_asr_hits, buf_det = [], 0, 0

    for img, gt_idx in zip(gen_pils, gt_id_indices):
        # 1. FaceNet-VGGFace2 scoring (MTCNN crop on generated portrait)
        crop = mtcnn(img)
        if crop is not None:
            fn_det += 1
            t = (crop / 255.0).clamp(0.0, 1.0).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = eval_fn(t)[0]  # [512]
                all_sims = (gallery_fn @ emb).detach().cpu().numpy()
            sim_gt = float(all_sims[gt_idx])
            fn_sims.append(sim_gt)
            if int(np.argmax(all_sims)) == gt_idx and sim_gt > tau_fn:
                fn_asr_hits += 1

        # 2. InsightFace buffalo_l native RetinaFace scoring
        bgr = np.array(img.convert("RGB"))[:, :, ::-1].copy()
        faces = eval_buf.app.get(bgr)
        if faces:
            buf_det += 1
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            emb_b = np.asarray(face.normed_embedding, dtype=np.float32)
            emb_b = emb_b / (float(np.linalg.norm(emb_b)) + 1e-12)
            all_b = gallery_buf @ emb_b
            sim_b = float(all_b[gt_idx])
            buf_sims.append(sim_b)
            if int(np.argmax(all_b)) == gt_idx and sim_b > tau_buf:
                buf_asr_hits += 1

    return {
        "n_attempts": n,
        "facenet_fdr_pct": 100.0 * fn_det / max(n, 1),
        "facenet_ism": float(np.mean(fn_sims)) if fn_sims else 0.0,
        "facenet_asr_pct": 100.0 * fn_asr_hits / max(n, 1),
        "buffalo_fdr_pct": 100.0 * buf_det / max(n, 1),
        "buffalo_ism": float(np.mean(buf_sims)) if buf_sims else 0.0,
        "buffalo_asr_pct": 100.0 * buf_asr_hits / max(n, 1),
    }


def _install_diffusers_compat_shims() -> None:
    """Restore moved diffusers symbols expected by PhotoMaker and InstantID on diffusers >= 0.35."""
    import types
    import diffusers.loaders as _dl

    if not hasattr(_dl, "StableDiffusionXLLoraLoaderMixin"):
        from diffusers.loaders.lora_pipeline import StableDiffusionXLLoraLoaderMixin as _SDXLLoraMixin
        _dl.StableDiffusionXLLoraLoaderMixin = _SDXLLoraMixin

    if "diffusers.pipelines.controlnet.multicontrolnet" not in sys.modules:
        try:
            __import__("diffusers.pipelines.controlnet.multicontrolnet")
        except ModuleNotFoundError:
            from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel
            mod = types.ModuleType("diffusers.pipelines.controlnet.multicontrolnet")
            mod.MultiControlNetModel = MultiControlNetModel
            sys.modules["diffusers.pipelines.controlnet.multicontrolnet"] = mod

    # Guard against Colab's preinstalled torchao==0.10.0 breaking peft LoRA dispatch
    try:
        import peft.import_utils as _peft_iu
        import peft.tuners.lora.torchao as _peft_tao
        _peft_iu.is_torchao_available = lambda: False
        _peft_tao.is_torchao_available = lambda: False
    except Exception:
        pass


def _best_sdxl_dtype(device: str) -> torch.dtype:
    if device == "cuda" and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def run_photomaker_eval(
    method_tensors: Dict[str, Dict[str, torch.Tensor]],
    e2e_indices: List[int],
    gt_id_indices: List[int],
    gallery_fn: torch.Tensor,
    gallery_buf: np.ndarray,
    eval_fn: FaceNetEncoder,
    eval_buf: InsightFaceNativeEvaluator,
    tau_fn: float,
    tau_buf: float,
    device: str,
    steps: int = 20,
    gen_size: int = 768,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    _install_diffusers_compat_shims()
    from diffusers import DDIMScheduler
    from huggingface_hub import hf_hub_download

    pm_repo = ensure_repo("https://github.com/TencentARC/PhotoMaker.git", Path("/content/PhotoMaker"))
    if str(pm_repo) not in sys.path:
        sys.path.insert(0, str(pm_repo))
    from photomaker import PhotoMakerStableDiffusionXLPipeline

    gen_dtype = _best_sdxl_dtype(device)
    log(f"Loading PhotoMaker + SG161222/RealVisXL_V3.0 ({gen_dtype})...")
    pm_ckpt = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")
    pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
        "SG161222/RealVisXL_V3.0",
        torch_dtype=gen_dtype,
        use_safetensors=True,
        variant="fp16",
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.load_photomaker_adapter(
        os.path.dirname(pm_ckpt),
        subfolder="",
        weight_name=os.path.basename(pm_ckpt),
        trigger_word="img",
        pm_version="v1",
    )
    pipe.id_encoder.to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.fuse_lora()

    prompt = "a realistic studio portrait photograph of a person img, neutral soft lighting, sharp focus"
    neg_prompt = "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoon, sketch), grayscale"
    start_merge_step = max(2, int(0.2 * steps))

    out_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for m_name, regimes in method_tensors.items():
        out_metrics[m_name] = {}
        for r_name, x_batch in regimes.items():
            gen_pils = []
            for k, idx in enumerate(e2e_indices):
                ref_pil = tensor_to_pil(x_batch[idx]).resize((512, 512), Image.BICUBIC)
                gen = torch.Generator(device=device).manual_seed(2101 + k)
                img = pipe(
                    prompt=prompt,
                    input_id_images=[ref_pil],
                    negative_prompt=neg_prompt,
                    num_images_per_prompt=1,
                    num_inference_steps=steps,
                    start_merge_step=start_merge_step,
                    height=gen_size,
                    width=gen_size,
                    generator=gen,
                ).images[0]
                gen_pils.append(img)
            sc = score_generated_pils(
                gen_pils, gt_id_indices, gallery_fn, gallery_buf, eval_fn, eval_buf, tau_fn, tau_buf, device
            )
            out_metrics[m_name][r_name] = sc
            log(
                f"PhotoMaker | {m_name} [{r_name}] -> "
                f"FN: ASR={sc['facenet_asr_pct']:.1f}%, ISM={sc['facenet_ism']:.3f}, FDR={sc['facenet_fdr_pct']:.0f}% | "
                f"Buf: ASR={sc['buffalo_asr_pct']:.1f}%, ISM={sc['buffalo_ism']:.3f}, FDR={sc['buffalo_fdr_pct']:.0f}%"
            )

    del pipe
    clear_cuda()
    return out_metrics


def run_instantid_eval(
    method_tensors: Dict[str, Dict[str, torch.Tensor]],
    e2e_indices: List[int],
    gt_id_indices: List[int],
    gallery_fn: torch.Tensor,
    gallery_buf: np.ndarray,
    eval_fn: FaceNetEncoder,
    eval_buf: InsightFaceNativeEvaluator,
    eval_ant: InsightFaceNativeEvaluator,
    tau_fn: float,
    tau_buf: float,
    device: str,
    steps: int = 20,
    gen_size: int = 768,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    _install_diffusers_compat_shims()
    import cv2
    from diffusers.models import ControlNetModel
    from huggingface_hub import hf_hub_download, snapshot_download

    id_repo = ensure_repo("https://github.com/instantX-research/InstantID.git", Path("/content/InstantID"))
    if str(id_repo) not in sys.path:
        sys.path.insert(0, str(id_repo))
    from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps

    gen_dtype = _best_sdxl_dtype(device)
    log(f"Loading InstantID (ControlNet + ip-adapter.bin + SG161222/RealVisXL_V3.0, {gen_dtype})...")
    cnet_dir = Path("/content/InstantID_ckpt/ControlNetModel")
    if not (cnet_dir / "config.json").exists():
        snapshot_download(
            repo_id="InstantX/InstantID",
            allow_patterns=["ControlNetModel/*"],
            local_dir="/content/InstantID_ckpt",
        )
    ip_path = hf_hub_download(repo_id="InstantX/InstantID", filename="ip-adapter.bin", repo_type="model")

    controlnet = ControlNetModel.from_pretrained(str(cnet_dir), torch_dtype=gen_dtype)
    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        "SG161222/RealVisXL_V3.0",
        controlnet=controlnet,
        torch_dtype=gen_dtype,
        use_safetensors=True,
        variant="fp16",
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.load_ip_adapter_instantid(ip_path)

    prompt = "a realistic studio portrait photograph of a person, neutral soft lighting, sharp focus"
    neg_prompt = "(lowres, low quality, worst quality, illustration, 3d, 2d, painting, cartoon, sketch), grayscale"

    out_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for m_name, regimes in method_tensors.items():
        out_metrics[m_name] = {}
        for r_name, x_batch in regimes.items():
            gen_pils = []
            valid_gt = []
            input_det = 0
            for k, idx in enumerate(e2e_indices):
                ref_pil = tensor_to_pil(x_batch[idx]).resize((gen_size, gen_size), Image.BICUBIC)
                bgr = cv2.cvtColor(np.array(ref_pil), cv2.COLOR_RGB2BGR)
                faces = eval_ant.app.get(bgr)
                if not faces:
                    continue
                input_det += 1
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                face_emb = face["embedding"]
                face_kps = draw_kps(ref_pil, face["kps"])
                gen = torch.Generator(device=device).manual_seed(2101 + k)
                img = pipe(
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    image_embeds=face_emb,
                    image=face_kps,
                    controlnet_conditioning_scale=0.8,
                    ip_adapter_scale=0.8,
                    num_inference_steps=steps,
                    guidance_scale=5.0,
                    height=gen_size,
                    width=gen_size,
                    generator=gen,
                ).images[0]
                gen_pils.append(img)
                valid_gt.append(gt_id_indices[k])

            sc = score_generated_pils(
                gen_pils, valid_gt, gallery_fn, gallery_buf, eval_fn, eval_buf, tau_fn, tau_buf, device
            )
            # Adjust ASR denominator to total planned attempts (standard protocol)
            n_plan = len(e2e_indices)
            sc["input_fdr_pct"] = 100.0 * input_det / max(n_plan, 1)
            sc["facenet_asr_pct"] = sc["facenet_asr_pct"] * (len(gen_pils) / max(n_plan, 1))
            sc["buffalo_asr_pct"] = sc["buffalo_asr_pct"] * (len(gen_pils) / max(n_plan, 1))
            sc["n_attempts"] = n_plan
            out_metrics[m_name][r_name] = sc
            log(
                f"InstantID | {m_name} [{r_name}] -> InFDR={sc['input_fdr_pct']:.0f}% | "
                f"FN: ASR={sc['facenet_asr_pct']:.1f}%, ISM={sc['facenet_ism']:.3f}, FDR={sc['facenet_fdr_pct']:.0f}% | "
                f"Buf: ASR={sc['buffalo_asr_pct']:.1f}%, ISM={sc['buffalo_ism']:.3f}, FDR={sc['buffalo_fdr_pct']:.0f}%"
            )

    del pipe, controlnet
    clear_cuda()
    return out_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scheme 1 Real E2E Benchmark vs Official IDProtector CVPR'25")
    p.add_argument("--num-identities", type=int, default=8)
    p.add_argument("--src-per-id", type=int, default=2)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--fp-weight", type=float, default=0.65)
    p.add_argument("--gen-steps", type=int, default=20)
    p.add_argument("--gen-size", type=int, default=768)
    p.add_argument("--skip-e2e", action="store_true")
    p.add_argument("--e2e-fast", action="store_true", help="Skip Stage 1 ablation & extra purifiers to focus runtime on Stage 2 E2E generators")
    p.add_argument("--generators", type=str, default="both", choices=["both", "photomaker", "instantid"])
    p.add_argument("--out-dir", type=str, default=str(REPO_ROOT / "results"))
    p.add_argument("--out-name", type=str, default="e2e_scheme1_benchmark.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_all = time.time()
    torch.manual_seed(20261123)
    np.random.seed(20261123)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    vram_gb = (torch.cuda.get_device_properties(0).total_memory / 1e9) if device == "cuda" else 0.0
    microbatch = 4 if vram_gb >= 20 else 2
    log(f"Scheme 1 E2E Benchmark on {gpu_name} ({vram_gb:.1f}GB) | ids={args.num_identities}, steps={args.steps}, fp_w={args.fp_weight}")

    # Load 16 real aligned 256x256 portraits (8 distinct identities x 2 photos)
    x_src, x_ref, x_tar, labels = load_lfw_cohort(
        num_identities=args.num_identities,
        photos_per_id=4,
        src_per_id=args.src_per_id,
        device=device,
    )
    unique_ids = list(dict.fromkeys(labels))
    n_imgs = x_src.shape[0]
    log(f"Loaded {n_imgs} portraits across {len(unique_ids)} identities: {unique_ids[:4]}...")

    # Surrogates + Official IDProtector CVPR'25
    log("Loading zero-overlap surrogates + Tweedie UNet + Official IDProtector (CVPR'25)...")
    surr_irse50 = IRSE50Encoder(device=device, use_bgr=True)
    surr_casia = FaceNetEncoder(pretrained="casia-webface", device=device)
    surr_clip_b16 = OpenCLIPViTEncoder(model_name="ViT-B-16", pretrained="openai", device=device)
    surr_clip_b32 = OpenCLIPViTEncoder(model_name="ViT-B-32", pretrained="openai", device=device)
    tweedie_op = TweediePurifierOperator(model_id="google/ddpm-celebahq-256", device=device)
    lpips_fn = LPIPSMetric(net="alex", device=device)
    mask_builder = FaceMaskBuilder(use_mtcnn=True)
    face_masks = mask_builder.build(x_src, device=device)
    idprotector = IDProtectorCVPR25(repo_dir="/content/IDProtector", device=device)

    # Unseen Evaluators + Purifiers
    log("Loading strictly unseen evaluators & cross-model purifiers...")
    eval_fn = FaceNetEncoder(pretrained="vggface2", device=device)
    eval_clip_l14 = UnseenViTEvaluator("ViT-L-14", "openai", img_size=224, device=device)
    eval_eva_l14 = UnseenViTEvaluator("EVA02-L-14-336", "merged2b_s6b_b61k", img_size=336, device=device)
    ctx_id = 0 if device == "cuda" else -1
    eval_buf = InsightFaceNativeEvaluator("buffalo_l", ctx_id=ctx_id)
    eval_ant = InsightFaceNativeEvaluator("antelopev2", ctx_id=ctx_id)
    unseen_sdvae = SDVAEResidualDecoder(device=device, dtype=best_vae_dtype(device))
    diffpure = DiffPurePurifier(device=device, dtype=torch.float16, microbatch=min(4, microbatch))

    with torch.no_grad():
        ref_fn = eval_fn(x_ref)
        ref_clip_l = eval_clip_l14.encode(x_ref)
        ref_eva_l = eval_eva_l14.encode(x_ref)
        clean_clip_b16 = surr_clip_b16.encode_image(x_src)
    ref_buf, _ = eval_buf.extract_batch(x_ref)
    ref_ant, _ = eval_ant.extract_batch(x_ref)

    # Build 1-per-identity clean gallery for end-to-end Rank-1 ASR scoring
    e2e_indices = [labels.index(uid) for uid in unique_ids]
    gt_id_indices = list(range(len(unique_ids)))
    gallery_fn = ref_fn[e2e_indices]  # [num_ids, 512]
    gallery_buf = ref_buf[e2e_indices]  # [num_ids, 512]

    def _cal_far01(se: np.ndarray, re: np.ndarray) -> float:
        imps = [float(np.dot(se[i], re[j])) for i in range(n_imgs) for j in range(n_imgs) if labels[i] != labels[j]]
        return float(np.quantile(imps, 0.99))

    tau_fn = _cal_far01(eval_fn(x_src).detach().cpu().numpy(), ref_fn.detach().cpu().numpy())
    tau_buf = _cal_far01(eval_buf.extract_batch(x_src)[0], ref_buf)
    tau_ant = _cal_far01(eval_ant.extract_batch(x_src)[0], ref_ant)
    log(f"Calibrated FAR=1% thresholds: FaceNet={tau_fn:.3f}, buffalo_l={tau_buf:.3f}, antelopev2={tau_ant:.3f}")

    @torch.no_grad()
    def score_interfaces(x_ev: torch.Tensor) -> Dict[str, float]:
        cos_fn = (eval_fn(x_ev) * ref_fn).sum(dim=-1)
        cos_cl = (eval_clip_l14.encode(x_ev) * ref_clip_l).sum(dim=-1)
        cos_ev = (eval_eva_l14.encode(x_ev) * ref_eva_l).sum(dim=-1)
        eb, db = eval_buf.extract_batch(x_ev)
        cb = np.sum(eb * ref_buf, axis=-1)
        bv = cb[db] if db.any() else np.array([0.0])
        ea, da = eval_ant.extract_batch(x_ev)
        ca = np.sum(ea * ref_ant, axis=-1)
        av = ca[da] if da.any() else np.array([0.0])
        return {
            "pm_clip_l14_cos": float(cos_cl.mean().item()),
            "pulid_eva_cos": float(cos_ev.mean().item()),
            "faceid_buffalo_cos": float(bv.mean()),
            "faceid_buffalo_dodge_pct": float(100.0 * np.mean(bv < tau_buf)),
            "faceid_buffalo_fdr_pct": float(100.0 * np.mean(db)),
            "instantid_antelope_cos": float(av.mean()),
            "instantid_antelope_dodge_pct": float(100.0 * np.mean(av < tau_ant)),
            "instantid_antelope_fdr_pct": float(100.0 * np.mean(da)),
            "facenet_vgg_cos": float(cos_fn.mean().item()),
            "facenet_vgg_dodge_pct": float(100.0 * (cos_fn < tau_fn).float().mean().item()),
        }

    # Precompute subspace helper for uniform 5-axis CLIP Attr-Drift on ALL methods (including IDProtector)
    tmp_prot = PurificationFixedPointProtector(
        [surr_irse50, surr_casia], [surr_clip_b16, surr_clip_b32], tweedie_op, lpips_fn, mask_builder, PFPConfig(steps=1)
    )
    text_axes = tmp_prot.text_axes_raw
    axis_names = tmp_prot.axis_names

    @torch.no_grad()
    def eval_5axis(x_ev: torch.Tensor) -> Dict[str, float]:
        diff = surr_clip_b16.encode_image(x_ev) - clean_clip_b16
        proj = torch.einsum("bd,kd->bk", diff, text_axes).abs()
        out = {ax: float(proj[:, i].mean().item()) for i, ax in enumerate(axis_names)}
        out["mean_5axis_drift"] = float(proj.mean().item())
        return out

    # Generate protected tensors for all methods
    protected: Dict[str, torch.Tensor] = {}
    mech_diags: Dict[str, Dict] = {}

    log("--- Generating IDProtector (CVPR'25 Official) ---")
    t0 = time.time()
    protected["IDProtector (CVPR'25)"] = idprotector.protect(x_src)
    mech_diags["IDProtector (CVPR'25)"] = {"elapsed_per_image_s": (time.time() - t0) / n_imgs}

    pfp_variants = [
        (
            "Baseline: Pixel PGD-40 (Balanced)",
            PFPConfig(
                steps=args.steps,
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
            "Ablation: PFP w/o Fixed-Point Ratio",
            PFPConfig(
                steps=args.steps,
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
            "Ours: PFP-OrthoID (Scale-Fixed L_FP)",
            PFPConfig(
                steps=args.steps,
                use_face_mask=True,
                use_purification_in_loop=True,
                use_fixed_point_loss=True,
                use_pythagorean_target=True,
                use_mgda=True,
                fixed_point_weight=args.fp_weight,
                target_mode="target_free",
            ),
        ),
    ]

    if args.e2e_fast:
        pfp_variants = [v for v in pfp_variants if not v[0].startswith("Ablation:")]

    for name, cfg in pfp_variants:
        log(f"--- Generating {name} ---")
        prot = PurificationFixedPointProtector(
            [surr_irse50, surr_casia],
            [surr_clip_b16, surr_clip_b32],
            tweedie_op,
            lpips_fn,
            mask_builder,
            cfg,
        )
        out = prot.protect_batched(x_src, x_target=x_tar, face_mask=face_masks, microbatch=microbatch)
        protected[name] = out["x_prot"]
        mech_diags[name] = out["diagnostics"]
        clear_cuda()

    # Stage 1 Evaluation across all 5 interfaces x 5 purifiers
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name

    results = {
        "meta": {
            "gpu_name": gpu_name,
            "num_portraits": int(n_imgs),
            "num_identities": int(len(unique_ids)),
            "steps": args.steps,
            "fp_weight": args.fp_weight,
            "gen_steps": args.gen_steps,
            "gen_size": args.gen_size,
            "far01_thresholds": {"facenet": tau_fn, "buffalo_l": tau_buf, "antelopev2": tau_ant},
            "budget_spec": {"eps_255": 8.0, "rms_max_255": 5.5, "quantization": "uint8"},
        },
        "clean_interfaces": score_interfaces(x_src),
        "stage1_interfaces": [],
        "stage2_e2e_generators": {},
    }

    # Precompute DiffPure(t=100) outputs for both Stage 1 and Stage 2 reuse
    dp100_cache: Dict[str, torch.Tensor] = {"Clean": diffpure.purify(x_src, t_start=100, steps=5, seed=42)}

    for name, x_prot in protected.items():
        budget = compute_budget_stats(x_prot, x_src)
        psnr_val = compute_psnr(x_prot, x_src)
        ssim_val = compute_ssim(x_prot, x_src)
        with torch.no_grad():
            lpips_val = float(lpips_fn(x_prot, x_src).mean().item())
        attr_d = eval_5axis(x_prot)
        x_dp100 = diffpure.purify(x_prot, t_start=100, steps=5, seed=42)
        dp100_cache[name] = x_dp100

        row = {
            "method": name,
            "budget": budget,
            "visual": {"psnr_db": psnr_val, "ssim": ssim_val, "lpips": lpips_val},
            "attr_drift": attr_d,
            "direct": score_interfaces(x_prot),
            "diffpure_t100": score_interfaces(x_dp100),
            "mechanism": mech_diags[name],
        }
        if not args.e2e_fast:
            row["jpeg75"] = score_interfaces(purify_jpeg(x_prot, quality=75))
            row["unseen_sdvae"] = score_interfaces(purify_unseen_sdvae(x_prot, unseen_sdvae, microbatch=microbatch))
            row["diffpure_t150"] = score_interfaces(diffpure.purify(x_prot, t_start=150, steps=8, seed=42))
        results["stage1_interfaces"].append(row)
        sdvae_str = (
            f"SDVAE[Ant={row['unseen_sdvae']['instantid_antelope_cos']:.3f} FN_D={row['unseen_sdvae']['facenet_vgg_dodge_pct']:.0f}%] | "
            if "unseen_sdvae" in row
            else ""
        )
        log(
            f"[Stage 1] {name} | PSNR={psnr_val:.2f} LPIPS={lpips_val:.4f} 5Ax={attr_d['mean_5axis_drift']:.4f} "
            f"FP_ratio={mech_diags[name].get('final_fp_ratio')} | "
            f"DIR[PM={row['direct']['pm_clip_l14_cos']:.3f} Buf={row['direct']['faceid_buffalo_cos']:.3f} "
            f"Ant={row['direct']['instantid_antelope_cos']:.3f} FN_D={row['direct']['facenet_vgg_dodge_pct']:.0f}%] | "
            f"{sdvae_str}"
            f"DP100[PM={row['diffpure_t100']['pm_clip_l14_cos']:.3f} Buf={row['diffpure_t100']['faceid_buffalo_cos']:.3f} "
            f"Ant={row['diffpure_t100']['instantid_antelope_cos']:.3f} FN_D={row['diffpure_t100']['facenet_vgg_dodge_pct']:.0f}%]"
        )

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Saved Stage 1 checkpoint to {out_path}")

    # Free surrogate / purifier models before loading SDXL generators (keeps VRAM < 12GB on L4!)
    del surr_irse50, surr_casia, surr_clip_b16, surr_clip_b32, tweedie_op, lpips_fn, idprotector
    del eval_clip_l14, eval_eva_l14, unseen_sdvae, diffpure, tmp_prot
    clear_cuda()

    if not args.skip_e2e:
        e2e_methods = {
            "Clean": {"direct": x_src.cpu(), "diffpure_t100": dp100_cache["Clean"].cpu()},
            "IDProtector (CVPR'25)": {
                "direct": protected["IDProtector (CVPR'25)"].cpu(),
                "diffpure_t100": dp100_cache["IDProtector (CVPR'25)"].cpu(),
            },
            "Baseline: Pixel PGD-40 (Balanced)": {
                "direct": protected["Baseline: Pixel PGD-40 (Balanced)"].cpu(),
                "diffpure_t100": dp100_cache["Baseline: Pixel PGD-40 (Balanced)"].cpu(),
            },
            "Ours: PFP-OrthoID (Scale-Fixed L_FP)": {
                "direct": protected["Ours: PFP-OrthoID (Scale-Fixed L_FP)"].cpu(),
                "diffpure_t100": dp100_cache["Ours: PFP-OrthoID (Scale-Fixed L_FP)"].cpu(),
            },
        }
        del protected, dp100_cache
        clear_cuda()

        # 1. Real End-to-End PhotoMaker SDXL
        if args.generators in ("both", "photomaker"):
            try:
                results["stage2_e2e_generators"]["PhotoMaker"] = run_photomaker_eval(
                    e2e_methods,
                    e2e_indices,
                    gt_id_indices,
                    gallery_fn,
                    gallery_buf,
                    eval_fn,
                    eval_buf,
                    tau_fn,
                    tau_buf,
                    device=device,
                    steps=args.gen_steps,
                    gen_size=args.gen_size,
                )
            except Exception as exc:
                import traceback
                tb = traceback.format_exc()
                log(f"[PhotoMaker ERROR] {exc}\n{tb}")
                results["stage2_e2e_generators"]["PhotoMaker"] = {"error": str(exc), "traceback": tb}
                clear_cuda()
            results["meta"]["total_elapsed_s"] = float(time.time() - t_all)
            out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

        # 2. Real End-to-End InstantID SDXL
        if args.generators in ("both", "instantid"):
            try:
                results["stage2_e2e_generators"]["InstantID"] = run_instantid_eval(
                    e2e_methods,
                    e2e_indices,
                    gt_id_indices,
                    gallery_fn,
                    gallery_buf,
                    eval_fn,
                    eval_buf,
                    eval_ant,
                    tau_fn,
                    tau_buf,
                    device=device,
                    steps=args.gen_steps,
                    gen_size=args.gen_size,
                )
            except Exception as exc:
                import traceback
                tb = traceback.format_exc()
                log(f"[InstantID ERROR] {exc}\n{tb}")
                results["stage2_e2e_generators"]["InstantID"] = {"error": str(exc), "traceback": tb}
                clear_cuda()
            results["meta"]["total_elapsed_s"] = float(time.time() - t_all)
            out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    log(f"ALL DONE in {time.time() - t_all:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()

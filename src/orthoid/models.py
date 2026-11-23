"""Modular encoder, SD-VAE zero-init residual decoder, and DiffPure purifier wrappers."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def best_vae_dtype(device: str) -> torch.dtype:
    """
    Select the fastest numerically stable dtype for SD-VAE backward passes:
      - CUDA SM >= 8.0 (A100 / L4 / H100 / RTX 4090): torch.bfloat16 (native HW bf16, FP32 range, no underflow)
      - Older CUDA (T4 SM 7.5), Apple MPS, or CPU: torch.float32 (avoids FP16 underflow and emulated bf16 overhead)
    """
    if str(device).startswith("cuda") and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        if major >= 8 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
    return torch.float32


class FaceNetEncoder(nn.Module):
    """InceptionResnetV1 Face Recognition encoder wrapper (vggface2 or casia-webface)."""

    def __init__(self, pretrained: str = "vggface2", device: str = "cpu"):
        super().__init__()
        from facenet_pytorch import InceptionResnetV1

        self.name = f"facenet_{pretrained}"
        self.model = InceptionResnetV1(pretrained=pretrained).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        x_in = F.interpolate(x01 * 2.0 - 1.0, size=(160, 160), mode="bilinear", align_corners=False)
        return F.normalize(self.model(x_in), dim=-1)


class OpenCLIPViTEncoder(nn.Module):
    """
    Aligned OpenCLIP image + text encoder (default: ViT-B-16 openai).
    Provides shared 512-d unit-sphere image and zero-shot text embeddings.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        device: str = "cpu",
    ):
        super().__init__()
        import open_clip

        self.name = f"openclip_{model_name}_{pretrained}"
        self.device = torch.device(device)
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.register_buffer(
            "mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1),
        )

    def encode_image(self, x01: torch.Tensor) -> torch.Tensor:
        x_in = F.interpolate(x01, size=(224, 224), mode="bilinear", align_corners=False)
        x_norm = (x_in - self.mean) / self.std
        return F.normalize(self.model.encode_image(x_norm), dim=-1)

    @torch.no_grad()
    def encode_text(self, prompts: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer(list(prompts)).to(self.device)
        return F.normalize(self.model.encode_text(tokens), dim=-1)

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        return self.encode_image(x01)


class SDVAEResidualDecoder(nn.Module):
    """
    Zero-Initialized Face-Localized SD-VAE Latent-Residual Synthesis:
        x'(dz) = clip(x + M_face * (D_VAE(z_0 + dz) - D_VAE(z_0)), 0, 1)
    At dz = 0, x'(0) == x identically (PSNR = inf, zero background error).
    """

    def __init__(
        self,
        model_id: str = "stabilityai/sd-vae-ft-mse",
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        local_files_only: bool = False,
    ):
        super().__init__()
        from diffusers import AutoencoderKL

        self.device = torch.device(device)
        self.dtype = dtype
        self.vae = AutoencoderKL.from_pretrained(
            model_id,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        ).eval().to(self.device)
        for p in self.vae.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode_clean(self, x01: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode clean image batch x01 -> (z0, clean_recon_01)."""
        x_in = (x01 * 2.0 - 1.0).to(dtype=self.dtype)
        z0 = self.vae.encode(x_in).latent_dist.mean
        x_hat0 = ((self.vae.decode(z0).sample.float() + 1.0) * 0.5).clamp(0.0, 1.0)
        return z0.float(), x_hat0

    def synthesize(
        self,
        x_clean: torch.Tensor,
        z0: torch.Tensor,
        x_hat0: torch.Tensor,
        dz: torch.Tensor,
        face_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable zero-init residual synthesis in [0, 1]."""
        z_cur = (z0 + dz).to(dtype=self.dtype)
        x_hat = ((self.vae.decode(z_cur).sample.float() + 1.0) * 0.5).clamp(0.0, 1.0)
        delta_vae = x_hat - x_hat0
        return (x_clean + face_mask * delta_vae).clamp(0.0, 1.0)


class DiffPurePurifier(nn.Module):
    """Real DDPM/DDIM diffusion purifier using google/ddpm-celebahq-256."""

    def __init__(
        self,
        model_id: str = "google/ddpm-celebahq-256",
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = False,
        microbatch: int = 2,
    ):
        super().__init__()
        from diffusers import DDIMScheduler, UNet2DModel

        self.device = torch.device(device)
        self.dtype = dtype
        self.microbatch = int(microbatch)
        self.unet = UNet2DModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        ).eval().to(self.device)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        self.scheduler = DDIMScheduler.from_pretrained(model_id, local_files_only=local_files_only)
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

    @torch.no_grad()
    def purify(
        self,
        x01: torch.Tensor,
        t_start: int = 100,
        steps: int = 5,
        seed: int = 0,
    ) -> torch.Tensor:
        if t_start <= 0:
            return x01.clone()
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        noise_full = torch.randn(x01.shape, generator=gen, dtype=torch.float32).to(self.device)
        ts = np.linspace(t_start, 0, steps + 1, dtype=int)
        a_bar = float(self.alphas_cumprod[t_start])

        outs = []
        for i in range(0, x01.shape[0], self.microbatch):
            xb = x01[i : i + self.microbatch] * 2.0 - 1.0
            nb = noise_full[i : i + self.microbatch]
            cur = (math.sqrt(a_bar) * xb + math.sqrt(1.0 - a_bar) * nb).to(dtype=self.dtype)
            for idx in range(steps):
                t_cur = int(ts[idx])
                t_prev = int(ts[idx + 1])
                a_cur = float(self.alphas_cumprod[t_cur])
                a_prev = float(self.alphas_cumprod[t_prev]) if t_prev > 0 else 1.0
                eps_pred = self.unet(cur, t_cur).sample.float()
                x0_pred = ((cur.float() - math.sqrt(1.0 - a_cur) * eps_pred) / math.sqrt(a_cur)).clamp(-1.0, 1.0)
                if t_prev > 0:
                    cur = (math.sqrt(a_prev) * x0_pred + math.sqrt(1.0 - a_prev) * eps_pred).to(dtype=self.dtype)
                else:
                    cur = x0_pred.to(dtype=self.dtype)
            outs.append(((cur.float() + 1.0) * 0.5).clamp(0.0, 1.0))
        return torch.cat(outs, dim=0)


class LPIPSMetric(nn.Module):
    """Differentiable AlexNet LPIPS wrapper for active-set perceptual retraction & eval."""

    def __init__(self, net: str = "alex", device: str = "cpu"):
        super().__init__()
        import lpips

        self.fn = lpips.LPIPS(net=net, verbose=False).eval().to(device)
        for p in self.fn.parameters():
            p.requires_grad_(False)

    def forward(self, x01: torch.Tensor, y01: torch.Tensor) -> torch.Tensor:
        return self.fn(x01 * 2.0 - 1.0, y01 * 2.0 - 1.0).view(-1)

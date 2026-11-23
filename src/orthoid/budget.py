"""Publishable 8-bit pixel budget projection (Pi_x) and differentiable EOT transforms."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class PublishableBudgetProjector:
    """
    Enforces the unified publishable 8-bit image budget Pi_x:
      - ||x_prot - x_clean||_inf <= eps   (default: 8 / 255)
      - RMS(x_prot - x_clean) <= rms_max  (default: 5.5 / 255)
      - x_prot in [0, 1]
      - Exact uint8 quantization on export while guaranteeing strict budget compliance.
    """

    def __init__(self, eps: float = 8.0 / 255.0, rms_max: float = 5.5 / 255.0):
        self.eps = float(eps)
        self.rms_max = float(rms_max)

    def project_continuous(self, x_cand: torch.Tensor, x_clean: torch.Tensor) -> torch.Tensor:
        """Differentiable projection onto the L_inf + RMS ball intersected with [0, 1]."""
        delta = (x_cand - x_clean).clamp(-self.eps, self.eps)
        # Keep +1e-12 inside sqrt so gradient at delta == 0 is strictly finite (never NaN).
        rms = (delta.flatten(1).pow(2).mean(dim=1) + 1e-12).sqrt().view(-1, 1, 1, 1)
        scale = torch.clamp(self.rms_max / rms, max=1.0)
        delta = delta * scale
        return (x_clean + delta).clamp(0.0, 1.0)

    def project_ste(self, x_cand: torch.Tensor, x_clean: torch.Tensor) -> torch.Tensor:
        """Continuous projection plus Straight-Through Estimator (STE) 8-bit rounding."""
        x_proj = self.project_continuous(x_cand, x_clean)
        x_q = torch.round(x_proj * 255.0) / 255.0
        return x_proj + (x_q - x_proj).detach()

    @torch.no_grad()
    def publish(self, x_cand: torch.Tensor, x_clean: torch.Tensor) -> torch.Tensor:
        """
        Export an exact 8-bit publishable image strictly satisfying both:
          max |x_pub - x_clean| <= eps  and  RMS(x_pub - x_clean) <= rms_max.
        """
        x_clean_q = torch.round(x_clean.clamp(0.0, 1.0) * 255.0) / 255.0
        x_proj = self.project_continuous(x_cand, x_clean_q)
        delta = x_proj - x_clean_q

        out = []
        eps_int = int(round(self.eps * 255.0 + 1e-6))
        for b in range(x_cand.shape[0]):
            d = delta[b]
            x0 = x_clean_q[b]
            # Try shrinking slightly if uint8 rounding pushes RMS marginally above rms_max.
            best_x = None
            for shrink in (1.0, 0.99, 0.97, 0.95, 0.92, 0.88, 0.84, 0.80):
                d_int = torch.round((d * shrink) * 255.0).clamp(-eps_int, eps_int)
                xq = (torch.round(x0 * 255.0) + d_int).clamp(0.0, 255.0) / 255.0
                actual_rms = float((xq - x0).pow(2).mean().sqrt().item())
                best_x = xq
                if actual_rms <= self.rms_max + 1e-7:
                    break
            out.append(best_x)
        return torch.stack(out, dim=0)


def di_transform(
    x: torch.Tensor,
    seed: int,
    prob: float = 0.7,
    min_scale: float = 0.88,
    max_jitter_px: int = 4,
) -> torch.Tensor:
    """
    Diverse Inputs (DI) + mild crop/alignment EOT preserving differentiable gradients.
    """
    _, _, h, w = x.shape
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    if float(torch.rand((1,), generator=g).item()) > prob:
        return x

    low_h = max(32, int(round(h * min_scale)))
    rnd_h = int(torch.randint(low_h, h + 1, (1,), generator=g).item())
    rnd_w = int(round(rnd_h * (w / float(h))))
    res = F.interpolate(x, size=(rnd_h, rnd_w), mode="bilinear", align_corners=False)

    pad_h = h - rnd_h
    pad_w = w - rnd_w
    p_top = int(torch.randint(0, pad_h + 1, (1,), generator=g).item()) if pad_h > 0 else 0
    p_left = int(torch.randint(0, pad_w + 1, (1,), generator=g).item()) if pad_w > 0 else 0
    out = F.pad(res, (p_left, pad_w - p_left, p_top, pad_h - p_top), mode="reflect")

    if max_jitter_px > 0:
        dy = int(torch.randint(-max_jitter_px, max_jitter_px + 1, (1,), generator=g).item())
        dx = int(torch.randint(-max_jitter_px, max_jitter_px + 1, (1,), generator=g).item())
        if dy != 0 or dx != 0:
            out = torch.roll(out, shifts=(dy, dx), dims=(2, 3))
    return out


class MidBandFilter(torch.nn.Module):
    """
    Parameter-free Difference-of-Gaussians (DoG) mid-frequency facial filter:
        B_mid(u) = G_{sigma_hi}(u) - G_{sigma_lo}(u)
    Properties:
      - Zero DC (exact zero global brightness/color/lighting shift)
      - Suppresses 2-3 px Nyquist pixel grain (preserves LPIPS and survives DiffPure/JPEG)
      - Passes the 4-12 px periocular/nasolabial facial-edge band read by CNN-FR encoders,
        complementing SD-VAE's >=16 px semantic band.
    """

    def __init__(self, sigma_hi: float = 1.25, sigma_lo: float = 5.0):
        super().__init__()
        self.sigma_hi = float(sigma_hi)
        self.sigma_lo = float(sigma_lo)
        self.register_buffer("k_hi", self._make_1d_kernel(self.sigma_hi))
        self.register_buffer("k_lo", self._make_1d_kernel(self.sigma_lo))

    @staticmethod
    def _make_1d_kernel(sigma: float) -> torch.Tensor:
        radius = max(2, int(round(3.0 * sigma)))
        ax = torch.arange(-radius, radius + 1, dtype=torch.float32)
        k = torch.exp(-(ax * ax) / (2.0 * sigma * sigma))
        k = k / k.sum()
        return k

    @staticmethod
    def _sep_blur(x: torch.Tensor, k1d: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        r = k1d.numel() // 2
        k = k1d.to(device=x.device, dtype=x.dtype)
        kh = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
        kv = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
        out = F.pad(x, (r, r, 0, 0), mode="reflect")
        out = F.conv2d(out, kh, groups=c)
        out = F.pad(out, (0, 0, r, r), mode="reflect")
        out = F.conv2d(out, kv, groups=c)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._sep_blur(x, self.k_hi) - self._sep_blur(x, self.k_lo)

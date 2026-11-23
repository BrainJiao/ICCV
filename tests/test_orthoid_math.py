"""Fast mathematical unit tests for OrthoID core primitives (Proposition 1, Proposition 2, Pi_x)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from orthoid.budget import MidBandFilter, PublishableBudgetProjector
from orthoid.optim import mgda_two_task
from orthoid.subspace import (
    _orthonormalize_rows,
    project_nullspace,
    project_onto_subspace,
    pythagorean_counterfactual_target,
)


def test_pythagorean_target_proposition_1() -> None:
    torch.manual_seed(0)
    b, d, k = 4, 512, 8
    f_src = F.normalize(torch.randn(b, d), dim=-1)
    e_tar = F.normalize(torch.randn(b, d), dim=-1)
    q_attr = _orthonormalize_rows(torch.randn(b, k, d))

    # Targeted mode
    e_tilde = pythagorean_counterfactual_target(f_src, e_tar, q_attr)
    # 1. Exact unit norm
    norms = e_tilde.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), f"Non-unit norms: {norms}"
    # 2. Exact attribute coordinate preservation: Q_attr^T e_tilde == Q_attr^T f_src
    c_src = torch.einsum("bkd,bd->bk", q_attr, f_src)
    c_tilde = torch.einsum("bkd,bd->bk", q_attr, e_tilde)
    assert torch.allclose(c_src, c_tilde, atol=1e-5), f"Attribute mismatch: {(c_src - c_tilde).abs().max()}"

    # Target-free mode
    e_free = pythagorean_counterfactual_target(f_src, None, q_attr)
    assert torch.allclose(e_free.norm(dim=-1), torch.ones(b), atol=1e-5)
    c_free = torch.einsum("bkd,bd->bk", q_attr, e_free)
    assert torch.allclose(c_src, c_free, atol=1e-5)
    # Orthogonal component is strictly antipodal
    perp_src = project_nullspace(f_src, q_attr)
    perp_free = project_nullspace(e_free, q_attr)
    cos_perp = F.cosine_similarity(perp_src, perp_free, dim=-1)
    assert torch.allclose(cos_perp, -torch.ones(b), atol=1e-5)


def test_mgda_proposition_2() -> None:
    torch.manual_seed(1)
    b = 8
    g_cnn = torch.randn(b, 4, 16, 16) * 5.0
    # Deliberately construct conflicting g_vit with negative cosine similarity
    g_vit = -0.6 * g_cnn / 5.0 + 0.8 * torch.randn(b, 4, 16, 16)

    d_star, stats = mgda_two_task(g_cnn, g_vit, normalize_scales=True)
    assert bool(stats.conflict_mask.any()), "Expected conflicting gradients in test case"

    dot1 = (d_star.reshape(b, -1) * g_cnn.reshape(b, -1)).sum(dim=1)
    dot2 = (d_star.reshape(b, -1) * g_vit.reshape(b, -1)).sum(dim=1)
    assert bool((dot1 > 0).all()), f"CNN descent violated: {dot1}"
    assert bool((dot2 > 0).all()), f"ViT descent violated: {dot2}"


def test_publishable_budget_pi_x() -> None:
    torch.manual_seed(2)
    proj = PublishableBudgetProjector(eps=8.0 / 255.0, rms_max=5.5 / 255.0)
    x_clean = torch.rand(4, 3, 64, 64)
    # Extreme perturbation violating both L_inf and RMS
    x_cand = (x_clean + 0.25 * torch.randn_like(x_clean)).clamp(0.0, 1.0)
    x_pub = proj.publish(x_cand, x_clean)

    x_clean_q = torch.round(x_clean * 255.0) / 255.0
    delta = (x_pub - x_clean_q).abs()
    assert float(delta.max().item()) <= 8.0 / 255.0 + 1e-6, f"L_inf exceeded: {delta.max().item() * 255}"
    rms = delta.flatten(1).pow(2).mean(dim=1).sqrt()
    assert float(rms.max().item()) <= 5.5 / 255.0 + 1e-6, f"RMS exceeded: {rms.max().item() * 255}"
    # Exact uint8 values
    q_err = (x_pub * 255.0 - torch.round(x_pub * 255.0)).abs().max().item()
    assert q_err < 1e-6, f"Not exact 8-bit: {q_err}"

    # Regression test: autograd at exact zero perturbation (delta == 0) must be finite (no NaN)
    p = torch.zeros_like(x_clean, requires_grad=True)
    x_zero = proj.project_continuous((x_clean + p).clamp(0.0, 1.0), x_clean)
    loss = (x_zero ** 2).mean()
    g = torch.autograd.grad(loss, p)[0]
    assert torch.isfinite(g).all(), "Non-finite gradient at delta == 0!"


def test_midband_dog_filter() -> None:
    filt = MidBandFilter(sigma_hi=1.25, sigma_lo=5.0)
    # 1. Zero-init exactness
    z = torch.zeros(2, 3, 64, 64)
    assert torch.allclose(filt(z), z, atol=1e-7)
    # 2. Exact zero-DC (no global brightness / color shift)
    ones = torch.ones(2, 3, 64, 64)
    assert filt(ones).abs().max().item() < 1e-5
    # 3. High-freq Nyquist checkerboard strongly suppressed vs 12-px mid-band wave passed
    yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    checker = (((xx + yy) % 2) * 2.0 - 1.0).view(1, 1, 64, 64).repeat(1, 3, 1, 1)
    mid_wave = torch.sin(2.0 * torch.pi * xx.float() / 12.0).view(1, 1, 64, 64).repeat(1, 3, 1, 1)
    rms_checker = filt(checker).pow(2).mean().sqrt().item()
    rms_mid = filt(mid_wave).pow(2).mean().sqrt().item()
    assert rms_checker < 0.05, f"Nyquist grain not suppressed: {rms_checker}"
    assert rms_mid > 0.35, f"Mid-band signal overly attenuated: {rms_mid}"


if __name__ == "__main__":
    test_pythagorean_target_proposition_1()
    test_mgda_proposition_2()
    test_publishable_budget_pi_x()
    test_midband_dog_filter()
    print("ALL_ORTHOID_MATH_TESTS_PASSED")

# OrthoID

Standalone PyTorch implementation of **OrthoID** (*Attribute-Nullspace Decoupling and Conflict-Free Cross-Paradigm Optimization for Proactive Portrait Protection*).

## Structure
- `src/orthoid/` — Core zero-training protection modules (zero-init face-localized SD-VAE + DoG mid-band residual, online 5-axis CLIP attribute nullspace & Pythagorean counterfactual target, conflict-free 2-task MGDA + flatness lookahead, differentiable 5-point ArcFace alignment EOT, publishable 8-bit budget projector `Pi_x`).
- `tests/test_orthoid_math.py` — Unit tests verifying Proposition 1, Proposition 2, `Pi_x` budget compliance, and DoG mid-band properties.
- `scripts/run_m0_local.py` — M0 sanity gate.
- `scripts/run_m1a_zero_overlap.py` — M1-A strict zero-encoder-overlap 5-interface benchmark.

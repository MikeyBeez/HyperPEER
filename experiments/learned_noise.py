"""Learned inter-step noise for HyperPEER recursion (additive, non-invasive).

Mikey's idea (2026-06-13): the act_noise control injected ISOTROPIC Gaussian
noise between recursion steps and did nothing. Replace it with a *separate head
that learns what noise to produce*.

Key design choice — energy matching. Under a plain next-token / distill loss a
fully-learned noise head has a degenerate optimum: drive the noise magnitude to
zero (noise only hurts a deterministic objective). To get a clean experiment we
therefore FIX the per-token noise energy to the same value the isotropic control
used (target_std * RMS(u)) and let the head learn only the DIRECTION. So:

    iso      : u += randn_like(u) * (target_std * RMS(u))          # random dir
    learned  : u += unit(head(u)) * (target_std * RMS(u) * sqrt d) # learned dir

Both inject identical energy; the only difference is whether the direction is
chosen at random or by a trained head. If `learned` beats `iso` and the plain
`rederive` arm, a structured perturbation genuinely helps the second pass — the
core of the recursion hypothesis. If it ties them, structured noise is worthless
too (a sharper null than act_noise gave).

This module is self-contained: it imports only the stable ExpertGenerator and
does NOT modify src/generator.py or experiments/recursive_stage3.py, so the live
wt2/rec pipeline and the verdict eval that re-import those files are unaffected.
"""

import math
import torch
import torch.nn as nn


class NoiseHead(nn.Module):
    """Per-token, per-dim learned perturbation DIRECTION of the inter-step state.

    Zero-init on the output projection => at step 0 the direction is the zero
    vector and the injected perturbation is exactly 0, so training starts
    numerically identical to the no-noise `rederive` arm (D2L stable-launch
    philosophy). The head then learns a non-zero direction via gradient.

    One head is shared across all layers (like the generator), so its parameter
    count is tiny relative to the generator.
    """

    def __init__(self, d_model, hidden=None):
        super().__init__()
        hidden = hidden or d_model
        self.in_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.proj = nn.Linear(hidden, d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, u, target_std):
        # u: [b, n, d]; target_std: scalar fraction of per-token RMS to inject.
        rms = u.pow(2).mean(dim=-1, keepdim=True).sqrt()           # [b, n, 1]
        d = u.shape[-1]
        z = self.act(self.fc1(self.in_norm(u)))
        raw = self.proj(z)                                         # [b, n, d], 0 at init
        unit = raw / (raw.norm(dim=-1, keepdim=True) + 1e-6)       # per-token unit dir
        mag = target_std * rms * math.sqrt(d)                      # match isotropic energy
        return unit * mag


class RecursiveNoiseFFN(nn.Module):
    """Same recursion cell as src.generator.RecursiveGeneratedFFN, plus a
    pluggable inter-step noise mode: 'none' | 'iso' | 'learned'.

        u_0 = x
        for t in 1..T:
            h   = norm(u)
            wd,wu = generate(h)                 # rederive='step' every t
            u   = u + sum_i wu_i * act(wd_i . h)
            if t < T-1: u += noise(u)           # iso or learned
        return u_T - x                          # block adds the residual once

    With t_steps=1 this is numerically identical to GeneratedFFN.
    """

    def __init__(self, orig_ffn, generator, noise_head, layer_idx, t_steps=2,
                 rederive="step", noise_mode="none", target_std=0.02,
                 grad_ckpt=True):
        super().__init__()
        self.orig = orig_ffn
        self.layer_idx = layer_idx
        self.t_steps = t_steps
        self.rederive = rederive
        self.noise_mode = noise_mode
        self.target_std = target_std
        self.grad_ckpt = grad_ckpt
        self.use_generated = False
        # plain attributes, NOT submodules: generator + head are optimized
        # outside the frozen base model (exactly like D2L's Perceiver).
        object.__setattr__(self, "_generator", generator)
        object.__setattr__(self, "_noise_head", noise_head)

    def _gen(self, h):
        if self.grad_ckpt and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                self._generator, h, self.layer_idx, use_reentrant=False)
        return self._generator(h, self.layer_idx)

    def forward(self, x, collect=False):
        if not self.use_generated:
            return self.orig(x, collect=collect)
        u = x
        wd = wu = None
        for t in range(self.t_steps):
            h = self.orig.norm(u)
            if wd is None or self.rederive == "step":
                wd, wu = self._gen(h)
            a = torch.einsum("bnd,bnkd->bnk", h, wd)
            a = self.orig.activation(a)
            u = u + torch.einsum("bnk,bnkd->bnd", a, wu)
            if t < self.t_steps - 1 and self.noise_mode != "none":
                if self.noise_mode == "iso":
                    rms = u.pow(2).mean(dim=-1, keepdim=True).sqrt()
                    u = u + torch.randn_like(u) * (self.target_std * rms)
                elif self.noise_mode == "learned":
                    u = u + self._noise_head(u, self.target_std)
                else:
                    raise ValueError(f"unknown noise_mode {self.noise_mode}")
        return u - x, {}


def install_recursive_noise_ffn(model, generator, noise_head, t_steps=2,
                                rederive="step", noise_mode="none",
                                target_std=0.02, grad_ckpt=True):
    """Wrap every block's FFN with the noise-recursion cell; return wrappers."""
    wrappers = []
    for li, blk in enumerate(model.blocks):
        w = RecursiveNoiseFFN(blk.ffn, generator, noise_head, li,
                              t_steps=t_steps, rederive=rederive,
                              noise_mode=noise_mode, target_std=target_std,
                              grad_ckpt=grad_ckpt)
        blk.ffn = w
        wrappers.append(w)
    return wrappers

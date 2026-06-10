"""HyperPEER generator: D2L (doc-to-lora) logic applied to PEER experts.

Source method: ~/Code/doc-to-lora/experiments/phase03_perceiver_train.py.
Kept from D2L, element for element:

  - Perceiver trunk: learned latents -> cross-attention to the conditioning
    input -> self-attention blocks (PerceiverBlockCross / PerceiverBlockSelf).
  - Hypernet head: (layer_embed + row queries) attention-pool over the
    latents, then two linear projections emit the adapter matrices.
  - ZERO-INIT on the up/B projection so the generated adapter starts as a
    zero contribution (D2L's fp16-stability + stable-launch trick).
  - Swappable injection: GeneratedFFN wrappers play the role of D2L's
    LoRAInjectedLinear + set_lora/clear_lora — teacher mode runs the real
    PEER FFN, generated mode runs the hypernet's experts. Same frozen base.

The one adaptation: D2L conditions on a PASSAGE and emits one adapter; PEER's
expert matrix is a function of each TOKEN's hidden state (the teacher's
retrieval queries depend only on the post-attention hidden state). So the
Perceiver here conditions on the per-token hidden state and the head's rank
queries become k_gen row queries emitting the [k x d] down/up rows. A layer
embedding lets one generator serve all n_layers (as D2L's layer_embed does).

Gate folding: the teacher's per-token function is
    out = sum_i w_i * up_i * GELU(down_i . rmsnorm(x))
The generator folds w_i into the up rows (w_i * up_i is just another row
vector), so its target is a clean pair of [k x d] matrices:
    out = sum_i gen_up_i * GELU(gen_down_i . rmsnorm(x))
"""

import torch
import torch.nn as nn


# ============================================================
# Perceiver blocks — same as D2L phase03
# ============================================================
class PerceiverBlockCross(nn.Module):
    def __init__(self, d, kv_d, heads=8):
        super().__init__()
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(kv_d)
        self.attn = nn.MultiheadAttention(d, heads, kdim=kv_d, vdim=kv_d, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, latents, kv):
        h, _ = self.attn(self.ln_q(latents), self.ln_kv(kv), self.ln_kv(kv))
        latents = latents + h
        latents = latents + self.mlp(self.ln2(latents))
        return latents


class PerceiverBlockSelf(nn.Module):
    def __init__(self, d, heads=8):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, latents):
        h, _ = self.attn(self.ln1(latents), self.ln1(latents), self.ln1(latents))
        latents = latents + h
        latents = latents + self.mlp(self.ln2(latents))
        return latents


# ============================================================
# Expert hypernet head — D2L's LoRAHypernet with rank -> k_gen rows
# ============================================================
class ExpertHypernetHead(nn.Module):
    """(layer_embed + row queries) pool over latents -> [k_gen x d] down/up rows.

    D2L mapping: rank_query -> row_query (k_gen of them), proj_a -> proj_down,
    proj_b -> proj_up (ZERO-INIT, so generated experts start as a no-op).
    """

    def __init__(self, latent_d, n_layers, k_gen, d_model, heads=8):
        super().__init__()
        self.k_gen = k_gen
        self.layer_embed = nn.Embedding(n_layers, latent_d)
        self.row_query = nn.Parameter(torch.randn(k_gen, latent_d) * 0.02)
        self.ln_q = nn.LayerNorm(latent_d)
        self.ln_kv = nn.LayerNorm(latent_d)
        self.pool_attn = nn.MultiheadAttention(latent_d, heads, batch_first=True)
        self.proj_down = nn.Linear(latent_d, d_model)
        self.proj_up = nn.Linear(latent_d, d_model)
        nn.init.normal_(self.proj_down.weight, std=0.005)
        nn.init.zeros_(self.proj_down.bias)
        nn.init.zeros_(self.proj_up.weight)      # D2L proj_b zero-init
        nn.init.zeros_(self.proj_up.bias)

    def forward(self, latents, layer_idx):
        # latents: [B, latent_n, latent_d]  (B = one entry per token)
        B = latents.size(0)
        le = self.layer_embed(torch.tensor(layer_idx, device=latents.device))
        q = self.ln_q(self.row_query + le).unsqueeze(0).expand(B, -1, -1)   # [B, k, ld]
        kv = self.ln_kv(latents)
        pooled, _ = self.pool_attn(q, kv, kv)                               # [B, k, ld]
        wd = self.proj_down(pooled)                                         # [B, k, d]
        wu = self.proj_up(pooled)                                           # [B, k, d]
        return wd, wu


class ExpertGenerator(nn.Module):
    """hidden state -> [k_gen x d] down/up expert matrices (per token, per layer)."""

    def __init__(self, d_model, n_layers, k_gen=256, latent_n=8, latent_d=256,
                 n_cross=2, n_self=2, heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.k_gen = k_gen
        self.latents = nn.Parameter(torch.randn(1, latent_n, latent_d) * 0.02)
        self.cross_blocks = nn.ModuleList(
            [PerceiverBlockCross(latent_d, d_model, heads) for _ in range(n_cross)]
        )
        self.self_blocks = nn.ModuleList(
            [PerceiverBlockSelf(latent_d, heads) for _ in range(n_self)]
        )
        self.head = ExpertHypernetHead(latent_d, n_layers, k_gen, d_model, heads)

    def forward(self, x, layer_idx):
        """x: [b, n, d] hidden states -> wd, wu each [b, n, k_gen, d]."""
        b, n, d = x.shape
        kv = x.reshape(b * n, 1, d)                       # each token = 1 kv entry
        latents = self.latents.expand(b * n, -1, -1)
        for blk in self.cross_blocks:
            latents = blk(latents, kv)
        for blk in self.self_blocks:
            latents = blk(latents)
        wd, wu = self.head(latents, layer_idx)            # [b*n, k, d]
        return wd.view(b, n, self.k_gen, d), wu.view(b, n, self.k_gen, d)

    def config(self):
        return {
            "d_model": self.d_model, "n_layers": self.n_layers, "k_gen": self.k_gen,
            "latent_n": self.latents.shape[1], "latent_d": self.latents.shape[2],
            "n_cross": len(self.cross_blocks), "n_self": len(self.self_blocks),
        }


# ============================================================
# Swappable injection — D2L's set_lora / clear_lora, FFN-shaped
# ============================================================
class GeneratedFFN(nn.Module):
    """Wraps a trained AdaptivePEER FFN. teacher mode -> the real PEER forward;
    generated mode -> experts produced by the hypernet from this token's hidden
    state. Reuses the teacher's own RMSNorm and activation so the two modes
    compute in the same frame."""

    def __init__(self, orig_ffn, generator, layer_idx, grad_ckpt=True):
        super().__init__()
        self.orig = orig_ffn
        self.layer_idx = layer_idx
        self.use_generated = False
        self.grad_ckpt = grad_ckpt
        # plain attribute, NOT a submodule: the generator is owned/optimized
        # outside the base model, exactly like D2L's Perceiver vs base.
        object.__setattr__(self, "_generator", generator)

    def forward(self, x, collect=False):
        if not self.use_generated:
            return self.orig(x, collect=collect)
        xn = self.orig.norm(x)                            # teacher's RMSNorm
        if self.grad_ckpt and torch.is_grad_enabled():
            # checkpoint ONLY the generator: its per-token Perceiver internals
            # dominate memory; recompute them in backward. (Checkpointing the
            # whole transformer Block trips a saved-tensor mismatch; the bare
            # generator is a clean pure function and checkpoints fine.)
            wd, wu = torch.utils.checkpoint.checkpoint(
                self._generator, xn, self.layer_idx, use_reentrant=False)
        else:
            wd, wu = self._generator(xn, self.layer_idx)  # [b,n,k,d] x2
        h = torch.einsum("bnd,bnkd->bnk", xn, wd)
        h = self.orig.activation(h)
        out = torch.einsum("bnk,bnkd->bnd", h, wu)
        return out, {}


def install_generated_ffn(model, generator):
    """Wrap every block's FFN; returns the wrappers (D2L install_lora_wrappers)."""
    wrappers = []
    for li, blk in enumerate(model.blocks):
        w = GeneratedFFN(blk.ffn, generator, li)
        blk.ffn = w
        wrappers.append(w)
    return wrappers


def set_generated(wrappers, on):
    """on=True -> student (generated experts); on=False -> teacher (real PEER)."""
    for w in wrappers:
        w.use_generated = on

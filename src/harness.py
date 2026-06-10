"""HyperPEER Stage-0 harness: capture (hidden_state -> [k x d] expert matrix) pairs.

Runs the trained PEER teacher (peer-adaptive-k, fixed-k checkpoint) over token
batches and yields, per PEER layer and per token:

  hidden      : the residual-stream hidden state entering the PEER FFN  [b, n, d]
  expert_ids  : the k_max retrieved expert indices                      [b, n, h, k]
  gate_w      : renormalised expert weights w_i (softmax over used set) [b, n, h, k]
  logits      : teacher LM logits (for logit-KL distillation)           [b, n, V]

Nothing is extracted or stored: the [k x d] down/up matrices are assembled on
the fly from expert_ids via `expert_matrices()` (a gather from the teacher's
embedding tables), exactly reproducing the wd/wu the teacher used internally.

The generator's training target for a token with hidden state x is:

  out = sum_i  w_i * up_i * GELU(down_i . rmsnorm(x))

`verify_reconstruction()` checks this identity against the teacher's actual
FFN output and is the smoke test (run this file as a script).

Usage (on pop, from ~/Code/HN/HyperPEER):
  ~/Code/HN/peer-adaptive-k/.venv/bin/python -m src.harness            # smoke test
"""

import os
import sys
import importlib.util

import torch
import torch.nn.functional as F

TEACHER_REPO = os.path.expanduser("~/Code/HN/peer-adaptive-k")
DEFAULT_CKPT = os.path.join(TEACHER_REPO, "checkpoints", "p0_matched_k256.pt")


def _load_teacher_pkg(repo=TEACHER_REPO):
    """Import peer-adaptive-k's `src` package under the alias `peer_src`,
    so it cannot collide with HyperPEER's own `src` package."""
    if "peer_src" in sys.modules:
        return sys.modules["peer_src"]
    pkg_dir = os.path.join(repo, "src")
    spec = importlib.util.spec_from_file_location(
        "peer_src",
        os.path.join(pkg_dir, "__init__.py"),
        submodule_search_locations=[pkg_dir],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["peer_src"] = mod
    spec.loader.exec_module(mod)
    return mod


class TeacherHarness:
    """Wraps the trained PEER LM; captures per-layer (hidden, ids, gate, logits)."""

    def __init__(self, ckpt_path=DEFAULT_CKPT, device="cuda", repo=TEACHER_REPO):
        _load_teacher_pkg(repo)
        from peer_src.config import Config
        from peer_src.model import PEERLanguageModel
        from peer_src.data import TokenData

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg_d = dict(ckpt["cfg"])
        if isinstance(cfg_d.get("k_clamp"), list):
            cfg_d["k_clamp"] = tuple(cfg_d["k_clamp"])
        self.cfg = Config(**cfg_d)
        self.device = device

        self.model = PEERLanguageModel(self.cfg)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.data = TokenData(
            data_dir=os.path.join(repo, "data"),
            context_len=self.cfg.context_len,
            device=device,
        )
        self.n_layers = self.cfg.n_layers
        self.k_max = self.cfg.k_max
        self.d_model = self.cfg.d_model

        # forward hooks: grab each PEER FFN's input hidden state (and output, for verify)
        self._ffn_in = [None] * self.n_layers
        self._ffn_out = [None] * self.n_layers
        for li, blk in enumerate(self.model.blocks):
            blk.ffn.register_forward_hook(self._make_hook(li))

    def _make_hook(self, li):
        def hook(module, inputs, output):
            self._ffn_in[li] = inputs[0].detach()
            self._ffn_out[li] = output[0].detach()
        return hook

    @torch.no_grad()
    def capture(self, tokens):
        """Run the teacher over `tokens` [b, n] (long). Returns dict:
           logits [b,n,V] and per-layer list of
           {hidden [b,n,d], expert_ids [b,n,h,k], gate_w [b,n,h,k]}."""
        out = self.model(tokens, collect=True)
        layers = []
        for li, info in enumerate(out["infos"]):
            scores = info["scores"]                      # [b,n,h,k]
            m = info["gate"]                             # [b,n,h,k] hard top-k mask (fixed mode)
            p = scores.softmax(dim=-1)
            denom = (m * p).sum(dim=-1, keepdim=True).clamp_min(1e-9)
            w = (m * p) / denom                          # exactly the teacher's _gate()
            layers.append({
                "hidden": self._ffn_in[li],              # pre-norm input to the PEER FFN
                "expert_ids": info["expert_ids"],
                "gate_w": w,
                "ffn_out": self._ffn_out[li],            # kept for verification only
            })
        return {"logits": out["logits"], "layers": layers}

    def expert_matrices(self, layer_idx, expert_ids):
        """Assemble the teacher's [.., k, d] down/up matrices for given ids
        (on-the-fly gather; this IS the generator's regression target)."""
        ffn = self.model.blocks[layer_idx].ffn
        wd = ffn.weight_down(expert_ids)                 # [.., k, d]
        wu = ffn.weight_up(expert_ids)
        return wd, wu

    @torch.no_grad()
    def verify_reconstruction(self, capture, atol=1e-4):
        """Recompute each FFN output from (hidden, ids, gate_w) + assembled wd/wu;
        compare to the hooked teacher output. Returns max abs error per layer."""
        errs = []
        for li, lay in enumerate(capture["layers"]):
            ffn = self.model.blocks[li].ffn
            x = ffn.norm(lay["hidden"])                  # the teacher's internal RMSNorm
            wd, wu = self.expert_matrices(li, lay["expert_ids"])
            h = torch.einsum("bnd,bnhkd->bnhk", x, wd)
            h = ffn.activation(h) * lay["gate_w"]
            out = torch.einsum("bnhk,bnhkd->bnd", h, wu)
            errs.append((out - lay["ffn_out"]).abs().max().item())
        return errs

    def iter_pairs(self, split="train", batch_size=8, n_batches=None, seed=None):
        """Generator over capture dicts; the training stream for the hypernet."""
        g = None
        if seed is not None:
            g = torch.Generator().manual_seed(seed)
        i = 0
        while n_batches is None or i < n_batches:
            x, _ = self.data.get_batch(split, batch_size, generator=g)
            yield self.capture(x)
            i += 1


def smoke_test():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}  ckpt={DEFAULT_CKPT}")
    th = TeacherHarness(device=dev)
    print(f"loaded: d_model={th.d_model} n_layers={th.n_layers} "
          f"k_max={th.k_max} fixed_k={th.cfg.fixed_k} mode={th.cfg.mode} "
          f"params={th.model.num_params():,}")

    x, y = th.data.get_batch("val", 4)
    cap = th.capture(x)
    lay0 = cap["layers"][0]
    print(f"logits {tuple(cap['logits'].shape)}")
    print(f"layer0 hidden {tuple(lay0['hidden'].shape)} "
          f"expert_ids {tuple(lay0['expert_ids'].shape)} "
          f"gate_w {tuple(lay0['gate_w'].shape)}")
    nz = (lay0["gate_w"] > 0).sum(dim=-1).float().mean().item()
    print(f"mean active experts per token (gate_w > 0): {nz:.1f} (expect fixed_k={th.cfg.fixed_k})")

    wd, wu = th.expert_matrices(0, lay0["expert_ids"][:1, :4])
    print(f"assembled wd {tuple(wd.shape)} wu {tuple(wu.shape)}  <- the [k x d] targets")

    errs = th.verify_reconstruction(cap)
    print("reconstruction max-abs-err per layer:", [f"{e:.2e}" for e in errs])
    ok = all(e < 1e-3 for e in errs)
    print("SMOKE TEST", "PASS" if ok else "FAIL")

    # teacher quality sanity: val loss on this batch
    with torch.no_grad():
        loss = th.model(x, targets=y)["loss"].item()
    print(f"teacher val loss on batch: {loss:.4f}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if smoke_test() else 1)

"""Which experts did Stage-1 distillation actually exercise?

Replays the EXACT token stream the distill run saw (same seeded sampler,
same batch size / ctx / step count) through the teacher with collect=True,
and counts, per layer, how many of the 16,384 experts were ever retrieved —
plus how skewed the usage was.

The generator only ever had to mimic the part of the bank the teacher
actually used on this stream; this measures that part.

Usage (on pop, from ~/Code/HN/HyperPEER):
  ~/Code/HN/peer-adaptive-k/.venv/bin/python -m experiments.expert_coverage \
      [--steps 5000 --batch 2 --ctx 256 --seed 1337]
"""

import argparse
import time

import torch

from src.harness import TeacherHarness


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--report-at", type=int, nargs="*",
                    default=[100, 500, 1000, 2500, 5000])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    th = TeacherHarness(device=device)
    th.data.context_len = args.ctx
    th.model.eval()
    L = th.n_layers
    E = th.cfg.num_experts

    counts = [torch.zeros(E, dtype=torch.long, device=device) for _ in range(L)]
    g = torch.Generator().manual_seed(args.seed)
    t0 = time.time()

    for step in range(1, args.steps + 1):
        x, _ = th.data.get_batch("train", args.batch, generator=g)
        out = th.model(x, collect=True)
        for li, info in enumerate(out["infos"]):
            ids = info["expert_ids"].reshape(-1)
            counts[li] += torch.bincount(ids, minlength=E)
        if step in args.report_at:
            toks = step * args.batch * args.ctx
            seen = [int((c > 0).sum().item()) for c in counts]
            print(f"after {step:5d} batches ({toks/1e6:.2f}M tokens): "
                  f"unique experts/layer = {seen} "
                  f"({100.0 * sum(seen) / (L * E):.1f}% of bank overall)",
                  flush=True)

    print(f"\nreplay done in {time.time() - t0:.0f}s — usage skew per layer:")
    for li in range(L):
        c = counts[li].float()
        total = c.sum()
        sorted_c, _ = c.sort(descending=True)
        cum = sorted_c.cumsum(0) / total
        n50 = int((cum < 0.5).sum().item()) + 1
        n90 = int((cum < 0.9).sum().item()) + 1
        dead = int((c == 0).sum().item())
        print(f"  layer {li}: {E - dead:5d}/{E} ever used, {dead:5d} never; "
              f"50% of retrievals covered by top {n50:5d} experts, "
              f"90% by top {n90:5d}", flush=True)


if __name__ == "__main__":
    main()

"""Precise val evaluation of the distilled student (generated experts).

Loads the Stage-1 generator checkpoint, runs N val batches at the teacher's
native context length, and reports student CE/ppl vs teacher CE/ppl on the
SAME batches, plus the FFN-disabled baseline (generator zeroed = what the
student looked like at init).

Usage (on pop, from ~/Code/HN/HyperPEER):
  ~/Code/HN/peer-adaptive-k/.venv/bin/python -m experiments.eval_student \
      [--ckpt results/stage1_distill_k256/generator_checkpoint.pt] [--batches 50]
"""

import argparse
import math

import torch

from src.harness import TeacherHarness
from src.generator import ExpertGenerator, install_recursive_ffn, set_generated


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/stage1_distill_k256/generator_checkpoint.pt")
    ap.add_argument("--batches", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    # Stage-3 recursion shape; defaults reproduce the original single-pass eval
    ap.add_argument("--t-steps", type=int, default=1)
    ap.add_argument("--rederive", choices=["once", "step"], default="step")
    ap.add_argument("--noise-std", type=float, default=0.0)
    ap.add_argument("--noise-seed", type=int, default=1234)
    ap.add_argument("--teacher-ckpt", type=str, default=None)
    ap.add_argument("--data-dir", type=str, default=None)
    args = ap.parse_args()

    import os
    device = "cuda" if torch.cuda.is_available() else "cpu"
    th_kw = {}
    if args.teacher_ckpt:
        th_kw["ckpt_path"] = os.path.expanduser(args.teacher_ckpt)
    if args.data_dir:
        th_kw["data_dir"] = os.path.expanduser(args.data_dir)
    th = TeacherHarness(device=device, **th_kw)
    th.data.context_len = args.ctx
    th.model.eval()

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    generator = ExpertGenerator(**ck["generator_config"]).to(device).float()
    generator.load_state_dict(ck["generator_state_dict"])
    generator.eval()
    wrappers = install_recursive_ffn(th.model, generator, t_steps=args.t_steps,
                                     rederive=args.rederive, noise_std=args.noise_std)
    if args.t_steps > 1 or args.noise_std > 0:
        print(f"recursion: t_steps={args.t_steps} rederive={args.rederive} "
              f"noise_std={args.noise_std}")
        torch.manual_seed(args.noise_seed)
    print(f"loaded generator from step {ck['step']}  "
          f"config={ck['generator_config']}", flush=True)

    g = torch.Generator().manual_seed(args.seed)
    sums = {"teacher": 0.0, "student": 0.0}
    n = 0
    for i in range(args.batches):
        x, y = th.data.get_batch("val", args.batch_size, generator=g)
        set_generated(wrappers, False)
        sums["teacher"] += th.model(x, targets=y)["loss"].item()
        set_generated(wrappers, True)
        sums["student"] += th.model(x, targets=y)["loss"].item()
        n += 1
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.batches}  teacher={sums['teacher']/n:.4f}  "
                  f"student={sums['student']/n:.4f}", flush=True)

    t, s = sums["teacher"] / n, sums["student"] / n
    print(f"\nval over {n} batches x {args.batch_size} x ctx {args.ctx}:")
    print(f"  teacher (real PEER k256): CE={t:.4f}  ppl={math.exp(t):.3f}")
    print(f"  student (generated):      CE={s:.4f}  ppl={math.exp(s):.3f}")
    print(f"  delta: {s - t:+.4f} nats")


if __name__ == "__main__":
    main()

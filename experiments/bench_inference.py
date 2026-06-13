"""Inference latency: classic PEER retrieval vs hypernetwork generation.

Times the full LM forward pass in three modes on identical batches, same
device, with warmup and many timed iters (CUDA-synchronized):

  teacher   : real PEER product-key retrieval (the bank must be resident)
  generated : experts synthesized per token by the hypernetwork (no bank read)
  no_ffn    : expert layers disabled (floor — attention+embed only)

Reports ms per forward and tokens/sec at a few batch/context sizes, plus the
generation overhead ratio. Run AFTER the training chain (needs the GPU alone).

Usage (on pop, from ~/Code/HN/HyperPEER):
  ~/Code/HN/peer-adaptive-k/.venv/bin/python -m experiments.bench_inference \
      --gen-ckpt results/stage3_t1_control_wt2anneal/ckpt_best.pt \
      --teacher-ckpt ~/Code/HN/peer-adaptive-k/checkpoints/wt_k256_long.pt \
      --data-dir ~/Code/HN/peer-adaptive-k/data_wikitext
"""

import argparse
import time

import torch

from src.harness import TeacherHarness
from src.generator import ExpertGenerator, install_recursive_ffn, set_generated


def time_mode(th, wrappers, generated, t_steps, bs, ctx, iters=30, warmup=8):
    set_generated(wrappers, generated)
    dev = th.device
    x = torch.randint(0, th.cfg.vocab_size, (bs, ctx), device=dev)
    with torch.no_grad():
        for _ in range(warmup):
            th.model(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            th.model(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
    set_generated(wrappers, False)
    return dt, bs * ctx / dt


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-ckpt", default="results/stage3_t1_control_wt2anneal/ckpt_best.pt")
    ap.add_argument("--teacher-ckpt", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    import os
    device = "cuda" if torch.cuda.is_available() else "cpu"
    th_kw = {}
    if args.teacher_ckpt:
        th_kw["ckpt_path"] = os.path.expanduser(args.teacher_ckpt)
    if args.data_dir:
        th_kw["data_dir"] = os.path.expanduser(args.data_dir)
    th = TeacherHarness(device=device, **th_kw)
    th.model.grad_checkpoint = False
    th.model.eval()

    ck = torch.load(args.gen_ckpt, map_location=device, weights_only=False)
    gen = ExpertGenerator(**ck["generator_config"]).to(device).float()
    gen.load_state_dict(ck["generator_state_dict"])
    gen.eval()
    wrappers = install_recursive_ffn(th.model, gen, t_steps=1, rederive="step")

    # report parameter/memory framing
    bank = sum(p.numel() for n, p in th.model.named_parameters()
               if "weight_down" in n or "weight_up" in n)
    gp = sum(p.numel() for p in gen.parameters())
    print(f"expert bank params {bank/1e6:.1f}M  |  generator params {gp/1e6:.1f}M  "
          f"(bank/gen = {bank/gp:.1f}x)", flush=True)
    print(f"device={device}  iters={args.iters}\n", flush=True)

    configs = [(1, 256), (1, 512), (8, 256), (8, 512)]
    print(f"{'batch x ctx':>12} | {'teacher ms':>11} | {'gen ms':>9} | "
          f"{'gen/teacher':>11} | {'teach tok/s':>11} | {'gen tok/s':>10}", flush=True)
    for bs, ctx in configs:
        t_ms, t_tps = time_mode(th, wrappers, False, 1, bs, ctx, args.iters)
        g_ms, g_tps = time_mode(th, wrappers, True, 1, bs, ctx, args.iters)
        print(f"{bs:>4} x {ctx:>4}   | {t_ms*1e3:>11.2f} | {g_ms*1e3:>9.2f} | "
              f"{g_ms/t_ms:>10.2f}x | {t_tps:>11.0f} | {g_tps:>10.0f}", flush=True)

    print("\nNote: 'teacher' requires the full expert bank resident in memory; "
          "'gen' carries only the generator and reads no bank. The generator's "
          "cost is FLOPs (synthesizing experts); the bank's cost is memory "
          "residency + gather. This times a full training-shaped forward; "
          "single-token autoregressive decode would shift the balance.", flush=True)


if __name__ == "__main__":
    main()

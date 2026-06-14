"""Learned-noise recursion probe (Stage-3 follow-up, additive runner).

Four arms, all warm-started from the SAME converged wt2 anneal checkpoint and
trained at matched budget so they are comparable to each other:

  t1_control : T=1, no recursion            (baseline: more training only)
  rederive   : T=2, rederive step, no noise (plain function-space recursion)
  iso_step   : T=2, rederive step, ISO noise (random inter-step direction)
  learned_dir: T=2, rederive step, LEARNED noise direction at matched energy

Questions this 2x2-ish design answers at matched budget:
  rederive  vs t1_control : does recursion help at all?
  iso_step  vs rederive   : does ANY inter-step perturbation help?
  learned   vs iso_step   : does a LEARNED direction beat a random one (same energy)?

Does NOT import or modify recursive_stage3.py / generator.py beyond the stable
ExpertGenerator + harness, so the live pipeline and verdict eval are untouched.

Usage (on pop, from ~/Code/HN/HyperPEER):
  ~/Code/HN/peer-adaptive-k/.venv/bin/python -m experiments.run_learned_noise \
      --arm learned_dir --init-from results/stage3_t1_control_wt2anneal/ckpt_best.pt \
      --teacher-ckpt checkpoints/wt_k256_long.pt --data-dir data_wikitext \
      --steps 3000 --lr 6e-5 --run-suffix _ln
"""

import argparse
import json
import math
import os
import shutil
import time
from collections import deque
from pathlib import Path

import torch

from src.harness import TeacherHarness
from src.generator import ExpertGenerator, set_generated
from experiments.distill_stage1 import kl_loss
from experiments.learned_noise import (
    NoiseHead, NoiseHeadVar, install_recursive_noise_ffn)

# noise_mode: none | iso | learned | learned_var ; needs_head True for learned*
ARMS = {
    "t1_control":  {"t_steps": 1, "rederive": "step", "noise_mode": "none",        "needs_head": False},
    "rederive":    {"t_steps": 2, "rederive": "step", "noise_mode": "none",        "needs_head": False},
    "iso_step":    {"t_steps": 2, "rederive": "step", "noise_mode": "iso",         "needs_head": False},
    "learned_dir": {"t_steps": 2, "rederive": "step", "noise_mode": "learned",     "needs_head": True},
    # variational: head learns mean+variance of the noise; KL (weight --beta)
    # keeps the magnitude from collapsing. Lets the model CHOOSE how much noise.
    "learned_var": {"t_steps": 2, "rederive": "step", "noise_mode": "learned_var", "needs_head": True},
}

TRAILING_WINDOW = 200


@torch.no_grad()
def evaluate(th, wrappers, n_batches, batch_size, noise_seed=1234):
    th.model.eval()
    torch.manual_seed(noise_seed)        # fixed noise draw so evals are comparable
    tot_s, tot_t, tot_kl = 0.0, 0.0, 0.0
    for _ in range(n_batches):
        x, y = th.data.get_batch("val", batch_size)
        set_generated(wrappers, False)
        t_out = th.model(x, targets=y)
        set_generated(wrappers, True)
        s_out = th.model(x, targets=y)
        tot_t += t_out["loss"].item()
        tot_s += s_out["loss"].item()
        tot_kl += kl_loss(t_out["logits"], s_out["logits"]).item()
    set_generated(wrappers, False)
    n = n_batches
    return {"student_ce": tot_s / n, "teacher_ce": tot_t / n, "kl": tot_kl / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS.keys()))
    ap.add_argument("--init-from", required=True,
                    help="generator checkpoint to warm-start from (wt2 anneal ckpt_best)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--target-std", type=float, default=0.02)
    ap.add_argument("--beta", type=float, default=0.1,
                    help="KL weight for learned_var arm (0 => noise collapses to ~0)")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--stamp-at", type=int, nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--run-suffix", type=str, default="_ln")
    ap.add_argument("--teacher-ckpt", type=str, default=None)
    ap.add_argument("--data-dir", type=str, default=None)
    args = ap.parse_args()

    arm = ARMS[args.arm]
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = f"ln_{args.arm}{args.run_suffix}"
    results_dir = Path("results") / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    wandb = None
    if not args.no_wandb:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project="hyperpeer", name=run_name, config={**vars(args), **arm})

    print(f"Learned-noise probe — arm={args.arm}  {arm}  device={device}", flush=True)
    print(f"  warm start: {args.init_from}", flush=True)

    th_kw = {}
    if args.teacher_ckpt:
        th_kw["ckpt_path"] = os.path.expanduser(args.teacher_ckpt)
    if args.data_dir:
        th_kw["data_dir"] = os.path.expanduser(args.data_dir)
    th = TeacherHarness(device=device, **th_kw)
    th.data.context_len = args.ctx
    th.model.grad_checkpoint = False
    th.model.eval()

    ck = torch.load(args.init_from, map_location=device, weights_only=False)
    generator = ExpertGenerator(**ck["generator_config"]).to(device).float()
    generator.load_state_dict(ck["generator_state_dict"])
    d_model = generator.d_model
    print(f"  generator from step {ck['step']}  d_model={d_model}", flush=True)

    if arm["noise_mode"] == "learned_var":
        noise_head = NoiseHeadVar(d_model, target_std=args.target_std).to(device).float()
    else:
        noise_head = NoiseHead(d_model).to(device).float()

    wrappers = install_recursive_noise_ffn(
        th.model, generator, noise_head,
        t_steps=arm["t_steps"], rederive=arm["rederive"],
        noise_mode=arm["noise_mode"], target_std=args.target_std)
    print(f"  wrappers: t_steps={arm['t_steps']} rederive={arm['rederive']} "
          f"noise_mode={arm['noise_mode']} target_std={args.target_std} "
          f"head={'yes' if arm['needs_head'] else 'no'}", flush=True)

    params = list(generator.parameters())
    if arm["needs_head"]:
        params = params + list(noise_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95))

    def lam(s):
        if s < args.warmup:
            return s / max(1, args.warmup)
        prog = (s - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lam)

    metrics_path = results_dir / "metrics.jsonl"
    best_ce = float("inf")

    def log_eval(step, tag):
        nonlocal best_ce
        ev = evaluate(th, wrappers, args.eval_batches, args.batch)
        print(f"  [eval @ {step} | {tag}] student-CE={ev['student_ce']:.4f}  "
              f"teacher-CE={ev['teacher_ce']:.4f}  KL={ev['kl']:.4f}", flush=True)
        with open(metrics_path, "a") as f:
            f.write(json.dumps({"step": step, "tag": tag, **ev}) + "\n")
        if wandb:
            wandb.log({"eval/student_ce_val": ev["student_ce"],
                       "eval/teacher_ce_val": ev["teacher_ce"],
                       "eval/kl_to_teacher": ev["kl"]}, step=max(step, 1))
        if ev["student_ce"] < best_ce:
            best_ce = ev["student_ce"]
            torch.save({"step": step, "student_ce": best_ce,
                        "generator_state_dict": generator.state_dict(),
                        "noise_head_state_dict": noise_head.state_dict(),
                        "generator_config": generator.config(),
                        "arm": {**arm, "name": args.arm}, "args": vars(args)},
                       results_dir / "ckpt_best.pt")
        return ev

    log_eval(0, "step0-insertion")

    g = torch.Generator().manual_seed(args.seed)
    trailing = deque(maxlen=TRAILING_WINDOW)
    t0 = time.time()
    step, skipped, consec = 0, 0, 0

    while step < args.steps:
        x, y = th.data.get_batch("train", args.batch, generator=g)
        set_generated(wrappers, True)
        th.model.eval()
        loss = th.model(x, targets=y)["loss"]
        set_generated(wrappers, False)
        ce_only = float(loss.detach().item())
        kl_val, sigma_val = 0.0, float("nan")
        if arm["noise_mode"] == "learned_var":
            kl = sum(w._last_kl for w in wrappers) / len(wrappers)
            loss = loss + args.beta * kl
            kl_val = float(kl.detach().item())
            sigma_val = float(sum(w._last_sigma for w in wrappers) / len(wrappers))

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            skipped += 1
            consec += 1
            if consec >= 200:
                print("  ERROR: 200 consecutive NaN/Inf; halting.", flush=True)
                break
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 0.5)
        if torch.isfinite(norm):
            optimizer.step()
        scheduler.step()
        consec = 0
        step += 1
        lv = ce_only                       # report CE only (KL excluded from the headline)
        trailing.append(lv)
        if wandb:
            wandb.log({"train/ce": lv, "train/lr": scheduler.get_last_lr()[0]}, step=step)

        if step % args.log_every == 0:
            sps = step / max(1.0, time.time() - t0)
            extra = ""
            if arm["noise_mode"] == "learned_var":
                extra = f"  sigma={sigma_val:.4f} (prior {args.target_std})  KL={kl_val:.3f}"
            print(f"  step {step:6d}/{args.steps}  ce={lv:7.4f}  "
                  f"trail={sum(trailing)/len(trailing):7.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.1e}  {sps:.2f} step/s  "
                  f"eta {(args.steps-step)/max(0.1,sps):.0f}s  skipped {skipped}{extra}", flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            log_eval(step, "train")

        if step in args.stamp_at or step == args.steps:
            p = results_dir / "generator_checkpoint.pt"
            torch.save({"step": step, "generator_state_dict": generator.state_dict(),
                        "noise_head_state_dict": noise_head.state_dict(),
                        "generator_config": generator.config(),
                        "arm": {**arm, "name": args.arm}, "args": vars(args),
                        "from_ckpt": args.init_from}, p)
            shutil.copy(p, results_dir / f"ckpt_step{step}.pt")
            print(f"  -> stamped checkpoint at step {step}", flush=True)

    print(f"\nArm {args.arm} done in {time.time()-t0:.0f}s (skipped {skipped}); "
          f"best student-CE {best_ce:.4f}", flush=True)
    if wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

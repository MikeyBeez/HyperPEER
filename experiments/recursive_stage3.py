"""Stage 3: add within-layer recursion to the trained generator, continue NTP.

Spec: notes/spec_stage3_recursion.md (rev 2). All arms warm-start from the
Stage-2 model and train plain next-token cross-entropy. Arms:

  t1_control : --t-steps 1                                  (more training, no recursion)
  fixed_fn   : --t-steps 2 --rederive once                  (activation-space recursion)
  rederive   : --t-steps 2 --rederive step                  (function-space recursion)
  act_noise  : --t-steps 2 --rederive once --noise-std 0.02 (GRAM-style perturbation)

A step-0 eval records the insertion bump before any training.

Usage (on pop, from ~/Code/HN/HyperPEER):
  ~/Code/HN/peer-adaptive-k/.venv/bin/python -m experiments.recursive_stage3 \
      --arm rederive [--steps 5000]
"""

import argparse
import json
import math
import shutil
import time
from collections import deque
from pathlib import Path

import torch

from src.harness import TeacherHarness
from src.generator import ExpertGenerator, install_recursive_ffn, set_generated
from experiments.distill_stage1 import kl_loss

ARMS = {
    "t1_control": {"t_steps": 1, "rederive": "step", "noise_std": 0.0},
    "fixed_fn":   {"t_steps": 2, "rederive": "once", "noise_std": 0.0},
    "rederive":   {"t_steps": 2, "rederive": "step", "noise_std": 0.0},
    "act_noise":  {"t_steps": 2, "rederive": "once", "noise_std": 0.02},
}

TRAILING_WINDOW = 200


@torch.no_grad()
def evaluate(th, wrappers, n_batches, batch_size, noise_seed=1234):
    th.model.eval()
    torch.manual_seed(noise_seed)          # fixed noise draw for act_noise evals
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
    ap.add_argument("--init-from", default="results/stage2_ntp_naive/generator_checkpoint.pt")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--stamp-at", type=int, nargs="*", default=[2500, 5000])
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--run-suffix", type=str, default="",
                    help="appended to run name/results dir (e.g. _hot10k)")
    args = ap.parse_args()

    arm = ARMS[args.arm]
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = f"stage3_{args.arm}{args.run_suffix}"
    results_dir = Path("results") / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    wandb = None
    if not args.no_wandb:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project="hyperpeer", name=run_name, config={**vars(args), **arm})

    print(f"Stage 3 — arm={args.arm}  {arm}", flush=True)
    print(f"  warm start: {args.init_from}", flush=True)

    th = TeacherHarness(device=device)
    th.data.context_len = args.ctx
    th.model.grad_checkpoint = False
    th.model.eval()

    ck = torch.load(args.init_from, map_location=device, weights_only=False)
    generator = ExpertGenerator(**ck["generator_config"]).to(device).float()
    generator.load_state_dict(ck["generator_state_dict"])
    print(f"  generator from step {ck['step']}  config={ck['generator_config']}", flush=True)

    wrappers = install_recursive_ffn(th.model, generator, **arm)
    print(f"  recursive wrappers: t_steps={arm['t_steps']} rederive={arm['rederive']} "
          f"noise_std={arm['noise_std']}", flush=True)

    optimizer = torch.optim.AdamW(generator.parameters(), lr=args.lr, betas=(0.9, 0.95))

    def lam(s):
        if s < args.warmup:
            return s / max(1, args.warmup)
        prog = (s - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lam)

    metrics_path = results_dir / "metrics.jsonl"

    def do_eval(step, tag):
        ev = evaluate(th, wrappers, args.eval_batches, args.batch)
        print(f"  [eval @ {step} | {tag}] student-CE={ev['student_ce']:.4f}  "
              f"teacher-CE={ev['teacher_ce']:.4f}  KL={ev['kl']:.4f}", flush=True)
        with open(metrics_path, "a") as f:
            f.write(json.dumps({"step": step, "tag": tag, **ev}) + "\n")
        if wandb:
            wandb.log({"eval/student_ce_val": ev["student_ce"],
                       "eval/teacher_ce_val": ev["teacher_ce"],
                       "eval/kl_to_teacher": ev["kl"]}, step=max(step, 1))

    do_eval(0, "step0-insertion")     # the warm-start-into-recursion bump

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
        norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), 0.5)
        if torch.isfinite(norm):
            optimizer.step()
        scheduler.step()
        consec = 0
        step += 1
        lv = float(loss.detach().item())
        trailing.append(lv)
        if wandb:
            wandb.log({"train/ce": lv, "train/lr": scheduler.get_last_lr()[0]}, step=step)

        if step % args.log_every == 0:
            sps = step / max(1.0, time.time() - t0)
            print(f"  step {step:6d}/{args.steps}  ce={lv:7.4f}  "
                  f"trail={sum(trailing)/len(trailing):7.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.1e}  {sps:.2f} step/s  "
                  f"eta {(args.steps-step)/max(0.1,sps):.0f}s  skipped {skipped}", flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            do_eval(step, "train")

        if step in args.stamp_at or step == args.steps:
            p = results_dir / "generator_checkpoint.pt"
            torch.save({"step": step, "generator_state_dict": generator.state_dict(),
                        "generator_config": generator.config(),
                        "arm": {**arm, "name": args.arm}, "args": vars(args),
                        "from_ckpt": args.init_from}, p)
            shutil.copy(p, results_dir / f"ckpt_step{step}.pt")
            print(f"  -> stamped checkpoint at step {step}", flush=True)

    print(f"\nArm {args.arm} done in {time.time()-t0:.0f}s (skipped {skipped})", flush=True)
    if wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

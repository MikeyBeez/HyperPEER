"""Stage 1: distill the PEER teacher into the expert GENERATOR — D2L logic.

Same training logic as doc-to-lora phase03_perceiver_train.py:
  teacher pass  = frozen base with the REAL adapters (here: real PEER retrieval)
  student pass  = same frozen base with GENERATED adapters (here: hypernet experts)
  loss          = KL(teacher || student) on the logits
  loop          = AdamW + warmup/cosine on --steps, grad-clip 0.5, NaN-skip with
                  heartbeat and a hard stop on a skip cascade, trailing-window
                  loss logger, periodic checkpoints.

Differences forced by the setting (not the logic): KL is over ALL token
positions (the teacher differs from the student everywhere, not just on an
answer span), and eval has a real held-out split — val-KL is THE make-or-break
gate metric: if train-KL falls but val-KL does not, the generator memorized
and the premise fails.

Usage (on pop, from ~/Code/HN/HyperPEER):
  ~/Code/HN/peer-adaptive-k/.venv/bin/python -m experiments.distill_stage1 \
      --steps 5000 [--no-wandb]
"""

import argparse
import json
import math
import os
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F

from src.harness import TeacherHarness
from src.generator import (ExpertGenerator, install_generated_ffn,
                           install_recursive_ffn, set_generated)

TRAILING_WINDOW = 200


def kl_loss(teacher_logits, student_logits, tau=1.0):
    """KL(teacher || student), mean over all token positions (D2L's KL, full seq)."""
    t_lp = F.log_softmax(teacher_logits.float() / tau, dim=-1)
    s_lp = F.log_softmax(student_logits.float() / tau, dim=-1)
    kl = (t_lp.exp() * (t_lp - s_lp)).sum(dim=-1)         # [b, n]
    return kl.mean() * (tau ** 2)


@torch.no_grad()
def evaluate(th, wrappers, split, n_batches, batch_size, tau):
    """Teacher/student KL + CE on a split. model in eval mode, generated experts."""
    th.model.eval()
    tot_kl, tot_s_ce, tot_t_ce = 0.0, 0.0, 0.0
    for _ in range(n_batches):
        x, y = th.data.get_batch(split, batch_size)
        set_generated(wrappers, False)
        t_out = th.model(x, targets=y)
        set_generated(wrappers, True)
        s_out = th.model(x, targets=y)
        tot_kl += kl_loss(t_out["logits"], s_out["logits"], tau).item()
        tot_t_ce += t_out["loss"].item()
        tot_s_ce += s_out["loss"].item()
    set_generated(wrappers, False)
    n = n_batches
    return {"kl": tot_kl / n, "student_ce": tot_s_ce / n, "teacher_ce": tot_t_ce / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--k-gen", type=int, default=256)
    ap.add_argument("--latent-n", type=int, default=8)
    ap.add_argument("--latent-d", type=int, default=256)
    ap.add_argument("--n-cross", type=int, default=2)
    ap.add_argument("--n-self", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--init-from", type=str, default=None,
                    help="warm-start generator from a saved checkpoint")
    ap.add_argument("--t-steps", type=int, default=1,
                    help="micro-steps per FFN (Stage-3 native recursion)")
    ap.add_argument("--rederive", choices=["once", "step"], default="step")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = args.run_name or f"stage1_distill_k{args.k_gen}"
    results_dir = Path("results") / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    wandb = None
    if not args.no_wandb:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project="hyperpeer", name=run_name, config=vars(args))

    print(f"Stage 1 — distill PEER teacher -> expert generator (D2L logic)", flush=True)
    print(f"  steps={args.steps} batch={args.batch} ctx={args.ctx} "
          f"k_gen={args.k_gen} lr={args.lr:.0e} tau={args.tau}", flush=True)

    th = TeacherHarness(device=device)
    th.data.context_len = args.ctx
    # grad checkpointing recompute clashes with the generator call inside the
    # block (saved-tensor count mismatch); off — batch 2 x ctx 256 fits without it
    th.model.grad_checkpoint = False
    print(f"  teacher: d_model={th.d_model} layers={th.n_layers} "
          f"k={th.cfg.fixed_k} params={th.model.num_params():,}", flush=True)

    generator = ExpertGenerator(
        d_model=th.d_model, n_layers=th.n_layers, k_gen=args.k_gen,
        latent_n=args.latent_n, latent_d=args.latent_d,
        n_cross=args.n_cross, n_self=args.n_self,
    ).to(device).float()
    if args.init_from:
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        generator.load_state_dict(ck["generator_state_dict"])
        print(f"  warm-started from {args.init_from} (step {ck['step']})", flush=True)
    n_gen = sum(p.numel() for p in generator.parameters())
    print(f"  generator: {n_gen / 1e6:.1f}M params", flush=True)

    if args.t_steps > 1:
        wrappers = install_recursive_ffn(th.model, generator,
                                         t_steps=args.t_steps, rederive=args.rederive)
        print(f"  installed RECURSIVE wrappers: t_steps={args.t_steps} "
              f"rederive={args.rederive}", flush=True)
    else:
        wrappers = install_generated_ffn(th.model, generator)
        print(f"  installed GeneratedFFN wrappers on {len(wrappers)} blocks", flush=True)

    optimizer = torch.optim.AdamW(generator.parameters(), lr=args.lr, betas=(0.9, 0.95))

    def lr_lambda(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    g = torch.Generator().manual_seed(args.seed)
    log = []
    trailing = deque(maxlen=TRAILING_WINDOW)
    metrics_path = results_dir / "metrics.jsonl"
    t_train = time.time()
    step = 0
    skipped = 0
    consecutive_skips = 0
    max_consecutive_skips = 200      # D2L hard stop — no silent burning

    while step < args.steps:
        x, _ = th.data.get_batch("train", args.batch, generator=g)

        # teacher pass: real PEER, eval, no grad
        set_generated(wrappers, False)
        th.model.eval()
        with torch.no_grad():
            t_logits = th.model(x)["logits"]

        # student pass: generated experts, grads flow to the generator only
        set_generated(wrappers, True)
        s_logits = th.model(x)["logits"]
        loss = kl_loss(t_logits, s_logits, args.tau)
        set_generated(wrappers, False)

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            skipped += 1
            consecutive_skips += 1
            if consecutive_skips % 50 == 0:
                print(f"  [skip-burst] step {step}/{args.steps} "
                      f"consecutive={consecutive_skips} total={skipped}", flush=True)
            if consecutive_skips >= max_consecutive_skips:
                print(f"  ERROR: {consecutive_skips} consecutive NaN/Inf steps; "
                      f"halting.", flush=True)
                break
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), 0.5)
        if torch.isfinite(total_norm):
            optimizer.step()
        scheduler.step()
        consecutive_skips = 0
        step += 1
        lv = float(loss.detach().item())
        trailing.append(lv)
        rec = {"step": step, "loss": lv, "lr": scheduler.get_last_lr()[0]}
        log.append(rec)
        if wandb:
            wandb.log({"train/kl": lv, "train/lr": rec["lr"],
                       "train/grad_norm": float(total_norm)}, step=step)

        if step % args.log_every == 0:
            elapsed = time.time() - t_train
            trail = sum(trailing) / len(trailing)
            sps = step / max(1.0, elapsed)
            eta = (args.steps - step) / max(0.1, sps)
            print(f"  step {step:6d}/{args.steps}  kl={lv:7.4f}  "
                  f"trail@{len(trailing)}={trail:7.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.1e}  {sps:.2f} step/s  "
                  f"eta {eta:.0f}s  skipped {skipped}", flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            ev_val = evaluate(th, wrappers, "val", args.eval_batches, args.batch, args.tau)
            ev_trn = evaluate(th, wrappers, "train", args.eval_batches, args.batch, args.tau)
            gap = ev_val["kl"] - ev_trn["kl"]
            print(f"  [eval @ {step}] val-KL={ev_val['kl']:.4f}  "
                  f"train-KL={ev_trn['kl']:.4f}  GAP={gap:+.4f}  "
                  f"student-CE={ev_val['student_ce']:.4f}  "
                  f"teacher-CE={ev_val['teacher_ce']:.4f}", flush=True)
            erec = {"step": step, "val": ev_val, "train_eval": ev_trn, "gap": gap}
            with open(metrics_path, "a") as f:
                f.write(json.dumps(erec) + "\n")
            if wandb:
                wandb.log({"eval/val_kl": ev_val["kl"],
                           "eval/train_kl": ev_trn["kl"],
                           "eval/gap": gap,
                           "eval/student_ce_val": ev_val["student_ce"],
                           "eval/teacher_ce_val": ev_val["teacher_ce"]}, step=step)

        if step % args.ckpt_every == 0 or step == args.steps:
            torch.save({
                "step": step,
                "generator_state_dict": generator.state_dict(),
                "generator_config": generator.config(),
                "args": vars(args),
            }, results_dir / "generator_checkpoint.pt")
            print(f"  -> saved checkpoint at step {step}", flush=True)

    (results_dir / "training_log.json").write_text(json.dumps(log, indent=2))
    print(f"\nDone in {time.time() - t_train:.0f}s (skipped {skipped})", flush=True)
    if wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

"""Stage 2: switch the distilled generator from distillation to NEXT-TOKEN
training — and measure whether the distilled ability survives the switch.

This is the test of the catastrophic-forgetting warning (Rosanne Liu / Sakana,
re: Text-to-LoRA — anecdotal, unverified) and of the LP-FT cure (Kumar et al.,
ICLR 2022: train a probe on frozen features first, THEN unfreeze, so the fresh
objective's gradients nudge the distilled features instead of bulldozing them).

Two arms, same script:

  naive : --probe-steps 0 --kl-anchor 0
          unfreeze the whole generator immediately, pure NTP loss.
          If the anecdote transfers, val-CE spikes (or val-KL-to-teacher
          blows up) right after the switch.

  lpft  : --probe-steps P --kl-anchor L
          Phase A: freeze the Perceiver trunk; train ONLY the head's output
          projections (proj_down / proj_up — the change-of-basis between
          distilled features and expert rows) on NTP.
          Phase B: unfreeze everything at low LR with a KL anchor to the
          teacher that anneals to zero over the remaining steps.

Forgetting is read off the eval trace: step-0 eval gives the distilled
starting point (student CE ~2.48 on the small ckpt); a transient spike above
the no-FFN baseline (~3.0) = features wrecked; smooth descent = survived.
Success beyond survival = student CE dropping BELOW the teacher's (~2.32),
which mimicry alone cannot do.

Usage (on pop, from ~/Code/HN/HyperPEER):
  ~/Code/HN/peer-adaptive-k/.venv/bin/python -m experiments.ntp_stage2 \
      --ckpt results/stage1_distill_k256_big/generator_checkpoint.pt \
      --mode-name lpft --probe-steps 500 --kl-anchor 0.5 --steps 5000
"""

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F

from src.harness import TeacherHarness
from src.generator import ExpertGenerator, install_generated_ffn, set_generated
from experiments.distill_stage1 import kl_loss, evaluate

TRAILING_WINDOW = 200


def freeze_trunk(generator, frozen):
    """Freeze everything except the head's output projections."""
    for name, p in generator.named_parameters():
        is_proj = name.startswith("head.proj_down") or name.startswith("head.proj_up")
        p.requires_grad_(not frozen or is_proj)


def trainable_params(generator):
    return [p for p in generator.parameters() if p.requires_grad]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/stage1_distill_k256_big/generator_checkpoint.pt")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--probe-steps", type=int, default=0,
                    help="LP-FT phase A length; 0 = naive arm")
    ap.add_argument("--kl-anchor", type=float, default=0.0,
                    help="initial KL-to-teacher weight in phase B; anneals to 0")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--lr-probe", type=float, default=1e-4)
    ap.add_argument("--lr-ft", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--mode-name", type=str, default=None,
                    help="run-name suffix; default derived from flags")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode = args.mode_name or ("lpft" if args.probe_steps > 0 else "naive")
    run_name = f"stage2_ntp_{mode}"
    results_dir = Path("results") / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    wandb = None
    if not args.no_wandb:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project="hyperpeer", name=run_name, config=vars(args))

    print(f"Stage 2 — distill->NTP switch, arm={mode}", flush=True)
    print(f"  ckpt={args.ckpt}", flush=True)
    print(f"  steps={args.steps} probe={args.probe_steps} "
          f"kl_anchor={args.kl_anchor} lr_probe={args.lr_probe:.0e} "
          f"lr_ft={args.lr_ft:.0e}", flush=True)

    th = TeacherHarness(device=device)
    th.data.context_len = args.ctx
    th.model.grad_checkpoint = False
    th.model.eval()

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    generator = ExpertGenerator(**ck["generator_config"]).to(device).float()
    generator.load_state_dict(ck["generator_state_dict"])
    print(f"  generator from step {ck['step']}  config={ck['generator_config']}",
          flush=True)
    wrappers = install_generated_ffn(th.model, generator)

    def make_opt_sched(lr, horizon):
        opt = torch.optim.AdamW(trainable_params(generator), lr=lr, betas=(0.9, 0.95))
        def lam(s):
            if s < args.warmup:
                return s / max(1, args.warmup)
            prog = (s - args.warmup) / max(1, horizon - args.warmup)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
        return opt, torch.optim.lr_scheduler.LambdaLR(opt, lam)

    in_probe = args.probe_steps > 0
    freeze_trunk(generator, frozen=in_probe)
    n_train = sum(p.numel() for p in trainable_params(generator))
    print(f"  phase {'A (probe)' if in_probe else 'B (full)'}: "
          f"{n_train/1e6:.2f}M trainable params", flush=True)
    if in_probe:
        optimizer, scheduler = make_opt_sched(args.lr_probe, args.probe_steps)
    else:
        optimizer, scheduler = make_opt_sched(args.lr_ft, args.steps)

    g = torch.Generator().manual_seed(args.seed)
    metrics_path = results_dir / "metrics.jsonl"
    trailing = deque(maxlen=TRAILING_WINDOW)
    t0 = time.time()
    skipped = 0
    consecutive_skips = 0

    def do_eval(step, phase):
        ev = evaluate(th, wrappers, "val", args.eval_batches, args.batch, 1.0)
        print(f"  [eval @ {step} | {phase}] student-CE={ev['student_ce']:.4f}  "
              f"teacher-CE={ev['teacher_ce']:.4f}  "
              f"KL-to-teacher={ev['kl']:.4f}", flush=True)
        with open(metrics_path, "a") as f:
            f.write(json.dumps({"step": step, "phase": phase, **ev}) + "\n")
        if wandb:
            wandb.log({"eval/student_ce_val": ev["student_ce"],
                       "eval/teacher_ce_val": ev["teacher_ce"],
                       "eval/kl_to_teacher": ev["kl"]}, step=max(step, 1))
        return ev

    do_eval(0, "pre-switch")            # the distilled starting point

    step = 0
    while step < args.steps:
        # phase transition
        if in_probe and step >= args.probe_steps:
            in_probe = False
            freeze_trunk(generator, frozen=False)
            optimizer, scheduler = make_opt_sched(args.lr_ft, args.steps - step)
            n_train = sum(p.numel() for p in trainable_params(generator))
            print(f"  --- phase B: unfroze trunk at step {step} "
                  f"({n_train/1e6:.2f}M trainable, lr={args.lr_ft:.0e}) ---",
                  flush=True)
            do_eval(step, "unfreeze-point")

        phase = "probe" if in_probe else "full"
        x, y = th.data.get_batch("train", args.batch, generator=g)

        # KL anchor weight: full during probe (harmless: trunk frozen),
        # annealed linearly to 0 over phase B
        if args.kl_anchor > 0:
            if in_probe:
                lam_kl = args.kl_anchor
            else:
                prog = (step - args.probe_steps) / max(1, args.steps - args.probe_steps)
                lam_kl = args.kl_anchor * max(0.0, 1.0 - prog)
        else:
            lam_kl = 0.0

        t_logits = None
        if lam_kl > 0:
            set_generated(wrappers, False)
            with torch.no_grad():
                t_logits = th.model(x)["logits"]

        set_generated(wrappers, True)
        out = th.model(x, targets=y)
        ce = out["loss"]
        loss = ce
        kl = None
        if lam_kl > 0:
            kl = kl_loss(t_logits, out["logits"])
            loss = ce + lam_kl * kl
        set_generated(wrappers, False)

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            skipped += 1
            consecutive_skips += 1
            if consecutive_skips >= 200:
                print("  ERROR: 200 consecutive NaN/Inf steps; halting.", flush=True)
                break
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(trainable_params(generator), 0.5)
        if torch.isfinite(total_norm):
            optimizer.step()
        scheduler.step()
        consecutive_skips = 0
        step += 1
        cev = float(ce.detach().item())
        trailing.append(cev)
        if wandb:
            rec = {"train/ce": cev, "train/lr": scheduler.get_last_lr()[0],
                   "train/lambda_kl": lam_kl}
            if kl is not None:
                rec["train/kl_anchor_term"] = float(kl.detach().item())
            wandb.log(rec, step=step)

        if step % args.log_every == 0:
            trail = sum(trailing) / len(trailing)
            sps = step / max(1.0, time.time() - t0)
            print(f"  step {step:6d}/{args.steps} [{phase}]  ce={cev:7.4f}  "
                  f"trail={trail:7.4f}  lam_kl={lam_kl:.3f}  "
                  f"lr={scheduler.get_last_lr()[0]:.1e}  {sps:.2f} step/s  "
                  f"skipped {skipped}", flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            do_eval(step, phase)

        if step % args.ckpt_every == 0 or step == args.steps:
            torch.save({
                "step": step,
                "generator_state_dict": generator.state_dict(),
                "generator_config": generator.config(),
                "args": vars(args),
                "from_ckpt": args.ckpt,
            }, results_dir / "generator_checkpoint.pt")
            print(f"  -> saved checkpoint at step {step}", flush=True)

    print(f"\nDone in {time.time() - t0:.0f}s (skipped {skipped})", flush=True)
    if wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

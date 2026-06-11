# Stage 3 Spec: Within-Layer Recursion — Function-Space vs Activation-Space

**Project**: HyperPEER (`~/Code/HN/HyperPEER` on pop). Teacher and data from `~/Code/HN/peer-adaptive-k` (k256 checkpoint), exactly as Stages 1–2.
**Drafted**: 2026-06-10 (chat). **Executor**: Claude Code on the 5070 Ti.

## Hypothesis

Re-deriving the expert MLP from the evolving hidden state at each micro-step (function-space recursion) buys measurable quality over (a) applying one generated MLP repeatedly (activation-space recursion, the TRM/GRAM shape) and (b) stochastic perturbation of the state between applications (GRAM-style noise, Mikey's original activation-space idea). This is the core architectural claim of the hypernet-recursive design note: the reasoning operation should be a function of where the state currently is, not a fixed transform re-applied.

## Setup

- Teacher: `peer-adaptive-k/checkpoints/p0_matched_k256.pt`, frozen, eval mode. Precise-eval reference: teacher CE 2.3227 (50 batches × 4 × ctx 512, seed 42, `experiments/eval_student.py` protocol).
- Generator architecture: the scaled config from Stage 1 — latent_n 16, latent_d 512, n_cross 2, n_self 3, k_gen 256 (17.1M params). Fresh init for every arm (no warm start; recursion changes the function being learned, and warm-starting from the T=1 solution would muddy the comparison).
- Loss and recipe: logit-KL distillation vs the teacher, identical to `distill_stage1.py` — AdamW (0.9, 0.95), clip 0.5, warmup 200, cosine, batch 2, ctx 256, lr 3e-5 (the post-divergence rate; recursion adds feedback, do NOT use 1e-4).
- Generator-only activation checkpointing ON (whole-block checkpointing is known-broken with the capture hooks). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. If T=2 OOMs at ctx 256, drop to ctx 192 FOR ALL ARMS and note it in the README.

## The recursion cell (precise definition)

Inside each wrapped FFN, with x the block input and norm the teacher's own RMSNorm:

```
u_0 = x
for t in 1..T:
    h_t   = norm(u_{t-1})
    wd,wu = experts for this micro-step        # arm-dependent, see below
    u_t   = u_{t-1} + einsum(GELU(h_t · wd), wu)
return (u_T - x)                               # block's own residual adds this once
```

Per-micro-step residual, teacher norm re-applied each step, no detach between micro-steps (T=2 is shallow enough; if instability appears, detach u_{t-1} inside the generator call only and note it).

## Arms (all T=2 except A)

- **A — t1_matched**: T=1, exactly today's architecture, retrained from scratch for the SAME step budget as the other arms. The clean baseline (the existing big2 checkpoint had 20k steps; do not reuse it for comparison).
- **B — fixed_fn**: generate (wd, wu) ONCE from norm(x) at t=1; apply the same matrices at both micro-steps. Recursion in activation space with a frozen per-token function. Isolates "more depth/iteration" from "re-derived function." One generator call per layer per token.
- **C — rederive**: regenerate (wd, wu) at every micro-step from norm(u_{t-1}). The hypothesis arm. Two generator calls per layer per token (2× generator compute vs B — say so honestly in the README).
- **D — act_noise**: arm B plus Gaussian noise added to u_1 between micro-steps (std 0.02 × RMS of u_1, train and eval with a fixed eval seed). Mikey's original activation-space perturbation, the GRAM move. Controls whether any C-gain is really about re-derivation or just about the second step seeing a perturbed state.

## Procedure

1. Implement `experiments/recursive_stage3.py`: clone of `distill_stage1.py` with `--t-steps {1,2}`, `--rederive {once,step}`, `--noise-std FLOAT` wired into the GeneratedFFN wrapper as defined above. Smoke-test each arm 20 steps (`--steps 20 --no-wandb`) before the queue; all four must run NaN-free and the T=1 smoke should match today's behavior.
2. Run all four arms at **10,000 steps** each, chained in tmux window `gram:hyperpeer`, W&B runs `stage3_{t1_matched,fixed_fn,rederive,act_noise}`, logs `logs_stage3_<arm>.log`, results in `results/stage3_<arm>/`.
3. Standard eval cadence every 250 steps (val-KL, train-KL, gap, student/teacher CE).
4. After each arm: precise eval via `eval_student.py` extended with the same `--t-steps/--rederive/--noise-std` flags (eval must run the arm's own forward shape; noise arm evals WITH its noise, fixed seed).
5. Step-stamped checkpoint copies at 5,000 and 10,000 (`ckpt_step{N}.pt`), per the training-run-management protocol. On any divergence: kill before the next rolling save, resume from last good stamp at lr/3, note in README.

## Metrics

Primary: precise-eval student CE per arm (50 × 4 × ctx 512, seed 42), delta vs teacher 2.3227, and placement on the retrieval frontier (k4 2.3673, k8 2.3387, k16 2.3233). Secondary: final val-KL, train/val gap trace (memorization check, expect ~0 as in Stages 1–2), steps/s and wall-clock per arm (the honest compute accounting for C).

## Pre-committed predictions (calibration check on return)

- **A** lands near 2.39–2.41 (10k steps is half the big2 budget; big2 was 2.371 at 20k).
- **B vs A**: small improvement, ≤0.01 nats. Extra application of the same function is nearly free depth, but TinyStories likely doesn't reward it much.
- **C vs B**: the bet — C better by ≥0.015 nats. Confidence: genuinely uncertain, this is the experiment. If C ≤ B, function-space recursion buys nothing at this scale and the recursive design note needs revision, which is a publishable negative.
- **D vs B**: no improvement (GRAM's own ablation: unstructured noise was worthless; all gain came from structure). If D ≈ C > B, the gain is perturbation, not re-derivation — that would be the surprising outcome worth a follow-up.
- Stability risk ranked: C > D > B > A. Expect at most one divergence-and-resume across the queue at lr 3e-5.

## Success criteria

- All four arms complete with gap ~0 (|gap| < 0.02 sustained): PASS/FAIL per arm.
- Hypothesis confirmed iff C − B ≤ −0.015 nats on the precise eval (C better) AND C stable.
- Any arm beating the 20k-step big2 number (2.3710) at half the budget is a headline regardless of which arm.

## Runtime budget

A ≈ 2h (1.4 step/s), B ≈ 2.5h, C ≈ 3.5h (double generator calls), D ≈ 2.5h. Queue total ≈ 10–11h + evals — an overnight run. Ceilings, not hang thresholds.

## Files to create

- `experiments/recursive_stage3.py`
- `results/stage3_<arm>/` × 4 (metrics.jsonl, training_log.json, stamped checkpoints)
- `results/stage3/README.md` (the phase README)

After the runs complete, write `results/stage3/README.md` following the phase-README format: headline paragraph with the C-vs-B verdict and the most important number, headline table (4 arms × precise CE / delta-vs-teacher / final val-KL / wall-clock), per-arm prose, pre-committed predictions vs measured outcomes, architectural interpretation (2–4 sentences, paper-ready framing in a blockquote if publishable), file manifest, open questions (the next spec should be draftable from them — candidates: T=3, depth-collapse via step embedding, engram conditioning into the generator's latents).

`git add` all new/modified files (script, results, READMEs), commit as `Stage 3: function-space vs activation-space recursion — <one-line verdict>`, and push to origin.

# Stage 3 Spec (rev 2): Add Recursion to the Trained Model, Continue NTP

**Project**: HyperPEER (`~/Code/HN/HyperPEER` on pop). **Drafted**: 2026-06-10 chat; **rev 2 2026-06-11** per Mikey: warm-start from the Stage-2 model and add recursion under next-token training — NOT fresh distillation per arm (rev 1's design; kept at bottom as a deferred variant).

## Hypothesis

Adding within-layer recursion to the already-trained generator and continuing next-token training improves the model beyond what the same amount of extra T=1 training buys. Strong version (the design-note bet): re-deriving the experts from the evolving state each micro-step (function-space recursion) beats both re-applying the same generated experts (activation-space recursion) and noise-perturbed re-application (GRAM-style stochasticity, Mikey's original activation-space idea).

## Setup

- **Warm start, all arms**: `results/stage2_ntp_naive/generator_checkpoint.pt` (the 17.1M generator after distillation + 5k NTP steps; precise eval 2.3573, teacher 2.3227, no-FFN ~3.0; frontier k4 2.3673, k8 2.3387).
- Teacher model frozen as ever; loss is **plain next-token cross-entropy** (no KL anchor — Stage 2 showed the naive path is benign), lr 2e-5, warmup 100, cosine, AdamW (0.9, 0.95), clip 0.5, batch 2, ctx 256. Generator-only activation checkpointing ON; `expandable_segments:True`. If T=2 OOMs at ctx 256, drop ctx to 192 for ALL arms and note it.

## The recursion cell (precise definition)

Inside each wrapped FFN, x = block input, norm = the teacher's RMSNorm:

```
u_0 = x
for t in 1..T:
    h_t   = norm(u_{t-1})
    wd,wu = experts for this micro-step        # arm-dependent
    u_t   = u_{t-1} + einsum(GELU(h_t · wd), wu)
return (u_T - x)                               # block residual adds this once
```

Per-micro-step residual, teacher norm re-applied, no detach between micro-steps (T=2; if unstable, detach inside the generator call only and note it).

## Arms (shared warm start, 5,000 NTP steps each)

- **A — t1_control**: T=1, just 5,000 more NTP steps. Attributes gains to recursion rather than to more training. Expected to grind slowly (Stage 2's 5k steps bought 0.014 nats; diminishing).
- **B — fixed_fn**: T=2, generate experts once from norm(x), apply twice. Activation-space recursion, frozen per-token function.
- **C — rederive**: T=2, regenerate experts each micro-step from norm(u_{t-1}). Function-space recursion — the hypothesis arm. 2× generator calls (honest compute note in README).
- **D — act_noise**: arm B + Gaussian noise on u_1 between steps (std 0.02 × RMS(u_1); eval with fixed seed). The GRAM perturbation control: if C ≈ D > B, gains come from the second step seeing a different state, not from re-derivation.

## Procedure

1. Implement `experiments/recursive_stage3.py`: NTP loop (clone of ntp_stage2 naive path) + `--t-steps {1,2}`, `--rederive {once,step}`, `--noise-std FLOAT`, `--init-from` wired into the GeneratedFFN wrapper per the cell above.
2. **Step-0 eval per arm before any training.** B/C/D insert a single-application generator into a two-application loop, so expect an immediate CE *bump* at step 0 (the warm start is off-distribution for T=2). Record it — the recovery curve is part of the result, and Stage 2's transition trace is the template.
3. Smoke 20 steps per arm NaN-free, then queue all four in tmux `gram:hyperpeer`, W&B `stage3_{t1_control,fixed_fn,rederive,act_noise}`, logs `logs_stage3_<arm>.log`, results `results/stage3_<arm>/`.
4. Eval every 100 steps (student CE, teacher CE same batches, KL-to-teacher). Stamped checkpoint copies at 2,500 and 5,000.
5. Precise eval per finished arm (`eval_student.py` extended with the same flags; noise arm evals with its noise, fixed seed; 50 × 4 × ctx 512, seed 42).
6. Divergence playbook per training-run-management: kill before next rolling save, stamp, resume at lr/3.

## Metrics

Primary: precise-eval CE per arm vs the 2.3573 warm-start number and vs teacher 2.3227; frontier placement. Secondary: step-0 bump size and recovery half-life per arm; final KL-to-teacher (drift); steps/s and wall-clock (compute-honesty for C).

## Pre-committed predictions

- **A**: 2.350–2.355 (slow grind, diminishing returns).
- **B**: recovers its step-0 bump within ~500 steps, ends ≈ A or marginally better (free depth, same function).
- **C**: the bet — ends ≥0.015 nats below A (i.e., ≤ ~2.340, at/past k8 2.3387 would be the headline). Genuinely uncertain; if C ≤ B the function-space claim fails at this scale and the design note gets revised — publishable either way.
- **D**: ≈ B (GRAM ablation: unstructured noise is worthless). D ≈ C > B would be the surprise that redirects the program toward stochasticity.
- Step-0 bumps: C < B ≈ D is plausible (C's second step at least conditions on the actual current state); not confident.
- Stability: NTP at 2e-5 from a warm start should not diverge; at most one resume across the queue.

## Success criteria

- Hypothesis confirmed iff C − A ≤ −0.015 nats AND C − B ≤ −0.010 on the precise eval, C stable.
- Any arm at or below k8 (2.3387) is a headline.
- All arms keep KL-to-teacher < 0.15 (sanity: nobody wandered off the language model).

## Runtime budget

A ≈ 65 min (1.3 step/s), B ≈ 75 min, C ≈ 100 min (2× generator), D ≈ 75 min. Queue ≈ 5.5 h + evals. Ceilings, not hang thresholds.

## Files to create

- `experiments/recursive_stage3.py`
- `results/stage3_<arm>/` × 4 (metrics.jsonl, stamped ckpts)
- `results/stage3/README.md`

After the runs, write `results/stage3/README.md` in the phase-README format: headline paragraph with the C-vs-A/B verdict and the key number, headline table (4 arms × precise CE / delta vs warm start / step-0 bump / wall-clock), per-arm prose, pre-committed predictions vs measured, architectural interpretation (paper-ready blockquote if warranted), file manifest, open questions (candidates: T=3, depth-collapse via step embedding, engram conditioning, task-shifted NTP forgetting arm).

`git add` everything new/modified, commit as `Stage 3: recursion added to trained model under NTP — <one-line verdict>`, push.

---

## Deferred variant (rev 1, for the record)

Fresh-init distillation per arm at matched budget — isolates the pure architectural question ("does function-space recursion *learn* better?") from the warm-start convenience. Costs ~4 full distillations. Run it only if rev 2's C-result is positive and we want the cleaner causal claim for a paper, or if warm-started C fails in a way that smells like a bad init rather than a bad idea (e.g., step-0 bump never recovers while fresh smoke-training a tiny T=2 model works).

# Stage 3 — Within-Layer Recursion: NULL (and the recipe that mattered)

Recursion was tested six ways against the trained generator and contributed nothing on TinyStories; the entire apparent gain of the best recursive run was reproduced by its matched-heat T=1 control. The accidental finding is the training recipe: hot NTP after distillation (lr 6e-5, 10k steps) moved the single-pass generator to **2.3372** precise vs teacher 2.3227 — past the k=8 retrieval frontier (2.3387), gap 0.0145 nats, one-third of the post-distillation gap.

## Headline table (precise eval: 200 seqs × ctx 512, teacher = 2.3227)

| Arm | Schedule | Precise CE | Δ vs teacher |
|---|---|---|---|
| warm start (stage-2) | — | 2.3573 | +0.0346 |
| t1_control | 5k @ 2e-5 | 2.3509 | +0.0281 |
| fixed_fn (T=2 same experts) | 5k @ 2e-5 | 2.3527 | +0.0300 |
| rederive (T=2 fresh experts) | 5k @ 2e-5 | 2.3524 | +0.0297 |
| act_noise (T=2 + state noise) | 5k @ 2e-5 | 2.3546 | +0.0319 |
| rederive HOT | 10k @ 6e-5 | 2.3390 | +0.0163 |
| **t1_control HOT (attribution)** | 10k @ 6e-5 | **2.3372** | **+0.0145** |
| native T=2 distill (from scratch) | killed @10.7k/20k | ~3.1 (stuck) | failed |

## Per-condition notes

Gentle 5k arms: statistically tied; the T=2 arms spent the budget recovering their insertion bump. Correct criticism (Mikey): decaying 2e-5 on a converged model can't teach a new strategy — null here is weak evidence, prompting the hot rerun.

Hot rederive: best model for a few hours; on the small fixed-batch eval it edged the teacher (2.2917 vs 2.2929) but not on the full protocol.

Attribution control: same heat, no recursion, slightly better. Criterion was pre-registered before it ran.

Native T=2: KL flat ~0.7, CE pinned at no-FFN level for 10k+ steps. Curriculum result: the one-pass solution is a prerequisite; recursion is not distillable from scratch at this config.

## Pre-committed predictions vs measured

Predicted A 2.350–2.355 (measured 2.3509 ✓); B ≈ A (✓); C ≥0.015 over A (✗ — null after attribution); D ≈ B (✓); hot-C ≤2.300 win threshold (✗ — 2.3390, and credit went to the schedule). The one unpredicted result: the schedule itself as the headline.

## Architectural interpretation

> On a corpus whose teacher sits near the entropy floor, a second reasoning pass has nothing to fix: within-layer recursion — reapplied, re-derived, or noise-perturbed; retrofitted gently or hot; or trained natively — adds nothing that a matched training schedule doesn't. Meanwhile distill-then-hot-NTP closes three-quarters of the remaining teacher gap with zero inference cost. Recursion's hypothesis survives only as: hard tokens are required. Retest once, on WikiText.

## Files

`experiments/recursive_stage3.py`, `results/stage3_{t1_control,fixed_fn,rederive,act_noise}{,_hot10k}/`, `logs_stage3_*.log`, `eval_stage3.out`, `eval_hot10k.out`, native run: `results/stage1_native_t2/`, `logs_native_t2.log`.

## Open questions

(1) WikiText rung: retrain the teacher on WikiText-103, repeat distill → hot-NTP → one recursion arm with matched control. (2) Does a longer/restarted hot schedule cross the teacher (gap 0.0145 was still shrinking at cosine end) or asymptote? (3) T-annealing or per-microstep KL targets as a native-recursion curriculum, only if (1) shows recursive headroom.

# Idea: Hypernetwork-Recursive Reasoning with Engram Conditioning (+ gate)
Origin: Mikey brainstorm 2026-06-09, after watching the GRAM explainer. DEFERRED — pursue after the GRAM comparison (P1/P2). Capturing while fresh.

## Core architecture
Move the variability from ACTIVATION space (GRAM perturbs the hidden state h) to FUNCTION space:
  context h_t -> HYPERNET g(h_t [, engram]) -> generates a transform / small latent space
            -> run context THROUGH that transform -> refined h_{t+1}
            -> recurse.
So the reasoning operation is RE-DERIVED from the data each step, not a fixed shared function re-applied (the TRM/GRAM way). Lineage: fast-weights / fast-weight programmers (Schmidhuber), HyperNetworks (Ha). Buildable form = hypernet emits a LOW-RANK / FiLM modulation (A:512x16, B:16x512; h += B*tanh(A*h)), not full weights. The small bottleneck is also what makes it tunable; rhymes with Mikey engrams/compression line.

## Two distinct roles for the injected vector (keep them separate!)
1. CONDITIONING (deterministic, retrieval-like): inject the RIGHT knowledge so the model reasons with facts it didnt fully learn. Engram = a precomputed, interpretable knowledge vector, e.g. the MEAN-POOLED Wikipedia article per entity. Chemistry example: bank = {mean-pool(article) for each element}; for 2H2+O2->2H2O inject the H and O engrams.
2. STOCHASTICITY (GRAM-style): sample different engrams/codes -> different reasoning pathways. Helps on MULTI-solution / pure-search problems.

## Mikey hypothesis / my read
- Conditioning is likely the BIGGER lever for knowledge-rich tasks (most practical ones); GRAMs ablation showed pure noise is worthless, ALL gain came from STRUCTURE -> conditioning is structure taken to its end. Cheaper too (one directed pass vs N samples + selector).
- BUT on pure combinatorial search with no external knowledge (hard Sudoku, N-Queens, ARC) conditioning has nothing to inject; stochasticity is the only lever and wins. So "better" depends on problem class.
- Risk of conditioning: only as good as engram relevance; a WRONG engram injects confident wrong knowledge (worse than noise). Stochasticity fails gracefully.

## Unifiers (Mikey)
- A HEAD that generates a LIST of candidate engrams -> a distribution. Commit to top = conditioning; sample = stochasticity. Knob, not fork.
- A GATE (soft, learned) that decides per-case: conditioning vs stochasticity vs BOTH. Lets the model LEARN the boundary instead of us hard-coding it. Soft (non-exclusive) gate gives "both" for free.

## Experiment design (when we get to it)
Ablation arms on the SAME task: (a) no engram; (b) engram as plain input (concat / cross-attn, NO hypernet); (c) engram -> hypernet-generated transform. The whole bet is (c) > (b) — must show the hypernet earns its complexity.
Killer result: GENERALIZATION TO HELD-OUT ELEMENTS — train on reactions for some elements, test on an unseen element purely via its injected Wikipedia engram. If it works, the model is USING injected knowledge, not memorizing.
Run on BOTH a knowledge-rich task (chemistry/held-out elements) AND a pure-search task (Sudoku) to MAP where conditioning helps vs where only stochasticity does — that boundary is the finding.

## Caveats / prior art to position against (be honest, not de-novo)
fast weights, HyperNetworks, FiLM conditioning, retrieval-augmented generation, discrete-VAE / VQ, mixture-of-experts, test-time latent optimization, and DeepSeeks "Engram" conditional-memory paper (arXiv 2601.07372, N-gram-addressed memory). Discrete engram pick is hard to train (Gumbel/straight-through/REINFORCE) -> start with a SOFT attention-weighted blend over the bank. Watch posterior collapse (model ignores engram) and recursion stability (residual+norm+detach like TRM). Mean-pooling a whole article is lossy (gist > mechanism) but fine as v1.

## Training methodology: borrow Drag-and-Drop LLMs (DnD) — added 2026-06-09
Mikey: "use the same methodology as Doc-to-LoRA." That is Drag-and-Drop LLMs (arXiv 2506.16406; sibling: Text-to-LoRA). DnD = lightweight text ENCODER (prompts -> condition embedding) + structured DECODER (-> full LoRA matrices in one shot), trained on (prompt -> LoRA-checkpoint) pairs with plain MSE to ground-truth weights. Generates adapters for NEW datasets zero-shot in one forward pass; reportedly beats the training LoRAs on unseen tasks.

ONE-TO-ONE MAP onto our idea:
- DnD prompt->condition embedding  ==  our ENGRAM (e.g. mean-pooled Wikipedia article / PEER key)
- DnD ground-truth LoRA checkpoints ==  real PEER EXPERTS (rank-1 up/down) as ground truth
- DnD MSE-to-weights               ==  Mikey "use actual experts as ground truth"
- DnD structured decoder (whole adapter set at once) == Mikey "generate experts in GROUPS"

CORRECTION to an earlier note: I had cautioned "match behavior not weights" (LoRA gauge symmetry). DnD shows DIRECT WEIGHT MSE WORKS, because you regress to ONE specific trained checkpoint -> no factorization ambiguity. So plain MSE is the proven, simpler recipe.

ENTRY EXPERIMENT (now concrete + de-risked):
1. Train/obtain a small PEER (lucidrains impl) on a toy task -> real expert bank with product keys.
2. DnD-style generator: encoder(key/engram) -> decoder -> expert (rank-1 up/down). Loss = MSE to the real PEER expert. (Optionally + a functional/task loss.)
3. Test on HELD-OUT keys/engrams: does it generate useful unseen experts? (this is the feasibility + generalization test; DnD evidence says plausible.)
4. Only then wire the generator into the recursive reasoning loop (generate active group per step, run context through, recurse), with engram conditioning + sampled group-seed for stochasticity.

Differences to watch: PEER = ~1M rank-1 experts vs DnD one adapter set (mitigate: only generate the ACTIVE ~256 group per context, not all); DnD conditions on prompt BATCHES vs our single engram.

## Refinement (Mikey 2026-06-09): logit DISTILLATION, not weight MSE; two-stage
Instead of DnD weight-MSE, use FUNCTIONAL distillation:
- TEACHER = model running the REAL PEER experts. STUDENT = same model with GENERATED experts (hypernet).
- Loss = KL between teacher and student LOGITS (softened, temperature) on a data stream. Matching OUTPUTS not weights -> sidesteps the LoRA factorization/gauge symmetry entirely. (This re-vindicates the earlier "match behavior not weights" instinct; DnD shows weight-MSE also works, but logit-KL is the more robust choice.)
- STAGE 1: distill (KL on logits) until student ~ teacher -> proves the generator makes FUNCTIONING experts; strong init.
- STAGE 2: switch to END-TO-END next-token prediction -> generator now produces experts that OPTIMIZE the task, beyond mimicry.

Honest tradeoffs: logit distillation runs BOTH teacher+student per step (more compute than weight-MSE). Stage-2 can DRIFT off the distilled init (forgetting) -> mitigate with low LR and/or a small retained KL-anchor. Requires a runnable PEER teacher (train-a-small-PEER is now load-bearing). Define a switch criterion (distill KL plateau / student within eps of teacher).

Crystallized recipe: generator(engram-or-key encoder -> group decoder) ; Stage1 logit-KL distill vs PEER teacher ; held-out-key generation test ; Stage2 end-to-end next-token (low LR + KL anchor).

## REALITY CHECK (Mikey skepticism, 2026-06-09) — do NOT assume DnD/Text-to-LoRA worked
Mikey: "I do not think this worked with Doc-to-LoRA." Checked: found NO independent replication — almost all material is the authors own claims + their self-acknowledged limitations. The "beats trained LoRAs by 30%" claim is unreplicated -> treat as unproven. (I had over-weighted it earlier; correcting.)
DnD self-admitted limitations, all relevant to us:
- fails when training pairs are FEW (generation is data-hungry).
- generalization HIGHLY sensitive to diversity/representativeness of training pairs (threatens held-out-element generation).
- full fine-tuning eventually BEATS it (generation is a fast approximation, not a quality ceiling).
Implication for our plan: those mostly damage the ZERO-SHOT generate-and-go story; our Stage-2 end-to-end fine-tune already does not rely on that. BUT the real risk Mikey is pointing at: if generated experts add nothing useful, "generate experts" is just an expensive random init and Stage-2 re-learns from scratch. So the PREMISE hinges on the Stage-1 feasibility gate: can distillation produce FUNCTIONING experts that generalize to held-out keys? DO NOT ASSUME YES. Run that cheap gate FIRST, expect it may fail, invest only if it passes.

## RISK: catastrophic forgetting on the distill->next-token switch (Mikey, 2026-06-09)
Mikey recalls a paper in this family reporting CATASTROPHIC FORGETTING when, after teacher-distillation, they switched to next-token (end-to-end) training. Checked DnD (2506.16406): does NOT discuss it (DnD uses weight-MSE, not teacher distillation) -> the memory is likely a DIFFERENT paper, probably Text-to-LoRA (Sakana). Citation UNVERIFIED; TODO pull Text-to-LoRA to confirm.
Why it matters: Stage-2 (next-token) is the linchpin of our plan. If next-token training destroys the distilled experts, the distilled and task-optimal solutions are in different basins => the generated init bought nothing => premise fails. This is make-or-break, not a footnote; reinforces Mikey overall skepticism.
MITIGATION (adopt regardless): do NOT hard-switch. Blend/anneal distillation-KL DOWN while ramping next-token UP; keep a small KL-anchor to the teacher throughout; low LR across the transition; optionally regularize toward / partially freeze the distilled init.
TEST: after Stage 1, run the transition WITH vs WITHOUT the anchor, watch for forgetting BEFORE committing. Forgetting even with mitigation = strong fast negative signal on the whole premise.

## FIX for the switch-forgetting (Mikey, 2026-06-09): freeze + train a projection, then unfreeze
Mikey: to switch from student-teacher distillation to next-token, FREEZE the model, train a PROJECTION MATRIX on next-token, THEN train end-to-end. This re-derives a published recipe:
- LP-FT = Linear-Probe-then-Fine-Tune (Kumar et al., ICLR 2022, "Fine-Tuning can Distort Pretrained Features and Underperform OOD"): fine-tuning from a mismatched head DISTORTS good features; fix = train the head on FROZEN features first, then unfreeze. Exactly Mikey proposal.
- Gradual unfreezing (ULMFiT, Howard & Ruder 2018): unfreeze layers progressively (head -> generator -> base), not all at once.
Mechanism: forgetting at the switch = a fresh/mismatched output layer back-props HUGE gradients into the distilled features and wrecks them; freezing the body + training the projection first aligns output to features cheaply, so unfreezing gives small well-conditioned gradients (features nudged, not bulldozed).
Pin down for us: WHICH projection = the next-token readout (LM head/unembedding), maybe + the generated-expert->model projection. WHAT to freeze = base + hypernet generator during probe phase, then unfreeze selectively.
Honest: LP-FT reduces but does not always eliminate forgetting; Stage-2b can still drift -> combine with the KL-anchor to teacher AND watch the metric across the transition. Strong principled mitigation, not a guarantee. Net: converts the make-or-break switch into a managed staged procedure with a published recipe.

## Provenance correction (2026-06-09): the forgetting warning is from Rosanne Liu / Sakana, not a paper
The "catastrophic forgetting when switching distillation -> next-token" warning did NOT come from a paper. SOURCE = Rosanne Liu, in a Zoom meeting/talk with Sakana AI (Mikey recollection). Sakana is the Text-to-LoRA lab, so it is very likely her describing their actual Text-to-LoRA experience verbally. STATUS = PRACTITIONER/ANECDOTAL — real and worth heeding, but we do NOT know the exact training setup, so cannot confirm it generalizes or that the freeze-projection (LP-FT) fix is always required. Earlier "TODO pull Text-to-LoRA to confirm" stands but the primary source is the talk, not the paper; the paper may or may not mention it.

## THE SYNTHESIS (Mikey, 2026-06-09) — "the new PEER project" = D2L-method applied to PEER adapters
Claude kept failing to connect these; they are ONE plan, not separate options:
- ORIGINAL PEER project = ~/Code/HN/peer-adaptive-k (on pop). Produces the real PEER adapters/experts (rank-1 up/down) = GROUND TRUTH. Has trained checkpoints.
- D2L = ~/Code/doc-to-lora (on Mac). The METHOD: a Perceiver HYPERNETWORK that GENERATES a (rank-8) LoRA adapter, trained STUDENT-TEACHER (logit-KL distillation) against a target. Proven to work (Bleak House install).
- NEW PEER PROJECT = apply D2L method to PEER adapters: a hypernetwork that GENERATES the PEER experts, distilled against the real ones from peer-adaptive-k. (PEER experts are rank-1 ~ tiny LoRAs, so D2L machinery transfers.)
Parts list already exists: peer-adaptive-k = ground-truth adapters; D2L = generator + distillation pipeline. Do NOT need D2Ls lost checkpoint or a from-scratch PEER distillation.
## THE SYNTHESIS 2026-06-09 - the new PEER project = D2L method applied to PEER adapters
Claude kept failing to connect these; they are ONE plan, not separate options:
- ORIGINAL PEER project = ~/Code/HN/peer-adaptive-k on pop. Produces the real PEER adapters/experts, rank-1 up/down = GROUND TRUTH. Has trained checkpoints.
- D2L = ~/Code/doc-to-lora on Mac = the METHOD. A Perceiver hypernetwork that GENERATES a LoRA adapter, trained STUDENT-TEACHER via logit-KL distillation. Proven on the Bleak House install.
- NEW PEER project = apply D2L method to PEER adapters: a hypernetwork that GENERATES the PEER experts, distilled against the real ones from peer-adaptive-k. PEER experts are rank-1 so basically tiny LoRAs and D2L machinery transfers.
Parts already exist: peer-adaptive-k gives ground-truth adapters, D2L gives the generator + distillation pipeline. Do NOT need D2L lost checkpoint or a from-scratch PEER distillation.
FORGETTING TEST rides on top: the new PEER project is student-teacher trained by construction, so NTP fine-tuning IT is exactly the distill-to-NTP switch where catastrophic forgetting was reported by Rosanne Liu at Sakana. Build: extract real adapters from peer-adaptive-k, train a D2L-style hypernet to reproduce them via distillation, then NTP fine-tune and measure forgetting, plus the LP-FT freeze-projection fix.

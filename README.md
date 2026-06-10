# HyperPEER

Generate PEER experts with a hypernetwork instead of retrieving them.

PEER builds a per-token MLP by retrieving k single-neuron experts via product keys, then stacking them along the rank dimension into a k-by-d matrix, and applying out = sum_i w_i * up_i * GELU(down_i . x). HyperPEER trains a hypernetwork to take the input hidden state and generate that k-by-d matrix directly, with no retrieval and no stored bank. The experts become a function of the input.

Teacher and ground truth: the trained PEER model in ../peer-adaptive-k, which uses K_max=256 candidate experts and has fixed-k checkpoints. For each input we read off the assembled expert matrix and the output logits it produces, and train the generator to reproduce them. Nothing is extracted or stored.

Plan, staged to de-risk:
1. Harness: run peer-adaptive-k over inputs, capture hidden-state to k-by-d expert-matrix pairs via its collect path.
2. Generator: a Perceiver or D2L-style hypernet mapping hidden state to the matrix.
3. Train by distillation: logit-KL to match teacher outputs, and/or matrix-MSE to match the matrix.
4. Forgetting test: switch to next-token fine-tuning and measure whether the distilled ability collapses; mitigate with LP-FT, meaning freeze the body, train a projection, then unfreeze.

Full design, the conditioning-vs-stochasticity discussion, the gate idea, and the honest risks are in notes/design.md. Built on ../peer-adaptive-k. Runs on the RTX 5070 Ti.

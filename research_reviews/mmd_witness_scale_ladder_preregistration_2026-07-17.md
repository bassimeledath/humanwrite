# MMD-witness 4B data-scale ladder preregistration

Status: frozen before the measurement-v4 candidate outputs were generated or opened and before any scale-ladder source was selected.

## Question

Does the retained lower-variance MMD-witness treatment acquire a reliable human-writing advantage over matched SFT as clean, structured training data increases from 4,096 to 16,384 and then 46,080 documents?

This is a data-scale experiment, not a claim to reproduce Rosmine's proprietary DFT algorithm. The largest cell is approximately the quarter-data scale discussed in the article and remains about four times smaller than its 185,000-document corpus.

## Launch condition from measurement v4

The 4,096 cell may launch unless measurement v4 shows all of the following: adequate treatment displacement, a hard validity pass, and clear harm or movement in the wrong direction on both independent embedding families and blinded human-style judgment. A null caused by weak displacement does not block the scale ladder because additional unique examples and optimizer steps are the intervention under test.

No coefficient, sampler, representation, or model-size choice may be selected from measurement-v4 results.

## Why the scale cells are independent

A single 46,080-document run checkpointed after 4,096 and 16,384 examples would confound data diversity with continued optimization and would let a full-corpus witness influence the smaller checkpoints. Therefore each scale cell restarts from the same frozen seed-11 initial adapter and computes its witness only from its own immutable nested corpus. The extra compute is accepted in exchange for identifying a genuine data-scale effect.

## Nested data

- Accepted training sizes: 4,096, 16,384, and 46,080 documents.
- The 4,096 corpus is an immutable hash-selected prefix of 16,384; 16,384 is an immutable prefix of 46,080.
- Every document is disjoint by source fingerprint and cleaned fingerprint from all prior Humanwrite training and evaluation artifacts.
- Domain concentration is capped prospectively; the exact cap and deterministic tie-breaking rule must be frozen before source streaming.
- Cleaning uses Qwen3-32B ordered-line removal under the existing strict provenance validator.
- Qwen3-32B generates prompt, use case, and style metadata. GPT-5-mini generates outlines. Exactly 25% of each nested prefix has an empty outline.
- Target lengths are normalized with the pinned Qwen3-4B tokenizer and serialized as tokens.
- Provider failures are resumable and provenance-labeled; no row may be silently substituted or hand-authored.

## Matched arms at each scale

- Model: pinned Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`.
- Accelerator: L40S for baseline generation, witness materialization, both training arms, candidate generation, and embeddings. There is no automatic H100 fallback. The completed local benchmark processed 140,196 4B training tokens in 493.373 accelerator-seconds for $0.320890, while the recent H100 arms had nearly identical token throughput at approximately twice the cost per token.
- Starting artifact: the same frozen seed-11 initial SFT adapter used by measurement v4.
- Arms: uniform SFT and MMD-witness only.
- Exposure: one complete seeded epoch, batch size 2, identical order and random streams.
- Optimizer: learning rate 1e-5, AdamW weight decay 0.01, gradient clipping 1.0.
- Teacher-forced horizon: 128 tokens with EOS-aware masking.
- Witness representation, bandwidth family, and temperature 0.035 remain fixed from measurement v4.
- Each cell computes an exact witness from only that cell's human and baseline-generated distributions.

The 46,080-scale witness must use a chunked exact RBF computation. Unit tests must prove equality with the existing dense implementation on small matrices, invariance to chunk size, correct leave-one-out diagonals, and bounded memory.

Baseline generations, frozen human/base embeddings, and witness values are materialized once per nested corpus and shared by both matched arms. The SFT arm must not recompute MMD-only artifacts.

## Evaluation structure

Two new evaluation surfaces are required and are disjoint from all training data and prior panels:

1. A scale-development panel used for the 4,096 and 16,384 gates. Opening it does not authorize hyperparameter changes.
2. A final untouched confirmation panel opened only for a 46,080 cell that passes the 16,384 gate.

Every scale comparison uses matched prompts and sampling seeds, 128-token EOS-aware generation, BGE-small, Llama-Embed-Nemotron-8B, token unigram L2, repetition and script validity, byte identity, and blinded human-style plus overall-quality judgments.

## 4,096 gate

The 4,096 result is primarily diagnostic. Stop immediately only for non-finite training, artifact mismatch, hard validity failure, a significant overall-quality regression, or movement in the wrong direction on both embedding families with human-style win rate below 50%.

Otherwise unlock 16,384.

## Hard early-stop gate at 16,384

The 46,080 cell is unlocked only if all safety conditions and at least one signal condition pass.

Safety conditions:

1. No hard validity regression versus matched SFT.
2. Token unigram L2 is non-inferior within 0.25 human-split standard deviations.
3. Overall judged quality is not significantly worse at one-sided p <= 0.05.
4. At least 30% of MMD-witness outputs differ byte-for-byte from SFT.

Signal conditions; at least one is required:

1. Both independent embedding-family MMD point estimates improve versus SFT; or
2. one embedding family improves by at least 0.5 human-split standard deviations and blinded human-style win rate is at least 52%.

If this intersection fails, stop at 16,384. Do not reinterpret a single noisy metric, alter the coefficient, or launch 46,080.

## Final 46,080 decision

The final candidate is promoted only if the untouched confirmation panel shows:

1. improvement in both embedding families under prospectively calibrated meaningful-effect and paired-significance rules;
2. token unigram L2 non-inferiority;
3. human-style win rate at least 55% with one-sided p <= 0.05;
4. no significant overall-quality regression;
5. byte identity with matched SFT below 60%; and
6. all hard validity and repetition checks pass.

The final panel should contain at least 400 prompts for blinded judgments. Automatic distribution references and human floors are fixed before candidate generation. No 14B, sealed, GPTZero, or Pangram escalation follows an automatic-only result.

## Budget and time boundaries

- Internal Modal monthly cap: $100.
- Internal OpenRouter monthly cap: $100.
- Expected stop-at-16,384 total: approximately $40-$60 across Modal and OpenRouter.
- Expected full independent 4,096/16,384/46,080 ladder: approximately $115-$160 across Modal and OpenRouter, with approximately $70-$95 attributable to Modal.
- Each scale is resumable only at completed artifact or optimizer-step boundaries.
- Data preparation may be pipelined, but 46,080 paid synthesis should not materially outrun the 16,384 decision.

Budget exhaustion causes a safe pause, never threshold weakening or artifact substitution.

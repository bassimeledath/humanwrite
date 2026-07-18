# MMD-witness 4B confirmation preregistration

Status: frozen before source materialization, training, candidate generation, or candidate inspection.

## Question

Can a prospectively stronger MMD-witness weighting produce a reliable human-style and distributional improvement over a matched SFT control at 4B and a 128-token horizon?

## Why this is the only retained treatment

The completed measurement-v3 cycle found that token-moment improved some automatic metrics while significantly worsening judged overall quality. It is retired. MMD-witness improved all automatic distribution metrics and produced 54.7% human-style wins without a significant quality regression, but 78.1% of its outputs were byte-identical to SFT.

The completed MMD-witness training audit shows why displacement was weak:

- Witness weights at temperature 0.7 had mean 1.0 and standard deviation 0.01056.
- The witness-delta gradient norm averaged 0.00885 versus 1.578 for uniform SFT, a mean ratio of 0.00580.
- This made the treatment effectively ordinary SFT with a very small reweighting perturbation.

## Frozen treatment

- Model and starting adapter: the same pinned Qwen3-4B revision and seed-11 initial adapter used by measurement v3.
- Training corpus: the same frozen 1,024 cleaned, structured training briefs.
- Arms: matched uniform SFT control and MMD-witness treatment only. No token-moment arm.
- Exposure: two complete seeded epochs per arm, batch size 2, 1,024 optimizer steps, 2,048 optimizer examples.
- Teacher-forced completion horizon: 128 tokens with EOS-aware masking.
- Learning rate: 1e-5; AdamW weight decay 0.01; gradient clipping 1.0.
- MMD-witness representation and bandwidths: unchanged from measurement v3.
- Witness weighting: mean-one softmax with temperature 0.035.

Using the already-frozen witness values, temperature 0.035 deterministically implies weight standard deviation about 0.199, range about 0.378 to 1.654, and effective sample size about 985/1,024. This is approximately 18.8 times the prior weight dispersion without concentrating training on a small subset.

## Training stop rules

Stop the treatment before evaluation if any occur:

- non-finite loss or gradient;
- more than 25% of steps are gradient-clipped;
- witness-delta/uniform-SFT gradient-norm ratio has median below 0.05 or above 0.30 after the first 64 steps;
- completed artifact or input hashes do not match the frozen contract.

The control and treatment must share the exact prompt order, optimizer exposure, seed, runtime, and starting adapter.

## Fresh evaluation

- Materialize and clean a new FineWeb pool with all prior training and evaluation domains and fingerprints excluded.
- Freeze 128 prompt sources, 256 semantic references, and two disjoint 128-document human floors before candidate generation.
- Synthesize briefs only for prompt sources; never condition evaluation on candidate outputs.
- Generate 128 tokens with normal EOS stopping, identical per-prompt seeds, temperature 1.0, top-p 1.0, and no top-k truncation.
- Evaluate with pinned BGE-small and Llama-Embed-Nemotron-8B, token unigram L2, repetition, validity, and blinded GPT-5.4-mini pairwise judging.

## Promotion rule

The MMD-witness treatment passes only if all are true on the untouched panel:

1. Both embedding families improve in the preregistered direction and satisfy the prospectively calibrated meaningful-effect and paired-significance rules.
2. Token unigram L2 is non-inferior.
3. Unexpected non-Latin output rate is no worse than control by more than 2 percentage points and is at most 10% absolutely.
4. Replacement-character rate is zero; uniqueness and repetition checks pass.
5. Byte identity with SFT is below 60%.
6. Human-style judge win rate is at least 55% with one-sided p <= 0.05.
7. Overall judged quality is not significantly worse.

No 14B run, sealed submission, GPTZero, or Pangram call is allowed unless this complete intersection passes. A second automatic-only trend is not sufficient.

## Interpretation

- Pass: the reconstructed MMD-witness route earns multi-seed 4B confirmation and then consideration for 14B.
- Fail with adequate displacement: retire this reconstruction; the proprietary Rosmine method is materially different or the disclosed evidence is insufficient.
- Fail because displacement remains below 40%: permit at most one implementation audit, not another blind coefficient sweep.

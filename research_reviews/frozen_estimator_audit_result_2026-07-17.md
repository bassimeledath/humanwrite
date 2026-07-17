# Frozen estimator audit result

Date: 2026-07-17

Run: `dftr-1784303078-239a1732`

Commit: `d82529f647fbd4e6b7e7e6d0867f2f39e834e011`

Config SHA-256: `67c5b0b96321bd0f27e2d78ed7edf3e34ce294d74edec52cd3f8905484f1f682`

Result SHA-256: `74a7ccd9b6baf6f5f6b264c552125914108796c74e391358f65c46b3a5f15e0a`

## Decision

Do not advance the score-function semantic-MMD route to global-K32 training.
The length-matched K32 gradient-norm coefficient of variation passed its
prospective maximum, but split-half gradient direction failed by a wide
margin. The independently recomputed result contains all 128 expected audit
rows and reproduces every reported split-half cosine.

| Human reward support | K | Split-half gradient cosine | Gradient-norm CV | Mean gradient norm | Mean MMD2 |
|---|---:|---:|---:|---:|---:|
| Full documents | 4 | -0.164 | 0.996 | 1.750 | 0.00833 |
| Full documents | 8 | -0.070 | 0.644 | 1.088 | 0.00801 |
| Full documents | 16 | 0.154 | 0.442 | 0.658 | 0.00666 |
| Full documents | 32 | 0.041 | 0.291 | 0.459 | 0.00694 |
| 64-token humans | 4 | -0.144 | 0.937 | 1.648 | -0.00122 |
| 64-token humans | 8 | -0.073 | 0.699 | 0.828 | -0.00165 |
| 64-token humans | 16 | 0.118 | 0.383 | 0.463 | -0.00200 |
| 64-token humans | 32 | -0.014 | 0.394 | 0.273 | -0.00174 |

Prospective K32 gate for 64-token support:

- split-half cosine required at least `0.50`; observed `-0.01385`;
- gradient-norm CV required at most `1.0`; observed `0.39377`;
- conjunction result: fail.

## Interpretation

Increasing K clearly reduced advantage variability, gradient-norm variability,
and mean gradient magnitude. It did not make the estimated gradient direction
consistent across independent replicate halves. That pattern is consistent
with a noisy or prompt-dependent semantic-MMD direction whose variance cannot
be repaired merely by increasing K from 4 to 32.

Length matching materially changed absolute MMD2, from approximately `+0.0069`
with full humans to `-0.0017` with truncated humans at K32. It did not improve
gradient coherence. This confirms that support length contaminated the old
absolute reward geometry but was not the sole reason the score-function
gradient failed.

All rollouts used the full 64-action horizon, so no EOS occurred in this audit.
The EOS-aware implementation remains necessary for longer or different-policy
runs, but post-EOS actions do not explain this result.

The CountSketch dimension, seeds, group sizes, support variants, and thresholds
were frozen before output. The same projection was used for every replicate,
so projection noise cannot selectively favor one group size or support. Both
support variants independently miss the direction threshold by a large margin.

## Next branch

Retire score-function semantic MMD as the next training mechanism. Build a
matched lower-variance 4B screen on cleaned data with:

1. an ordinary SFT continuation control;
2. teacher-forced token-distribution moment matching, directly targeting the
   article's token-L2 motivation;
3. a one-round shared-rollout MMD-witness-weighted SFT arm, testing whether
   distributional reweighting works when gradients flow through human targets
   rather than sampled actions.

Evaluate on a fresh unpaired panel with two independent embedders, token
unigram L2, validity, repetition, length, and prompt-grounded quality. Do not
use 14B, the sealed endpoint, or detector feedback until one 4B arm shows a
replicable improvement.

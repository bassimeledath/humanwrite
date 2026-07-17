# Independent-review reconciliation and next-cycle decision

Date: 2026-07-17

## Decision

The completed A64 result remains a valid failure of the exact delivered recipe:
eight optimizer steps, learning rate `1e-5`, MMD coefficient `0.1`, four
cross-prompt rollouts per step, 64 sampled tokens, and the frozen seed-11
adapter initialization. It is not a decision-grade negative result for
adequately dosed score-function MMD, Qwen3-4B capacity, distribution objectives
generally, or Rosmine's proprietary DFT.

The next cycle is ordered as follows:

1. rebuild and qualify an unpaired public distribution instrument;
2. repair target-length units, EOS handling, language integrity, power, and
   authorship decision semantics;
3. audit frozen-policy reward sensitivity and gradient stability at effective
   group sizes `4, 8, 16, 32` before any parameter update;
4. choose the next treatment from the audit rather than from the spent panel;
5. run a clean-data scale ladder in parallel with the winning mechanism;
6. reserve 14B, sealed evaluation, and Tier 3 for a confirmed 4B effect.

## Findings independently reproduced

- All 64 old prompt briefs were synthesized from the same 64 documents used
  as the `human_eval` distribution panel. Matched-pair kernel similarity was
  approximately `0.739`, versus `0.518-0.519` for unmatched cross-pairs.
- Topic pairing shifted absolute MMD by approximately `-0.0069`. Removing the
  64 matched cross-pairs changes A0 MMD2 from `-0.004513` to `+0.002391` and
  A64 from `-0.004580` to `+0.002339`.
- The treatment contrast is robust to that offset: approximately `-0.000067`
  with all cross-pairs and `-0.000052` with matched pairs excluded. The exact
  A64 recipe therefore still fails.
- The authorship improvement threshold was `0.15`, while realized A0
  separability was only `0.0937`. The gate was impossible after control
  observation and cannot be reused.
- The prospective power code simulated significance versus zero rather than
  the scored intersection of minimum effect and paired significance. Its
  reported `0.897` power at the exact effect threshold is not valid.
- The absolute MMD diagnostic uses `abs(unbiased_MMD2)` in a two-sided
  permutation test. Large negative unbiased estimates are not evidence of
  distributional separation, so this diagnostic is retired.
- Brief synthesis defines `target_length` as tokens, while prompt rendering
  labels the same integer as words.
- Raw-policy sampling continues for 64 actions through EOS and rewards decoded
  post-EOS resumptions.
- Unexpected non-Latin letters occur in `6/64` A0 outputs and `7/64` A64
  outputs. Operational stability did not establish acceptable writing quality.
- The materialized 256-record corpus is not meaningfully cleaned. It contains
  addresses, stock notices, navigation fragments, review widgets, checkout
  messages, and other page artifacts despite the intended cleaning spec.
- The logged scalar variance of `advantage * sequence_log_probability` is not
  parameter-gradient variance. MMD surrogate-loss sign changes likewise do
  not establish gradient-direction oscillation.

## Reconciliation of the proposed treatments

The K32 score-function proposal is the cleaner direct test of the existing
mechanism, but only if a frozen-policy audit shows that the reward responds to
style rather than primarily topic/length and that larger groups produce a
coherent gradient estimate.

Iterated reward-weighted SFT is a credible lower-variance Arm C and a plausible
interpretation of staged distribution training. The proposed three-round
"identical rollout bytes" control is not valid unchanged: after round one,
control and treatment policies differ, so their rollouts cannot be both
identical and on-policy. The first causal screen should use one shared frozen
rollout set with uniform versus reward-derived weights. Iterative on-policy
rounds require a separately preregistered estimand and control.

## Fresh-instrument requirements

- Prompt-matched references are reserved for quality/adherence only.
- Distribution references and both human-floor panels are disjoint from the
  prompt source documents, all training records, and all historical panels.
- Primary semantic direction must agree under BGE-small and a second frozen
  embedder family that is never used for training reward.
- Token unigram L2 returns as a Rosmine-comparable secondary endpoint.
- Authorship is a non-worsening/equivalence endpoint, not an impossible fixed
  improvement from a near-chance control.
- Repetition, length, script integrity, and diversity use human-calibrated
  two-sided ranges.
- Qualification must detect known positive controls before candidate outputs:
  human-vs-human null, 64-token-prefix versus full-human length difference,
  SFT versus unpaired humans, and base-model versus humans.
- Power must execute the exact intended decision function and demonstrate
  power at an effect strictly beyond the decision boundary.

## Budget boundary

The remaining Modal and provider budgets are sufficient for instrument
qualification, the frozen-policy audit, one serious 4B treatment/control
screen, a clean 256/1024 data ladder, and confirmation of one winner. Increased
compute authorization does not waive the fresh-panel, manipulation-check, or
cross-representation gates.

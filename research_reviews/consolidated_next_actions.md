# Consolidated next actions

Date: 2026-07-16

This plan reconciles `native_postmortem_next_cycle.md` with the independent
`dftr-postmortem-2026-07-16.md`. Repository inspection confirmed the external
report's two most important additional findings: the visible human-floor MMD
bootstraps each fixed half with replacement, and the checked-in deployment
sampler remains unfrozen. No post-merge Tier-1 replay is published.

## Decision

The sealed result rejects the exact seed-29 Qwen3-4B SFT artifact. It does not
reject 4B generally or DFT, because no DFT arm has been trained. Stay at 4B,
repair the screening instrument, verify artifact/interface fidelity, establish
a matched data-scale SFT control, and then test cheap distribution objectives.

## Ordered plan

1. **Measurement and fidelity audit, no new training.**
   - Replay the seed-29 adapter and merged-v5 artifact on identical full-brief
     inputs and sampling seeds; require parity within sampling variation.
   - Operator-audit the sealed prompt renderer, tokenizer/chat template,
     truncation, output cap, batching, and sampler against the visible recipe.
   - Replace the human-floor calculation with disjoint resampling without
     replacement through an operator-owned harness change and attestation.
   - Freeze one canonical deployment-generation contract. Retire checkpoint
     selection by visible AUC proximity to 0.5.
   - Expected cap: $2-$4. If parity fails, repair and repeat this phase before
     interpreting the sealed magnitudes or running new science.

2. **Build an independent public shadow evaluator.**
   - Use 512-1,024 public human documents from a different crawl/domain slice,
     with fingerprint/domain exclusions against all training and evaluation
     material.
   - Use an evaluation embedder from a family that will not be used as a
     training reward, fresh authorship probes, matched sample cardinality, and
     prompt-clustered uncertainty.
   - Separate corpus-level distribution references from prompt-matched human
     answers used for quality/adherence.
   - Power-test the instrument on a synthetic known gap, then score the existing
     frozen 144 outputs. Screening only; never a promotion authority.
   - Expected cap: $1-$2.

3. **Run a staged 4B SFT scale/exposure control.**
   - First screen seed 11 at 64 and 128 output tokens on: current 256 examples,
     256 examples repeated to match exposure, and 1,024 diverse examples for
     one epoch.
   - Keep model revision, full-brief renderer, optimizer, LoRA capacity,
     sampler, prompts, output counts, and evaluation fixed.
   - Unlock seeds 29 and 47 only if the 1,024-example arm beats the
     exposure-matched control on shadow semantic distance without worsening
     authorship AUC, factual control, or collapse.
   - Unlock a 4,000-example cell only after a positive 1,024-example trend.
   - Initial expected cap: $4-$8.

4. **Run teacher-forced moment matching against the matched SFT control.**
   - Test coefficient zero, low, and high at seed 11 and 64/128 tokens.
   - Match data, training tokens, steps, and LoRA capacity to SFT.
   - Training features must be independent of the shadow and sealed evaluation
     representations.
   - Confirm seeds 29 and 47 only if independent semantic distance and fresh
     authorship AUC both improve without factual/diversity regression.
   - Stop the arm if its own moment loss improves while independent metrics stay
     flat. Expected cap: $3-$6.

5. **Conditional research branches.**
   - If data scale improves both semantic and authorship metrics monotonically,
     extend to 4K before adding RL.
   - If semantic distance improves but authorship AUC plateaus, test n-gram
     residual/unlikelihood, then reward-weighted SFT.
   - If cheap objectives fail, test score-function MMD at 64/128 tokens with an
     SFT/KL anchor and independent evaluation representation.
   - Defer GAIL/GRPO, the 13-LoRA staging ablation, and long 512/1,024-token
     curricula until a cheaper objective produces a stable 128-token signal.

6. **Promotion gates.**
   - No sealed submission until merge/interface parity is proved, the public
     shadow evaluator passes its power check, and a candidate beats a matched
     current-cycle 4B SFT control across three training seeds.
   - The next sealed comparison should be matched SFT versus a genuine DFT arm,
     with endpoint-independent checkpoint selection.
   - No 14B bridge until a genuine 4B method receives sealed confirmation.
   - No GPTZero/Pangram until the frozen final candidate passes Tier 2; detector
     results remain a one-shot, human-triggered final exam.

## Interpretation rules

- Do not cite the current visible MMD delta versus human floor or standardized
  visible S as valid distribution evidence until the floor and sample-size
  mismatch are repaired.
- Keep the current sealed rejection in force for the exact checkpoint while
  treating its point magnitudes as provisional until merge/interface parity is
  verified.
- Do not describe the current result as a DFT failure or as proof that 4B is
  inadequate.

# Native Phase 1 adversarial red-team

Date: 2026-07-16  
Scope: public repository evidence only. I did not access private evaluator
source or data, secrets, provider services, hidden volumes, or paid compute.
This review is prospective: it asks what could still be wrong even if the
planned merge/interface replay and public-evaluator repair both report PASS.

## Bottom line

Phase 1 is necessary but not yet sufficient to authorize new scientific
claims. Its two audits should be treated as instrument-qualification work:

- A merge/interface replay can establish transport fidelity, but it cannot
  validate the private evaluator or rehabilitate the rejected seed-29 result.
- A repaired public evaluator can establish a better screening instrument, but
  replaying the already-seen 144 outputs cannot be confirmatory evidence.
- The proposed 256-versus-1,024 experiment does not currently identify a
  "data diversity" effect because the 1,024 examples come from a new disjoint
  slice while the 256 examples are the old slice. Corpus composition,
  synthesis batch, unique-example count, and exposure all move together.

The consolidated plan should proceed only after the P0 safeguards below are
made explicit pass/fail criteria.

## Prioritized failure modes and safeguards

### P0.1 - The evaluator owner and the research optimizer are not technically separated

**Why this can create false reassurance.** `CLAUDE.md:24-34` says the Tier-1
harness is immutable to the research agent. The local guard itself says it is
"NOT a security boundary" (`.claude/hooks/guard_bash.sh:1-5`), and the deny
rules are tool-specific (`.claude/settings.json:1-15`). More importantly, the
same repository history contains research-cycle commits that changed the
harness and then evaluated the active candidate: `8922ea4`, `1dbccdb`,
`1612614`, `184ab39`, `5155d1c`, and `3a8d23b`. This does not prove that an
agent, rather than an operator, authored each change, but the public record
does not independently attest the claimed separation.

The callable boundary is also porous. `experiments/m1/tier1_batch.py:95-100`
accepts injected `evaluate_fn` and `embedder`; lines 62-92 select bank,
calibration, and judge through environment variables. The harness itself
accepts injected embedder, probe, judge, and generator
(`harness/src/harness/cli.py:632-640`) and permits a calibration-path override
(`harness/src/harness/cli.py:547-549`). These are reasonable testing hooks,
but a report generated through them is not independently governed merely
because it has a harness-shaped JSON schema.

**Required safeguards.** Put the public evaluator in a separately owned repo
or protected branch, release it as a content-addressed container/wheel, and
require an operator signature over code image, data manifest, metric manifest,
and output. Promotion evidence must run the signed CLI entry point with all
test injections and environment overrides disabled. Require CODEOWNERS review
from someone who did not train/select the candidate. Research code may consume
the signed aggregate, but must not import evaluator internals. Record both the
research commit and evaluator image digest in every report.

### P0.2 - A same-runner adapter/merge replay can pass while the sealed interface is wrong

**Why this can create false reassurance.** The proposed replay says parity
should hold "within sampling variation" (`research_reviews/consolidated_next_actions.md:20-24`).
That is too weak and too ambiguous. With identical prompts and per-example RNG,
adapter and merged representations should first be compared before sampling.
Sampled text is a chaotic parity test: tiny logit differences can change an
early token and make whole completions diverge even when the merge is correct;
conversely, a small canary can match while a shared serializer or truncation
bug exists in both paths.

There are concrete representation differences to audit. Merge loads the base
explicitly as BF16 with `device_map={"": 0}`
(`experiments/m1/workflow.py:75-84`), whereas visible adapter inference loads
without an explicit dtype or device map (`experiments/m1/workflow.py:696-724`).
Training tokenizes the prompt with `add_special_tokens=False`
(`experiments/m1/workflow.py:351-367`), while generation tokenization does not
set that flag (`experiments/m1/workflow.py:729-735`). The fixed manifest reports
a maximum prompt of 1,017 tokens under a 1,024-token cap
(`configs/m1/manifests/m1_realdata_adherence_fixed_inputs_v1.json:7-10,23`),
so even a small special-token or tokenizer-version difference can alter the
tail of the longest prompts. The checked-in deployment sampler is still
unfrozen and null (`harness/deployment_sampler.json:1-12`), and the public
checkpoint-generation path only admits `{user_prompt}`
(`harness/src/harness/cli.py:185-227`) while the successful visible recipe uses
`dft.full-brief.v1`.

**Required safeguards.** Use a four-way public-canary matrix:

1. base plus PEFT adapter in the visible runner;
2. merged artifact in the visible runner;
3. merged artifact through the exact sealed generation container on public
   canary prompts;
4. a second, independently implemented contract checker over the emitted
   attestation.

Freeze and compare rendered prompt bytes, input IDs, attention masks,
truncation position/count, tokenizer and chat-template digests, model dtype,
library/container digests, sampler arguments, per-example RNG derivation, and
output-token IDs. Require tight teacher-forced logit/top-k tolerances and exact
greedy-token parity before looking at stochastic text. If stochastic parity is
also measured, preregister a distributional tolerance over many seeds; do not
use an undefined "within variation" judgment. The sealed service can expose
attestations for public canaries without exposing hidden prompts or metrics.
Human source inspection alone is not a parity proof.

### P0.3 - Artifact identity can change without model weights changing

**Why this can create false reassurance.** The public sealed client hashes the
entire local directory, including mutable metadata
(`harness/src/harness/cli.py:1065-1096`), while `artifact_uri` may be overridden
independently by environment (`harness/src/harness/cli.py:968-1009`). The public
record shows v2-v5 submission configs changing metadata, URI form, snapshot
generation, sampler metadata, and evaluator revision while retaining the same
source adapter hash (`configs/m2/m2_sealed_4b_seed29_merged_v2.json:14-19` and
`m2_sealed_4b_seed29_merged_v5.json:6-27`). Thus directory hashes can differ for
the same weights, and a local checkpoint hash need not by itself prove the
bytes at the remote URI. A quota system deduplicating only the submitted
16-hex directory hash could be bypassed accidentally through metadata churn.

**Required safeguards.** Define a canonical model identity as separate hashes
for weight tensors, tokenizer/chat template, and frozen generation contract.
The service must independently fetch and hash the remote artifact, compare it
to a signed manifest, and deduplicate on the canonical weight-plus-contract
tuple, not on caller-controlled metadata or path. Bind `artifact_uri` into the
signed manifest and reject client-side environment substitution. Metadata-only
repairs must not create a scientifically new checkpoint or fresh quota claim.

### P0.4 - Fixing human-floor resampling alone does not make visible MMD comparable

**Why this can create false reassurance.** The current MMD implementation
chooses bandwidths from the pooled samples on every call
(`harness/src/harness/metrics/distribution.py:62-83`). Candidate-versus-human
MMD therefore uses a candidate-dependent kernel. The human-floor function
then resamples human halves and calls the same helper separately for every
bootstrap (`harness/src/harness/metrics/distribution.py:103-122`), so the floor
and candidate can be measured under different kernels even after replacement
sampling is removed. Subtracting them is not a clean common-scale delta.
Candidate and historical baseline cardinalities also differ, as already noted
in `native_postmortem_next_cycle.md:55-65`.

**Required safeguards.** Freeze the kernel and bandwidth from an operator-owned
human calibration split only, then use that exact kernel for human-human,
baseline-human, and candidate-human comparisons. Match generated and human
cardinality, length policy, and prompt/domain strata. Estimate paired or
permutation uncertainty at the prompt-cluster level. Demonstrate calibrated
null behavior over many disjoint human-human splits, not merely one synthetic
known gap. Keep raw MMD-squared and confidence intervals primary; a negative
unbiased point estimate is not an ordinal claim that text is "more human."
Any repaired score is a new evaluator version and cannot be numerically mixed
with the old standardized `S`.

### P0.5 - A replay on the existing 144 outputs is post-selection diagnosis, not validation

**Why this can create false reassurance.** The candidate outputs motivated the
floor audit, the repetition correction, checkpoint selection, and the new
shadow-evaluator design. The 144 rows are also only 16 prompt clusters repeated
across three training and three sampling seeds, not 144 independent prompts
(`experiments/m1/tier1/adherence_4b_three_seed_summary_v2.json:22-32`). A repaired
instrument can look sensible on the artifact that caused its redesign and
still fail prospectively.

**Required safeguards.** Freeze the evaluator, null controls, perturbation
suite, thresholds, and analysis code before unblinding candidate labels. Use
the old 144 solely as a labeled regression/diagnostic fixture. Qualify the
instrument on: disjoint human-human nulls; duplicated/template-collapse text;
topic-shifted humans; length-matched synthetic prose; outline-copying; and
several graded corruption strengths. Reserve a new prompt set and fresh model
outputs for the first prospective estimate. A Phase 1 public-evaluator PASS
must not change the existing sealed rejection or promote the old artifact.

### P1.1 - The staged plan still permits adaptive multiple-comparison bias

**Why this can create false reassurance.** The cycle considers three data
cells, two output lengths, training seeds, sampling seeds, several metrics,
then three moment-loss coefficients and later several objectives. The plan
screens on seed 11, conditionally adds a second sampling seed, and later counts
seeds 11/29/47 together (`native_postmortem_next_cycle.md:145-160`). This reuses
the discovery seed as confirmation and creates informative missingness if only
passing cells receive more samples. The consolidated version also weakens the
native plan's pooled-CI condition to merely "beats" on semantic distance
(`consolidated_next_actions.md:45-54`). Selecting a coefficient/length/seed and
then reporting an ordinary CI on the same holdout does not undo selection.

**Required safeguards.** Preregister one primary contrast, one primary length,
one effect direction/minimum effect, the full metric hierarchy, and a group-
sequential alpha/error-spending rule. Treat seed 11 as discovery only; confirm
on fresh training seeds or clearly label the final estimate as two-seed
confirmation. Run every sampling seed for every cell admitted to a stage; if a
cell stops early, count it as a failure rather than omitting it from averages.
Use one bank for arm/coefficient selection and a different locked bank for the
confirmatory CI. Report all attempted cells and family-wise or false-discovery
control; do not select on AUC proximity to 0.5 or on the best of 64/128 tokens.

### P1.2 - The proposed data-scale contrast does not yet identify diversity versus exposure

**Why this can create false reassurance.** The plan proposes the current 256
examples, the current 256 repeated, and 1,024 new disjoint examples
(`consolidated_next_actions.md:45-54`; `native_postmortem_next_cycle.md:145-153`).
But the new corpus is to come from a different public slice and a later brief
synthesis batch. The main contrast therefore changes source/time/domain mix,
brief-generation batch, document-length distribution, unique examples, and
possibly token exposure. Four epochs over 256 examples only match 1,024
example presentations; it need not match loss-bearing completion tokens or
gradient weighting. The current trainer masks prompt tokens and counts only
completion labels (`experiments/m1/workflow.py:355-370`), uses batch size one,
and takes one optimizer update per document (`workflow.py:520-540`). Variable
completion lengths make "same documents seen" and "same labeled tokens" two
different exposure estimands.

**Required safeguards.** Build one frozen 1,024-document superpopulation under
one source/synthesis contract. Draw preregistered, domain/length/target-stratified
256 subsets from that superpopulation; ideally use multiple independent 256
subsets to estimate subset-composition variance. Compare a 256 subset repeated
against the 1,024 superpopulation at matched optimizer updates, labeled-token
budget, batch/packing policy, learning-rate schedule, LoRA capacity, and
example-order policy. Keep the historical 256 corpus as an external replication
cell, not the sole causal control. Report unique documents, document
presentations, prompt tokens, labeled completion tokens after truncation,
optimizer updates, and per-stratum coverage. A diversity claim requires the
1,024 arm to beat repeated-256 across several preregistered subset draws, not
just one favorable 256 composition.

### P1.3 - Output length can masquerade as both diversity and adherence

**Why this can create false reassurance.** Full briefs request target length in
words (`experiments/m1/workflow.py:273-283`), while the proposed screens cap
generation at 64 or 128 tokens. Current source targets can exceed those caps.
Short truncation mechanically changes outline recall, sentence/paragraph
structure, self-BLEU, and probe features. A 1,024-data model that happens to be
more concise could appear more factual or less separable without a distribution
improvement.

**Required safeguards.** Either select prompts whose preregistered target
length fits each cap, or define a common prefix estimand and separate completion
rate/EOS/length-adherence gates. Match realized length distributions before
interpreting semantic, structural, lexical, or AUC differences, while also
reporting the unmatched primary analysis. Predeclare 128 tokens as primary and
64 as safety/early-screen evidence, or correct for testing both.

### P1.4 - The planned shadow evaluator can misstate uncertainty and authorship separation

**Why this can create false reassurance.** The current authorship probe uses
ordinary stratified document folds and an item-level bootstrap
(`harness/src/harness/metrics/quality.py:125-145,148-189`). If outputs from the
same prompt appear across model/sampling seeds, related prompt-specific lexical
signals can leak across folds, and item bootstraps treat correlated outputs as
independent. A 512-1,024-human bank does not fix this when a candidate cell has
far fewer independent prompt clusters. Nor does a different crawl slice by
itself define a fair corpus target; topic/time mismatch may dominate the model
effect.

**Required safeguards.** Split and bootstrap by prompt/source/domain cluster,
never by row. Keep reports cell-level and use a hierarchical model/bootstrap
for training-seed, sampling-seed, and prompt variation. Use matched cardinality
or an explicitly calibrated unequal-sample estimator. Freeze the target-domain
mixture. Maintain separate banks for corpus distribution, prompt-matched human
quality, and instrument calibration, as the consolidated plan proposes; prove
their fingerprint/domain exclusions in both directions. Report effective
cluster count, not just row count.

### P1.5 - The visible validity pass can still be a copying pass

**Why this can create false reassurance.** The unsupported-claim metric is
token overlap, not entailment (`harness/src/harness/metrics/validity.py:47-80`),
and the current full brief exposes supported facts verbatim
(`experiments/m1/workflow.py:281-282`). The very low unsupported rate and high
recall can therefore be achieved by close outline copying. The perfect
secondary preference scores are already known to use prompt-unmatched humans
(`native_postmortem_next_cycle.md:67-79`).

**Required safeguards.** Keep the frozen overlap proxy only for continuity.
Before any promotion, add an independently owned, locked audit that separates
entailed paraphrase, verbatim copying, contradiction, unsupported addition,
and empty-outline behavior. Predefine a copy-rate/source-span ceiling and test
the evaluator on adversarial copy-only fixtures. Use prompt-matched human
answers for preference; never let the distribution bank double as the quality
answer.

## Prospective pass criteria for the two Phase 1 audits

### Audit A: merge and generation-interface fidelity

PASS only if all of the following are public and hash-bound:

1. Adapter and merged weights are tied to a canonical tensor digest, not a
   metadata-sensitive directory hash.
2. Fixed canaries match on rendered prompt bytes, input IDs, masks, truncation,
   tokenizer/chat template, and greedy token IDs; logit tolerances are
   preregistered and met.
3. The exact sealed generation image produces the same public-canary
   attestation as the visible image, including per-example RNG and batching.
4. Remote artifact bytes are independently hashed and match the signed local
   manifest.
5. Any mismatch fails the audit. "Close-looking prose" or same aggregate
   metrics is not parity.

Even a PASS only says the artifact was transported faithfully. It does not
say the sealed metric implementation or hidden sample is correct.

### Audit B: repaired public evaluator

PASS only if all of the following were frozen before candidate replay:

1. Independent ownership and a signed evaluator/data/analysis release.
2. Common human-calibrated MMD kernel, matched cardinality, and cluster-aware
   intervals.
3. Human-human type-I/null calibration plus graded sensitivity, topic-shift,
   copy, collapse, and length controls.
4. Grouped authorship cross-validation and cluster bootstrap.
5. Separate distribution and prompt-matched quality banks.
6. A new prospective holdout remains untouched after the legacy 144-output
   diagnostic replay.

Even a PASS qualifies the evaluator for future screening; it does not turn a
post-hoc rescore of seed 29 into promotion evidence.

## Minimum defensible data-scale design

The smallest design that can answer the stated question is:

1. Freeze one 1,024-example superpopulation and at least two preregistered,
   stratified 256-example subsets from it, all synthesized under one contract.
2. At seed 11, compare each repeated-256 subset with 1,024 at equal optimizer
   updates and labeled-token exposure. Keep 256-once as an exposure diagnostic.
3. Use a new locked prompt holdout. Run the same primary 128-token sampling
   seeds for every arm; treat 64 tokens as an early safety stage, not a second
   chance to win.
4. Estimate the 1,024-minus-repeated-256 effect with prompt-clustered paired
   intervals and subset-composition variance. Require a preregistered minimum
   effect on the independent semantic endpoint, no worse AUC distance from
   0.5, and absolute validity/collapse/length gates.
5. If seed 11 passes, confirm on fresh training seeds without using the screen
   bank or retuning thresholds. Do not count conditional missing cells as
   successes.

If only one old 256 corpus is compared with a new disjoint 1,024 corpus, label
the result honestly as a **corpus-plus-scale package comparison**, not evidence
that diversity rather than exposure caused the change.

## Reviewer-of-reviewers verdict

The consolidated plan correctly narrows the sealed conclusion, keeps 14B and
Tier 3 closed, calls for an independent shadow instrument, and introduces an
exposure-matched control. Those are real improvements. The remaining danger is
procedural: each repair can be implemented and judged by the same loop that
benefits from a PASS, and each old artifact can be repeatedly rescored until an
instrument looks reassuring.

Accordingly, neither Phase 1 audit should self-certify. Each needs an owner
separate from candidate training, a preregistered falsification suite, signed
artifacts, and a second adversarial review that sees the audit specification
before results. Phase 1 may clear the instruments for prospective use; it may
not erase the sealed rejection, select a new winner from the legacy outputs,
or authorize scale-up by itself.

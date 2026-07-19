# M3 4K scientific-screen method contract

Status: frozen before 4K corpus construction, training, candidate generation,
or evaluation. This instantiates, but does not weaken, the parent
`m3_rewriting_14b_preregistration_2026-07-19.md`.

## Purpose

The screen asks one question: does `HUMANWRITE14` rewrite AI-styled prose more
humanly than a matched `SFT14` control without losing facts? The 128-example
mechanical smoke is not evidence on that question.

## Frozen 4,096-record task mixture

Use the first 4,096 identities of the prefix-stable M3 assignment:

- 1,843 multi-provider AI rewrites;
- 819 pinned base-Qwen3-14B drafts reconstructed from the visible source brief;
- 205 controlled light-edit rewrites;
- 205 already-human restraint/no-op rewrites;
- 1,024 structured generation tasks, exactly 256 with empty outlines.

Multi-provider and light-edit candidates use Claude Haiku 4.5 and Gemini 3.1
Flash Lite as source generators, balanced by fingerprint and template before
outcomes are observed. Qwen3-32B independently verifies every API-generated
pair. Base-model drafts come from the pinned local Qwen3-14B revision and are
verified by Qwen3-32B. No-op inputs equal their human targets by construction
and request minimal editing. Every non-noop pair must pass the parent factual,
literal, language, length, and non-identity gates.

The accepted corpus is immutable and SHA-bound before either arm starts. Failed
construction attempts may be retried, but their provenance remains recorded.

## Matched arms

Both arms start independently from the same pinned Qwen3-14B base and use:

- bfloat16 base weights;
- rank-32 LoRA on `q/k/v/o/gate/up/down` projections, alpha 64, dropout 0;
- two stages of 2,048 examples each, one total target exposure per record;
- a prospectively seeded permutation, split at the 2,048-example boundary;
- microbatch 2, gradient accumulation 4, effective batch 8;
- AdamW, learning rate `2e-5`, zero weight decay, gradient clipping at 1.0;
- maximum prompt/completion/sequence lengths 640/383/1,024 tokens;
- merge the stage-1 LoRA into the base, initialize a fresh rank-32 LoRA, reset
  optimizer state, and run stage 2;
- checkpoint every 128 optimizer steps and at both stage boundaries.

`SFT14` uses uniform teacher-forced cross-entropy in both stages.

`HUMANWRITE14` uses the same cross-entropy plus the following frozen additions.
The 4K screen prospectively omits paired preference correction, as allowed by
the parent protocol, because no frozen SFT14 rejected-output corpus exists
before this screen.

### Frequent-token moment correction

Select the 256 most frequent non-special Qwen3-14B tokens from training targets
only. Over 32 fixed training-only calibration microbatches, choose one loss
coefficient from `{0.01, 0.03, 0.1, 0.3, 1.0}` whose median unclipped moment-
gradient norm is closest to 20% of the median cross-entropy gradient norm,
breaking ties toward the smaller coefficient. Freeze that coefficient before
the first treatment optimizer step.

For selected token `v`, match both the mean teacher-forced probability and mean
squared probability to the empirical human one-hot moments. Weight token terms
by inverse square root human frequency, normalized to mean one. Log component
losses, unclipped gradient norms, cosine, and clipping separately.

### Stage-2 witness reweighting

At the treatment stage-1 boundary, generate one rewrite from each item in a
fixed 512-item training-only subset. No evaluation input is used. Build paired
features from:

1. the target-minus-source and policy-output-minus-source embedding residuals
   under pinned `BAAI/bge-small-en-v1.5`; and
2. twelve preregistered surface features: token count, sentence count,
   paragraph count, mean and standard deviation of sentence length,
   type-token ratio, comma, semicolon, colon, parenthesis, em-dash, and newline
   rates.

Standardize dimensions using human-target training statistics. Let `gap` be
the mean human residual minus mean policy residual. Each human target receives
the linear witness score `dot(human_residual, gap)`. Convert standardized scores
to weights `exp(0.5*z)`, clip to `[0.5, 2.0]`, and renormalize to mean one.
Stage 2 samples without replacement from the frozen second half, and multiplies
each example's cross-entropy and moment term by its witness weight. This combines
topic-residualized embedding and surface evidence; semantic MMD alone never sets
weights.

## Fresh 256-item evaluation panel

Freeze a new source/domain-disjoint panel before generating any candidate:

- 128 naturally AI-styled passages from at least two model families;
- 48 fact-dense passages;
- 48 explicit light-edit or register requests;
- 32 already-human restraint/no-op cases.

Compare base prompting, `SFT14`, and `HUMANWRITE14` using identical visible
instructions and sampling. Primary SFT14-vs-HUMANWRITE14 comparisons are blinded
and order-randomized by a recorded seed. Human-style and overall-quality votes
use two non-Qwen judge families; an item score is the mean of the two binary
votes, with unparseable responses retried once and then recorded as missing.
Content preservation is evaluated separately and may veto a style win.

Meaningful edit means normalized character similarity below 0.98 and at least
one changed non-whitespace token, while passing content preservation. Restraint
means a no-op output preserves content and normalized character similarity is
at least 0.90. All automatic lexical and embedding metrics use the full frozen
panel; no opened M2 panel is reused.

## Exact 4K decision

Advance only if every parent 4K-to-16K gate passes. Report point estimates and
cluster-bootstrap 95% intervals, but do not retrofit a significance requirement
that the parent screen did not preregister. Any failed gate ends this method at
4K. Infrastructure failures may be repaired without changing identities,
coefficients, exposure, judges, or thresholds.

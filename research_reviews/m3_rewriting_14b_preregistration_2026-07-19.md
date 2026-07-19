# M3 rewriting-first Qwen3-14B preregistration

Status: frozen before M3 data construction, training, candidate generation, or evaluation.

## Decision objective

Determine whether a Qwen3-14B model can rewrite AI-styled English prose into
materially more human-like prose than a matched rewrite-SFT control while
preserving the source's facts, intent, and usable quality. This is a product
test, not a claim to reproduce Rosmine's undisclosed DFT algorithm.

The total combined Modal and provider-API spend may not exceed USD 200 under
the existing USD 100 GPU and USD 100 API gateway caps. Historical spend counts
against those caps. No cap may be raised automatically.

## Model

- Base: `Qwen/Qwen3-14B` at one immutable revision resolved before the first
  accelerator smoke.
- Runtime: bfloat16 base weights, LoRA adapters, gradient checkpointing, one
  H100 per active arm, and Qwen non-thinking chat formatting.
- No 4-bit training is introduced unless a pre-result memory smoke proves that
  bfloat16 cannot fit one H100. Such a change requires a new contract and a
  matched control; it may not be made in the middle of an arm.
- The initial 14B checkpoint is shared byte-for-byte by all matched arms.

## Corpus sizes and automatic gates

- Mechanical smoke: 128 examples.
- First scientific screen: 4,096 examples.
- Decisive screen: 16,384 examples.
- Quarter-scale confirmation: 46,080 examples.

The 4K and 16K artifacts are prefixes of one frozen 46,080-record training
corpus. A larger stage may start only after the preceding stage passes its
frozen gate. A failed gate stops the program without consuming the remaining
budget.

## Training-task mixture

The 46,080 records are grouped by source document before splitting. No source
document, near duplicate, or domain assigned to evaluation may appear in
training.

### Rewrite tasks: 75%

Each target is a cleaned human-written document. Each input is a content-
aligned transformation of that target, with one of four prospectively tagged
origins:

1. Multi-provider AI paraphrase, 45% of all records. At least two model families
   and multiple prompt templates must contribute; no provider/template may
   exceed half of this stratum.
2. Baseline-model draft reconstructed from a faithful source-derived brief,
   20% of all records.
3. Controlled light-edit corruption, 5% of all records. Transformations may
   alter discourse structure and common formulaic phrasing but may not insert
   facts or operate from a hand-authored detector phrase blacklist.
4. Already-human/no-op examples, 5% of all records. The source is the human
   target and the instruction requests restraint. These prevent obligatory
   paraphrasing and measure whether the model damages good prose.

Rewrite instructions are varied prospectively: general naturalness, audience
or register adjustment, concision, clarity, and minimal editing. The input
schema exposes the source text, optional user instruction, requested style,
target token range, and explicit preservation requirements for names, numbers,
and quotations. It never exposes the hidden human target or target-derived
phrases outside the visible source.

### Generate tasks: 25%

Generation records retain the disclosed structured schema: prompt, use case,
style, target length in tokenizer tokens, em-dash permission, and outline.
Exactly 25% of generation outlines are empty. These records preserve useful
from-scratch behavior without dominating the rewriting objective.

## Task-construction acceptance rules

A rewrite pair is accepted only when all applicable checks pass:

- UTF-8 round trip with no replacement character.
- English targets and English synthesized inputs; quoted foreign text may be
  explicitly exempted by the record.
- Every numeral, currency amount, date, URL, email address, and exact quotation
  in the target is either present in the input or recorded as intentionally
  absent. Records with unexplained factual additions are rejected.
- Named-entity and atomic-fact preservation are checked by deterministic
  extraction plus a provider verifier that is different from the source
  synthesizer whenever possible.
- Source/target semantic similarity clears a threshold calibrated on held-out
  accepted and rejected fixtures before bulk construction.
- Source length is within `[0.70, 1.35]` of target tokenizer length unless the
  instruction explicitly requests length change.
- Source and target may not be byte-identical except in the no-op stratum.
- Navigation, cookie text, contact blocks, missing-image captions, malformed
  fragments, and document truncation are rejected.

Rejected records are never silently repaired into the accepted corpus. A
repair creates a new attempt with its own provider/template provenance.

## Training arms

### SFT14 control

Uniform teacher-forced cross-entropy over the frozen task mixture.

### HUMANWRITE14 treatment

The same examples, order family, optimizer-step budget, and LoRA stage count as
SFT14, with three additions:

1. Paired preference correction using the human target as chosen and a frozen
   SFT14 rewrite as rejected, only for records whose rejected output passes the
   same content-preservation gate.
2. Teacher-forced frequent-token first/second-moment correction against the
   human-target corpus. Token sets and coefficients are frozen from training
   data only.
3. Stagewise witness reweighting of human targets undercovered by the current
   policy. Witness features combine topic-residualized embedding features and
   preregistered surface statistics; semantic embedding MMD alone may not set
   weights.

Both arms use the same LoRA merge/reset boundaries. The treatment may not
receive additional target tokens. Component gradients, clipping, loss scales,
and pair eligibility are logged separately.

The 4K screen may omit preference correction if insufficient frozen SFT14
outputs exist; in that case it compares uniform SFT with moment-plus-witness
training and the omission is recorded before training. Preference correction
must be present and frozen before the 16K decisive screen.

## Evaluation panel

Freeze 256 rewrite inputs before candidate generation, disjoint by source and
domain from all training data:

- 128 naturally AI-styled passages from multiple model families.
- 48 fact-dense passages containing names, dates, numbers, or quotations.
- 48 explicit light-edit or audience/register requests.
- 32 already-human restraint/no-op cases.

Evaluation compares base Qwen3-14B prompting, SFT14, and HUMANWRITE14. Deft may
be included as an external context benchmark if access is available, but it
cannot select checkpoints or tune prompts.

Primary endpoints are HUMANWRITE14 versus SFT14:

1. Blinded human-style preference.
2. Blinded overall-quality preference.
3. Atomic-fact, name, number, date, and quotation preservation.
4. Meaningful-edit rate on AI-styled inputs and restraint on no-op inputs.

Secondary endpoints are token 1/2/3-gram distance, two independent embedding
MMD families, repetition, length adherence, unexpected-language rate, and
malformed-character rate. Detector scores are excluded from training and model
selection.

### Frozen blinded-judge execution

Human-style and overall-quality comparisons use both
`anthropic/claude-haiku-4.5` and `google/gemini-3.1-flash-lite` through
OpenRouter. Each family judges all 256 HUMANWRITE14-versus-SFT14 pairs on both
dimensions. Candidate order is independently fixed by
`sha256(8903:model:dimension:fingerprint)` parity. The response contract is
exactly `A`, `B`, or `TIE`; ties count as one half only in the descriptive
preference rate and are also reported separately. The rubrics and execution
contract are implemented in `data/m3_rewrite_judge.py` before the panel or
candidate outputs are available. The two model-family results remain separate;
they may not be pooled to conceal disagreement.

## Frozen promotion gates

### 4K to 16K

- At least 70% of AI-styled inputs receive a meaningful edit.
- HUMANWRITE14 point estimates are at least 55% for human style and 50% for
  overall quality versus SFT14.
- Content preservation is no worse than SFT14 by more than 3 percentage points.
- No replacement characters and no increase above 2 percentage points in
  unexpected-language rate.
- At least one lexical distribution metric improves and none worsens by more
  than one human-split standard deviation.

### 16K to 46K

- Human-style win rate at least 60% with a one-sided paired/randomized
  `p <= 0.05`.
- Overall-quality win rate at least 55%, with no significant evidence of
  treatment inferiority.
- Content preservation non-inferior within 2 percentage points overall and no
  hard regression for names, numbers, dates, or quotations.
- At least 75% meaningful edits on AI-styled inputs and at least 80% acceptable
  restraint on no-op inputs.
- Zero replacement characters; treatment unexpected-language rate no higher
  than SFT14 by more than 2 percentage points.
- Token unigram distance does not worsen and both embedding-family MMD effects
  are directionally favorable.

### Final success at 46K

The 16K criteria must reproduce on a fresh 256-item confirmation panel. The
final report must include representative randomly selected outputs, all
failures, cost, exact artifact hashes, and uncertainty intervals. One optional
human-triggered detector evaluation may follow only after the artifact is
immutable.

## Stop conditions

- Combined gateway commitments reach USD 200 or either existing provider cap
  is exhausted.
- Non-finite loss/gradients, repeated checkpoint corruption, or irreproducible
  resume.
- More than 10% malformed/unexpected-language training rollouts.
- A scientific gate fails.
- The treatment improves style only by dropping or altering facts.

Infrastructure failures may be repaired and retried without weakening any
scientific gate. All retries preserve the same input identities and contract.

## Autonomous continuation contract

After this protocol and its implementation tests pass, the event-driven
coordinator may validate terminal artifacts, repair recoverable infrastructure
failures, launch the next already-authorized stage, and stop at any frozen
gate. It must not raise budgets, open Tier 3 detectors, reuse an opened panel
for method changes, or waive a failed gate. Routine stage transitions do not
require user sign-off.

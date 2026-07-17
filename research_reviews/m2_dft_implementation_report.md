# M2 score-function MMD implementation report

Date: 2026-07-17 (America/Los_Angeles)

## Status

The first prospective 4B score-function MMD training path is implemented but
has not been launched. The public runner now recognizes
`dftr.m2.score_function_mmd.v1` and trains two fresh PEFT adapters from one
hash-bound Qwen3-4B SFT LoRA source:

- `A0`, with the MMD coefficient exactly zero; and
- `A64`, with the preregistered nonzero coefficient and exactly 64 generated
  tokens per rollout.

Both arms reload the same source adapter and use the same seeded rollout and
anchor batches, rollout seed schedule, optimizer, KL reference, CE anchor,
runtime contract, and generated-token accounting. Outputs remain adapter
native; the implementation does not merge adapters or import harness or
measurement-v2 code.

Sampling is a manual 64-step raw policy categorical loop with an
autoregressive cache. It does not use `generate`, temperature, top-k/top-p,
EOS stopping, minimum-length processors, or model generation defaults. The
same raw policy distribution is therefore used for rollout and score-function
log probabilities.

## Implemented contract

`experiments/m2/dft.py` provides a strict, exact-key config schema and binds
the scientific method with a canonical SHA-256 over the run, compute, model,
source adapter, data, representation, kernel, runtime, optimizer, arms, and
stop rules. Runtime validation fails closed on package versions. Input
validation fails closed on the complete source-adapter file map, adapter
weight/config hashes, all training JSONL hashes, and a human-only bandwidth
artifact bound to the human-target hash, frozen representation, and tokenizer
artifact identity.

Rollout and anchor records are serialized with the exact
`dft.full-brief.v1` field set and formatting used by the source SFT path. The
schema and serializer hash are part of the method contract, so a reduced
single-field prompt cannot silently substitute for full-brief conditioning.
The gateway also recognizes `train_dft` as an evidentiary job, requires an
immutable revision, and rejects protocol or task-kind substitutions before
reservation.

The training representation is the frozen base model's final hidden state,
attention-mask mean pooled and normalized in bounded batches. It is explicitly
training-only. One Qwen3-4B base is resident per arm: the trainable policy and
frozen reference are separate PEFT adapters on that base, and reward
embeddings are computed with adapters disabled. The score reward is the conditional gradient contribution of
the unbiased generated/generated and generated/human MMD terms. Its factor of
two on generated pairs accounts for either sampled document carrying the
shared policy score. Each sample's control is computed from the generated
panel with that sample removed, so the control cannot depend on the held-out
sample. The objective adds sampled KL to a separately loaded frozen starting
adapter and a teacher-forced SFT anchor.

Hash-bound intermediate checkpoints include the policy adapter, optimizer
state, CPU/CUDA/Python RNG state, prior logs, step, and exact token accounting.
Resume reloads the policy checkpoint while always loading the frozen reference
from the original SFT source and rejects method/source/state mismatches.

Per-step logs include reward components, leave-one-out controls, advantages,
MMD/KL/SFT/total losses, gradient norm, scalar score-estimator variance,
rollout and anchor indices, rollout seed, generated tokens, effective rollout
count, unique fraction, and trigram repetition. Stops cover non-finite
loss/gradients, KL drift, duplicate collapse, trigram repetition, training-only
outline fact recall, unsupported-claim rate, and target-length adherence. The
scalar score variance is diagnostic only and cannot stop an arm. Final manifests bind config,
method, Git revision, source adapter, complete non-manifest adapter payload
maps, and exact matched exposure.

## Verification

The focused tests cover:

- strict A0/A64 config and method-hash enforcement;
- source adapter, data, runtime, and human-only bandwidth artifact binding;
- vectorized reward equality with explicit kernel loops;
- invariance of sample `i`'s control when only sample `i` changes;
- finite score gradients and exact zero-coefficient A0 equivalence;
- deterministic without-replacement batches and repetition accounting;
- exact source-SFT full-brief serialization and rejection of warped sampling;
- fixed-length raw categorical sampling that continues through EOS tokens;
- left-padding-aware continuation log probabilities;
- an enumerated MMD policy-gradient oracle checked against the exact expected
  objective and a finite difference;
- optimizer plus Python/CPU RNG interruption/resume equality; and
- training-only factuality/adherence sentinel emission;
- runner dispatch and protocol-substitution rejection; and
- absence of harness or measurement-v2 imports.

Verification command:

```text
python -m pytest -q <non-independent experiment tests> infra/tests/test_policy.py infra/tests/test_local_backend.py
140 passed in 2.95s
```

No model weights were loaded and no GPU/API job was launched.

## Remaining pre-launch caveats

1. No materialized prospective config is checked in because the real seed-11
   source-adapter file map, training-human bandwidth artifact, worker package
   versions, and chosen nonzero coefficient are not yet frozen. Inventing
   placeholders would weaken the fail-closed boundary.
2. The 4B path has not had the required small-model plumbing smoke. GPU memory,
   deterministic CUDA compatibility, and the installed Transformers/PEFT
   execution path therefore remain unverified. The smoke must also compare
   cached incremental-sampling logits with full teacher-forced logits closely
   enough for the frozen bfloat16 runtime.
3. Exact resume is implemented and covered with an interrupted optimizer/RNG
   equality test, but it has not yet been exercised with a real PEFT adapter on
   CUDA. That remains part of the required plumbing smoke.
4. Logged scalar score-contribution variance is diagnostic only, not a
   parameter-gradient variance estimate or a hard stop.
5. The factuality/adherence sentinels are training-only lexical proxies. Their
   thresholds must be frozen prospectively and they cannot replace public
   evaluation gates.
6. Adapter-native A0/A64 evaluation, power qualification, and any deployment
   decision remain future prospective work. This implementation produces no
   scientific result by itself.

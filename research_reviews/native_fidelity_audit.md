# Phase 1 artifact and interface fidelity audit

Date: 2026-07-16

Scope: repository and workspace-visible, non-hidden artifacts only. This audit did not access the private evaluator repository, hidden prompts/data, secrets, or provider services; it did not launch GPU compute.

## Bottom line

The public lineage identifies the correct seed-29 adapter and the merge source unambiguously. A workspace-visible materialization of the final merged directory reproduces the sealed checkpoint hash `0f437f62bc1cca0c`, and its three weight-shard hashes match the original merge manifest. That is strong byte-provenance evidence.

However, no post-merge visible replay exists. The repository therefore has not demonstrated that the merged model produces the same outputs as the adapter on the 16 full-brief inputs used for Tier 1. It also lacks a single canonical generation contract covering full-brief serialization, tokenizer behavior, dtype, batching, and RNG policy. The sealed result remains valid for the exact merged checkpoint, but current public evidence cannot prove that it is behaviorally interchangeable with the adapter whose visible outputs were scored.

## Verified provenance

### Adapter

- Training run: `dftr-1784216516-91130dd3`.
- Training config: `configs/m1/m1_realdata_adherence_sft_qwen3_4b_three_seed_v1.yaml`, canonical config hash `c2136c45512abd48eb8fe31b388ddb18e7706f241d59ab1524574f174dbe576f`.
- Base: `Qwen/Qwen3-4B` at revision `1cfa9a7208912126459214e8b04321603b3df60c`.
- Seed: `29`; 256 train records; 46,732 completion tokens; loss `0.7009145491`.
- Adapter path: `/checkpoints/runs/dftr-1784216516-91130dd3/seed-29`.
- Adapter SHA-256: `a34c14230f4847001a3a0c4362a3bc26b3a43c1d0ef049e12a7a0d029aacea91`.
- The SHA in the training provenance, merge config, merged manifest, sealed configs, and final aggregate record agrees. A direct hash of the workspace-visible adapter file also agrees.

### Merge

- Merge run: `dftr-1784224462-c1f83ed3`.
- Merge config: `configs/m2/m2_merge_4b_seed29_v1.yaml`, canonical config hash `37e4360b88b511616024381c4123ffd3ea28e62ccac684a4330a60225d7cba0c`.
- Launch commit: `7b5c8a7a00b1e236028cbbcc19be5cb41e6a1c3b`.
- The merge code verifies the adapter hash, loads the immutable base in `torch.bfloat16`, calls PEFT `merge_and_unload(safe_merge=True)`, saves safetensors shards, and copies the tokenizer.
- Recorded cost: 40.173 L40S-seconds, `$0.026128`.
- Workspace-visible merged shard hashes exactly match the merge manifest:
  - `model-00001-of-00003.safetensors`: `a71dbb017fabaca073df4f9a30f32dd005be7bc3cbc6fbf29df915ef74dc634a`
  - `model-00002-of-00003.safetensors`: `cc16bff3aa61d9c6dc3fa986b4ae9585c8de8955923c461f3f221fa406b07933`
  - `model-00003-of-00003.safetensors`: `9fde8fbbba787c3f92981351da3cf6be74b01ec6f91fb538f5ac0c61fc95f177`
  - index: `5b36dbd79cdd00f76600e471c61157c10090952160d49c5a3d7e05840bda029a`
- Computing the public harness content hash over the materialized merged directory returns `0f437f62bc1cca0c`, exactly the checkpoint hash in `experiments/m2/sealed_4b_seed29_v1.json`.

### Qualification

The workspace materialization lives under ignored `.swarmy/` storage, not Git. Its content hash ties it to the final scored directory, but future reproducibility should export a small, committed, non-weight manifest for the exact final artifact. The original merge manifest's `train_config.json` hash is stale because operator-owned submission metadata was later rewritten; the model shards did not change. This is a metadata-lineage defect, not evidence of a weight change.

## No post-merge replay exists

The only visible seed-29 generations are adapter outputs from run `dftr-1784217143-f01137b4`: 16 inputs under sampling seeds `[101,202,303]`, for 48 outputs. Searches of experiment records, configs, materialized sample trees, and the ledger find no sample JSONL, report, or replay bound to merge run `dftr-1784224462-c1f83ed3` or to the merged checkpoint hash.

The merge preregistration hypothesized functional equivalence, but the merge job only emitted weights and a manifest. Functional equivalence was never tested.

## What the public generation path actually specifies

### Full-brief serializer

`experiments/m1/workflow.py::_render_full_brief` is the canonical visible serializer. It emits exactly eight newline-separated lines:

1. `Writing request: {user_prompt}`
2. `Use case: {use_case}`
3. `Style category: {style_kind}`
4. `Style: {style}`
5. `Detail mode: {detail_mode}`
6. `Target length: about {target_length} words`
7. `Em dashes allowed: yes|no`
8. `Grounding outline ...: {canonical compact sorted JSON}`

The outer prompt is exactly `USER:\n{brief}\nASSISTANT:`. The code does not call `tokenizer.apply_chat_template`; the saved `chat_template.jinja` is unused by this visible path. There is a schema inconsistency: `data/PIPELINE.md` calls `target_length` tokens, while the serializer labels it words.

The exact repaired 64-row dev file is not checked into Git, but the 16 replay inputs are reconstructable from visible material: join the archived seed-29 sample rows to `.swarmy/operator-materialized/m1-realdata-pilot-v1/dev_briefs.jsonl` by `fineweb_id`, replace only `user_prompt` with the archived repaired prompt, and retain every other field. All 16 join; all 16 outlines and reference completions match; all 16 user prompts differ as expected. An authorized replay should instead load the hash-bound canonical file at `/checkpoints/data/m1-realdata-adherence-v1/dev_briefs.jsonl` and verify SHA `78014ab7dd9e3ae1b96dac196001ab5ecbb65e534e90b7c8e9a971edb1acd4c2`.

### Tokenizer and generation

- Adapter and merged artifacts have byte-identical `tokenizer.json`, `chat_template.jinja`, merges, vocabulary, added-token, and special-token files.
- Their `tokenizer_config.json` files differ: the merged copy adds `max_length=384`, `stride=0`, right truncation, and `longest_first`. The replay must hash and report both configs and pass explicit tokenization arguments rather than rely on these defaults.
- Training tokenizes prompt and completion separately with `add_special_tokens=False`, then appends EOS. Visible generation calls the tokenizer without an explicit `add_special_tokens`, so the library default applies. This train/generation asymmetry is not necessarily wrong, but it must be frozen explicitly.
- Visible limits: `max_input_tokens=1024`, `max_new_tokens=384`.
- Visible decoding: `do_sample=true`, `temperature=1.0`, `top_p=1.0`, left padding, pad token set to EOS if missing.
- Visible batching: one prompt at a time.
- Visible seed policy: one `torch.manual_seed(sampling_seed)` and `torch.cuda.manual_seed_all` call before the 16-prompt sequential loop; deterministic algorithms enabled. Sampling seeds are exactly `[101,202,303]`.

### Sealed-interface differences

- `harness/deployment_sampler.json` is still `frozen=false` with null fields and only accepts a legacy `{user_prompt}` format. It cannot represent or reproduce the successful full-brief path.
- `configs/m2/m2_sealed_4b_seed29_merged_v5.json` records decoding values and `max_new_tokens=384`, but omits prompt serializer/hash, tokenizer hashes, `max_input_tokens`, batch size, sampling-seed value, seed derivation, dtype, padding, and special-token policy.
- The aggregate audit says sealed generation was batched in eights and seeded deterministically. The public visible workflow was sequential, and the exact sealed seed policy is not public. Batched stochastic generation can change outputs even with identical weights and a nominally identical seed.

Therefore the historical sealed interface cannot be reproduced exactly from the public config alone.

## Does local code support an adapter-versus-merged replay?

Partially, but not end to end.

`_generate_outputs` already loads either a PEFT adapter or a full merged directory based on the presence of `adapter_config.json`, uses the full-brief serializer, and supports the exact visible seeds and limits. However, `sample_sweep` accepts a training checkpoint manifest shaped around adapter seeds and cannot express a paired adapter/merged comparison. It also writes independent sweep outputs without paired hashes, prompt-token hashes, logits, or equivalence decisions.

A private-function one-liner could generate both sides, but it would not be a preregistered, provenance-checked scientific artifact. A small dedicated workflow step is required.

## Minimal audit-and-replay specification

### Required code/config additions

1. Add `workflow.step: replay_equivalence` dispatch in `experiments/m1/workflow.py`.
2. Add `_replay_equivalence` there, reusing `_load_training_records`, `_directional_dev_subset`, `_render_prompt`, and `_generate_outputs`. It must:
   - verify the adapter SHA and merged directory content hash before loading;
   - load exactly the 16 frozen fingerprints and seeds `[101,202,303]`;
   - serialize prompts once and store SHA-256 of each rendered UTF-8 prompt and token-id sequence;
   - run the adapter and merged model after resetting the same RNG state before each side;
   - write paired JSONL plus a manifest with environment/library versions, dtype, tokenizer hashes, model hashes, per-row output hashes, and the acceptance decision;
   - never call a judge, harness sealed-submit, or provider.
3. Add `configs/m2/m2_adapter_merge_fidelity_replay_v1.yaml` with:
   - comparison `M2-adapter-merge-fidelity-replay-v1`, task `experiment`, L40S, 35-minute screen limit;
   - base and revision above;
   - adapter path/SHA above;
   - merged path `/checkpoints/runs/dftr-1784224462-c1f83ed3/merged-model` and expected content hash `0f437f62bc1cca0c`;
   - the exact adherence data/fixed-manifest hashes, 16 fingerprints, prompt format/schema, limits, sampler, and seeds from `configs/m1/m1_realdata_adherence_directional_qwen3_4b_three_seed_v1.yaml`;
   - `batch_size: 1` for the historical visible replay and an explicit `add_special_tokens` value matching the archived visible path.
4. Add `experiments/tests/test_m2_fidelity_replay.py` covering fail-closed hashes, exact 16-row selection/order, prompt/token hashes, RNG reset, paired cardinality 48/48, and no hidden/provider surface.
5. Add `replay_equivalence` to the immutable-revision guards in `infra/gpu` and `infra/backend/policy.py`.

Do not modify historical sealed configs or `harness/deployment_sampler.json` for this replay. A separate prospective full-brief deployment contract should be created only after the historical equivalence result is known.

### Commands after review and commit

```bash
ledger/ledger.py add \
  --comparison M2-adapter-merge-fidelity-replay-v1 \
  --hypothesis "The exact seed-29 adapter and merged checkpoint are byte-replay equivalent on the frozen 16 full-brief inputs under seeds 101,202,303."

infra/gpu submit \
  --config configs/m2/m2_adapter_merge_fidelity_replay_v1.yaml \
  --budget-class screen
```

After materialization, a local verifier in the new workflow should validate the paired manifest without regenerating text. No Tier-1 judge or sealed call is needed.

### Cost

The original three-seed, 144-document generation cost `$1.279227` and 1,966.832 L40S-seconds. Two paths for one checkpoint across 48 outputs imply about 1,311 L40S-seconds and `$0.85`. Allowing two model loads and verification, expected cost is `$0.9-$1.3`; a 35-minute screen reservation is approximately `$1.64` including the wrapper's 20% reserve. Provider and sealed-evaluator cost is zero.

### Acceptance criteria

Strict acceptance requires all of the following:

1. Adapter SHA, merged content hash, shard hashes, tokenizer hashes, data hash, subset hash, prompt hashes, and token-id hashes match preregistration.
2. The adapter replay reproduces the 48 archived visible `generated_completion` strings byte-for-byte. If it does not, stop: environment/interface replay is not established, so adapter-versus-merged output comparison is uninterpretable.
3. Adapter and merged outputs match byte-for-byte for all 48 paired stochastic generations after identical RNG reset.
4. Generated token counts and every deterministic visible metric are therefore exactly equal; no metric-level epsilon substitutes for output equality.

If strict output equality fails, run a diagnostic forward comparison, not an automatic pass: explicit bfloat16 on both sides, identical token IDs, teacher-forced reference completion for up to 64 tokens per record. Record mean absolute logit difference, maximum absolute difference, and top-1 agreement. Numeric closeness may be described when mean absolute difference is `<=0.002`, maximum is `<=0.05`, and top-1 agreement is `>=99.9%`, but it does **not** satisfy byte-replay equivalence. These are diagnostic tolerances for bfloat16 merge rounding, not promotion gates.

### Interpretation by outcome

| Outcome | Permitted conclusion |
| --- | --- |
| Adapter reproduces archive and merged matches 48/48 | The merge is behaviorally equivalent on the tested visible interface. The sealed rejection can reasonably be attributed to the same seed-29 model behavior, subject to the still-unresolved sealed batching/prompt-contract difference. |
| Adapter reproduces archive; merged differs, but logits meet diagnostic tolerances | The merge is numerically close but not byte-replay equivalent. Visible adapter metrics cannot be treated as exact merged metrics; score the merged 48 outputs visibly before any further scientific interpretation. The sealed verdict still applies to the merged artifact only. |
| Adapter reproduces archive; merged materially differs or greedy/top-1 differs | Merge fidelity failed. The sealed verdict remains valid for the merged artifact but does not test the visible adapter. Do not reuse the merged artifact or infer Tier-1/Tier-2 generalization until the merge path is corrected and replayed. |
| Adapter fails to reproduce the archived 48 outputs | The public generation environment is not reproducible. Freeze tokenizer/library/dtype/RNG behavior before testing merge fidelity; no conclusion about merge equivalence is permitted. |

## Prospective interface fix after the replay

Create one versioned full-brief deployment contract containing: serializer source/hash, prompt format, the target-length unit, tokenizer and chat-template hashes, `add_special_tokens`, padding/truncation side, input/output limits, model dtype, generation-config hash, sampler, batch size, record order, and an explicit per-record seed derivation such as `SHA256(global_seed || fingerprint) mod 2^63`.

Per-record generators make stochastic outputs invariant to batch size and would remove the current sequential-versus-batch ambiguity. This prospective rule cannot retroactively prove what the historical sealed evaluator did; it prevents the same ambiguity in the next submission.

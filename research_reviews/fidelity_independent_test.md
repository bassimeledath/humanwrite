# Fidelity replay independent verification

Date: 2026-07-16

Target: `operator/fidelity-replay-v1` at `97bc504b0f2c58ed2c90ba7549ee350eb787bcaf`

Verdict: **FAIL — do not authorize the replay launch at this commit.**

This was a public, CPU-only review. No GPU job, provider, judge, sealed service,
private repository, deployment, or merge was accessed. `/checkpoints` is not
available locally, so real adapter/merged bytes and the 48 archived outputs were
not regenerated.

## Results

| Requirement | Result | Evidence |
| --- | --- | --- |
| Checked-in identities and hashes | PARTIAL PASS | The generation contract, serializer, historical config, fixed manifest, and archive-index hashes match the replay config. Artifact paths, adapter files, merged shards/content, data, archive samples, and tokenizer equality are checked before model loading, but could not be exercised without `/checkpoints`. The reported `tensor_identity_sha256` is a hash of file-hash JSON, not a tensor-value digest independent of shard layout. |
| Exact 16 fingerprints and seeds | FAIL CLOSEDNESS | The checked-in list matches the historical list and seeds are exactly `[101, 202, 303]`. However, reversing fingerprint order is accepted because the subset hash is computed over the sorted list; the hash-bound historical config is never parsed to bind the ordered values. |
| Prompt/token/mask/truncation provenance | PASS BY INSPECTION/UNIT TEST | Both tokenizers must produce identical prompt bytes, IDs, masks, hashes, and truncation counts before comparison. The contract explicitly fixes special tokens, left padding, right truncation, and limits. |
| Greedy and teacher-forced diagnostics before stochastic comparison | PASS BY INSPECTION | Greedy exact-token parity and teacher-forced logit/top-1 thresholds gate both historical and prospective stochastic generation. |
| Per-record RNG and batch invariance | FAIL | Seed derivation is order-independent, but the actual path passes `generator=` to `model.generate`. Transformers `4.57.6` rejects it as an unused model kwarg, so prospective pairing cannot execute in the tested runtime. No dependency/container digest is frozen before launch. |
| No hidden/provider/judge/sealed surfaces | FAIL CLOSEDNESS | Workflow validation recursively rejects exact forbidden keys, but accepts aliases such as nested `judge_url`. Backend policy and `infra/gpu` inspect only top-level exact keys; a nested exact `workflow.runtime.judge` surface is accepted. |
| Infra policy enforcement | PARTIAL PASS | Immutable model revision, protocol version, task kind, command prefix, GPU count, timeout, and top-level forbidden keys are enforced. Recursive forbidden-surface enforcement is missing. |
| Config canonicality | FAIL | The launch payload hash detects transit mutation but does not bind the replay to the checked-in canonical config. Reordered fingerprints remain valid, and other self-consistent config substitutions are not compared with the hash-bound historical config contents. |
| Historical artifact immutability | PASS | No files under `configs/m1`, historical `configs/m2/m2_sealed*`, `harness`, or `experiments/m1/tier1` changed in the target commit. The sampled historical files have identical Git object IDs in `HEAD^` and `HEAD`. |

## Blocking findings

1. **Runtime failure in prospective stochastic comparison.**
   `experiments/m1/fidelity.py::_prospective_pairs` creates per-record
   `torch.Generator` objects and `_generate_one` forwards them to
   `GenerationMixin.generate`. The independent minimal GPT-2 probe fails with:
   `ValueError: The following model_kwargs are not used by the model: ['generator']`.
   This occurs before any parity decision can be produced.

2. **The frozen record order is not canonical.** Reversing all 16 fingerprints
   without changing `dev_subset_hash` passes `validate_replay_spec`. This misses
   the requested exact selection/order guarantee and permits a replay artifact
   whose row order differs from the preregistered historical interface.

3. **Forbidden-surface policy is bypassable by nesting or aliases.** A nested
   exact `judge` key passes backend launch validation, and `judge_url` passes
   workflow validation. The equivalent top-level-only scan is present in
   `infra/gpu`, so local and backend policy are consistently incomplete.

4. **Canonical tensor identity is not implemented.** Exact shard/file hashes
   are useful byte identities, but canonicalizing their JSON map does not hash
   tensor names, shapes, dtypes, and values independently of metadata/sharding,
   as required by the red-team acceptance criterion.

The four gaps above are preserved as strict expected-failure tests in
`experiments/tests/test_m2_fidelity_replay_independent.py`.

## Commands and outcomes

```bash
python -m pytest -q experiments/tests/test_m2_fidelity_replay.py \
  experiments/tests/test_m2_fidelity_replay_independent.py \
  infra/tests/test_policy.py
# 28 passed, 4 xfailed

python -m pytest -q -rxX \
  experiments/tests/test_m2_fidelity_replay_independent.py
# 4 xfailed, with one expected failure for each uncovered executable gap

python -m pytest -q
# collection failed: four existing infra modules require backend on PYTHONPATH

PYTHONPATH=infra python -m pytest -q
# 137 passed, 4 xfailed

python -m py_compile experiments/m1/fidelity.py experiments/m1/workflow.py \
  infra/backend/policy.py infra/gpu
git diff --check
# both passed

git diff --name-only HEAD^ HEAD -- configs/m1 \
  ':(glob)configs/m2/m2_sealed*' harness experiments/m1/tier1
# no output

for p in \
  configs/m1/m1_realdata_adherence_directional_qwen3_4b_three_seed_v1.yaml \
  configs/m1/manifests/m1_realdata_adherence_fixed_inputs_v1.json \
  experiments/m1/tier1/adherence_4b_three_seed_v1/index.json \
  harness/deployment_sampler.json; do
  git rev-parse "HEAD^:$p"
  git rev-parse "HEAD:$p"
done
# each parent/current object-ID pair was identical
```

The committed tests cover the CPU-reproducible findings only. Actual model,
archive reproduction, greedy/logit parity, and 48-pair output parity remain
unverified until the blockers are fixed and a separately authorized GPU replay
is run.

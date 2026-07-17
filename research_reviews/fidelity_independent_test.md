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

## Re-test addendum: repair commit `9f350e1`

Final verdict: **FAIL — two launch-readiness blockers remain.**

The repair was tested independently at
`9f350e135e5e56353a6fd64397d780b618b1e56f`. It genuinely closes the four
immediate implementation defects from the first review:

- The stochastic path no longer forwards `generator=` to Transformers. With
  the locally installed Transformers `4.57.6`, a real tiny GPT-2 model generated
  successfully inside the scoped RNG context. The ambient PyTorch RNG state was
  restored exactly, and per-record outputs were identical under forward,
  reverse, singleton, and regrouped execution.
- Reversing the checked-in 16-fingerprint list is rejected against the
  hash-verified historical config, and the three seed values/order remain bound.
- Workflow, backend, and `infra/gpu` policy now recursively reject tested exact
  and aliased API/provider/judge/sealed/hidden keys.
- Identity fields and the implementation report now accurately describe hashes
  of serialized weight files/shard maps rather than claiming a canonical
  tensor-value identity. A tensor-value digest independent of serialization and
  sharding is still not produced, so that stronger native red-team criterion
  remains unverified rather than misrepresented.

Two fail-closed gaps remain:

1. **Transformers is not pinned.** The GPU worker image still declares
   `transformers>=4.53,<5`. Recording the installed version after execution does
   not freeze the generation environment before launch. The local 4.57.6 test
   therefore does not guarantee the same sampling behavior for a later image
   rebuild.
2. **The historical order binding is self-substitutable.** Both
   `historical_sampling_config` and its expected SHA-256 come from the mutable
   replay config. An adversarial config can point to a substitute historical
   YAML, update the supplied hash, reverse the fingerprints in both files, and
   pass `validate_replay_spec`. Neither preregistration nor policy binds the
   launch to the repository's canonical replay-config digest or to hard-coded
   historical path/hash constants.

These two gaps are strict expected-failure tests. Repair verification commands:

```bash
python -m pytest -q -rxX \
  experiments/tests/test_m2_fidelity_replay_independent.py
# 6 passed, 2 xfailed

PYTHONPATH=infra python -m pytest -q \
  experiments/tests/test_m2_fidelity_replay.py \
  experiments/tests/test_m2_fidelity_replay_independent.py \
  infra/tests/test_policy.py experiments/tests/test_m1_sampler_loader.py
# 43 passed, 2 xfailed

PYTHONPATH=infra python -m pytest -q
# 149 passed, 2 xfailed

python -m py_compile experiments/m1/fidelity.py experiments/m1/workflow.py \
  experiments/tests/test_m2_fidelity_replay_independent.py \
  infra/backend/policy.py infra/gpu
git diff --check
# both passed

git diff --name-only a4cae58 HEAD -- configs/m1 \
  ':(glob)configs/m2/m2_sealed*' harness experiments/m1/tier1
# no output; historical artifacts remain unchanged
```

No GPU, deploy, provider, judge, sealed/private repository, or hidden-data
access was performed during this re-test.

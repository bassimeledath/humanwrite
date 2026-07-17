# Public adapter/merge fidelity implementation report

Date: 2026-07-16

Branch: `operator/fidelity-replay-v1`

Scope: public repository implementation only; no GPU launch, provider call,
judge call, sealed-evaluator call, private repository access, or hidden-data access.

## Outcome

Implemented a fail-closed `replay_equivalence` workflow and a prospective,
versioned full-brief generation contract. The workflow is ready for a later
human-authorized public GPU run, but this implementation task did not launch it.

The replay preserves all historical configs and artifacts. It adds a new
workflow, contract, and replay config instead of changing the archived M1
generation behavior.

## Frozen identities

The new replay config binds:

- the seed-29 adapter directory, adapter tensor SHA-256, checkpoint-manifest
  SHA-256, and the manifest's complete per-file identity map;
- the merged directory's public 16-hex content hash plus all three tensor-shard
  SHA-256 values and the tensor-index SHA-256;
- tokenizer files through the adapter checkpoint manifest, with byte equality
  required against the merged directory and a runtime chat-template digest;
- base model `Qwen/Qwen3-4B` at immutable revision
  `1cfa9a7208912126459214e8b04321603b3df60c`;
- the fixed-input manifest, exact dev JSONL SHA-256, historical sampling-config
  SHA-256, archived Tier-1 index SHA-256, and each archived sample-file SHA-256;
- exactly 16 preregistered fingerprints, their subset hash, and sampling seeds
  `[101, 202, 303]`;
- the generation-contract SHA-256 and canonical serializer source SHA-256.

## Execution order and evidence

The workflow records prompt UTF-8 bytes and SHA-256, explicit token IDs,
attention masks, their sequence hashes, untruncated and truncated token counts,
and right-truncation counts. Both adapter and merged tokenizers must emit
identical attestations before model comparison proceeds.

Both models load explicitly as BF16, and their observed parameter dtypes and
the Python, PyTorch, Transformers, and PEFT versions are recorded. The runner
enables deterministic algorithms.

The comparison is ordered as follows:

1. exact greedy-token parity on all 16 records;
2. teacher-forced comparison for up to 64 reference tokens per record, with
   preregistered thresholds of mean absolute logit difference `<= 0.002`,
   maximum absolute difference `<= 0.05`, and top-1 agreement `>= 0.999`;
3. only after that gate passes, historical adapter replay with the original
   sequential global-RNG behavior and exact byte comparison to all 48 archived
   outputs;
4. only after archive reproduction passes, prospective adapter-versus-merged
   sampling with a fresh generator per record and seed derived as
   `SHA256(u64be(global_seed) || NUL || fingerprint) mod 2^63`;
5. exact output-token and UTF-8 byte parity for all 48 prospective pairs.

If deterministic diagnostics fail, no stochastic replay runs. If adapter
archive reproduction fails, the adapter-versus-merged stochastic comparison is
not run and interpretation is explicitly disallowed. Numeric closeness cannot
substitute for exact replay parity.

## Public-only enforcement

The replay requires the ordinary `experiment` task kind and the existing
`python -m experiments.runner` command prefix. Runtime validation rejects paid
or hidden surfaces. Both the local GPU client and backend policy now:

- treat `replay_equivalence` as an evidentiary workflow requiring an immutable
  model revision;
- require protocol `dftr.adapter_merge_replay.v1`;
- reject top-level API, provider, judge, sealed, or hidden surfaces.

The replay implementation imports no harness judge, sealed client, provider
SDK, or network client.

## Verification

Focused tests:

```text
python -m pytest -q experiments/tests/test_m2_fidelity_replay.py \
  infra/tests/test_policy.py experiments/tests/test_m1_sampler_loader.py
31 passed in 0.08s
```

Full relevant experiment and infrastructure tests:

```text
python -m pytest -q experiments/tests infra/tests
64 passed in 1.13s
```

Repository-wide tests with the infrastructure package on the import path:

```text
PYTHONPATH=infra:. python -m pytest -q
137 passed in 4.10s
```

A bare repository-root `python -m pytest -q` cannot collect four existing
infrastructure test modules because they import the `backend` package with
`infra` expected on `PYTHONPATH`; this is an existing test invocation
requirement, not a fidelity change. The corrected repository-wide invocation
above passes.

Additional checks:

```text
python -m py_compile experiments/m1/fidelity.py experiments/m1/workflow.py \
  infra/backend/policy.py infra/gpu
git diff --check
```

Both completed successfully.

## Launch boundary

No preregistration was opened and no `infra/gpu submit` command was run. A
future operator can review and preregister `M2-adapter-merge-fidelity-replay-v1`
before authorizing the bounded screen run.

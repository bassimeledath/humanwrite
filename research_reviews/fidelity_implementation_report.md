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

- the seed-29 adapter directory, serialized adapter-weight file SHA-256, checkpoint-manifest
  SHA-256, and the manifest's complete per-file identity map;
- the merged directory's public 16-hex content hash plus all three serialized
  weight-shard SHA-256 values and the shard-index SHA-256;
- tokenizer files through the adapter checkpoint manifest, with byte equality
  required against the merged directory and a runtime chat-template digest;
- base model `Qwen/Qwen3-4B` at immutable revision
  `1cfa9a7208912126459214e8b04321603b3df60c`;
- the fixed-input manifest, exact dev JSONL SHA-256, historical sampling-config
  SHA-256, archived Tier-1 index SHA-256, and each archived sample-file SHA-256;
- exactly 16 preregistered fingerprints in the order bound by the hash-verified
  historical config, their subset hash, and sampling seeds `[101, 202, 303]`;
- the generation-contract SHA-256 and canonical serializer source SHA-256.

The canonical generation contract now pins Transformers `4.57.6`, and the
checked-in replay config carries the same exact runtime requirement. The Modal
worker image installs `transformers==4.57.6` rather than a version range.

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
   sampling inside a scoped PyTorch RNG context per record, with the global RNG
   restored afterward and the record seed derived as
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
- recursively reject API, provider, judge, sealed, or hidden surfaces, including
  aliases such as `judge_url`, `apiKey`, and `sealedEvaluator`.

The replay implementation imports no harness judge, sealed client, provider
SDK, or network client.

The reported weight identities are explicitly serialization-byte identities:
the adapter weight-file map and merged weight-shard file map. They are not
described as canonical tensor-value identities independent of sharding. That
stronger identity remains the responsibility of a future remote artifact
attestation.

## Independent-test correction

Independent CPU verification at commit `a4cae58` identified four launch
blockers in the first implementation (`97bc504`): Transformers 4.57.6 rejected
`generate(generator=...)`; fingerprint order was not bound; nested/aliased paid
surfaces bypassed policy; and file-map hashes were mislabeled as tensor
identities.

This correction addresses all four findings. The independent tests are kept as
ordinary strict regression tests and now pass. In particular, stochastic
generation no longer forwards `generator` into Transformers. It extracts the
per-record seed, enters `torch.random.fork_rng`, seeds the scoped CPU/CUDA RNG,
runs `generate`, and restores prior RNG state. A stochastic fake-model test
also verifies identical per-record outputs across forward, reverse, singleton,
and regrouped execution orders.

### Second independent re-test

Independent re-test commit `befabaf` confirmed the four earlier repairs and
identified two remaining launch gates: the worker still allowed a Transformers
version range, and the caller could substitute both the historical binding file
and its expected hash.

Both are now closed:

- Transformers `4.57.6` is exact in the worker image, generation contract,
  replay config, backend policy, local GPU policy, and workflow runtime check.
  The workflow aborts if the installed library reports any other version.
- The canonical generation-contract path/SHA and historical sampling-config
  path/SHA are repository-code constants in both workflow and launch policy.
  Caller fields must equal those constants, and the workflow independently
  hashes the canonical repository files. A self-consistent substitute path and
  hash is rejected before compute by policy and again by the workflow.

The two new independent expected failures were retained as normal regression
tests and now pass.

## Verification

Focused tests:

```text
python -m pytest -q experiments/tests/test_m2_fidelity_replay.py \
  experiments/tests/test_m2_fidelity_replay_independent.py \
  infra/tests/test_policy.py experiments/tests/test_m1_sampler_loader.py
47 passed in 2.70s
```

Repository-wide tests with the infrastructure package on the import path:

```text
PYTHONPATH=infra:. python -m pytest -q
153 passed in 5.66s
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

## Historical launch boundary

At the time of the original implementation, no preregistration had been opened
and no `infra/gpu submit` command had been run. The v1 comparison was later
preregistered and launched, then failed on the artifact-identity mismatch
described below.

## Prospective replay v2 identity repair

The historical v1 replay is preserved byte-for-byte. Its config SHA-256 remains
`8015afd23f7d21953e0e7f0f1045db824a87377ec38de4c7c478b7455570ef4c`,
including its now-known incorrect binding of the original merged-model path to
the later submitted-snapshot directory hash. The failed v1 attempt is not
reinterpreted as evidence.

The prospective v2 replay separates the two artifact identities:

- original merged directory
  `/checkpoints/runs/dftr-1784224462-c1f83ed3/merged-model`, canonical directory
  hash `7f095c31e83f8b03`;
- later submitted snapshot, canonical directory hash `0f437f62bc1cca0c`.

The independently hashed identity manifest
`configs/m2/manifests/m2_adapter_merge_snapshot_identity_v2.json` has SHA-256
`602cb05fed6fe3a0ecc1e37bc811ae5bb255c2624b57b051ae0744c7a0973b2c`.
It binds the complete original file map, the identical snapshot file map, and
the only two differences:

| File | Original SHA-256 | Submitted-snapshot SHA-256 |
| --- | --- | --- |
| `generation_config.json` | `64d86df2173901c58389974bde21f7d2ab9eb7d79f35a337753329d39cf265c0` | `0ba4fa0fce9b70e3a1a830c618de7fd1a1e4adb3008eaa147fa22aa35550d0f0` |
| `train_config.json` | `a09d02a3fce6aa2b2e4447dd69b24493d97694b63be0151e455e513ae4b93ef2` | `bdeb26ea942bd28eeae6d4522849636c3459b8406b00288380ba8b41c7a3ba18` |

All weights, tokenizer files, and the weight index are exact serialization-byte
matches between the original merge and later snapshot. Replay generation does
not consume the differing snapshot metadata: its arguments remain explicit and
bound to `configs/m2/canonical_full_brief_generation_v1.json`.

The workflow, local GPU client, and backend policy require the exact v2
comparison ID, protocol, original hash, snapshot hash, manifest path/hash, and
ordered two-file difference declaration. The workflow also independently hashes
the manifest, validates both file maps, and verifies the original artifact's
full file map before generation. Substitution of either directory identity or
metadata-difference set fails closed; no check was weakened.

This repair is prospective only. No v2 preregistration was created, no v2 job
was submitted, and no budget was spent. A future operator must review and open
`M2-adapter-merge-fidelity-replay-v2` before any bounded screen launch.

Repair verification:

```text
PYTHONPATH=infra:. python -m pytest -q \
  experiments/tests/test_m2_fidelity_replay_v2.py \
  experiments/tests/test_m2_fidelity_replay.py \
  experiments/tests/test_m2_fidelity_replay_independent.py \
  infra/tests/test_policy.py
53 passed in 3.18s

PYTHONPATH=infra:. python -m pytest -q
162 passed in 5.64s
```

## Final public-surface classifier correction

Final independent review at `7bb4b8c` found that the recursive public-only
scanner missed wrapped camel/Pascal credential aliases while treating standard
tokenizer metadata as credential tokens. The worker and canonical backend
policy now classify structured key words after splitting camel/Pascal, snake,
kebab, dotted, and other non-alphanumeric boundaries. The local GPU client uses
that canonical backend classifier.

Sensitive words are rejected in any key position, including the tested
`remoteServiceUrl`, `privateEndpointUrl`, `externalAuthConfig`,
`clientSecretValue`, and `gatewayAccessTokenValue` forms. Token-bearing keys
remain rejected unless every word belongs to a bounded public tokenizer or
generation-metadata vocabulary with a non-token modifier. This permits audited
fields such as `special_tokens_map`, `added_tokens`, `max_new_tokens`, and
`pad_token_policy`, while continuing to reject `access_token`, `provider_token`,
and unqualified token-value fields.

No replay schema, comparison, artifact identity, canonical contract, exact
serialization, or v1 historical-config guard changed. The v1 YAML SHA-256 is
still `8015afd23f7d21953e0e7f0f1045db824a87377ec38de4c7c478b7455570ef4c`;
the v2 YAML SHA-256 is still
`a5f0504dfdcfd12cfda5081e068919a603a395c7155a725bd9e7c13016ba1d8c`;
and the identity-manifest SHA-256 is still
`602cb05fed6fe3a0ecc1e37bc811ae5bb255c2624b57b051ae0744c7a0973b2c`.

Final verification:

```text
# Raw final tester pack with inherited strict-xfail markers.
PYTHONPATH=infra:. python -m pytest -q \
  experiments/tests/test_m2_fidelity_replay_v2_final_independent.py
15 passed, 8 failed

# Forced bodies: all 22 semantic assertions pass; the sole remaining failure
# is the tester-only assertion that implementation still equals old target d36b2e2.
PYTHONPATH=infra:. python -m pytest -q --runxfail \
  experiments/tests/test_m2_fidelity_replay_v2_final_independent.py
22 passed, 1 failed

PYTHONPATH=infra:. python -m pytest -q --runxfail \
  experiments/tests/test_m2_fidelity_replay_v2_final_independent.py \
  -k 'not test_tester_commit_does_not_modify_fidelity_implementation_surfaces'
22 passed, 1 deselected in 0.20s

# Focused replay/policy suite.
109 passed, 1 deselected in 3.78s

# Repository-wide forced suite, excluding only that stale scope assertion.
PYTHONPATH=infra:. python -m pytest -q --runxfail \
  -k 'not test_tester_commit_does_not_modify_fidelity_implementation_surfaces'
218 passed, 1 deselected in 6.09s
```

`py_compile`, structured key spot checks, `git diff --check`, and exact artifact
hash checks also passed. No preregistration, deployment, launch, or spend was
performed.

## Fidelity-v2 launch-boundary repair

Independent tester commit `25c2f60` confirmed the direct artifact-identity
checks but found three launch bypasses. The follow-up repair closes them at the
worker workflow, backend policy, and local GPU client without changing either
tester-owned artifact:

- replay protocol and comparison IDs are now bidirectional: v2 can only use
  `M2-adapter-merge-fidelity-replay-v2`; v1 can only use the historical v1
  comparison, submitted-snapshot hash, and exact canonical parsed-config hash
  `859798f2...4587b9f` whose source YAML remains byte-pinned at
  `8015afd2...0ef4c`;
- backend and client now enforce `exact_serialization_bytes` and the canonical
  explicit generation-arguments authority, matching worker validation; and
- recursive public-only scans reject credential, secret, token, auth, key,
  endpoint, and service aliases, including nested/compound forms, while not
  misclassifying ordinary tokenizer fields.

Forced execution of the three tester-owned blockers now passes all 19 tests.
The focused replay/policy/sampler suite passes 90 tests with strict-xfail
markers forced to ordinary execution; the repository-wide forced run passes
196 tests. A normal repository run excluding only the tester file passes 177
tests; keeping that strict-xfail file in a normal run intentionally reports
strict XPASS until the independent tester updates its verdict artifact. No
preregistration, deployment, launch, remote artifact access, or spend was
performed by this repair.

## Expanded model-token vocabulary correction

Tester commit `ecb37a4` broadened the classifier matrix beyond the first five
examples. It confirmed the private-alias boundary across OAuth, bearer, JWT,
provider, service-account, and API-key forms, then identified nine legitimate
Transformers tokenizer/generation fields that the bounded public vocabulary
still rejected. It also showed that OAuth/OIDC `id_token` was ambiguous with
model token-ID metadata.

The classifier now admits the audited public concepts `split`, `extra`, `all`,
`extended`, `spaces`, `between`, `healing`, `image`, `video`, and `vision` only
inside the existing bounded model-token vocabulary. It explicitly rejects the
ordered key `id_token`; model metadata uses the inverse `*_token_id` order.
This preserves rejection of every private alias in the broadened matrix while
allowing all standard public model-token fields tested by the reviewer.

Verification after the correction:

```text
# Broadened independent semantic matrix, excluding only its old-target scope check
44 passed, 1 deselected

# All fidelity independent packs plus policy tests
120 passed, 2 deselected

# Implementation fidelity/policy tests
67 passed

# Full repository with all strict-xfail bodies forced, excluding two old-target scope checks
282 passed, 2 deselected
```

No protocol, artifact, manifest, generation-contract, or historical-v1 bytes
changed. No preregistration, deployment, launch, or spend occurred.

## Strict canonical replay-config boundary

The broadened tester matrix also exposed that fuzzy key classification is not
the correct authorization boundary for this one-off replay. Its passing-field
cases injected an invented `runtime.public_metadata` extension, but neither
canonical replay YAML contains that extension: both runtime sections contain
only `transformers_version`. All tokenizer and generation behavior lives in
`configs/m2/canonical_full_brief_generation_v1.json`, whose path and raw
SHA-256 are already frozen.

Replay v2 is therefore now restricted to the exact checked-in prospective
config, just as v1 was already restricted to its exact historical config. The
worker binds raw YAML SHA-256
`a5f0504dfdcfd12cfda5081e068919a603a395c7155a725bd9e7c13016ba1d8c`
and parsed canonical hash
`ee76ca0ecda72321f07cecd1c70fba5905779321e3169579e357bafdad4cd1da`;
the backend and GPU client bind the same parsed hash. Any unknown field now
fails closed regardless of whether its name looks public or private. The
structured classifier remains defense in depth, not an extension mechanism.

The independent reviewer explicitly confirmed this strict boundary matches
both canonical configs and that the earlier arbitrary-public-metadata
expectation was invalid. Focused worker/policy verification passes 72 tests.
No model, artifact, protocol, generation-contract, or historical-v1 bytes
changed, and no external action or spend occurred.

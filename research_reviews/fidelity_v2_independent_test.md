# Fidelity replay v2 independent verification

Date: 2026-07-16 (America/Los_Angeles; completed 2026-07-17 UTC)

Target commit: `092d3c0b8b93c95734a30e708dfe6a2d47c68219`

Branch: `swarmy/humanwrite-next-cycle`

Verdict: **FAIL — do not preregister or authorize the fidelity-v2 replay launch
at this commit**

This was a public, CPU-only verification. No preregistration, deployment, GPU
launch, provider/judge/sealed call, private repository access, hidden-data
access, artifact merge, or spend occurred. `/checkpoints` was not accessed, so
this report does not claim the scientific adapter/merge replay succeeded.

## Executive result

The direct v2 identity repair is correct and unusually well bound. The checked-
in v2 config targets the original merged directory hash `7f095c31e83f8b03`,
audits the later submitted snapshot separately as `0f437f62bc1cca0c`, and
pins identity-manifest SHA-256
`602cb05fed6fe3a0ecc1e37bc811ae5bb255c2624b57b051ae0744c7a0973b2c`.
The workflow opens and hashes that exact manifest, verifies the complete
original file map, verifies the shared snapshot file map, restricts differences
to `generation_config.json` and `train_config.json`, and then recomputes the
canonical hash of the original directory. Direct substitution of either hash,
the manifest digest, difference-file order/set, or the v2 comparison ID is
rejected by workflow, backend, and local client.

The repair is still launch-bypassable through protocol downgrade. Starting
from the exact v2 config, changing only:

```yaml
workflow.protocol_version: dftr.adapter_merge_replay.v1
artifacts.merged_content_hash: 0f437f62bc1cca0c
```

is accepted by `validate_replay_spec`, backend `validate_launch`, and
`infra/gpu`. This preserves the v2 comparison ID but skips the entire v2
identity manifest and permits the submitted-snapshot hash to replace the
original target. The original `7f095…` binding therefore is not unconditional.

Two defense-in-depth gaps also remain: a false
`weights_tokenizer_index_identity: not-exact` claim is rejected by the worker
but admitted by backend/client, and a nested `credential` surface is accepted
by all three forbidden-surface scanners.

## Requirement matrix

| Requirement | Result | Independent evidence |
| --- | --- | --- |
| Exact v2 config and identity manifest | PASS | Config and manifest hashes match the checked-in constants; original/snapshot identities are distinct and paths are bound. |
| Original merged-directory identity | FAIL end-to-end | Direct v2 substitutions reject and full file-map/canonical hashing is strict, but v2→v1 downgrade plus snapshot-hash substitution is accepted by all three guards. |
| Submitted-snapshot identity | PASS in direct v2 path | Snapshot hash is audit-only, exact manifest digest is pinned, and it cannot directly replace the original hash under protocol v2. |
| Metadata file map/difference set | PASS in workflow | Dropped/changed shared-file hashes, an added metadata difference, changed difference order/set, and manifest-digest substitution reject. |
| v1/v2 comparison and schema | FAIL | v2 binds its comparison only while protocol remains v2. A v2-comparison config declaring protocol v1 is accepted and bypasses v2 identity checks. |
| Backend policy | FAIL | Direct v2 hash/manifest/difference/comparison substitutions reject, but schema downgrade, snapshot-hash substitution, false serialization claim, and `credential` alias are admitted as described above. |
| `infra/gpu` policy | FAIL | Same result as backend; validation can return an accepted comparison before any worker-side identity rejection. |
| Forbidden public-only surfaces | FAIL | Declared aliases `api_key`, `provider_token`, `judge_url`, `sealed_endpoint`, and `hidden_fixture` reject recursively. `credential` is not in the token list and passes every layer. |
| v1/historical immutability | PASS | v1 config SHA-256 remains `8015afd…`; parent/target Git object IDs match; no files in the audited historical paths changed. |
| CPU implementation regression suite | PASS except independent blockers | Existing v1/v2 workflow, RNG, runtime, policy, and artifact-hash tests pass. |

## Blocking findings

### F1 — Critical: protocol downgrade restores snapshot-hash substitution

`validate_replay_spec` accepts both replay schemas, but only checks the v2
comparison and calls `load_snapshot_identity_audit` inside
`if protocol_version == REPLAY_SCHEMA_V2`. Backend policy and `infra/gpu`
follow the same pattern. There is no reverse constraint requiring comparison
`M2-adapter-merge-fidelity-replay-v2` to use protocol v2.

Consequently the v2 config can declare v1, replace the original merge hash with
the submitted snapshot hash, retain a self-consistent payload hash and open v2
preregistration identity, and pass all pre-launch/worker spec validators. The
v2 manifest fields simply become ignored extras.

Required repair: bind comparison ID and protocol schema bidirectionally in all
three layers. The v2 comparison must require v2, and the v1 protocol must
require the historical v1 comparison plus its historical artifact identity.
Prefer an exact checked-in v2 config digest or equivalent immutable field-set
contract so a schema downgrade cannot weaken validation.

### F2 — High: backend/client omit part of the v2 identity contract

Changing `submitted_snapshot_audit.weights_tokenizer_index_identity` from
`exact_serialization_bytes` to `not-exact` is rejected by the workflow but
accepted by backend and local client. A paid GPU job could therefore be
submitted only to fail after worker startup.

Required repair: share one canonical replay-spec validator at both launch
boundaries or mirror every identity-manifest authority field, including exact
serialization identity and generation-argument authority.

### F3 — High: credential alias bypasses public-only policy

The scanners enumerate `api`, `provider`, `judge`, `sealed`, and `hidden` key
parts. A nested key named `credential` with a private value is accepted by
workflow, backend, and client. The replay must not admit credential-bearing
surfaces under neutral aliases.

Required repair: reject credential/secret/token/auth/key/endpoint/service
aliases appropriate to this credential-free experiment, or validate against a
strict allowlisted replay schema instead of a partial denylist.

## Passing adversarial evidence

The tester-owned pack independently confirms:

- exact original `7f095…` versus submitted snapshot `0f437…` roles;
- exact identity-manifest digest and original merged path;
- direct substitution of either directory hash fails at every layer;
- manifest SHA substitution and v2→v1 comparison substitution fail directly;
- reversed/changed metadata difference sets fail;
- a dropped shared file, added metadata difference, or changed shared-file hash
  fails manifest semantic validation even when the substitute manifest's own
  digest is updated;
- canonical directory hashing changes on metadata byte changes and extra files;
- the exact file map rejects byte mutation;
- declared API/provider/judge/sealed/hidden aliases reject recursively; and
- v1 and all audited historical surfaces are unchanged at the target commit.

The real original directory is not locally present, so actual `7f095…` bytes,
adapter archive reproduction, greedy/logit diagnostics, and 48 stochastic
pairs remain future authorized-run checks.

## Commands and results

Existing focused fidelity/policy suite plus the independent v2 pack:

```text
PYTHONPATH=infra python -m pytest -q \
  experiments/tests/test_m2_fidelity_replay_v2.py \
  experiments/tests/test_m2_fidelity_replay_v2_independent.py \
  experiments/tests/test_m2_fidelity_replay.py \
  experiments/tests/test_m2_fidelity_replay_independent.py \
  infra/tests/test_policy.py experiments/tests/test_m1_sampler_loader.py
72 passed, 3 xfailed
```

Repository-wide suite:

```text
PYTHONPATH=infra python -m pytest -q
178 passed, 3 xfailed
```

The three strict xfails are tester-owned executable blockers. Forced ordinary
execution shows the accepted layers exactly:

```text
PYTHONPATH=infra python -m pytest -q --runxfail \
  experiments/tests/test_m2_fidelity_replay_v2_independent.py
16 passed, 3 failed
```

- schema downgrade + snapshot hash: accepted by workflow, backend, gpu client;
- false exact-serialization claim: accepted by backend and gpu client;
- credential surface: accepted by workflow, backend, gpu client.

Additional checks:

- manifest SHA-256: `602cb05f…0973b2c` (matches config and code);
- v2 config SHA-256: `a5f0504d…ba1d8c`;
- v1 config SHA-256: `8015afd2…0ef4c`;
- v1 parent/target Git object IDs: both
  `dfad94032d5cacc1e18b895ac351945385ac00b1`;
- historical target diff: empty;
- `py_compile` and `git diff --check`: pass.

## Runtime manifest

```json
{
  "target_commit": "092d3c0b8b93c95734a30e708dfe6a2d47c68219",
  "identity_manifest_sha256": "602cb05fed6fe3a0ecc1e37bc811ae5bb255c2624b57b051ae0744c7a0973b2c",
  "v2_config_sha256": "a5f0504dfdcfd12cfda5081e068919a603a395c7155a725bd9e7c13016ba1d8c",
  "tester_fixture_sha256": "8a26250c2c5e122cae2b481577d7897c9b3b39a71687ba825d3163b90f86b7be",
  "python": "3.10.9",
  "pytest": "7.4.3",
  "pyyaml": "6.0.1",
  "platform": "macOS-26.4.1-arm64-arm-64bit",
  "tested_at": "2026-07-17T01:03:38Z"
}
```

## Post-verdict comparison with the implementation report

The implementation report accurately describes the direct v2 path and v1
byte preservation. Its statement that workflow, client, and backend “require
the exact v2 comparison ID, protocol, original hash, snapshot hash” is not true
under downgrade: all three accept the v2 comparison with protocol v1 and the
snapshot hash in the original-hash field. The report also overstates launch-
boundary coverage of exact serialization identity; only the worker validates
that claim.

This comparison was performed after the independent FAIL verdict was frozen.

## Required disposition

Do not preregister, submit, or spend on fidelity replay v2 at `092d3c0`. Close
the three strict-xfail cases, rerun the full CPU suite with zero failures, and
then obtain separate human authorization for the bounded GPU replay. A future
CPU PASS would qualify only the implementation and launch guards; the actual
fidelity conclusion still requires hash verification of the original remote
artifact and all replay acceptance gates on the pinned worker.

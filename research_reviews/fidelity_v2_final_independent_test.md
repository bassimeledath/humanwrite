# Final independent fidelity replay v2 verification

Date: 2026-07-16

Target: `d36b2e2dc8b162bb3e3978d8841307afda24e879`

Verdict: **FAIL — do not preregister, launch, deploy, or spend on fidelity replay v2.**

Scope was public repository state and CPU-only validation. No preregistration,
launch, provider call, hidden/private artifact access, credential use, or spend
was performed. The pre-existing local edit to `progress/status.json` was not
changed or included. No implementation surface was modified by this tester
commit.

## What passed

- The prior independent pack's three blockers are closed. Forced execution is
  `19 passed`: v2 cannot downgrade to v1 with the submitted-snapshot hash, the
  false exact-serialization claim is rejected, and the plain `credential`
  alias is rejected at workflow, backend, and local-client layers.
- Protocol/comparison binding is bidirectional. V1 and v2 cross-pairings fail
  at all three layers.
- V1 is restricted to the exact historical parsed config identity. Its YAML
  remains SHA-256 `8015afd2...0ef4c`, canonical JSON hash
  `859798f2...4587b9f`, comparison `M2-adapter-merge-fidelity-replay-v1`, and
  merged-content identity `0f437f62bc1cca0c`. The exact config is accepted;
  any tested field substitution is rejected everywhere.
- V2 rejects substitutions of exact serialization, snapshot-declared
  generation authority, and ordered metadata differences at all three layers.
  Generation-contract path/hash substitutions are rejected by backend and
  client before launch and by the worker's canonical contract loader before
  artifact/model execution.
- Recursive scanning correctly permits the audited `tokenizer_path`,
  `tokenizer_config`, and `weights_tokenizer_index_identity` public fields.

## Blocking findings

### F1 — High: wrapped credential aliases bypass every layer

The new scanners recognize aliases only when a protected word is an
underscore-delimited part, selected prefix, or suffix. Natural compound names
with text on both sides are missed. Nested keys `remoteServiceUrl`,
`privateEndpointUrl`, `externalAuthConfig`, `clientSecretValue`, and
`gatewayAccessTokenValue` are accepted by `validate_replay_spec`, backend
`validate_launch`, and local-client `_validate_submit`.

Required repair: tokenize snake, kebab, dotted, and camel/pascal names into
semantic words, then reject credential/private-surface terms in any position.
Keep the same canonical scanner shared by backend and client, and independently
enforce equivalent behavior in the worker.

### F2 — Medium: public tokenizer fields are false positives

The scanner treats an exact `tokens` component as a credential token. As a
result, inert public metadata named `special_tokens_map` or `added_tokens` is
rejected by all three layers even though those are standard public tokenizer
artifact fields and both files are part of the audited replay identity.

Required repair: distinguish credential/authentication tokens from tokenizer
vocabulary fields. Add explicit positive tests for standard Hugging Face
tokenizer and generation metadata while retaining negative tests for
`access_token`, `provider_token`, and wrapped credential aliases.

## Test evidence

The pre-repair independent pack was run normally as requested. It reports
`16 passed, 3 strict XPASS failures`; those XPASS failures mean its old expected
failures are repaired, not that the implementation regressed. With markers
forced to ordinary tests it reports `19 passed`.

Before the new final cases, the focused forced suite reported `90 passed` and
the repository-wide forced suite reported `196 passed`. The final tester cases
retain F1/F2 as strict expected failures so a normal suite remains fail-closed;
`--runxfail` exposes them as ordinary failures until repaired.

Final commands and results:

```text
PYTHONPATH=infra:. python -m pytest -q \
  experiments/tests/test_m2_fidelity_replay_v2_final_independent.py
16 passed, 7 xfailed in 0.24s

PYTHONPATH=infra:. python -m pytest -q --runxfail \
  experiments/tests/test_m2_fidelity_replay_v2_final_independent.py
7 failed, 16 passed in 0.24s

# Focused replay, policy, and sampler suite including both independent packs.
PYTHONPATH=infra:. python -m pytest -q --runxfail <focused paths>
7 failed, 106 passed in 3.63s

# Full suite with every strict expected failure forced.
PYTHONPATH=infra:. python -m pytest -q --runxfail
7 failed, 212 passed in 6.17s

# Full normal suite excluding only the stale repaired-blocker pack whose
# strict XPASS behavior is reported above.
PYTHONPATH=infra:. python -m pytest -q \
  --ignore=experiments/tests/test_m2_fidelity_replay_v2_independent.py
193 passed, 7 xfailed in 6.78s
```

## Disposition

The downgrade and launch-identity repair is materially correct, but the
credential-free boundary is still bypassable and overbroad. Do not authorize
the replay. After F1 and F2 are repaired, rerun this file with `--runxfail`, the
earlier independent pack with `--runxfail`, the focused replay/policy/sampler
suite, and the full repository suite before considering CPU qualification.

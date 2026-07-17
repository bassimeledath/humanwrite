# Fidelity v2 strict-boundary independent verification

Date: 2026-07-16 (America/Los_Angeles; completed 2026-07-17 UTC)

Target: `ec3f74ba1fa72c2cfeead5b2f25866b5065286f0`

Verdict: **PASS for CPU-verifiable fidelity-v2 implementation and launch
guards.** This does not constitute a preregistration or authorize a replay,
deployment, provider call, hidden access, or spend.

This review used only public repository state and CPU tests. No implementation
file was edited by the tester. The two final tester packs were retargeted to the
strict-boundary commit and their expectations were rebased from an invented
extensible metadata namespace to the actual exact replay-config contract.

## Verified result

- V1 remains restricted to its exact historical YAML bytes, canonical parsed
  identity, comparison, protocol, and submitted-snapshot artifact identity.
- V2 is restricted to the exact checked-in prospective config: file SHA-256
  `a5f0504d...1d8c` and canonical parsed hash `ee76ca0e...1da`. Any unknown or
  changed top-level, `runtime`, or `workflow` field rejects at worker, backend,
  and local-client boundaries.
- The canonical v1 and v2 replay configs need no arbitrary metadata. Both
  contain only `runtime.transformers_version`; tokenizer and generation options
  live in `configs/m2/canonical_full_brief_generation_v1.json`, whose path and
  SHA-256 are independently frozen.
- The complete private-alias/public-tokenizer classifier matrix from tester
  commit `ecb37a4` passes. Public model-token names remain non-sensitive at the
  classifier-unit layer, but injecting them into replay config correctly fails
  the exact config boundary. All private aliases reject.
- OAuth/OIDC `id_token` is explicitly sensitive, while ordered public model
  names such as `image_token_id` remain non-sensitive in defense-in-depth
  classification.
- Fresh boundaries `password`, `passphrase`, `clientAssertion`,
  `sessionCookie`, `oauthClientId`, and `codeVerifier` all reject end-to-end
  even when their individual words are absent from the heuristic vocabulary.
  Exact config identity, rather than an unbounded alias denylist, is the
  decisive credential-free boundary.
- Exact serialization, snapshot/original artifact identity, ordered metadata
  differences, generation authority, canonical generation contract, and
  bidirectional v1/v2 comparison/protocol checks remain green.

## Test evidence

```text
PYTHONPATH=infra:. python -m pytest -q --runxfail \
  experiments/tests/test_m2_fidelity_replay_independent.py \
  experiments/tests/test_m2_fidelity_replay_v2_independent.py \
  experiments/tests/test_m2_fidelity_replay_v2_final_independent.py \
  experiments/tests/test_m2_fidelity_surface_final_independent.py
106 passed in 3.89s

PYTHONPATH=infra:. python -m pytest -q --runxfail
375 passed in 7.34s
```

The forced run executes every previously strict-xfailed blocker as an ordinary
requirement. No test was deselected. Tester scope assertions confirm no
fidelity implementation/config/identity surface changed after the target.

## Disposition

The public CPU-verifiable fidelity-v2 correction passes independent review.
Any actual replay still requires a separate human decision, open
preregistration, normal budget controls, and the already defined launch
workflow; none of those actions occurred here.

# Final fidelity public-surface verification

Date: 2026-07-16 (America/Los_Angeles; completed 2026-07-17 UTC)

Target: `a32ab25181766aff589619942b27526d9778654d`

Verdict: **FAIL — do not preregister, deploy, launch, or spend on fidelity
replay v2 at this commit.**

This was a public, CPU-only verification. No implementation file was edited,
no private or remote artifact was accessed, and no external action or spend
occurred. The pre-existing local `progress/status.json` edit was left untouched.

## What passed

- All 19 cases from the first v2 independent pack pass as ordinary
  requirements. Protocol downgrade, snapshot-hash substitution, false exact
  serialization, and the plain `credential` alias remain closed.
- All 23 cases from the prior final pack pass after rebasing its provenance
  target to `a32ab25`. Exact v1/v2 comparison, protocol, artifact, manifest,
  generation-authority, and historical-v1 config identity guards remain intact.
- The repaired word tokenizer correctly rejects all tested wrapped private
  camel, snake, dotted, and acronym forms, including:
  `remoteServiceUrl`, `privateEndpointUrl`, `externalAuthConfig`,
  `clientSecretValue`, `gatewayAccessTokenValue`, `remoteHTTPSServiceURL`,
  `OPENROUTER_API_KEY`, `oauth2ClientCredentials`, `HF_ACCESS_TOKEN`,
  `privateModelEndpoint`, `client.authorization.header`, `SecretStorePath`,
  `provider_authentication_service`, `refreshTokenValue`,
  `bearer_token_header`, `JWT_TOKEN_VALUE`, `serviceAccountKey`, and
  `xApiKeyValue`.
- These legitimate public tokenizer/generation names pass at all three layers:
  `tokenizer_path`, `tokenizer_config`, `weights_tokenizer_index_identity`,
  `special_tokens_map`, `added_tokens`, `decoder_start_token_id`,
  `forced_bos_token_id`, `forced_eos_token_id`, `begin_suppress_tokens`,
  `suppress_tokens`, `max_new_tokens`, `token_type_ids`,
  `additional_special_tokens`, `specialTokensMap`, `addedTokens`,
  `tokenizerClass`, `clean_up_tokenization_spaces`,
  `additional_special_tokens_ids`, `added_tokens_decoder`, `bos_token`,
  `eos_token`, `pad_token`, `unk_token`, `mask_token`, and
  `generationTokenCount`.
- With only the new tester matrix excluded, the complete forced repository
  suite passes 219 tests.

## Blocking findings

### F1 — Medium: standard public token metadata remains a false positive

The classifier rejects these legitimate public tokenizer/generation fields at
the worker, backend, and client layers:

- tokenizer configuration/properties: `split_special_tokens`,
  `extra_special_tokens`, `all_special_tokens`,
  `all_special_tokens_extended`, and `spaces_between_special_tokens`;
- generation configuration: `token_healing`; and
- multimodal tokenizer metadata: `image_token_id`, `video_token_id`, and
  `vision_start_token_id`.

The problem is structural rather than nine isolated omissions. A key containing
`token`/`tokens` is allowed only when every parsed word belongs to a finite
metadata-word allowlist. Any legitimate new modifier (`split`, `extra`, `all`,
`extended`, `spaces`, `between`, `healing`, `image`, `video`, `vision`) becomes
a credential false positive. Extending this word list one failure at a time is
not a stable public-schema boundary.

### F2 — High: OAuth `id_token` is admitted as public metadata

The nested private alias `id_token` is accepted by all three layers. Its two
words, `id` and `token`, are both in the public metadata allowlist, so the
classifier treats it like a model token-ID field. Word-set membership discards
order and context and therefore cannot distinguish OAuth `id_token` from a
public model field such as `bos_token_id`.

This is a real credential-bearing miss, not merely a usability false positive.
It means the credential-free replay boundary remains bypassable under a common
standard authentication key.

## Required repair

Prefer a strict, versioned allowlist for the replay configuration shape and
the exact public metadata fields admitted at each location. Unknown nested
runtime keys should fail closed. If extensible token metadata must remain,
classify complete ordered names/patterns rather than unordered word sets:

- explicitly deny credential forms such as `id_token`, `access_token`,
  `refresh_token`, bearer/session/JWT tokens, and their camel/acronym variants;
- allow registered tokenizer/generation fields and recognized ordered model
  patterns such as `<model-role>_token_id`; and
- keep worker behavior equivalent to the backend's single shared client/server
  classifier.

The next tester pass should retain this entire positive and negative matrix,
not only add the ten current boundary cases to a modifier-word list.

## Test evidence

Earlier blocker pack, forced to ordinary requirements:

```text
PYTHONPATH=infra:. python -m pytest -q --runxfail \
  experiments/tests/test_m2_fidelity_replay_v2_independent.py
19 passed
```

Prior final pack rebased to the target repair:

```text
PYTHONPATH=infra:. python -m pytest -q --runxfail \
  experiments/tests/test_m2_fidelity_replay_v2_final_independent.py
23 passed
```

New broadened surface matrix under normal marker handling:

```text
PYTHONPATH=infra:. python -m pytest -q \
  experiments/tests/test_m2_fidelity_surface_final_independent.py
35 passed, 10 xfailed
```

The same matrix forced to ordinary requirements reports the exact blockers:

```text
PYTHONPATH=infra:. python -m pytest -q --runxfail \
  experiments/tests/test_m2_fidelity_surface_final_independent.py
10 failed, 35 passed
```

Focused replay/policy/sampler suite including all independent packs:

```text
PYTHONPATH=infra:. python -m pytest -q --runxfail <focused fidelity paths>
10 failed, 148 passed
```

Repository-wide forced regression without only the new tester matrix:

```text
PYTHONPATH=infra:. python -m pytest -q --runxfail \
  --ignore=experiments/tests/test_m2_fidelity_surface_final_independent.py
219 passed
```

Repository-wide forced execution with the new matrix:

```text
PYTHONPATH=infra:. python -m pytest -q --runxfail
10 failed, 254 passed
```

The ten failures are the nine public-field false positives and the admitted
OAuth `id_token`. They occur identically at workflow, backend, and GPU-client
layers, confirming classifier consistency but not correctness.

## Disposition

The artifact-identity and downgrade repair remains sound, and the camel/acronym
tokenizer repair is directionally better. The public-surface classifier is
still both overbroad and bypassable. Keep fidelity replay v2 unauthorized until
F1/F2 are repaired and the full broadened matrix passes as ordinary tests.

# Measurement v2 independent retest report

Date: 2026-07-16 (America/Los_Angeles; completed 2026-07-17 UTC)

Repair commit tested: `b71aaa00152f09a84a51c332bc986a39b1cefd77`

Prior independent-test commit: `099cc78`

Branch: `operator/measurement-v2`

Final disposition: **FAIL — not qualified for protocol transfer, attestation,
prospective scoring, or promotion use**

## Outcome

The repair genuinely closes all eleven qualification defects reported against
`21c1592`. The prior tester pack now runs as thirteen ordinary passing tests:
panel ID cardinality, calibration/baseline/signature requirements, bandwidth
cross-hashes, post-hoc and underpowered promotion blocks, seed binding,
selection structure/BLEU rejection, generated-side reference fingerprints,
instrumented AUC fit counts, and Ed25519 signature requirements all pass.

The repaired boundary is materially stronger. It verifies Ed25519 signatures
against an external trust map, confines artifact paths to the supplied root,
hashes bound files, cross-binds protocol/report identities, requires nested
seed cells, and keeps the checked-in unmaterialized candidate fail-closed.

Qualification still fails because several signed hashes are accepted without
checking whether the referenced bytes satisfy the claimed experiment. Six new
public synthetic adversarial cases demonstrate that the validator accepts:

1. a matched baseline claiming 64 documents whose bound output JSONL contains
   only one row;
2. 192 per-item human `content_sha256` claims that do not match the 192 bound
   human-content records;
3. a power artifact containing six claimed rates but no simulation trials,
   minimally important effects, required prompt-cluster/seed design, or
   coverage-trial evidence;
4. a bare caller-provided `{"status":"pass"}` as historical inventory proof;
5. `hard_gates={"placeholder": true}` as the complete promotion hard-gate
   intersection; and
6. an eligible prospective report with no report/output signature.

All six objects were internally hash-consistent and signed where the current
API requests a signature. The failures therefore isolate missing semantic and
output verification rather than ordinary byte tampering.

## Blind qualification matrix

| Blind group | Result | Retest evidence |
| --- | --- | --- |
| disjoint floor | FAIL end-to-end | Panel IDs and manifest fingerprints are globally unique, but fingerprints are not recomputed from the bound content bundle and the eligibility-attestation hash is not backed by a verified artifact. |
| exact small-matrix oracle | PASS | Independent explicit loops match to `1e-12`, including the unchanged negative unbiased result. |
| kernel freeze | PASS at public interface | Candidate-invariant human-only derivation and common value hash pass; bandwidth artifact/panel hashes are cross-bound. No real bandwidth is materialized in the checked-in candidate. |
| cardinality fail closed | PASS for metric/report panels | 16-vs-32, 2-vs-32, missing counts, duplicate IDs, and empty panel IDs reject. Baseline-output cardinality remains a matched-control failure below. |
| matched control | FAIL | Prompt/brief/sampling/seed/output file hashes are cross-bound, but the output file is never parsed. One output row satisfies a baseline claiming 64 documents. |
| self-BLEU cardinality | PASS | Same n and n-1 references remain enforced. |
| repetition resolution | PASS | n=16 is underpowered/non-promoting, zero events do not fail a lower bound, and high repetition fails. |
| prompt match | PASS at metric interface | Prompt ID, brief hash, SHA-256 reference fingerprint, split, uniqueness, and row-order alignment pass the prior adversarial cases. |
| AUC refit | PASS | Full vectorizer/classifier refits are instrumented; signed AUC/separability, successful refit seeds, actual `.fit()` count, and small-cluster underpowering pass. |
| selection firewall | PASS | Fixed/all/hash/training rules are structurally checked; BLEU and earlier promotion endpoints reject; report training seeds and nested cells bind to protocol. |
| cluster/power | FAIL | Effective prompt clusters are not inflated by sampling rows, but a six-number signed power assertion passes without prospective simulation design/trial evidence. Arbitrary hard-gate names also satisfy the promotion intersection. |
| historical immutability | FAIL at attestation boundary | Direct SHA-256 inventory verification passes all six frozen sets, but `build_attestation` accepts a bare status boolean and emits `historical_inventory_verified=true` without binding the actual inventory-check artifact. |
| no sealed imitation | PASS visible scan | No private embedder identifier, hidden fixture/prompt/per-item output, credential, or sealed evaluator implementation entered the visible repair. |

Result: **9 groups pass and 4 fail. All 13 are required.** Report/output
signature verification is an additional cross-cutting blocker.

## Promotion-critical findings

### F1 — Critical: matched-control output materialization is not verified

`_verify_artifact_bindings` hashes the output file and
`_validate_bound_protocol_content` compares that digest with the matched
baseline/protocol. It never parses the JSONL or verifies row count, unique
prompt IDs, full-brief identity, training/sampling seed cells, decoding policy,
or one output per expected cell. A signed one-row file passes a declared
64-document baseline.

Required repair: define and validate the output-manifest schema; require the
exact prompt × training-seed × sampling-seed grid, unique row IDs, n rows per
cell, and per-row prompt/brief/sampler/checkpoint bindings before readiness.

### F2 — Critical: promotion reports and output decisions are unsigned

The frozen protocol and blind manifest are signed, but the report is not.
`validate_report_v2` therefore accepts caller-authored MMD values, endpoint
decisions, quality-linkage status, authorship fit count, and promotion state as
long as their surrounding metadata matches the protocol. This does not meet
the required operator signature over evaluator output.

Required repair: content-address and verify an Ed25519-signed report/output
payload, bound to the protocol, evaluator image/commit, input/output manifests,
runtime, and analysis artifact.

### F3 — Critical: the hard-gate intersection accepts arbitrary names

Promotion requires a nonempty mapping whose values are all `true`, but no
frozen required gate set is defined. `{"placeholder": true}` passes. The
instrument therefore does not prove factual, adherence, validity, collapse,
or other prespecified hard gates were evaluated.

Required repair: freeze the exact required gate names/versions in the protocol
and require equality with the signed report gate set plus pass evidence for
each gate.

### F4 — High: human content fingerprints are not tied to content bytes

The manifest and content bundle receive independent hashes, and the protocol
is signed, but the per-document fingerprints are not recomputed from the
bundle. The eligibility-attestation field is only checked for SHA-256 syntax;
no eligibility artifact is bound or verified. Arbitrary unique fingerprints
can therefore claim 3n disjoint humans even when unrelated to the content.

Required repair: use a parseable content manifest/bundle contract and recompute
every fingerprint, panel membership, and eligibility/exclusion proof—or bind a
separately signed verifier result that performed those checks.

### F5 — High: power evidence remains aggregate self-assertion

The validator checks numerical thresholds and a nonempty multiplicity method,
but not simulation trial count, frozen minimally important effects, null and
alternative generators, required n/prompt clusters/seed grid, interval method,
blind coverage trials, or a content-addressed simulation output. A signed JSON
with six favorable numbers passes.

Required repair: define the prospective power schema and bind its assumptions,
trial manifest/results, effect sizes, cluster/seed design, multiplicity rule,
and analysis code. Recompute or independently verify the reported aggregates.

### F6 — High: attestation trusts historical inventory status

`build_attestation` checks only `inventory_check.status == "pass"`. It does not
require the inventory-check schema, artifact-set rows, expected/observed
digests, canonical inventory hash, or a verified signature. The returned
attestation then asserts historical verification as true.

Required repair: invoke the inventory verifier over the frozen inventory and
repository root during attestation, or require and verify a signed,
content-addressed inventory-check artifact bound to that inventory.

## Candidate fail-closed verification

The checked-in `measurement_protocol_v2.candidate.json` remains correctly
unmaterialized and non-transferable. CLI validation exits 2 and reports null
hashes, empty panels, unfrozen bandwidth/power/seeds, missing approval,
missing trusted signature, invalid artifact evidence, non-ready state, and the
candidate schema. The checked-in trust store is empty. No real panel,
bandwidth, power, calibration, baseline, output, or signature was created by
this retest.

This operational fail-closed result must remain in force. The positive-path
problem is that a signed, internally hash-consistent but semantically empty
bundle can currently become `ready`.

## Test execution

Prior qualification defects, now ordinary tests:

```text
uv run --extra test pytest -q \
  ../research_reviews/test_measurement_v2_independent_adversarial.py
13 passed
```

New semantic-binding falsification run, with strict-xfail markers overridden
so every body executes as an ordinary requirement:

```text
uv run --extra test pytest -q --runxfail \
  ../research_reviews/test_measurement_v2_retest_adversarial.py
6 failed
```

The failures are the six findings above. Strict xfails keep the repository's
normal suite usable while making each unresolved requirement executable; a
repair will turn them into strict XPASS failures.

Locked harness suite plus both tester packs:

```text
uv run --extra test pytest -q tests \
  ../research_reviews/test_measurement_v2_independent_adversarial.py \
  ../research_reviews/test_measurement_v2_retest_adversarial.py
88 passed, 6 xfailed, 8 warnings
```

Repository-wide component suite:

```text
PYTHONPATH=<repo>:<repo>/harness/src:<repo>/infra pytest -q \
  data/tests experiments/tests harness/tests infra/tests ledger/tests \
  research_reviews/test_measurement_v2_independent_adversarial.py \
  research_reviews/test_measurement_v2_retest_adversarial.py
164 passed, 6 xfailed
```

Additional checks:

- historical inventory CLI: all six frozen sets pass;
- checked-in candidate protocol CLI: expected exit 2/fail-closed;
- target repair path diff: no existing v1 historical artifact changed;
- `git diff --check` and `compileall`: pass;
- no paid/provider calls, data materialization, private access, deployment, or
  merge operation occurred.

The eight warnings remain in the untouched v1 scikit-learn authorship path.

## Runtime manifest

```json
{
  "evaluator_commit": "b71aaa00152f09a84a51c332bc986a39b1cefd77",
  "dependency_lock_sha256": "32f60b643dc4b1799a27ad165f6dc0b203523de1e6a0a81a25f3dc5e07c59dd8",
  "prior_tester_fixture_sha256": "f441973bc12effc83d96a126251d0d46c684c90c4d1698ef94ada532c4cb45eb",
  "retest_fixture_sha256": "a20e3959066854b26ae4887d10023b0cf24279164fbb195c0b44b697d73707fc",
  "python": "3.11.7",
  "numpy": "2.4.6",
  "scikit_learn": "1.9.0",
  "pytest": "9.1.1",
  "cryptography": "49.0.0",
  "platform": "macOS-26.4.1-arm64-arm-64bit",
  "tested_at": "2026-07-17T00:53:08Z",
  "signature_status": "tester_report_unsigned_git_artifact"
}
```

No private fixture-pack hash or signing key was available or used.

## Post-verdict comparison with the implementation report

The implementation report accurately states that the checked-in real-data
state is fail-closed and that all eleven prior tester cases are repaired. Its
claim that the positive fixture proves “content contracts all verify” is too
strong:

- its own positive fixture uses a one-line `control-outputs.jsonl` while
  claiming 64 documents;
- its human bundle strings and per-item `content_sha256` inputs are not the same
  bytes, yet readiness passes;
- it supplies `inventory_check={"status":"pass"}` directly to attestation;
- its power artifact contains aggregate rates without a simulation manifest;
  and
- neither report signature nor a fixed hard-gate name set is validated.

These discrepancies were evaluated only after the retest verdict was frozen.

## Required disposition

Keep v2 quarantined as diagnostic infrastructure. Do not issue a ready
protocol, attestation, prospective report, or promotion claim from `b71aaa0`.
Repair the six strict-xfail cases with semantic content verification and signed
output evidence, then rerun all 13 blind groups with zero failures. The current
unmaterialized candidate must remain fail-closed throughout.

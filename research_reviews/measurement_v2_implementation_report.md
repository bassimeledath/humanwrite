# Operator-owned measurement v2 implementation report

Date: 2026-07-16

Branch: `operator/measurement-v2`

Disposition: implemented prospectively; real-data qualification remains fail-closed

## Outcome

Measurement v2 is implemented as a new, additive namespace. Historical v1 metric code, calibration files, baselines, reports, selection records, and the preserved sealed aggregate were not edited. The checked-in v1 inventory verifies byte-for-byte after installation.

The implementation is deliberately not a promotion-ready result. The visible human pool, three real no-replacement panels, bandwidth array, prospective power simulations, matched current SFT control, trusted operator key, and independent blind-test signature are still unmaterialized. The candidate protocol therefore reports `fail_closed`. Transfer now requires an exact candidate SHA-256, explicit operator approval, and a separately signed frozen protocol whose referenced artifact bytes and content contracts all verify.

No sealed evaluator source, hidden fixture, hidden prompt, private embedder identifier, private per-item output, or paid service was accessed.

## Implemented surfaces

- `harness.metrics.distribution_v2` provides immutable-ID embedding panels, three-panel disjoint/equal-cardinality checks, bandwidths derived only from the union of the two human floor panels, a fixed common kernel, unbiased MMD, deterministic label permutations, prompt-paired candidate/control swaps, cluster intervals, and raw candidate/control/human-floor reporting. Candidate and control are aligned by prompt ID; missing, duplicated, or replaced IDs fail before scoring.
- `harness.metrics.quality_v2` enforces one-to-one prompt/brief/split/SHA-256 fingerprint linkage from both generated and human rows, deterministic pair order, prompt-clustered quality intervals, repeated grouped authorship cross-fitting with the actual fitted-pipeline count and successful refit seeds, signed AUC plus distance from 0.5, small-cluster `underpowered` status, and a structurally validated endpoint-selection firewall.
- `harness.metrics.validity_v2` enforces same-n self-BLEU with n-1 references and a one-sided Newcombe repetition non-inferiority interface. Inadequate n or absent power evidence is `underpowered`/non-promoting; zero candidate events never fail for being below a lower bound.
- `harness.measurement_v2` verifies historical manifests; opens and hashes every protocol-bound artifact; checks panel-content, human-only bandwidth, prompt/brief, power, calibration, matched-baseline, seed-grid, evaluator-image, and dependency-lock semantics; verifies Ed25519 signatures against an external trust store; validates the promotion intersection rule; and emits an attestation only after the historical check and all 13 signed blind groups pass.
- Versioned protocol, report, and attestation JSON Schemas are checked in under `harness/schemas/`.
- Operator artifacts under `harness/measurement_v2/` include the historical inventory/quarantine, fail-closed protocol candidate, unmaterialized prompt/panel/bandwidth/power/calibration/matched-control/selection files, an empty external trust-store template, and an aggregate-only blind-test manifest candidate.

## Independent-test repair

Independent tester commit `099cc78` rejected the first implementation with 11 strict executable qualification failures. The repair closes all 11 without changing the tester-owned cases:

- claimed panel counts with empty ID/content lists fail;
- protocol readiness requires the real artifact root, exact file digests, required artifact schemas, semantic cross-bindings, and a valid Ed25519 signature from an externally trusted key;
- signed blind evidence binds the protocol, evaluator image, dependency lock, signer identity, runtime metadata, timestamp, unique 13-group result set, fixture hash, and no-sealed-imitation assertion;
- report bandwidths, prompt/panel contents, calibration, baseline, dependency lock, evaluator image, full brief, sampling grid, nested seed cells, and selection policy are cross-bound to the verified protocol;
- `post_hoc_shadow`, underpowered repetition, and underpowered authorship cannot be promotion eligible, and an eligible prospective report must satisfy the full endpoint/hard-gate intersection;
- fixed/all-seed selection rules require nonempty deterministic seed declarations, selected seeds must equal report training seeds, and BLEU is prohibited alongside the earlier endpoints;
- generated quality rows must declare the exact prompt-matched human reference fingerprint; and
- authorship `fit_count` is the actual number of `.fit()` calls across every grouped fold, not a count of nominal OOF evaluations.

The public synthetic positive-path fixture generates a temporary Ed25519 key and content-addressed evidence bundle. It proves a complete synthetic protocol/report/attestation can validate, while a one-byte dependency-lock change fails readiness. It contains no real human or private operator data.

## Historical quarantine and verification

The inventory covers six sets: the two harness calibration artifacts, harness baseline, all 98 Tier-1 JSON reports/indexes/summaries, five top-level M1 calibration/baseline artifacts, five seed-29 selection manifests, and the exact sealed aggregate. Each set carries a canonical manifest SHA-256 over sorted `relative_path NUL file_sha256 LF` records.

## Second-round semantic-binding repair

Independent tester commit `3e5b781` confirmed the first repair closed its 11
original defects, then supplied six new signed, internally hash-consistent
falsifications. This repair closes those cases without editing either
tester-owned pack:

- human panel fingerprints are recomputed from the exact `document_id`/`text`
  JSONL bundle, with exact 3n membership and no extra or missing IDs; the same
  rows carry passing eligibility basis and empty exclusion flags whose canonical
  digest must equal the signed panel eligibility attestation;
- matched-control output JSONL must contain the exact prompt x training-seed x
  sampling-seed grid, with unique cells and per-row bindings to the complete
  brief, prompt panel, sampling grid, checkpoint, decoding policy, and
  generation contract;
- the power artifact freezes minimally important effects, null and alternative
  generators, prompt-cluster/document/seed design, at least 1,000 trials per
  required scenario, multiplicity, analysis code, and content hashes, and the
  validator recomputes every reported rate from successes/trials;
- attestation accepts only a trusted-key-signed inventory result that embeds its
  source inventory, then reruns that inventory against the supplied repository
  root and requires exact row-for-row agreement;
- the protocol freezes the exact named/versioned hard-gate set; promotion
  requires equality, pass decisions, and byte-verified gate evidence files; and
- promotion-eligible reports require a trusted Ed25519 signature plus a signed,
  byte-verified candidate-output binding. Any post-signature mutation fails.

The checked-in protocol candidate remains deliberately unmaterialized and now
advertises these fields as null/empty prerequisites rather than satisfying
them with placeholders.

## Third-round hard-gate evidence repair

Independent tester commit `faa2e8e` demonstrated that the signed report could
reuse one plaintext `pass` file as all four hard-gate artifacts. This repair
leaves both tester-owned files untouched and closes that bypass:

- the protocol freezes exactly the factuality, brief-adherence, validity, and
  collapse gate schemas at their required v1 versions;
- every report gate entry has an exact four-field shape and every evidence file
  must parse as an exact three-field JSON object containing the expected
  `artifact_schema`, gate `name`, and `pass` decision;
- resolved evidence paths and byte identities must be distinct across gates, so
  path aliases, copied evidence, shared plaintext, and same-content files fail;
- the validated evidence byte identity is covered by the trusted report
  signature whose payload also contains the independently verified frozen
  protocol hash and candidate-output byte hash, producing an explicit unique
  evaluated identity for each gate; and
- the report and protocol schemas now freeze the same contracts, with a
  dedicated hard-gate evidence schema checked in alongside them.

Implementation tests cover the exact positive shape, wrong/missing/extra
fields, wrong gate/version/decision, non-object and plaintext evidence, outer
entry extras, resolved-path reuse, and copied-byte identity reuse. The
independent valid synthetic promotion remains accepted, while the shared
plaintext placeholder and post-signature candidate-output byte swap fail
closed. The real protocol candidate remains unmaterialized and non-promoting.

Post-implementation verification result: all six sets `pass`. The quarantine permits exact historical reproduction and preserves the original sealed rejection, but forbids using v1 artifacts as prospective v2 floors, calibrations, selection endpoints, uncertainty estimates, hard-gate evidence, or promotion evidence.

## Public invariant coverage

Public tests exercise:

1. explicit-loop small-matrix MMD agreement to `1e-12`;
2. human-only bandwidth invariance and one common hash;
3. duplicate, overlap, replacement, cardinality, and prompt-set rejection;
4. harmless candidate/control row reordering after ID alignment;
5. raw common-kernel candidate/control/floor cells and deterministic permutations;
6. same-n self-BLEU and n-1 references;
7. deterministic underpowered repetition and powered high-repetition failure;
8. prompt-linked quality provenance and row-order invariance;
9. grouped authorship pipeline refits, fold seeds, signed AUC, separability, and underpowered state;
10. rejection of AUC, MMD, JMQ, composite `S`, and other endpoint-driven selection;
11. prompt-cluster effective n rather than sampling-row count;
12. historical manifest pass and changed-byte detection; and
13. all-13-group/no-sealed-imitation attestation gating.

Validation runs:

```text
uv run --project harness --extra test pytest -q harness/tests --runxfail
86 passed, 8 warnings in 4.81s

uv run --project harness --extra test pytest -q \
  harness/tests/test_measurement_v2_hard_gate_evidence.py \
  harness/tests/test_measurement_v2_bindings.py \
  research_reviews/test_measurement_v2_semantic_repair_independent.py --runxfail
22 passed in 0.65s

PYTHONPATH=<repo>:<repo>/harness/src:<repo>/infra harness/.venv/bin/pytest -q \
  data/tests experiments/tests harness/tests infra/tests ledger/tests \
  research_reviews/test_measurement_v2_independent_adversarial.py \
  research_reviews/test_measurement_v2_retest_adversarial.py \
  research_reviews/test_measurement_v2_semantic_repair_independent.py --runxfail
190 passed, 8 warnings in 5.19s
```

Second-round tester pack, executed as ordinary requirements:

```text
uv run --project harness --extra test pytest -q --runxfail \
  research_reviews/test_measurement_v2_retest_adversarial.py
6 passed
```

`--runxfail` executes the tester-owned strict-xfail bodies as ordinary tests. The markers were deliberately preserved; without that flag, repaired strict xfails correctly appear as strict XPASS until the independent tester updates its verdict artifact.

The eight warnings originate in the untouched v1 authorship test path's explicit scikit-learn `penalty` argument. The new v2 path does not emit them.

The inventory CLI was also run after the full suite and returned `status: pass` for every frozen set. JSON candidate/schema artifacts parse successfully, and `git diff --check` is clean.

## Required independent follow-through

Before prospective scoring, a separate operator must materialize at least `3n` eligible unique visible humans, freeze the three disjoint n-sized content-addressed panels with `n >= 64`, freeze human-only bandwidths and all preprocessing/embedder/model/dependency hashes, complete the simulation power and multiplicity artifact, freeze endpoint-independent checkpoint selection and nested seed cells, and materialize an exactly prompt/brief/sampler/seed-matched current SFT control. A trusted operator key must be provisioned outside the protocol bundle. A second operator must run the private synthetic blind pack and return only its signed aggregate manifest.

Until those steps pass, no `measurement_protocol_v2.json` should be issued, no operator attestation can be produced, and no v2 result is promotion evidence. Preserved compatible outputs, if later replayed, must be labeled `post_hoc_shadow`.

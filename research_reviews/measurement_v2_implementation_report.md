# Operator-owned measurement v2 implementation report

Date: 2026-07-16

Branch: `operator/measurement-v2`

Disposition: implemented prospectively; real-data qualification remains fail-closed

## Outcome

Measurement v2 is implemented as a new, additive namespace. Historical v1 metric code, calibration files, baselines, reports, selection records, and the preserved sealed aggregate were not edited. The checked-in v1 inventory verifies byte-for-byte after installation.

The implementation is deliberately not a promotion-ready result. The visible human pool, three real no-replacement panels, bandwidth array, prospective power simulations, matched current SFT control, and independent blind-test signature are still unmaterialized. The candidate protocol therefore reports `fail_closed`, and protocol transfer requires both an exact candidate SHA-256 and explicit operator approval.

No sealed evaluator source, hidden fixture, hidden prompt, private embedder identifier, private per-item output, or paid service was accessed.

## Implemented surfaces

- `harness.metrics.distribution_v2` provides immutable-ID embedding panels, three-panel disjoint/equal-cardinality checks, bandwidths derived only from the union of the two human floor panels, a fixed common kernel, unbiased MMD, deterministic label permutations, prompt-paired candidate/control swaps, cluster intervals, and raw candidate/control/human-floor reporting. Candidate and control are aligned by prompt ID; missing, duplicated, or replaced IDs fail before scoring.
- `harness.metrics.quality_v2` enforces one-to-one prompt/brief/split/fingerprint linkage, deterministic pair order, prompt-clustered quality intervals, repeated grouped authorship cross-fitting with full-pipeline uncertainty refits, signed AUC plus distance from 0.5, small-cluster `underpowered` status, and an endpoint-selection firewall.
- `harness.metrics.validity_v2` enforces same-n self-BLEU with n-1 references and a one-sided Newcombe repetition non-inferiority interface. Inadequate n or absent power evidence is `underpowered`/non-promoting; zero candidate events never fail for being below a lower bound.
- `harness.measurement_v2` verifies historical manifests, validates frozen protocols and v2 reports, binds operator transfer to an expected SHA-256, and emits an attestation only after the historical check and all 13 named blind groups pass.
- Versioned protocol, report, and attestation JSON Schemas are checked in under `harness/schemas/`.
- Operator artifacts under `harness/measurement_v2/` include the historical inventory/quarantine, fail-closed protocol candidate, unmaterialized panel/bandwidth/power/calibration/matched-control files, and an aggregate-only blind-test manifest candidate.

## Historical quarantine and verification

The inventory covers six sets: the two harness calibration artifacts, harness baseline, all 98 Tier-1 JSON reports/indexes/summaries, five top-level M1 calibration/baseline artifacts, five seed-29 selection manifests, and the exact sealed aggregate. Each set carries a canonical manifest SHA-256 over sorted `relative_path NUL file_sha256 LF` records.

Post-implementation verification result: all six sets `pass`. The quarantine permits exact historical reproduction and preserves the original sealed rejection, but forbids using v1 artifacts as prospective v2 floors, calibrations, selection endpoints, uncertainty estimates, or promotion evidence.

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

Validation run from `harness/`:

```text
uv run --extra test pytest -q
70 passed, 8 warnings in 5.10s
```

The eight warnings originate in the untouched v1 authorship test path's explicit scikit-learn `penalty` argument. The new v2 path does not emit them.

The inventory CLI was also run after the full suite and returned `status: pass` for every frozen set. JSON candidate/schema artifacts parse successfully, and `git diff --check` is clean.

## Required independent follow-through

Before prospective scoring, a separate operator must materialize at least `3n` eligible unique visible humans, freeze the three disjoint n-sized panels with `n >= 64`, freeze human-only bandwidths and all preprocessing/model/dependency hashes, complete the simulation power artifact, freeze endpoint-independent checkpoint selection and all seeds, and materialize an exactly prompt/sampler/seed-matched current SFT control. A second operator must run the private synthetic blind pack and return only its signed aggregate manifest.

Until those steps pass, no `measurement_protocol_v2.json` should be issued, no operator attestation can be produced, and no v2 result is promotion evidence. Preserved compatible outputs, if later replayed, must be labeled `post_hoc_shadow`.

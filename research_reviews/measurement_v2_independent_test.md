# Measurement v2 independent blind test report

Date: 2026-07-16 (America/Los_Angeles; completed 2026-07-17T00:30:22Z)

Evaluator commit: `21c15921782faeefa0d64097bc454da169579e00`

Branch tested: `operator/measurement-v2`

Disposition: **FAIL — not qualified for protocol transfer, attestation, prospective scoring, or promotion use**

This verdict was derived from `native_measurement_audit.md` and
`native_phase1_redteam.md`. I did not read
`measurement_v2_implementation_report.md` until after the tests below had run
and the FAIL verdict was frozen. I did not use sealed/private fixtures, hidden
data, provider services, paid calls, deployment, or merge operations. No
implementation source was modified.

## Executive result

The additive v2 metric primitives repair several important v1 defects. The
historical inventory is byte-exact, the explicit-loop MMD oracle agrees to
`1e-12` including a negative unbiased result, bandwidth derivation is human
only, core panel/cardinality checks reject duplicates and overlap, n=16
repetition is non-promoting, same-n self-BLEU is implemented, and grouped AUC
does refit the complete vectorizer/classifier pipeline.

The commit nevertheless fails the blind qualification contract at its trust
boundary. Protocol, report, and attestation validators accept self-asserted
metadata that is not bound to the actual calibration, matched baseline,
bandwidth, panel contents, power artifact, selected training seed, or operator
signature. A post-hoc shadow report and an underpowered-authorship report can
both set `promotion.eligible=true` and pass validation. These are promotion-
critical fail-open paths, so passing primitive math is insufficient.

The checked-in `measurement_protocol_v2.candidate.json` itself does correctly
return `fail_closed` while its fields are null/unmaterialized. That local
sentinel is not enough: a caller can construct a nominally ready dictionary
using arbitrary valid-looking SHA-256 strings and `"pass"` labels, without the
real artifacts, and the public validators accept it.

## Blind-group manifest

| Blind group | Result | Independent evidence |
| --- | --- | --- |
| disjoint floor | PASS | Unique IDs are required within panels; overlap across the three human panels and unequal n are rejected. The implementation uses frozen panels rather than replacement floor draws. |
| exact small-matrix oracle | PASS | Tester explicit loops match to `1e-12`; the independent fixture produces raw MMD-squared `-0.3645944636600923`, which remains negative. |
| kernel freeze | FAIL | Human-only derivation is candidate-invariant, but `validate_report_v2` does not require `distribution.bandwidth_sha256 == hashes.bandwidths_sha256`; an arbitrary candidate-selected kernel hash passes. |
| cardinality fail closed | FAIL | Metric primitives reject 16-vs-32/2-vs-32 and duplicate IDs, but `protocol_readiness` accepts panels claiming `document_count=64` with empty `document_ids`. |
| matched control | FAIL | Prompt-set alignment and harmless row reordering pass in `common_kernel_report`; report validation does not bind the fixed checkpoint seed to `seeds.training`, full-brief bytes, a matched-baseline hash, or nested training/sampling seed cells. |
| self-BLEU cardinality | PASS (primitive) | Same-n panels and exactly n-1 references are enforced; old 32-reference calibration cannot enter this helper. The report schema does not require the result, so this pass does not cure the end-to-end report failures. |
| repetition resolution | PASS | n=16 returns `underpowered/not_promoting`; zero candidate events do not fail a lower bound; powered high repetition fails deterministically. |
| prompt match | FAIL | Prompt ID, brief hash, unique human fingerprint, and split are checked, but the generated row may omit `reference_fingerprint`; an unrelated lookalike reference with copied prompt/brief metadata is accepted. |
| AUC refit | FAIL | Full pipeline refits, grouped folds, signed AUC/separability, and small-n underpowering are present. Instrumentation observed 60 fitted pipelines while the report emitted `fit_count=12`, so the required fit-count audit trail is false. |
| selection firewall | FAIL | `fixed_seed` passes with no seed; `all_preregistered_seeds` passes with an empty seed list; fixed seed 29 is not compared with report training seed 11; visible BLEU ranking is not rejected. |
| cluster/power | FAIL | Cluster helpers count unique prompt IDs rather than repeated sampling rows. However, readiness trusts literal `"pass"` strings and an arbitrary power-plan hash; it does not verify a power artifact, its targets, type-I/coverage results, or multiplicity contract. |
| historical immutability | PASS | All six inventory sets pass their canonical SHA-256 manifests, and the target commit changes no pre-existing v1 artifact path. |
| no sealed imitation | PASS (visible scan) | Diff/source scan found no private embedder identifier, hidden fixture, hidden prompt, hidden per-item output, credential, or sealed implementation. References to the preserved public sealed aggregate and negative test split are expected public provenance. |

Overall: **5 pass/pass-at-primitive-level, 8 fail. All 13 groups must pass, so
attestation is prohibited.**

## Promotion-critical findings

### F1 — Critical: unbound protocol and unsigned attestation can self-qualify

`protocol_readiness` checks shapes, strings, and claimed statuses but does not
require the real calibration or matched-control artifacts at all, does not
require exactly n panel IDs when `document_count=n`, and does not bind the
bandwidth values to the declared bandwidth hash. `build_attestation` accepts a
manifest containing 13 caller-supplied `pass` rows and
`no_sealed_imitation=true`; there is no signature field, signer identity/key,
or signature verification.

This contradicts the required fail-closed state while real panels, bandwidth,
power, calibration, matched baseline, and signature remain unmaterialized.

### F2 — Critical: report validation does not enforce the intersection rule

The validator accepts both:

- `evidence_class="post_hoc_shadow"` with `promotion.eligible=true`; and
- `authorship.status="underpowered"` with `promotion.eligible=true`.

It also does not require pass decisions/power evidence for the distribution,
quality, authorship, repetition, and hard safety/adherence endpoints as one
intersection. Consequently a syntactically valid report can claim promotion
eligibility despite an explicitly non-promoting evidence class or endpoint.

### F3 — High: hashes and matched-design identity are not cross-bound

The top-level bandwidth hash may disagree with the distribution hash. Fixed
checkpoint seed 29 may accompany `seeds.training=[11]`. No calibration hash or
matched-current-SFT-baseline hash is present in the protocol/report validator,
and no candidate/control full-brief hash or nested seed grid is validated.
Valid-looking but unrelated SHA-256 strings therefore satisfy the interface.

### F4 — High: prompt-linked quality can accept an unrelated reference

Human reference fingerprints must be present and unique, but the generated row
is allowed to omit its declared reference fingerprint. An unrelated human row
with the same public prompt ID and copied brief hash then passes alignment. The
reference is not cryptographically bound one-to-one from both sides.

### F5 — High: selection firewall is syntactic and incomplete

Allowed rule names are not validated structurally. Missing/empty seed
declarations pass, report seeds are not compared with the selection manifest,
and a `ranking_metric="bleu"` endpoint passes the token blacklist. This does
not establish deterministic preselection or all-seed aggregation.

### F6 — Medium: AUC fit audit trail understates actual refits

The estimator is genuinely rebuilt for every grouped fold and uncertainty
replicate. However, `fit_count` counts complete OOF evaluations, not fitted
pipelines. The independent instrument observed 60 `.fit()` calls for a case
reported as 12. Fold-seed derivation for uncertainty refits is likewise not
fully enumerated in the returned report.

## Checked-in fail-closed state

Direct CLI validation of the checked-in candidate returned exit code 2 and
listed unavailable panels, null hashes, unfrozen bandwidths, unfinished power,
unfrozen seeds, missing approval, and non-ready status. The unmaterialized
panel, bandwidth, power, calibration, matched-control, and blind-manifest JSON
files contain no real evidence. This is the correct current operational
outcome: **no protocol transfer and no promotion use**.

The weakness is that the validator does not prove materialization; it trusts a
caller changing those assertions to ready/pass. Operator approval and an exact
candidate-file SHA bind only the supplied JSON bytes, not the existence or
contents of all contract artifacts.

## Test execution

Authoritative locked harness environment:

```text
cd harness
uv run --extra test pytest -q tests ../research_reviews/test_measurement_v2_independent_adversarial.py
72 passed, 11 xfailed, 8 warnings in 5.20s
```

The 11 strict xfails are executable qualification failures, not skipped
coverage. They become strict XPASS failures when repaired, signaling that the
tester markers should be removed and the cases promoted to ordinary passing
tests.

Repository-wide component run with explicit package roots:

```text
PYTHONPATH=<repo>:<repo>/harness/src:<repo>/infra pytest -q \
  data/tests experiments/tests harness/tests infra/tests ledger/tests \
  research_reviews/test_measurement_v2_independent_adversarial.py
148 passed, 11 xfailed
```

Additional checks:

- `git diff --check`: pass.
- `python -m compileall -q harness/src/harness`: pass.
- historical inventory CLI: all six artifact sets pass.
- target-commit path diff: no existing v1 calibration, baseline, Tier-1,
  selection, or sealed aggregate file changed.
- checked-in candidate protocol CLI validation: fails closed with exit code 2.
- visible source/diff scan for private/hidden identifiers and credentials: no
  prohibited identifier found.

The eight warnings are from the untouched v1 scikit-learn authorship path.

## Runtime and hashes

```json
{
  "evaluator_commit": "21c15921782faeefa0d64097bc454da169579e00",
  "dependency_lock_sha256": "53edd1f047be6e745fab6c255bc2608d99c7c763e2b5d6bcf256f75f307e7142",
  "tester_fixture_sha256": "dab2876560bd47820d3d1c2f45e20db78b7fc5313adc3d8bb1a1e0514e52c672",
  "python": "3.11.7",
  "numpy": "2.4.6",
  "scikit_learn": "1.9.0",
  "pytest": "9.1.1",
  "platform": "macOS-26.4.1-arm64-arm-64bit",
  "tested_at": "2026-07-17T00:30:22Z",
  "signature_status": "unavailable_unsigned_git_artifact"
}
```

No private fixture-pack hash or signature can truthfully be supplied because
no private fixture pack or signing key was available or used. The listed
tester-fixture hash is for the public tester-only adversarial file.

## Post-verdict comparison with the implementation report

The implementation report correctly says the real-data state is fail-closed
and historical bytes are preserved. Four claims overstate the implemented
boundary:

1. “fingerprint linkage” is optional on generated rows;
2. “other endpoint-driven selection” does not include plain BLEU and allowed
   seed rules need not contain seeds;
3. a “signed aggregate manifest” is described, but no signature is accepted or
   verified; and
4. AUC `fit_count` does not report the actual number of fitted pipelines.

## Required disposition

Keep measurement v2 quarantined as diagnostic infrastructure. Do not issue a
ready protocol, operator attestation, prospective report, or promotion claim
from this commit. Repair the strict-xfail cases, bind and verify every real
artifact and signature, rerun all 13 blind groups under an independently frozen
fixture pack, and require zero failures before qualification.

# Measurement v2 final independent verification

Date: 2026-07-16 (America/Los_Angeles; completed 2026-07-17 UTC)

Target implementation: `fec832f`

Disposition: **PASS — implementation qualifies for prospective operator
materialization; the checked-in candidate remains unqualified and fail-closed**

## Outcome

The hard-gate semantic repair closes the final public bypass reported at
`faa2e8e`. Each promotion gate now requires a distinct, byte-addressed JSON
artifact with the exact three-field schema, correct gate name/version, and pass
decision. Duplicate JSON keys, extra fields, wrong schemas, wrong names, failed
decisions, reused resolved paths, and reused byte identities all reject.

The full signed synthetic promotion path passes. The report signature covers
the exact gate declarations and evidence hashes together with the verified
protocol and candidate-output identities. The validator independently verifies
the protocol hash, candidate-output path and bytes, evidence paths and bytes,
and trusted report signature before constructing unique evaluated identities.
Post-signature changes to the protocol, report, candidate output, or gate
evidence fail closed.

No implementation file was changed during this verification. No protocol was
materialized, preregistered, deployed, launched, or submitted, and no external
service or paid compute was used.

## Independent requirement matrix

| Requirement | Result | Evidence |
| --- | --- | --- |
| Exact gate JSON parsing | PASS | Non-JSON, arrays, missing fields, duplicate keys, and trailing/extra fields reject. |
| Frozen gate set | PASS | Protocol must equal the four exact name-to-version mappings; report keys must equal that set. |
| Name/version/decision semantics | PASS | Every outer entry and parsed evidence object must match its gate and say `pass`. |
| Distinct evidence identity/path | PASS | Same resolved path and same SHA-256 bytes under another path both reject. |
| Protocol binding | PASS | The signed report protocol hash must equal the canonical verified protocol; a separately re-signed protocol mutation rejects. |
| Report binding | PASS | Any unsigned post-signature gate/report mutation fails Ed25519 verification. |
| Candidate-output binding | PASS | Signed path/hash and observed bytes must agree; a post-signature byte swap rejects. |
| Valid synthetic promotion | PASS | A complete trusted-key-signed protocol/report with four distinct exact gate artifacts validates. |
| Earlier semantic repairs | PASS | Baseline grid, human fingerprints/eligibility, power trials, inventory rerun, signatures, and selection tests all remain green. |
| Historical immutability | PASS | All six frozen v1 inventory sets match their file counts and manifest hashes. |
| Unmaterialized candidate | PASS fail-closed | CLI protocol validation exits 2; no real panel, power, calibration, key, or signature is present. |

## Test execution

All three prior independent packs plus the implementation hard-gate regression
pack, with inherited strict-xfail bodies forced to ordinary requirements:

```text
uv run --project harness --extra test pytest -q --runxfail \
  research_reviews/test_measurement_v2_independent_adversarial.py \
  research_reviews/test_measurement_v2_retest_adversarial.py \
  research_reviews/test_measurement_v2_semantic_repair_independent.py \
  harness/tests/test_measurement_v2_hard_gate_evidence.py
39 passed
```

New final tester-owned matrix:

```text
uv run --project harness --extra test pytest -q \
  research_reviews/test_measurement_v2_final_independent.py
11 passed
```

Focused measurement suite, including all independent packs:

```text
uv run --project harness --extra test pytest -q --runxfail \
  harness/tests/test_measurement_v2.py \
  harness/tests/test_measurement_v2_bindings.py \
  harness/tests/test_measurement_v2_hard_gate_evidence.py \
  harness/tests/test_distribution_v2.py \
  harness/tests/test_quality_v2.py \
  harness/tests/test_validity_v2.py \
  research_reviews/test_measurement_v2_independent_adversarial.py \
  research_reviews/test_measurement_v2_retest_adversarial.py \
  research_reviews/test_measurement_v2_semantic_repair_independent.py \
  research_reviews/test_measurement_v2_final_independent.py
76 passed
```

Repository-wide component suite plus every independent pack:

```text
PYTHONPATH=<repo>:<repo>/harness/src:<repo>/infra \
uv run --project harness --extra test pytest -q --runxfail \
  data/tests experiments/tests harness/tests infra/tests ledger/tests \
  research_reviews/test_measurement_v2_independent_adversarial.py \
  research_reviews/test_measurement_v2_retest_adversarial.py \
  research_reviews/test_measurement_v2_semantic_repair_independent.py \
  research_reviews/test_measurement_v2_final_independent.py
201 passed, 8 warnings
```

The eight warnings remain in the untouched v1 scikit-learn authorship path.
Schema JSON parsing and `git diff --check` pass. Historical inventory CLI
reports all six sets `pass`; candidate protocol validation exits 2 as expected.

## Residual operator boundary

This PASS qualifies the public validation implementation, not a scientific
result. Prospective use still requires real eligible human panels, frozen
bandwidth/power/calibration artifacts, an exact matched SFT control and output
grid, trusted operator keys/signatures, and independent blind qualification.
Until those artifacts exist, no ready protocol, attestation, score, or
promotion claim is authorized.

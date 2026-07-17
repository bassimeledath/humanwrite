# Measurement v2 semantic-repair independent test

Verdict: **FAIL**

Target implementation: `22a66521526087f55519943b61bbe3925d6a23d0`

Tester baseline: `3e5b781`

The second-round implementation closes the earlier cardinality, per-item human
content, power-simulation, signed-inventory, and signed candidate-output attacks.
It does not close the hard-gate placeholder boundary. A promotion-eligible,
trusted-key-signed report can bind every required hard gate to the same plaintext
file containing only `pass\n`; the validator accepts it.

## Test method

The existing tester-owned packs were left unchanged and rerun both with normal
pytest marker handling and with `--runxfail` so the six strict-xfail bodies ran
as ordinary assertions. New tester-owned attacks start from a valid signed
synthetic protocol, mutate the target evidence, recompute all affected artifact
hashes and dependent hashes, and re-sign the protocol/report. This prevents an
outer stale digest from standing in for the semantic check under test.

The new attacks cover:

- duplicate control prompt-seed cells with the declared 64-row cardinality;
- two human texts swapped between document IDs while preserving the global text
  set and rebinding the content bundle;
- recomputed power artifacts with a false reported rate, 999 trials, a changed
  minimally important effect, or 65 clusters instead of 64;
- a correctly signed inventory check followed by replacement of the inventoried
  repository file;
- a signed promotion report whose four exact named/versioned gates all point to
  one shared plaintext placeholder; and
- replacement of the bound candidate-output bytes after report signing.

## Results

| Area | Fresh attack | Result |
| --- | --- | --- |
| Baseline cardinality/content | Duplicate one registered prompt cell, omit another, preserve row count, rehash/re-sign chain | Rejected |
| Human per-item binding | Swap text between two IDs, preserve bundle cardinality/content set, rehash/re-sign chain | Rejected |
| Power trial recomputation | Change successes without changing claimed rate | Rejected |
| Power minimum trials | Reduce one scenario to 999 trials | Rejected |
| Power effect semantics | Change an alternative row's effect without changing the prospective effect contract | Rejected |
| Power cluster semantics | Change prospective prompt clusters from 64 to 65 | Rejected |
| Signed inventory rerun | Replace an inventoried file after signing the passing check | Rejected |
| Hard-gate evidence | Bind all exact gate declarations to the same `pass\n` file | **Accepted: bypass** |
| Signed report/output binding | Replace candidate-output bytes after report signing | Rejected |

The bypass is at `harness/src/harness/measurement_v2.py:909-934`. For each gate,
the validator checks the report's name-keyed entry, version, decision, path, and
file SHA-256. It never parses the evidence file or requires the evidence bytes
to declare the expected gate name, schema/version, and pass decision. It also
does not require distinct evidence paths or identities. Consequently, a shared
non-evidence plaintext file satisfies every hard gate as long as its hash is
copied into the signed report.

The promotion boundary should fail closed unless each gate evidence file is
parsed under an exact frozen schema and independently binds at least the gate
name, required version, decision, and evaluated output/report identity. Reusing
one evidence artifact across distinct gates should be rejected unless the frozen
schema explicitly represents and authenticates the complete gate set.

## Commands

Existing first-round pack, normal execution:

```text
uv run --project harness --extra test pytest -q \
  research_reviews/test_measurement_v2_independent_adversarial.py
13 passed in 1.02s
```

Existing second-round pack, normal execution:

```text
uv run --project harness --extra test pytest -q \
  research_reviews/test_measurement_v2_retest_adversarial.py
6 failed: all six are strict XPASS results because the inherited markers remain
```

Second-round bodies executed adversarially as ordinary tests:

```text
uv run --project harness --extra test pytest -q --runxfail \
  research_reviews/test_measurement_v2_retest_adversarial.py
6 passed in 0.90s
```

Implementation-owned measurement regression tests:

```text
uv run --project harness --extra test pytest -q \
  harness/tests/test_measurement_v2.py harness/tests/test_measurement_v2_bindings.py
10 passed in 0.61s
```

Fresh tester-owned semantic attacks:

```text
uv run --project harness --extra test pytest -q \
  research_reviews/test_measurement_v2_semantic_repair_independent.py
1 failed, 8 passed in 0.87s
```

Combined repaired bodies, implementation regressions, and fresh attacks:

```text
uv run --project harness --extra test pytest -q --runxfail \
  harness/tests/test_measurement_v2.py \
  harness/tests/test_measurement_v2_bindings.py \
  research_reviews/test_measurement_v2_independent_adversarial.py \
  research_reviews/test_measurement_v2_retest_adversarial.py \
  research_reviews/test_measurement_v2_semantic_repair_independent.py
1 failed, 37 passed in 0.80s
```

No implementation or existing tester file was edited. No protocol was deployed,
no external job was launched, and no budget was spent.

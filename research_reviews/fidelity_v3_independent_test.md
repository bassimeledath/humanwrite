# Independent prospective fidelity v3 verification

Date: 2026-07-17 (America/Los_Angeles)

Target: `3c082b6ad87b1a84bec06bcff968d6ca031ce3bb`

Verdict: **FAIL — repair the actual tokenizer-surface inventory before
preregistering, deploying, launching, or spending on fidelity replay v3.**

This was a public, CPU-only review. No implementation file was edited, no
checkpoint or private artifact was available locally, and no external action or
spend occurred.

## What passed

- The v3 config has SHA-256
  `71ac41a8cbf8eaa0fc4346e3c87cfa7c6e7ea196eeeb8797d0dba819a3d4405b`
  and canonical object hash
  `82ef89e5f78f205083392ad2a74f3a4795debc5856cd7ce5f7fe906f728fd6b9`.
- The v3 tokenizer manifest has SHA-256
  `54891d4320ee45db4f4ad08124c22b1696410b70210e63f0da5239e3958a7712`.
- The checked-in adapter and original-merge maps contain the same six exact
  shared files and distinct, frozen `tokenizer_config.json` hashes. The
  manifest declares only `tokenizer_config.json` as a metadata difference.
- The JSON exception is exactly four merged-only fields: `max_length: 384`,
  `stride: 0`, `truncation_side: right`, and
  `truncation_strategy: longest_first`. There are no adapter-only or
  changed-value fields.
- Independently rehashed manifest mutations fail closed for missing, added, or
  changed file-map entries; reordered/dropped exact-match lists; expanded or
  duplicated difference-file lists; missing, extra, or changed JSON-difference
  fields; and weakened runtime authority or attestation. Reordering keys within
  a JSON object remains semantically equivalent, as it should.
- Actual JSON-difference recomputation is deterministic across object insertion
  order and identifies added, removed, and changed values.
- The canonical v1, v2, and v3 protocol/comparison pairs pass at worker,
  backend, and GPU-client layers. Every cross-version protocol-only or
  comparison-only substitution rejects at all three layers. Unknown top-level
  and nested v3 fields also reject through the canonical-config binding.
- Behavioral tests show that a mismatch in either exact prompt token IDs or
  attention masks raises before `_run_diagnostics` is called.
- V1 config, v2 config, and v2 snapshot-manifest bytes retain their frozen
  SHA-256 identities:
  `8015afd23f7d21953e0e7f0f1045db824a87377ec38de4c7c478b7455570ef4c`,
  `a5f0504dfdcfd12cfda5081e068919a603a395c7155a725bd9e7c13016ba1d8c`,
  and `602cb05fed6fe3a0ecc1e37bc811ae5bb255c2624b57b051ae0744c7a0973b2c`.

## Blocking finding

### F1 — Medium: the runtime tokenizer file map is not a complete surface

The implementation report describes complete seven-file tokenizer surfaces,
but `_tokenizer_file_map` enumerates only a fixed six-name set and silently
filters every other file. It also omits `chat_template.jinja` from that map
(although the manifest and loaded-template comparison cover that particular
file separately).

An adapter directory containing the seven declared files plus an unmatched
adapter-only `tokenizer.model` produces the same `_tokenizer_file_map` as a
merged directory without `tokenizer.model`. The generic `_verify_file_map`
helper verifies that every declared file exists and hashes correctly, but does
not reject undeclared files. The merged directory's canonical content hash
protects that side; the adapter directory has no equivalent complete-directory
identity, so an added adapter tokenizer artifact is not rejected by file-map
attestation.

The exact prompt ID/mask gate substantially limits scientific impact: if the
extra artifact changes loaded tokenization on the frozen prompts, v3 still
fails before diagnostics. Nevertheless, the claimed complete artifact surface
and adapter/merge map comparison are not exact, and an undeclared recognized
tokenizer artifact should not be admitted by a strict prospective protocol.

## Required repair

Build the runtime tokenizer inventory from the actual tokenizer-related files
in each artifact directory, or bind a complete adapter directory/file-list
manifest and reject undeclared regular files and symlinks. The expected v3
exception can then be applied to the complete inventories: six files must match
byte-for-byte, `tokenizer_config.json` may have only the four frozen semantic
additions, and no unmatched tokenizer artifact may remain. Retain the exact
runtime prompt-ID and attention-mask gate.

## Test evidence

Independent pack with expected blocker marked strict-xfail:

```text
PYTHONPATH=infra:. python -m pytest -q \
  experiments/tests/test_m2_fidelity_replay_v3_independent.py
37 passed, 3 skipped, 1 xfailed
```

The skips are the three canonical protocol/comparison self-pairs in a
cross-product test; those pairs pass in a separate positive test.

V3 implementation and independent packs with strict-xfail forced:

```text
PYTHONPATH=infra:. python -m pytest -q --runxfail \
  experiments/tests/test_m2_fidelity_replay_v3.py \
  experiments/tests/test_m2_fidelity_replay_v3_independent.py
1 failed, 52 passed, 3 skipped
```

Focused fidelity/policy/sampler forced suite:

```text
PYTHONPATH=infra:. python -m pytest -q --runxfail <focused paths>
3 failed, 243 passed, 3 skipped
```

Two focused failures are old tester-only provenance assertions whose historical
targets correctly predate the v3 implementation. The sole semantic failure is
the unmatched actual tokenizer-file case.

Complete forced repository suite:

```text
PYTHONPATH=infra:. python -m pytest -q --runxfail
3 failed, 425 passed, 3 skipped
```

Complete forced suite with only the two stale historical tester-provenance
assertions deselected:

```text
PYTHONPATH=infra:. python -m pytest -q --runxfail \
  --deselect=<v2-final-stale-provenance> \
  --deselect=<surface-final-stale-provenance>
1 failed, 425 passed, 3 skipped, 2 deselected
```

No semantic regression beyond F1 appeared. The real `/checkpoints` artifact
directories were unavailable in this local environment, so this review did not
recompute the two checked-in tokenizer-config file hashes from the remote
artifacts or execute the GPU replay.

## Disposition

The v3 protocol is otherwise tightly and consistently bound and its runtime
attestation ordering is correct. Keep v3 prospective and unauthorized until
the tokenizer-file inventory rejects undeclared adapter-side tokenizer
artifacts and this independent strict-xfail passes as an ordinary requirement.

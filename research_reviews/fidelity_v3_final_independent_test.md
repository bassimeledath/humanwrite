# Final independent fidelity replay v3 verification

Date: 2026-07-17 (America/Los_Angeles)

Target: `8e718dd301cb878f07ccbbf9f73ba9bf17111027`

Verdict: **PASS for the public CPU-verifiable fidelity-v3 implementation and
guards.** This does not preregister or authorize a replay, deployment, provider
call, hidden access, or spend.

This review used only public repository state and synthetic CPU tests. No
implementation file was edited by the tester and no checkpoint/provider
surface was accessed.

## Verified result

- V3 config and tokenizer identity manifest retain their exact frozen byte and
  canonical identities. The adapter and original-merge maps contain the same
  six exact shared files and their distinct declared `tokenizer_config.json`
  hashes.
- The only permitted JSON exception remains exactly four merged-only fields:
  `max_length: 384`, `stride: 0`, `truncation_side: right`, and
  `truncation_strategy: longest_first`. Adapter-only and changed-value maps are
  empty; independently rehashed map/relationship mutations reject.
- Actual tokenizer discovery covers `tokenizer.model`, `spiece.model`,
  `sentencepiece.bpe.model`, arbitrary `tokenizer.*` and `tokenizer_*` files,
  and mixed-case equivalents. `chat_template.jinja` is included in the exact
  fixed tokenizer surface.
- Recognized symlinks and directories fail as non-regular tokenizer artifacts.
  Undeclared recognized regular files expand the observed map and therefore
  cannot equal the exact adapter or merge manifest map. The original
  adapter-only `tokenizer.model` blocker now passes as an ordinary requirement.
- Prompt token-ID and attention-mask equality is checked independently for the
  adapter and merged tokenizers before `_run_diagnostics`; either mismatch
  aborts with no diagnostic call.
- Every canonical v1/v2/v3 protocol-comparison pair is accepted and every
  cross-version protocol-only or comparison-only substitution rejects at
  worker, backend, and client boundaries.
- V1 config, v2 config, and the v2 snapshot manifest retain their frozen
  SHA-256 identities: `8015afd2...0ef4c`, `a5f0504d...1d8c`, and
  `602cb05f...b2c`.

## Fresh boundary coverage

Tester-owned cases added for this pass cover:

- all three standard tokenizer model filenames;
- representative `tokenizer.*` and `tokenizer_*` names, including mixed case;
- symlink and directory rejection for every requested filename family; and
- observed-map expansion for undeclared regular files from every requested
  filename family.

No new semantic blocker was found.

## Test evidence

```text
PYTHONPATH=infra:. python -m pytest -q --runxfail \
  experiments/tests/test_m2_fidelity_replay_v3.py \
  experiments/tests/test_m2_fidelity_replay_v3_independent.py
80 passed, 3 skipped in 1.60s

# All v1/v2/v3 fidelity implementation and independent packs.
PYTHONPATH=infra:. python -m pytest -q --runxfail <all fidelity paths> \
  --deselect=<v2-final-historical-provenance> \
  --deselect=<surface-final-historical-provenance>
222 passed, 3 skipped, 2 deselected in 4.22s

# Complete repository with every strict-xfail body forced.
PYTHONPATH=infra:. python -m pytest -q --runxfail \
  --deselect=<v2-final-historical-provenance> \
  --deselect=<surface-final-historical-provenance>
453 passed, 3 skipped, 2 deselected in 7.67s
```

The three skips are the redundant self-pairs in the v1/v2/v3 cross-product;
each canonical pair passes in the dedicated positive test. The two deselected
tests are tester-only provenance assertions frozen to the earlier v2 target;
their substantive v1/v2 immutability checks ran and passed. The v3 provenance
assertion was retargeted to `8e718dd` and passed.

## Disposition

Fidelity v3 passes independent public CPU verification. Any actual replay
still requires a separate human decision and the normal preregistration,
budget, and launch controls. None of those actions occurred in this review.

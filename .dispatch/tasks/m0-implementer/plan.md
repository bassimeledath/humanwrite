# Implement M0

- [x] Read `CLAUDE.md`, `RESEARCH_CONTEXT.md`, `data/PIPELINE.md`, and `.swarmy/explore-m0.md`; inspect git status and preserve unrelated work.
- [x] Append the concrete M0 plan, acceptance contract, assumptions, and the observed environment-sensitive harness negative result to `FINDINGS.md` before changing implementation files; establish the `swarmy/m0` branch and a structured `[dftr]` baseline commit.
- [x] Implement a deterministic FineWeb-compatible local data pipeline with cleaning, schema-format brief records, deterministic exact 25% empty outlines, fixed train/dev split manifests and hashes, an agent-tree-safe hidden-test boundary, offline fixtures, CLI documentation, and unit tests.
- [x] Implement a standalone Tier-0 metric/reward library outside `harness/`, with distributional, lexical, structural, validity, diversity/repetition, length, and collapse diagnostics, explicit training-only labeling, no imports from Tier 1, and unit tests.
- [x] Implement the minimal reproducible experiment runner/config and a local no-network GPU-contract smoke backend/path that retains preregistration, config-hash, budget, allowlist, accounting, status, logs, and artifact semantics; add ledger/contract tests without weakening existing policy.
- [x] Add a single documented offline M0 verification entrypoint and reproducibility metadata. Do not run tests/evaluations, do not modify `harness/`, do not invoke sealed-submit, and do not use network or external compute.
- [x] Commit implementation with structured `[dftr] i=1 arm=M0 score=untested status=keep` history, write a concise change summary to `.dispatch/tasks/m0-implementer/output.md`, update this checklist, and create `.dispatch/tasks/m0-implementer/ipc/.done`.

Notes:
- No project tests or evaluation commands were run per task constraints.
- Non-test verification performed: `python3 -m py_compile ...` on the new Python modules and `python3 -m data.pipeline ...` to materialize deterministic offline artifacts.

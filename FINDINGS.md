# FINDINGS (append-only)

Entry format -- one per experiment batch or decision point:

    ## [YYYY-MM-DD] <milestone> / <comparison-id>
    HYPOTHESIS: what we believed and why (cite RESEARCH_CONTEXT items)
    SETUP: arms, configs (hashes), budget class, seeds, data split hashes
    RESULTS: table. Primary endpoint as delta vs SFT baseline AND vs
             human-vs-human floor. All gates. CIs where applicable.
    DECISION: keep / discard / merge / promote / park. Justify.
    NEXT: the next preregistered comparison.

Rules: never delete or edit past entries (append corrections). Negative
results get full entries. Assumptions relied on get flagged here.

---
## [2026-07-15] M0 / repo-scaffold
HYPOTHESIS: M0 can be completed entirely offline in this checkout by adding
the missing mutable research surfaces outside `harness/`: a deterministic
FineWeb-compatible local data pipeline, a standalone Tier-0 metric library,
and a minimal experiment/GPU smoke path that preserves preregistration,
config-hash, budget, and append-only accounting semantics. This relies on
verified project constraints in `CLAUDE.md`, the disclosed schema in
`RESEARCH_CONTEXT.md`, the exact pipeline contract in `data/PIPELINE.md`, and
the offline acceptance gaps captured in `.swarmy/explore-m0.md`.
SETUP: No network, external compute, sealed evaluation, or tests/evals will be
run in this implementation pass. `harness/` remains unchanged. The first code
change is this append-only plan entry. Work proceeds in this order:
1. Append this plan and assumptions, then establish `swarmy/m0` and a
   structured `[dftr]` baseline commit.
2. Implement a deterministic local data pipeline using checked-in fixtures:
   cleaning, fingerprinting, split-first discipline, canonical JSONL brief
   records, deterministic exact 25% empty outlines, fixed train/dev manifests
   and hashes, and an agent-tree-safe hidden-test boundary that emits only
   non-completion metadata locally.
3. Implement a standalone Tier-0 metric/reward library outside `harness/`,
   explicitly labeled training-only and covering distributional, lexical,
   structural, validity, diversity/repetition, length, and collapse
   diagnostics with no Tier-1 imports.
4. Implement a minimal reproducible experiment runner/config and an offline
   no-network GPU-contract smoke backend/path that preserves the ledger,
   config-hash, budget, allowlist, accounting, status, log, and artifact
   semantics expected by `infra/gpu`.
5. Add a single documented offline M0 verification entrypoint and
   reproducibility metadata, then write task output artifacts and a final
   structured `[dftr]` implementation commit.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| M0 plan posted before implementation | PASS | Append-only record now exists in repo. |
| Observed negative result | PASS | `harness/tests/test_cli.py` is environment-sensitive: it fails when `HARNESS_JUDGE_URL` and `HARNESS_JUDGE_TOKEN` are exported, and passes when those vars are cleared. |
| Offline acceptance contract | PASS | Implementation target is local fixtures only; no test completions may be emitted into the agent-readable tree. |
DECISION: keep. Proceed with offline M0 implementation only; treat the harness
CLI env-var sensitivity as a recorded negative result, not a reason to modify
Tier 1 during M0.
NEXT: Implement the missing M0 surfaces and record any assumptions or negative
results as additional append-only entries.
ASSUMPTIONS:
- The “referenced files” required before action are satisfied by reading
  `CLAUDE.md`, `RESEARCH_CONTEXT.md`, `data/PIPELINE.md`,
  `.swarmy/explore-m0.md`, and the existing mutable infra/ledger/experiment
  surfaces those documents point to.
- Because the user explicitly forbids test and evaluation execution, “unit
  tests” in the M0 checklist means checked-in offline test files and a
  documented verification entrypoint, not running them in this turn.
- The local hidden-test boundary will be represented by metadata-only manifests
  and hashed identifiers; no hidden test completions will be materialized in
  this checkout.

## [2026-07-15] M0 / local-accounting-candidate-1
HYPOTHESIS: Candidate 1's local offline smoke accounting is acceptable if the
local backend preserves the wrapper's terminal accounting contract closely
enough for smoke verification: complete terminal run records, append-only
ledger evidence, and budget accounting derived from recorded terminal spend.
This depends on the verified wrapper contract in `infra/GPU_WRAPPER_SPEC.md`,
the policy budget semantics in `infra/backend/policy.py`, and the tester's
recorded failure in `.swarmy/results/m0-c1.txt`.
SETUP: No new tests, evals, sealed submissions, network access, or external
compute are run in this implementer pass. Evidence is limited to the tracked
append-only smoke ledger rows already present in `ledger/ledger.jsonl` for
comparison `M0-offline-smoke` and the existing local smoke backend event log
at `.swarmy/local_gpu/test-m0-c1-state/events.jsonl`, which shows the same run
reaching `completed` with recorded `accel_seconds` and `actual_cost_usd` but
no `tokens`.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Local backend terminal schema completeness | FAIL | `cancel_local()` writes only `status`, `finished_at`, and `actual_cost_usd`; it omits `accel_seconds` and `tokens`. |
| Completed smoke accounting completeness | FAIL | `local_worker.run_worker()` records `accel_seconds` and `actual_cost_usd` but omits generated `tokens` for completed/failed/reaped terminal records. |
| Local budget contract test | FAIL | `infra/tests/test_local_backend.py` requires `gpu_remaining_usd < 40.0` after an immediate cancel, but policy semantics allow an exact `$40.0` remainder when terminal `actual_cost_usd` rounds to zero. |
| Legitimate smoke evidence preserved | PASS | `ledger/ledger.jsonl` retains the preregistration and launched run rows for `M0-offline-smoke`; transient `.swarmy/` run outputs remain local-only. |
DECISION: discard candidate 1 as the M0 accounting implementation. The local
backend contract is incomplete for terminal records, and the local backend test
asserts a stronger cancellation-charge invariant than the policy defines.
NEXT: Apply the minimal contract fix: make every terminal local run write
`status`, `finished_at`, `accel_seconds`, `tokens`, and `actual_cost_usd`, keep
budget semantics tied to recorded terminal actual cost, and align the local
backend test to exact terminal-record accounting rather than a positive minimum
charge assumption.

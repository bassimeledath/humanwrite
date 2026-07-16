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

## [2026-07-15] M0 / milestone-result
HYPOTHESIS: M0 is complete if the append-only record shows the offline repo
scaffold requirements are satisfied without mutating Tier 1, candidate 1's
accounting defect is explicitly rejected, candidate 2's offline tester pass is
preserved as the acceptance result, and the remaining limitations are recorded
instead of being blurred into a stronger claim. This relies on the immutable
Tier-1 boundary in `CLAUDE.md`, the offline-only milestone scope already
documented for M0, and the tester artifacts in `.swarmy/results/m0-c1.txt`,
`.swarmy/results/m0-c2.txt`, and `.swarmy/logs/test-m0-c2.log`.
SETUP: Recorder pass only. No tests, evals, remote judge calls, network,
compute submission, or sealed validation are run in this turn. Evidence is
limited to the existing append-only records already present in the checkout:
prior M0 findings, the tracked `ledger/ledger.jsonl` delta produced by the
passing tester, and the tester's offline verification log showing candidate 2
passing with `harness/` unchanged.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Candidate 1 accounting acceptance | FAIL | `.swarmy/results/m0-c1.txt` records the local backend accounting mismatch: cancel-path spend stayed at `$40.0` remaining, so candidate 1 is not the M0 milestone result. |
| Candidate 2 milestone acceptance | PASS | `.swarmy/results/m0-c2.txt` records `score=pass`; `.swarmy/logs/test-m0-c2.log` shows offline smoke-path acceptance evidence and leaves only append-only `ledger/ledger.jsonl` rows as tracked delta. |
| Tier-1 immutability | PASS | Tester evidence shows no M0 `harness/` diff and keeps Tier 1 as an immutable boundary rather than part of the implementation surface. |
| Offline-only recorder scope | PASS | This recorder turn adds only documentation/finalization artifacts and preserves the tester's ledger rows without running tests, evals, network, compute, remote judge, or sealed-submit. |
| Remaining limitations | PASS | M0 remains an offline local scaffold milestone only: no remote GPU execution, no Tier-2 sealed validation, and no claim beyond the checked-in smoke/backend/data-contract evidence. |
DECISION: keep and record M0 as passed on candidate 2. Preserve the existing
append-only ledger delta exactly as produced by the passing tester, append this
milestone result, and park for human sign-off.
NEXT: Wait for human sign-off before any M1 work. Do not extend claims past
offline M0 evidence or rewrite prior ledger/findings history.

## [2026-07-15] M1 / milestone-plan
HYPOTHESIS: A reproducible Qwen3-1.7B SFT baseline, evaluated across a
preregistered sampler sweep and independent sampling seeds, will reproduce a
measurable SFT-to-human distributional gap while preserving the fixed validity
gates. Freezing the sampler from the joint distributional/quality evidence,
rather than from any single detector or reward representation, will make the
M1 baseline a defensible control for M2. This relies on the verified Qwen3
instruct-family setup and fixed-sampler observation in `RESEARCH_CONTEXT.md`;
it makes no claim of human indistinguishability.
SETUP: M0 has human sign-off. M1 is bounded to SFT baseline reproduction,
sampler selection, and calibration estimation. `harness/` (including metric
definitions and `harness/calibration.json`), fixed data splits/manifests,
source snapshots, prior `FINDINGS.md` entries, and prior ledger rows are
immutable. No Tier-2 sealed submission, Tier-3 evaluation, external provider,
direct accelerator, hidden data, or route outside the documented infra
contract may be used. Qwen3-0.6B is allowed only for a single plumbing smoke;
all evidentiary baseline results must come from Qwen3-1.7B. The execution order
is preregistered as follows:
1. Run a read-only explorer pass to inventory the checked-in M1-capable
   training/sampling configs, fixed split hashes, M0 contract evidence, budget
   state, harness CLI interface, and current git/ledger state. Record gaps or
   any extra compute/evaluation route as an append-only finding; do not use
   such a route. Establish a dedicated M1 research branch and preserve any
   pre-existing user changes.
2. Have a separate implementer add only the minimal mutable M1 configs and
   orchestration needed outside `harness/`. Pin the Qwen3 instruct checkpoint,
   data-manifest hashes, optimizer/LoRA settings, seeds, maximum lengths,
   sampler grid, artifact schema, and software/config hashes. Use the existing
   fixed M0 data; do not regenerate or alter splits. The sampler grid must vary
   only decoding controls exposed by the constrained wrapper, include the
   existing/default setting, and hold prompt schema, weights, examples, and
   output token budget fixed within each controlled comparison.
3. Before every launch, call `ledger/ledger.py add` with hypothesis,
   comparison, config hash, seed, and budget class. First check
   `infra/gpu budget`; then run at most one registered Qwen3-0.6B plumbing job
   under `smoke` if the remote M1 path has not already been demonstrated.
   Submit, inspect, and account for it only through `infra/gpu
   submit|status|logs|cancel`. A plumbing failure is a stop condition for
   evidentiary training, not permission to bypass the wrapper.
4. Train the Qwen3-1.7B SFT baseline through registered `screen` jobs. Use the
   preregistered training seeds selected in the checked-in config; retain each
   checkpoint/config hash and terminal accelerator-seconds/generated-token
   accounting. If the available budget cannot support the preregistered
   baseline set, record the shortfall and park rather than reducing the design
   after seeing results.
5. Generate the sampler-sweep samples only through registered wrapper jobs.
   For every eligible SFT checkpoint, evaluate the same fixed dev prompts at
   every sampler-grid point using preregistered sampling seeds. Run only
   `harness eval <ckpt_or_samples>` for Tier-1 screening. Report the fixed
   primary endpoint as delta versus SFT/default and versus the independently
   computed human-vs-human finite-sample floor, plus every hard gate, quality
   preference, authorship probe, diversity/repetition, and length statistic
   exposed by the immutable harness. Keep training-seed variance separate from
   sampling-seed variance and never cite Tier-0 reward as evidence.
6. Freeze exactly one deployment sampler using a preregistered rule: among
   settings that pass every hard validity gate and all existing immutable
   calibration constraints, choose the lowest mean primary gap; break a tie
   inside uncertainty in favor of the existing/default sampler, then lower KL
   drift, then the simpler/lower-temperature setting. Do not select on
   Rosmine-exact JMQ, a detector, an authorship probe, or the training
   representation alone. Commit the frozen sampler and hashes as an M1
   artifact; do not modify it after inspecting future M2 outcomes.
7. Compute proposed human-calibrated intervals from independent human dev
   subsets with fixed resampling seeds and checked-in code outside `harness/`.
   Produce a review artifact containing point estimates, interval method,
   confidence level, sample counts, split hashes, resampling seeds, and
   sensitivity to subset draw. These are proposals for the human to transfer
   into `harness/calibration.json`; the agent must not write that immutable
   file.
8. Dispatch an independent tester with no implementation rationale. The tester
   must verify config/ledger hash agreement, preregistration-before-launch,
   budget/accounting completeness, checkpoint provenance, fixed-split
   integrity, absence of `harness/` changes, sampler-sweep completeness,
   independent sampling, metric provenance, calibration reproducibility, and
   the frozen-sampler decision rule. Raw logs go under `.swarmy/logs/`; the
   concise verdict goes under `.swarmy/results/` with
   `score=pass|fail` and explicit failed checks.
9. After each experiment batch, append hypothesis -> setup -> full results ->
   decision -> next to this file, including negative results. Record structured
   `[dftr] i=<N> arm=SFT score=<...> status=<...>` commits as the scientific
   narrative while retaining `ledger/ledger.jsonl` as the compute registry.
   Finish with an M1 milestone summary and park for human sign-off; do not
   start Arms A-E, length-curriculum comparisons, LoRA staging ablations,
   sealed validation, or any M2 work.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| M0 human sign-off | PASS | Explicitly supplied by the user for this turn. |
| M1 plan recorded before implementation/compute | PASS | This append-only entry is the first M1 mutation. |
| Evidentiary acceptance criterion | PREREGISTERED | Independent tester passes all provenance, immutability, accounting, completeness, and reproducibility checks; 1.7B SFT-vs-human gaps and uncertainty are reported without overclaiming; one sampler is frozen; calibration proposals are ready for human transfer. |
| Milestone stop condition | PREREGISTERED | Park after the M1 summary for human sign-off, whether M1 passes or is blocked. |
DECISION: keep. Execute the bounded M1 plan through independently verified
evidence, subject to the wrapper budget and immutable boundaries.
NEXT: Dispatch the read-only M1 explorer, then implement and execute only the
preregistered M1 surfaces if the explorer confirms the constrained path.
ASSUMPTIONS:
- The checked-in repository contains or can support the M1 training and
  sampling schema without changing Tier 1; otherwise record a blocker instead
  of inventing an unapproved route.
- Exact seed values, model revision, sampler-grid values, maximum lengths, and
  interval procedure must be pinned in versioned configs before their first
  associated launch. They may be chosen from existing repository conventions
  during the read-only explorer/implementer pass, but not adapted after seeing
  experimental outcomes.

## [2026-07-15] M1 / repository-readiness
HYPOTHESIS: The signed-off M0 checkout contains enough constrained surfaces to
instantiate M1 without touching Tier 1, and any missing M1-specific artifacts
can be added outside `harness/` before preregistered compute.
SETUP: A role-separated read-only explorer inspected the directly relevant
contracts and ran only `infra/gpu budget`. It did not train, submit, test,
evaluate, access providers directly, change git state, or modify any protected
surface. Fixed visible split hashes were independently read as
train=`c59c853cdc03c7378308c8f35baa874e0f484fa035d4297948c3f3b755afa1a6`
and dev=`69dded207ccb2a7753666752ebcbdaee0e00260bf9848817979e50427bb2cf8b`.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Approved budget route | PASS | `infra/gpu budget` reports GPU `$40.0` remaining and API `$99.999337` remaining. |
| Successful remote M1 plumbing evidence | FAIL | Only local M0 smoke evidence exists; the sole tracked non-M0 remote run is `launch_failed`. |
| Executable checked-in M1 configs | FAIL | No 0.6B plumbing, 1.7B SFT, sampler-sweep, seed-grid, or independent-verification config is checked in. |
| Model revision provenance | FAIL | Current examples pin only `model.base`; remote `snapshot_download` resolves an unrecorded live revision. |
| Tier-1 readiness without mutation | FAIL | `harness/baseline_stats.json` is absent, while immutable `harness/calibration.json` and `harness/deployment_sampler.json` remain fail-closed placeholders. |
| Fixed data boundary | PASS | M0 manifests and hashes are present; hidden completions remain unmaterialized. Visible fixture counts are train=6 and dev=2, which must be reported as a limitation of any M1 estimate. |
| Extra route concern | WARN | The approved remote wrapper clones live GitHub at the ledger `git_sha`; this is inside the wrapper but creates availability/provenance dependence on the commit being pushed. Do not bypass it. |
DECISION: keep the legal path but do not claim the checkout is already M1-ready.
Add minimal mutable M1 runner/config/provenance artifacts, resolve the model
revision through a registered wrapper task before evidentiary training, and
generate samples outside Tier 1 for `harness eval`. Because the immutable
calibration and deployment sampler files are intentionally fail-closed, the
agent may only produce proposed baseline/calibration/frozen-sampler artifacts
for human transfer; it may not make the harness accept them during M1.
NEXT: Dispatch a separate implementer to create and preregister the bounded M1
surfaces without launching or evaluating. Review and commit exact seeds, grid,
revision-resolution flow, and config hashes before the first wrapper submit.

## [2026-07-15] M1 / constrained-workflow-candidate-1
HYPOTHESIS: A wrapper-only M1 workflow can safely reach the first remote
plumbing gate if it locks evidentiary work to Qwen3-1.7B, confines Qwen3-0.6B
to revision-resolution plumbing, and fails closed on every unresolved revision,
checkpoint, Tier-1 report, or preregistration.
SETUP: Candidate commit `3f778274b87a42ecfb4c1259caa0243bae079a39`
adds M1 configs and workflow/analysis code outside `harness/`. Fixed hashes are
train=`c59c853cdc03c7378308c8f35baa874e0f484fa035d4297948c3f3b755afa1a6`
and dev=`69dded207ccb2a7753666752ebcbdaee0e00260bf9848817979e50427bb2cf8b`;
training seeds are `[11,29,47]`; sampling seeds are `[101,202,303]`; the
five-point sampler grid includes default `(temperature=1.0, top_p=1.0)`.
Only comparison `M1-plumbing-revision-resolve-qwen3-0p6b` was preregistered,
after config hash
`8f0be62b88999d23143946f3c6dbf8db50d03e7eebf8032f924eb5fa1809f930`
was frozen. A separate tester received the commit and acceptance contract but
not the implementer's strategy or logs. The tester ran offline policy/local
backend checks and temp-file assertions only; it launched no job and ran no
Tier-1/Tier-2 evaluation.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Independent pre-compute verdict | PASS | `.swarmy/results/m1-precompute.txt` reports `score=pass`. |
| Protected surfaces | PASS | Candidate diff does not touch `harness/`, `sources/`, or fixed M0 data artifacts/manifests; the ledger change is one append-only prereg row. |
| Infra route confinement | PASS | Runtime remains behind `infra/gpu` and allowlisted `python -m experiments.runner`. |
| Config/ledger agreement | PASS | Resolver config hash exactly matches the sole open M1 preregistration. |
| Model boundary | PASS | 0.6B is resolver plumbing only; evidentiary SFT is locked to Qwen3-1.7B. |
| Fail-closed provenance | PASS | Unresolved 1.7B revisions, checkpoints, reports, and missing preregistration are rejected. |
| Offline checks | PASS | Policy and local-backend tests plus focused static/temp assertions passed without eval/provider access. |
DECISION: keep candidate 1 and permit only the preregistered 0.6B resolver
smoke as the next action. Do not launch 1.7B training or sampler generation
until the resolver result is recorded, exact revisions are pinned, configs are
rehash-preregistered, and a separate verification pass accepts them.
NEXT: Push the candidate commit required by the approved remote clone path,
check `infra/gpu budget`, submit the exact resolver config under `smoke`, and
record terminal accounting/provenance or the failure without bypass.

## [2026-07-15] M1 / remote-provenance-blocker
HYPOTHESIS: The independently accepted M1 candidate can enter the single
preregistered resolver smoke once its exact git SHA is available to the
approved remote wrapper clone path.
SETUP: Local candidate HEAD after recording pre-compute verification was
`a9764d8` on `agent/m1`, ahead of `origin/agent/m1`. The required normal push
was attempted with `git push origin agent/m1` before any project-compute
submission. After it failed, read-only checks inspected the configured remote
and `gh auth status`; no alternative compute/data/evaluator route was used.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Exact-SHA publication | FAIL | HTTPS push failed: Git could not read a GitHub username in this environment. |
| Existing GitHub CLI credential | FAIL | Active account exists, but `gh auth status` reports its token invalid and requires human re-authentication. |
| Resolver submission | NOT RUN | The wrapper clones and checks out the ledger SHA; submitting an unpublished commit would be a known provenance/launch failure. |
| GPU/API spend in this batch | PASS | No project compute, provider call, Tier-1 eval, or Tier-2 submission was made. |
| Boundary preservation | PASS | No direct provider, alternate git host, bundle injection, local accelerator, or wrapper bypass was attempted. |
DECISION: park as externally blocked rather than manufacture a launch failure
or relax provenance. Candidate implementation and its independent pre-compute
`score=pass` remain valid, but M1 scientific deliverables are incomplete: no
successful resolver, no pinned 1.7B revision, no SFT checkpoints, no sampler
sweep/freeze, and no calibration proposal from experimental outputs.
NEXT: Run an independent milestone-boundary audit, append its verdict, and
wait for human sign-off/direction. Resumption requires a valid credential that
can publish the exact M1 commit to the configured origin; after publication,
restart at `infra/gpu budget` and the already-preregistered resolver smoke.

## [2026-07-15] M1 / milestone-result
HYPOTHESIS: M1 may be recorded as complete only if independently verified
repository state contains the actual constrained-route 1.7B baseline,
sampler-sweep/freeze evidence, and reproducible human-calibration proposal;
safe workflow scaffolding alone is insufficient.
SETUP: A fresh boundary tester audited current `agent/m1` HEAD, immutable
contracts, git history/diff, ledger state, and checked-in artifacts without
reading implementation/explorer logs. It made no source/config/data/harness/
ledger/findings changes, launched no compute, and ran no Tier-1/Tier-2 eval.
The fixed score format was `score=pass|fail`, with failure required when
scientific M1 deliverables were incomplete even if safety checks passed.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Independent milestone verdict | FAIL | `.swarmy/results/m1-boundary.txt` reports `score=fail`. |
| Boundary immutability | PASS | No branch diff under `harness/`, `sources/`, or fixed M0 artifacts; findings and ledger history are append-only. |
| Commit provenance | PASS | M1 branch has five structured `[dftr]` commits above the published branch point at audit time. |
| Publication blocker | CONFIRMED | Audit observed local HEAD `9f8b999` ahead of published `fc880fe`; exact candidate SHA is unavailable to the wrapper clone path. |
| Post-prereg compute/eval activity | PASS | Ledger ends with the resolver preregistration; no M1 run/update row or Tier-1/Tier-2 result follows it. |
| Successful remote plumbing | FAIL | Not run because exact-SHA publication failed. |
| Pinned Qwen3-1.7B baseline/checkpoints | FAIL | No resolved immutable 1.7B revision or checkpoint provenance exists. |
| Sampler sweep and frozen sampler | FAIL | No Tier-1 sampler reports or output-derived freeze artifact exists. |
| Human calibration proposal | FAIL | Checked-in JSON is a fail-closed template pointing at missing reports, not an output-derived interval proposal. |
DECISION: park M1 as safe but incomplete and externally blocked. Do not
self-approve, do not reinterpret the pre-compute `score=pass` as milestone
acceptance, and do not start M2. The verified candidate remains the resume
point; no scientific conclusion about SFT-vs-human gaps or sampler quality is
supported by this milestone state.
NEXT: Wait for human sign-off/direction at the M1 boundary. To resume M1, the
human must restore a valid GitHub credential (or otherwise authorize an exact-
SHA publication path within the existing origin contract). Then publish HEAD,
recheck `infra/gpu budget`, submit only the already-preregistered 0.6B resolver
smoke, record it, pin/reverify/preregister the 1.7B configs, and continue the
remaining M1 plan. No M2 work is authorized.

## [2026-07-15] M1 / operator-authorized-resumption
HYPOTHESIS: M1 can safely resume at the already-preregistered Qwen3-0.6B
resolver smoke when the accepted scientific candidate remains published and
unchanged in history, and the current cloneable branch tip differs only by an
operator-owned progress record that explicitly authorizes automatic M1
resumption.
SETUP: The human operator explicitly authorized M1 to resume after publishing
scientific candidate `7fb2e98af7ae0734007267368cffb73209cbe9ac`. A PI audit
accepted current local and `origin/agent/m1` tip
`7f5a16691c4b42065b77d080d2afd17688314835` as a cloneable descendant of that
candidate; `git diff --name-status 7fb2e98..7f5a166` contains only
`M progress/status.json`. The worktree was clean before this append-only entry.
No prior blocker, milestone audit, finding, ledger row, config, protected
surface, or scientific candidate commit was rewritten.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Human authorization | PASS | Explicit authorization resumes M1 only; M2 remains forbidden. |
| Scientific candidate publication | PASS | Exact candidate `7fb2e98` remains an ancestor unchanged in published history. |
| Cloneable branch boundary | PASS | Local HEAD and local `origin/agent/m1` both resolve to progress-only descendant `7f5a166`. |
| Resumption scope | PASS | Continue only with the existing Qwen3-0.6B resolver config and `smoke` budget class through the constrained wrapper. |
DECISION: resume the bounded M1 batch without reinterpreting or deleting the
prior blocker and failed milestone result. Treat `7fb2e98` as the accepted
scientific candidate and `7f5a166` as the current cloneable progress-only tip.
NEXT: Run `infra/gpu budget`; only if sufficient, submit exact config
`configs/m1/m1_plumbing_revision_resolve_qwen3_0p6b_v1.yaml` with
`--budget-class smoke`, then monitor only with `infra/gpu status|logs|cancel`
to a terminal state and record full provenance/accounting without bypass.

## [2026-07-15] M1 / resolver-smoke-qwen3-0p6b
HYPOTHESIS: The independently accepted wrapper-only plumbing path can clone the
published M1 tip, resolve the requested Qwen3-0.6B upstream `main` revision to
an immutable commit, and return complete terminal provenance and accounting
within the preregistered smoke budget without using an alternate route.
SETUP: After `infra/gpu budget` reported GPU cap/remaining `$40.00`, GPU
committed `$0.00`, and API remaining `$99.999337`, exact unchanged config
`configs/m1/m1_plumbing_revision_resolve_qwen3_0p6b_v1.yaml` was submitted once
with `--budget-class smoke`. Run ID is `dftr-1784177307-97064e4f`; config hash
is `8f0be62b88999d23143946f3c6dbf8db50d03e7eebf8032f924eb5fa1809f930`;
clone git SHA is progress-only published tip
`7f5a16691c4b42065b77d080d2afd17688314835`. Fixed split hashes are
train=`c59c853cdc03c7378308c8f35baa874e0f484fa035d4297948c3f3b755afa1a6`
and dev=`69dded207ccb2a7753666752ebcbdaee0e00260bf9848817979e50427bb2cf8b`.
The run used one L4, `smoke`, and a 1,200-second timeout. Monitoring used only
`infra/gpu status` and `infra/gpu logs`. The approved append-only ledger CLI
recorded the terminal row after the gateway returned terminal accounting.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Submission/provenance gate | PASS | One authorized run; preregistered config hash, comparison, clone SHA, model, and split hashes agree. |
| Terminal execution | PASS | `completed`, return code 0; started `1784177307.8026717`, finished `1784177323.6196373`. |
| Immutable model resolution | PASS | `Qwen/Qwen3-0.6B` requested `main` resolved to `c1899de289a04d12100db370d81485cdf75e47ca`. |
| Artifact provenance | PASS | Resolver artifact: `/__modal/volumes/vo-EY2fT0CaoNDuXGLZLNZcGg/runs/dftr-1784177307-97064e4f/resolved_revision.json`; snapshot: `/checkpoints/hf-cache/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca`. |
| Terminal accounting | PASS | 11.589 accelerator-seconds, `$0.003087` actual GPU cost, 0 generated/train/total tokens; `$0.31968` was reserved. |
| Data limitation | PASS | Resolver reported visible fixture counts train=6/dev=2 as a provenance limitation only. |
| Scope/boundaries | PASS | No 1.7B job, Tier-1 eval, Tier-2/Tier-3 action, direct provider, alternate route, or protected/config mutation occurred. |
DECISION: keep the resolver smoke as successful plumbing evidence only. It is
not evidentiary SFT evidence, does not establish an SFT-to-human gap, and does
not complete M1. Do not launch Qwen3-1.7B from this batch or infer that the
0.6B repository revision is the immutable revision for a distinct 1.7B model.
NEXT: Park this uncommitted append-only findings/ledger batch for independent
PI verification. A later separately authorized batch must resolve and pin the
exact Qwen3-1.7B revision, rehash and preregister the evidentiary configs, and
pass its own pre-compute verification before any 1.7B launch. Do not begin M2.

## [2026-07-15] M1 / resolver-smoke-independent-verification
HYPOTHESIS: The successful Qwen3-0.6B resolver smoke is acceptable as plumbing
evidence only if an independent gate confirms append-only provenance, exact
publication/config/preregistration agreement, single-run terminal evidence,
scope confinement, and clean diff hygiene.
SETUP: The independent resolver gate audited the append-only FINDINGS/ledger
batch, published ancestry and tip, canonical resolver hash, preregistration
order, run cardinality, terminal wrapper evidence/accounting, protected scope,
and `git diff --check`. It launched no compute or evaluation.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Independent verdict | PASS | `.swarmy/results/m1-resolver-gate.txt` reports `score=pass` with no failures. |
| Append-only integrity | PASS | Prior FINDINGS and ledger bytes are unchanged; only 65 FINDINGS lines and two ledger rows were appended before this verdict. |
| Publication and config provenance | PASS | Published ancestor `7fb2e98`, published run tip `7f5a166`, canonical hash `8f0be62b88999d23143946f3c6dbf8db50d03e7eebf8032f924eb5fa1809f930`, preregistration, and run row agree. |
| Order and cardinality | PASS | Preregistration preceded launch by 6983.743112 seconds; exactly one resolver smoke run exists. |
| Terminal evidence/accounting | PASS | Return code 0; immutable revision `c1899de289a04d12100db370d81485cdf75e47ca`; artifact/snapshot provenance; 11.589 accelerator-seconds, `$0.003087`, and zero tokens. |
| Scope and hygiene | PASS | No protected/fixed, 1.7B, M2, Tier-2/3, or alternate-route action; `git diff --check` exits 0. |
DECISION: keep and close the Qwen3-0.6B resolver batch as independently
verified plumbing evidence only. This is not scientific SFT evidence and does
not complete M1.
NEXT: Publish this accepted resolver batch. Only after successful publication,
prepare and separately preregister the exact Qwen3-1.7B resolver derivative;
do not launch it in this preparation batch.

## [2026-07-16] M1 / resolver-prepare-qwen3-1p7b
HYPOTHESIS: The separately preregistered Qwen3-1.7B resolver preparation is
valid if it adds the exact non-evidentiary resolver derivative for
`Qwen/Qwen3-1.7B`, preserves the fixed M1 data boundary, and records an
append-only preregistration that pins canonical hash
`c3cb91e5fa1c2f854f3e9307ec18c3129df15ab8963ee8de50cf04587608b0e9`,
`smoke`, seeds `[11,29,47]`, requested revision `main`, and the rule that only
the returned immutable 1.7B revision may unlock later SFT preparation. This
depends on the accepted 0.6B resolver batch already being published and on the
explorer memo's exact-derivative requirement.
SETUP: Implementer-only preparation batch. Starting branch state was clean at
published `491abffc1795834eded47f9b16c651f6c185153d`, matching local `HEAD`
and `origin/agent/m1`. Added
`configs/m1/m1_plumbing_revision_resolve_qwen3_1p7b_v1.yaml` as the exact
derivative of `configs/m1/m1_plumbing_revision_resolve_qwen3_0p6b_v1.yaml`
with only comparison ID, model base, and resolver placeholder changed.
Canonical parsed-YAML hash was recomputed offline as
`c3cb91e5fa1c2f854f3e9307ec18c3129df15ab8963ee8de50cf04587608b0e9`.
Append-only preregistration was added through `ledger/ledger.py add` for
comparison `M1-plumbing-revision-resolve-qwen3-1p7b`, embedding the exact hash,
budget `smoke`, zero-token/non-evidentiary resolver scope, fixed train hash
`c59c853cdc03c7378308c8f35baa874e0f484fa035d4297948c3f3b755afa1a6`, fixed dev
hash `69dded207ccb2a7753666752ebcbdaee0e00260bf9848817979e50427bb2cf8b`,
requested revision `main`, and the Qwen3-1.7B model boundary. No compute,
`harness eval`, Tier 2, Tier 3, M2, or alternate route was used.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Exact 1.7B resolver derivative | PREPARED | New YAML changes only comparison ID, model base, and resolver placeholder from the accepted 0.6B resolver config. |
| Canonical config hash | PREREGISTERED | Offline recomputation matched the required hash `c3cb91e5fa1c2f854f3e9307ec18c3129df15ab8963ee8de50cf04587608b0e9`. |
| Unique preregistration row | PREREGISTERED | `ledger/ledger.py query --comparison M1-plumbing-revision-resolve-qwen3-1p7b` returns one open prereg row and no run rows. |
| Compute and evaluation activity | NOT RUN | This batch submitted no job and ran no Tier-1/Tier-3 action. |
| Protected and fixed surfaces | PRESERVED | No `harness/`, `sources/`, fixed data/manifests, sampler grid, or deployment/calibration files were mutated. |
DECISION: keep this as a preparation-only append. It does not claim scientific
acceptance, does not create a pass verdict, and does not authorize submission
by itself.
NEXT: Independent published-tip/hash/prereg verification before a single
submit of `configs/m1/m1_plumbing_revision_resolve_qwen3_1p7b_v1.yaml`.

## [2026-07-16] M1 / resolver-1p7b-prelaunch-verification
HYPOTHESIS: The 1.7B resolver batch can proceed only if the published 1.7B resolver preregistration exists, the independent resolver prelaunch verdict is an unqualified pass with no failures, and the publication tip is clean.
SETUP: From branch `agent/m1` at published tip `f1def157881dd6c6520015770d5e33184fba112e` (`agent/m1` tracking and live remote `origin/agent/m1` are both `f1def157881dd6c6520015770d5e33184fba112e`), appended a verification check against `.swarmy/results/m1-1p7b-resolver-prelaunch.txt`, prereg row state, and repo cleanliness before any launch.
RESULTS:
| check | status | evidence |
| --- | --- | --- |
| Independent prelaunch verdict | PASS | `.swarmy/results/m1-1p7b-resolver-prelaunch.txt` reports `score=pass` and `failed_checks=none`. |
| Repo cleanliness | PASS | `git status --short --branch` output is `## agent/m1...origin/agent/m1` (no tracked changes). |
| Publish tracking integrity | PASS | `git rev-parse HEAD`, `git rev-parse @{upstream}`, and `git ls-remote --heads origin agent/m1` all resolve to `f1def157881dd6c6520015770d5e33184fba112e`. |
| Preregistration state | PASS | `ledger/ledger.py query --comparison M1-plumbing-revision-resolve-qwen3-1p7b` returns exactly one open prereg row and no run rows. |
| Config canonical hash pointer | PASS | Resolver YAML is `configs/m1/m1_plumbing_revision_resolve_qwen3_1p7b_v1.yaml` with expected canonical hash `c3cb91e5fa1c2f854f3e9307ec18c3129df15ab8963ee8de50cf04587608b0e9` from prior append-only prep evidence. |
DECISION: keep and proceed only with constrained, single-submission resolver execution for Qwen3-1.7B.

## [2026-07-16] M1 / resolver-1p7b-terminal
HYPOTHESIS: The first registered Qwen3-1.7B resolver smoke can clone from published `agent/m1` and resolve `Qwen/Qwen3-1.7B` `main` to an immutable revision within the existing budget while preserving Qwen3-1.7B scope confinement and complete terminal accounting.
SETUP: Single constrained wrapper launch of `configs/m1/m1_plumbing_revision_resolve_qwen3_1p7b_v1.yaml` under `smoke` using one exact pre-registered comparison row (`M1-plumbing-revision-resolve-qwen3-1p7b`) with expected config hash `c3cb91e5fa1c2f854f3e9307ec18c3129df15ab8963ee8de50cf04587608b0e9` and fixed split hashes (`train=c59c853cdc03c7378308c8f35baa874e0f484fa035d4297948c3f3b755afa1a6`, `dev=69dded207ccb2a7753666752ebcbdaee0e00260bf9848817979e50427bb2cf8b`). Monitoring used only `infra/gpu status|logs`.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Budget gate | PASS | `infra/gpu budget` reported GPU `$40.00` remaining and API `$99.999337` remaining before launch. |
| Config hash/prereg agreement | PASS | Launch used exact config path and canonical hash from prereg; run row `dftr-1784178967-307cd34f` was created for comparison `M1-plumbing-revision-resolve-qwen3-1p7b`. |
| Publication/chain integrity | PASS | `git rev-parse HEAD`, `git rev-parse @{upstream}`, and live remote ref all resolve to `a66d34615f8c4fce4ecf7d18358852ac8e83337b`; run payload reports the same `git_sha`. |
| Terminal execution | PASS | One attempt. `status` and `logs` reached terminal state `completed` with `return_code=0`. |
| Start/finish | PASS | Started `1784178968.09574`, finished `1784179018.336987`. |
| Immutable revision resolution | PASS | Base `Qwen/Qwen3-1.7B` requested `main` resolved to immutable revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. |
| Artifact/snapshot provenance | PASS | Resolver artifact `/__modal/volumes/vo-EY2fT0CaoNDuXGLZLNZcGg/runs/dftr-1784178967-307cd34f/resolved_revision.json`; snapshot `/checkpoints/hf-cache/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. |
| Terminal accounting | PASS | `accel_seconds=43.702`, `actual_cost_usd=0.011642`, `tokens=0` in `ledger/ledger.py update` (`metrics_ptr` set to resolved revision artifact path). |
| Scope/boundaries | PASS | No local accelerator, no SFT, no harness eval, no Tier-2/3, no M2, and no alternate route. |
DECISION: keep.
NEXT: Use resolved immutable revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` to preregister and run exact 1.7B preparatory SFT configs only through approved M1 routes.

## [2026-07-16] M1 / sft-prepare-qwen3-1p7b
HYPOTHESIS: The first evidentiary Qwen3-1.7B SFT launch candidate is valid for preregistration only if it depends on the already accepted immutable resolver result for `Qwen/Qwen3-1.7B`, pins that revision only in the checked-in SFT surfaces named by the explorer memo, preserves every other SFT control exactly, and records the exact three-seed `screen` launch provenance before any compute or evaluation. This relies on the accepted resolver batch already recorded above, the fixed M1 data boundary, and the fail-closed SFT readiness memo in `.swarmy/explore-m1-sft-readiness.md`.
SETUP: Preparation-only implementer batch from clean published tip `3d52149f2d309bdb2fb5da259b66324910d02c3d`, matching local `HEAD` before edits. Pinned immutable revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` only in `configs/m1/manifests/revision_placeholders_v1.json` and `configs/m1/m1_sft_qwen3_1p7b_v1.yaml`. Canonical parsed-YAML hash for `configs/m1/m1_sft_qwen3_1p7b_v1.yaml` was recomputed offline after pinning and must equal `e213ed59e70ece4815b4b467b84df30eb6fccec8fb64c507d6699100df1575e8`. Launch provenance for later independent testing is fixed as comparison `M1-sft-baseline-qwen3-1p7b`, arm `SFT`, budget `screen`, seeds `[11,29,47]`, one `L40S`, timeout `120` minutes, train split hash `c59c853cdc03c7378308c8f35baa874e0f484fa035d4297948c3f3b755afa1a6`, dev split hash `69dded207ccb2a7753666752ebcbdaee0e00260bf9848817979e50427bb2cf8b`, and immutable fixed-manifest hashes `fixed_inputs_v1.json=e56e9cf2573b957b8491cf7733fa384de42908a71d6b8d2c2be581fbb402808d`, `sampler_grid_v1.json=662d21f269b7a8ea8bc70da105bd6b2b8164021e2fdc510120763aa19df800ae`. No compute, no `harness eval`, no Tier 2, no Tier 3, and no M2 work were run in this batch.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Accepted resolver dependency | PASS | This batch uses only immutable resolver revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` already recorded for `M1-plumbing-revision-resolve-qwen3-1p7b`; no new resolver or alternate route was used. |
| Authorized pinned surfaces | PASS | Only `configs/m1/manifests/revision_placeholders_v1.json` and `configs/m1/m1_sft_qwen3_1p7b_v1.yaml` received tracked config changes. |
| Canonical SFT config hash | PREREGISTERED | Offline recomputation after pinning must match `e213ed59e70ece4815b4b467b84df30eb6fccec8fb64c507d6699100df1575e8`. |
| SFT launch provenance | PREREGISTERED | Comparison `M1-sft-baseline-qwen3-1p7b` is fixed to seeds `[11,29,47]`, budget `screen`, one `L40S`, timeout `120` minutes, and fixed train/dev split hashes. |
| Compute and evaluation activity | NOT RUN | No submit, no training, no generation, and no Tier-1/Tier-3 action occurred in this preparation batch. |
| Protected/fixed surfaces | PRESERVED | No `harness/`, `sources/`, fixed M0 data artifacts, sampler sweep config, or M2 surface was mutated. |
DECISION: keep this as preparation and preregistration evidence only. It does not self-accept the launch tip and does not authorize compute without a separate tester.
NEXT: Separate tester must verify published-tip equality, allowed diff scope, exact hash/preregistration agreement, fixed-manifest hashes, and clean tracked worktree before any single `screen` SFT submit. Do not begin sampler, evaluation, or M2 work from this batch.

## [2026-07-16] M1 / resolver-independent-verification-qwen3-1p7b
HYPOTHESIS: The recorded Qwen3-1.7B resolver smoke remains acceptable as plumbing evidence only if an independent blind audit passed on exact published-tip provenance, source-run ancestry, terminal accounting, immutable revision resolution, and M1 boundary confinement without launching any new compute or evaluation.
SETUP: Reviewed `.swarmy/results/m1-1p7b-resolver-terminal.txt`, `.swarmy/logs/test-m1-1p7b-resolver-terminal.log`, the prior M1 resolver entries above, and local commit `a15077ebca1293e1912c07084f40794a392b2fc8`. The independent audit itself was executed read-only from clean published `agent/m1` tip `3d52149f2d309bdb2fb5da259b66324910d02c3d`, with local tracking and live remote `origin/agent/m1` equal to that same SHA, and it checked resolver run source SHA `a66d34615f8c4fce4ecf7d18358852ac8e83337b`.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Blind independent verdict | PASS | `.swarmy/results/m1-1p7b-resolver-terminal.txt` reports exact blind result `score=pass` and `failed_checks=none`. |
| Published/source resolver evidence | PASS | Audit `HEAD`, local upstream, and live remote all matched published `3d52149f2d309bdb2fb5da259b66324910d02c3d`; resolver run `dftr-1784178967-307cd34f` reports source `git_sha=a66d34615f8c4fce4ecf7d18358852ac8e83337b`, which the audit confirmed is an ancestor of the published tip. |
| Terminal accounting | PASS | Resolver terminal state was `completed` with `return_code=0`, `accel_seconds=43.702`, `actual_cost_usd=0.011642`, `tokens=0`, and `metrics_ptr=/__modal/volumes/vo-EY2fT0CaoNDuXGLZLNZcGg/runs/dftr-1784178967-307cd34f/resolved_revision.json`. |
| Immutable revision resolution | PASS | `Qwen/Qwen3-1.7B` requested at `main` resolved to immutable revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, with snapshot `/checkpoints/hf-cache/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. |
| Boundary result | PASS | The audit launched no compute and no evaluation, found no protected/fixed-surface mutation, and explicitly confirmed the earlier publication failure is historical/resolved rather than a current finding. |
DECISION: keep the Qwen3-1.7B resolver smoke as independently verified plumbing evidence only. It remains non-evidentiary for SFT and does not complete M1.

## [2026-07-16] M1 / sft-prelaunch-verification-qwen3-1p7b
HYPOTHESIS: The preregistered three-seed Qwen3-1.7B SFT screen can proceed
only if the independent SFT prelaunch verdict is an unqualified pass, the
tracked worktree is clean, and local `HEAD`, local upstream, and live
`origin/agent/m1` all match the required published tip
`c51cabedb3a1ffffa05a0105d3787f7ee6e733f4` before any new append-only
mutation.
SETUP: Read `CLAUDE.md`, `RESEARCH_CONTEXT.md`, the relevant M1 findings and
SFT preregistration evidence, `ledger/ledger.py`, and
`configs/m1/m1_sft_qwen3_1p7b_v1.yaml`; read only the terse independent
verdict file `.swarmy/results/m1-sft-prelaunch.txt`; then checked git
cleanliness and local/upstream/live-remote equality on branch `agent/m1`
before appending this verification entry.
RESULTS:
| check | status | evidence |
| --- | --- | --- |
| Independent prelaunch verdict | PASS | `.swarmy/results/m1-sft-prelaunch.txt` reports `score=pass` and `failed_checks=none`. |
| Required published tip before mutation | PASS | `git rev-parse HEAD`, `git rev-parse @{upstream}`, and `git ls-remote --heads origin agent/m1` each resolved to `c51cabedb3a1ffffa05a0105d3787f7ee6e733f4`. |
| Tracked worktree cleanliness | PASS | `git status --short --branch` reported `## agent/m1...origin/agent/m1` with no tracked changes before this append. |
| Preregistered SFT launch target | PASS | `configs/m1/m1_sft_qwen3_1p7b_v1.yaml` remains the exact preregistered SFT config for comparison `M1-sft-baseline-qwen3-1p7b`, budget `screen`, and seeds `[11,29,47]`. |
| Ledger uniqueness | PASS | Existing prereg evidence records exactly one open prereg row and zero prior run rows for `M1-sft-baseline-qwen3-1p7b`; submit has not occurred yet. |
DECISION: keep and proceed only with a single constrained `screen` submission
of `configs/m1/m1_sft_qwen3_1p7b_v1.yaml` after this exact evidence tip is
published cleanly. No sampler, Tier 1, Tier 2, Tier 3, M2, alternate compute,
or bypass is authorized by this append.

## [2026-07-16] M1 / sft-publication-boundary-qwen3-1p7b
HYPOTHESIS: The preregistered three-seed Qwen3-1.7B SFT screen can advance to
the budget gate only if the exact prelaunch evidence commit is published to
`origin/agent/m1` and the live remote equals the local clean tip before any
submit.
SETUP: From clean published tip `c51cabedb3a1ffffa05a0105d3787f7ee6e733f4`,
committed the append-only prelaunch evidence as local commit
`3bb756579a8926961cd30f709353c7fec597a048` with message
`[dftr] i=7 arm=SFT score=pass status=keep | verify three-seed SFT prelaunch`.
Attempted only the normal publication route `git push origin agent/m1`; after
local HTTPS credential failure, polled `git ls-remote --heads origin agent/m1`
three times over roughly 10 seconds for operator-side publication of the exact
SHA.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Local prelaunch evidence commit | PASS | Commit `3bb756579a8926961cd30f709353c7fec597a048` exists locally and contains the append-only `FINDINGS.md` prelaunch verification entry. |
| Normal publication route | FAIL | `git push origin agent/m1` returned `fatal: could not read Username for 'https://github.com': Device not configured`. |
| Remote exact-tip equality | FAIL | `git ls-remote --heads origin agent/m1` remained `c51cabedb3a1ffffa05a0105d3787f7ee6e733f4` on polls 1, 2, and 3, so the required exact SHA publication did not occur. |
| Submission gate | BLOCKED | Per the constrained contract, `infra/gpu budget`, `infra/gpu submit`, `infra/gpu status`, `infra/gpu logs`, and `ledger/ledger.py update` for SFT were not run after publication failed. |
| Boundary compliance | PASS | No alternate publish route, no bypass, no compute, no sampler, no Tier 1, no Tier 2, no Tier 3, no M2, and no protected-surface edit were used after the failure. |
DECISION: stop here with a negative infrastructure result. The three-seed SFT
screen was not submitted because the exact evidence tip was not published to
the live remote.
NEXT: Wait for the exact commit `3bb756579a8926961cd30f709353c7fec597a048` to
be published on `origin/agent/m1` or for local push credentials to be restored,
then restart from the publication/budget gate on a clean exact-match tip.

## [2026-07-16] M1 / sft-publication-resumption-qwen3-1p7b
HYPOTHESIS: The publication-only SFT boundary is resolved once the operator
publishes the exact independently accepted prelaunch evidence commit, without
changing the preregistered config, fixed inputs, sampler grid, or run
cardinality.
SETUP: The operator publication path pushed exact commit
`3bb756579a8926961cd30f709353c7fec597a048` to `origin/agent/m1` after the
isolated executor's HTTPS credential failure. Live `git ls-remote` then
returned that exact SHA. This append preserves the negative infrastructure
record above and records its resolution; no compute or evaluation was run by
this recovery action.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Exact evidence-tip publication | PASS | Local `HEAD`, local tracking ref, and live remote were synchronized to `3bb756579a8926961cd30f709353c7fec597a048` before this append. |
| Scientific configuration | UNCHANGED | The preregistered SFT YAML, immutable model revision, seeds, fixed manifests, and budget class were not changed. |
| SFT run cardinality | UNCHANGED | No SFT submit or run row occurred during publication recovery. |
| Boundary handling | PASS | The prior failure remains recorded append-only and is now historical rather than an active blocker. |
DECISION: resume M1 from a newly published clean record tip. Require a fresh
independent publication/config/preregistration/budget check on that exact tip
before the single three-seed SFT submit; do not bypass or reuse a stale verdict.

## [2026-07-16] M1 / sft-terminal-qwen3-1p7b
HYPOTHESIS: The exact preregistered single three-seed Qwen3-1.7B SFT screen
can complete from published source tip
`0531711c1a008325a2095c2a2ec9c9e2e87ef8f0` through the constrained wrapper
only if the approved surfaces expose complete terminal accounting and valid
per-seed checkpoint provenance.
SETUP: Read `CLAUDE.md`, `RESEARCH_CONTEXT.md`, all append-only M1 entries in
`FINDINGS.md`, `configs/m1/m1_sft_qwen3_1p7b_v1.yaml`,
`configs/m1/manifests/fixed_inputs_v1.json`,
`configs/m1/manifests/sampler_grid_v1.json`,
`configs/m1/manifests/revision_placeholders_v1.json`, `ledger/ledger.py`,
`ledger/ledger.jsonl`, `infra/gpu`, and only the terse independent verdict
`.swarmy/results/m1-sft-resume-gate.txt`. Verified local `HEAD`, local
upstream, and live `origin/agent/m1` all matched published tip
`0531711c1a008325a2095c2a2ec9c9e2e87ef8f0`; verified canonical SFT config hash
`e213ed59e70ece4815b4b467b84df30eb6fccec8fb64c507d6699100df1575e8`,
immutable model revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, and fixed
manifest hashes `e56e9cf2573b957b8491cf7733fa384de42908a71d6b8d2c2be581fbb402808d`
and `662d21f269b7a8ea8bc70da105bd6b2b8164021e2fdc510120763aa19df800ae`.
Observed that the single allowed SFT submit had already been consumed by run
`dftr-1784180693-f3c7ab5c` from that exact published tip, so no duplicate
submit was attempted. Monitored only with `infra/gpu status` and
`infra/gpu logs`, then preserved terminal accounting append-only through the
ledger CLI.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Fresh resume gate | PASS | `.swarmy/results/m1-sft-resume-gate.txt` reports `score=pass`, `failed_checks=none`, and `published_tip=0531711c1a008325a2095c2a2ec9c9e2e87ef8f0`. |
| Published-tip equality | PASS | `git rev-parse HEAD`, `git rev-parse @{upstream}`, and `git ls-remote --heads origin agent/m1` all resolved to `0531711c1a008325a2095c2a2ec9c9e2e87ef8f0`. |
| Config and fixed-manifest provenance | PASS | Canonical YAML hash matched `e213ed59e70ece4815b4b467b84df30eb6fccec8fb64c507d6699100df1575e8`; fixed manifest hashes matched the preregistered values; model revision remained `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. |
| Single-run cardinality boundary | PASS | The only SFT run row for `M1-sft-baseline-qwen3-1p7b` is `dftr-1784180693-f3c7ab5c` at git SHA `0531711c1a008325a2095c2a2ec9c9e2e87ef8f0`; no second submit was attempted. |
| Budget reservation and terminal wrapper status | PASS | `infra/gpu status dftr-1784180693-f3c7ab5c` reported `reserved_cost_usd=4.68288`, `status=completed`, `return_code=0`, `gpu=L40S`, `timeout_seconds=7200`, `accel_seconds=36.292`, `actual_cost_usd=0.023604`, and `tokens=1422`. |
| Three-seed SFT completion | PASS | `infra/gpu logs` showed three completed training blocks for seeds `[11,29,47]`; terminal run JSON reported `checkpoint_count=3`, `training_seeds=[11,29,47]`, `train_tokens=1422`, `generated_tokens=0`, and `total_tokens=1422`. |
| Per-seed checkpoint paths | PASS | Wrapper workflow fixes seed checkpoint directories to `/checkpoints/runs/dftr-1784180693-f3c7ab5c/seed-11`, `/checkpoints/runs/dftr-1784180693-f3c7ab5c/seed-29`, and `/checkpoints/runs/dftr-1784180693-f3c7ab5c/seed-47`; terminal logs exposed remote manifest pointer `/__modal/volumes/vo-EY2fT0CaoNDuXGLZLNZcGg/runs/dftr-1784180693-f3c7ab5c/checkpoints_manifest.json`. |
| Per-seed checkpoint hash retrieval | FAIL | Neither the host-resolved path `/__modal/volumes/vo-EY2fT0CaoNDuXGLZLNZcGg/runs/dftr-1784180693-f3c7ab5c/checkpoints_manifest.json` nor the in-container paths under `/checkpoints/runs/dftr-1784180693-f3c7ab5c/` were mounted locally in this executor, so the exact per-seed `checkpoint_files` hash maps could not be read from the approved surfaces. |
| Append-only ledger accounting | PASS | `ledger/ledger.jsonl` contains the launched run row plus completed `run_update` rows with matching `status=completed`, `cost=0.023604`, `accel_seconds=36.292`, `tokens=1422`, and identical manifest pointer `metrics_ptr=/__modal/volumes/vo-EY2fT0CaoNDuXGLZLNZcGg/runs/dftr-1784180693-f3c7ab5c/checkpoints_manifest.json`. The earlier completed row predates the explicit reconciliation append by 15.265519 seconds, so this is recorded as idempotent accounting duplication, not as a new scientific or infrastructure gate failure. |
DECISION: stop at the M1 SFT terminal evidence boundary. The exact
preregistered three-seed Qwen3-1.7B SFT run completed once from the required
published tip, but this executor cannot honestly close M1 as a terminal
`keep/pass` because the approved surfaces did not expose the per-seed
checkpoint hash payload required for valid checkpoint provenance. Do not begin
sampler work, Tier 1/2/3, M2, or 14B work from this state.
NEXT: Wait for an approved read path or operator-materialized copy of the
remote `checkpoints_manifest.json` or seed `provenance.json` payloads for run
`dftr-1784180693-f3c7ab5c`; do not resubmit or rerun the SFT screen.

## [2026-07-16] M1 / sft-checkpoint-provenance-boundary
HYPOTHESIS: The SFT terminal boundary can be recorded as `blocked/keep` if an
independent recorder verifies the operator-materialized checkpoint manifest
offline while preserving the earlier `legal_read_path=no` contract finding,
the immutable terminal ledger rows, and all protected surfaces.
SETUP: Recorder-only pass. Read the uncommitted SFT terminal finding above,
the SFT ledger rows, `.swarmy/explore-m1-sft-manifest.md`,
`.dispatch/tasks/m1-sft-manifest-explorer/output.md`, and the read-only
operator materialization at
`.swarmy/operator-materialized/dftr-1784180693-f3c7ab5c/`. Ran only offline
manifest, diff, and ledger hygiene checks; no `infra/gpu`, sampling,
evaluation, Tier 2/3, M2, 14B, protected-surface edit, alternate publication
path, or new ledger append was used.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Historical constrained-read boundary | BLOCKED | The explorer remains `legal_read_path=no`: status/logs/ledger exposed only terminal metadata, stdout, and an opaque manifest pointer, not artifact bytes. |
| Operator materialization | PASS | Attestation says the existing Modal `checkpoints_manifest.json` was read-only materialized; canonical sorted compact JSON SHA-256 independently recomputed as `2c255965575359ce8e92761befe0dd8db360b204b7385b924f4261194c0e2fb1`. |
| Manifest identity | PASS | `protocol_version=m1.checkpoints.v1`, `model_base=Qwen/Qwen3-1.7B`, and `model_revision=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. |
| Seed/token provenance | PASS | Exactly three checkpoint entries for seeds `[11,29,47]`, each with `train_tokens=474` for total ledger tokens `1422`. |
| Distinct adapter hashes | PASS | Per-seed `adapter_model.safetensors` hashes are distinct: seed 11 `714876d1ca760a4f8013b3377cd104297971bf0ef45c41425b44e427712e86fd`; seed 29 `e1fc8e1ee9069d3bb18c3dfe21196d223c3fb254989755a248e1798d7e357588`; seed 47 `d5665c4caa7e13eefc7e0c665292b433d25c38e3dbd0b9020556e402f0c34d8f`. |
| Ledger immutability | PASS | Preserved the run row plus both identical completed terminal updates as append-only idempotent duplication: same `status=completed`, `cost=0.023604`, `accel_seconds=36.292`, `tokens=1422`, and manifest pointer. |
DECISION: keep the completed SFT checkpoint-provenance record, but keep the
milestone parked as a boundary checkpoint rather than a pass. The operator
bytes verify the manifest contents; the constrained self-service contract still
does not expose those artifact bytes, and this recorder does not authorize
sampler, evaluation, Tier 2/3, M2, or 14B work.
NEXT: Wait for human direction on whether the operator materialization becomes
the approved durable read path or whether a read-only gateway/CLI artifact
surface should be added; do not resubmit the SFT screen.

## [2026-07-16] M1 / sft-checkpoint-provenance-resolution
HYPOTHESIS: The historical SFT checkpoint-provenance boundary from commit
`a07f0f5` is resolved if the exact operator action requested by the explorer
has now supplied the completed run's manifest bytes read-only, with no rerun,
state mutation, new ledger append, sampling, evaluation, or protected-surface
change.
SETUP: Resolution-only pass. Read
`.dispatch/tasks/m1-sft-boundary-recorder/ipc/blocker.md`, the
operator-materialized
`.swarmy/operator-materialized/dftr-1784180693-f3c7ab5c/checkpoints_manifest.json`,
its `ATTESTATION.md`, the latest M1 findings and ledger rows, and commit
`a07f0f586ab0fc504eea038eab25711f0e44fcbc`. Preserved `a07f0f5` and its
historical blocked finding unchanged. Recomputed and checked the planned
manifest fields offline with `jq` and SHA-256 only; did not run infra,
compute, sampling, `harness eval`, Tier 2/3, M2, or 14B work.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Operator unblocker identity | PASS | The blocker says the operator materialized the existing Modal `checkpoints_manifest.json` read-only and superseded the stale active-boundary instruction; this is the exact artifact-read action requested by the prior explorer, not a rerun or state mutation. |
| Canonical manifest hash | PASS | Recomputed canonical compact JSON SHA-256 as `2c255965575359ce8e92761befe0dd8db360b204b7385b924f4261194c0e2fb1`, matching the attestation and the required plan value. |
| Protocol and model provenance | PASS | Manifest has `protocol_version=m1.checkpoints.v1`, `model_base=Qwen/Qwen3-1.7B`, and `model_revision=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. |
| Seed and token provenance | PASS | Exactly three checkpoint entries exist for seeds `[11,29,47]`, each with `train_tokens=474`; total manifest tokens `1422` match the SFT ledger terminal rows. |
| Complete file maps | PASS | Every seed exposes the same complete SHA-256 map for `README.md`, `adapter_config.json`, `adapter_model.safetensors`, `added_tokens.json`, `chat_template.jinja`, `merges.txt`, `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json`, `training_metrics.json`, and `vocab.json`. |
| Distinct adapter hashes | PASS | `adapter_model.safetensors` hashes are distinct across the three seeds: `714876d1ca760a4f8013b3377cd104297971bf0ef45c41425b44e427712e86fd`, `e1fc8e1ee9069d3bb18c3dfe21196d223c3fb254989755a248e1798d7e357588`, and `d5665c4caa7e13eefc7e0c665292b433d25c38e3dbd0b9020556e402f0c34d8f`. |
| Historical evidence preservation | PASS | The blocked finding in `a07f0f5` remains historical and unchanged; the SFT run row plus both identical completed terminal updates in `ledger/ledger.jsonl` were not modified or extended. |
| Scope hygiene | PASS | Offline checks found no new ledger row, compute, sampling, Tier 1/2/3, M2, 14B, protected-surface, or immutable-surface action in this resolution pass. |
DECISION: keep. The operator-materialized attested manifest resolves the
checkpoint-provenance boundary and completes SFT checkpoint provenance for the
single completed three-seed Qwen3-1.7B SFT run. Sampler preparation may proceed
only after this resolution commit is published exactly.
NEXT: Publish this resolution commit exactly before any sampler-preparation
work. Do not treat unpublished local provenance resolution as authorization to
prepare samples.

## [2026-07-16] M1 / sampler-screen-prepare-qwen3-1p7b
HYPOTHESIS: The M1 SFT sampler screen is valid for preregistration only if the
sampler consumes the operator-verified three-seed SFT checkpoint manifest with
the exact `Qwen/Qwen3-1.7B` immutable revision, rejects any manifest or adapter
base/revision mismatch offline, pins only the two sampler YAML placeholders,
and records the downstream Tier-1 boundaries without changing fixed inputs,
sampler grid, inline reference behavior, compute shape, seeds, prompt format,
or budget.
SETUP: Preparation-only implementer batch from published sampler-readiness tip
`56fdcda3a8f35dae62874ccb8b670310759551e3`. Read the governing M1 findings,
sampler readiness and boundary-design memos, current mutable sampler workflow
and config, ledger, fixed manifest hashes, and operator-materialized SFT
manifest. Implemented only the sampler loader fix in `experiments/m1/workflow.py`:
`_load_checkpoint_index()` now validates manifest `model_base` and
`model_revision` against the sampler config, and PEFT adapter generation now
loads the base model and fallback tokenizer offline with pinned revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` after validating
`adapter_config.json` base identity. Added focused offline tests for revision
propagation, manifest revision mismatch, and adapter base mismatch. Pinned only
`configs/m1/m1_sampler_sweep_qwen3_1p7b_v1.yaml` `model.revision` and
`sampling.checkpoints_manifest=/checkpoints/runs/dftr-1784180693-f3c7ab5c/checkpoints_manifest.json`.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Sampler config hash | PREREGISTERED | Canonical parsed-YAML hash is exactly `09b7c974c5a3b49ade9447fa0619af819828bcc2da15a5703c06c6cf02bb0ec9`. |
| Checkpoint provenance | PRESERVED | Operator manifest canonical hash remains `2c255965575359ce8e92761befe0dd8db360b204b7385b924f4261194c0e2fb1`; model base/revision are `Qwen/Qwen3-1.7B` and `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; checkpoint seeds are `[11,29,47]`. |
| Sampler design | PRESERVED | Five grid points, sampling seeds `[101,202,303]`, checkpoint seeds `[11,29,47]`, prompt format, fixed dev data, max token settings, one `L40S`, 120-minute timeout, and `screen` budget are unchanged. |
| Ledger preregistration | PREREGISTERED | Exactly one open prereg row was appended through `ledger/ledger.py add` for comparison `M1-sft-sampler-sweep-qwen3-1p7b`, with no run row. |
| Expected cardinality | RECORDED | Future sampler screen is preregistered for `3 x 5 x 3 = 45` cells and `90` generated documents from fixed `dev_count=2`. |
| Offline checks | PASS | `python -m pytest experiments/tests/test_m1_sampler_loader.py infra/tests/test_policy.py` passed `15` tests; `git diff --check` passed; fixed input and sampler-grid hashes remained `e56e9cf2573b957b8491cf7733fa384de42908a71d6b8d2c2be581fbb402808d` and `662d21f269b7a8ea8bc70da105bd6b2b8164021e2fdc510120763aa19df800ae`. |
| Protected/fixed surfaces | PRESERVED | No `harness/`, `sources/`, fixed M0 data/manifests, `configs/m1/manifests/fixed_inputs_v1.json`, or `configs/m1/manifests/sampler_grid_v1.json` diff. |
| Compute/evaluation activity | NOT RUN | No infra command, sampler generation, `harness eval`, Tier 2, Tier 3, M2, or 14B action occurred in this batch. |
| Downstream human-bank boundary | EXPLICIT | Current sampler cells still write inline `reference_completion` for only the two fixed dev humans, so immutable Tier-1 harness evaluation cannot legally obtain four independent humans via `HARNESS_HUMAN_REFERENCE`; this task does not remove inline references or duplicate/train-bank humans. |
| Downstream baseline/freeze boundary | EXPLICIT | `harness/calibration.json` and baseline/freeze artifacts remain immutable/fail-closed, and the current freeze/baseline path remains circular until operator-approved immutable artifacts exist. |
DECISION: keep as preparation and preregistration evidence only. The sampler
loader is now exact-revision safe, and the sampler screen is preregistered, but
this commit does not launch sampling or claim Tier-1 executability.
NEXT: Publish this exact preparation commit before any future sampler launch.
Any launch must recheck budget and publication gates through the approved
wrapper; Tier-1 analysis still requires a separate authorized boundary design.

## [2026-07-16] M1 / sampler-screen-terminal-qwen3-1p7b
HYPOTHESIS: The preregistered Qwen3-1.7B sampler screen is complete evidence
only if the already-launched wrapper run terminates successfully and the
terminal payload proves exact source/config/revision/checkpoint/grid/seed
provenance, exact token/cost/accelerator accounting, the Tier-1 index pointer,
and the expected `45` index entries / `90` generated documents, while the
known downstream Tier-1 human-bank and baseline/freeze boundaries remain
explicitly unbypassed.
SETUP: Resume-only monitor pass on Thursday, July 16, 2026. Read the governing
M1 findings, the terse independent prelaunch verdict, the sampler config and
fixed manifests, the operator-materialized SFT checkpoint manifest and
attestation, the current sampler ledger rows, and the checked-in sampler
workflow contract. Resumed the already-launched run
`dftr-1784183624-2e567266` from published tip
`583dd8c149276f4544eb120649ba0b4952985216`. Monitoring used only
`infra/gpu status` and `infra/gpu logs` with polling waits no longer than
45 seconds. No submit, cancel, evaluation, calibration, freeze, Tier 2, Tier
3, M2, 14B, alternate route, or protected-surface mutation occurred. Because
the remote Modal artifact paths were not mounted locally in this checkout, the
`90` generated-document count was validated from primary sources by combining
the terminal `sample_count=45`, terminal visible `dev=2`, and the checked-in
`experiments/m1/workflow.py` contract that writes one JSONL row per dev record
for each index row.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Terminal wrapper run | PASS | `infra/gpu status dftr-1784183624-2e567266` returned `completed`, `return_code=0`, `started_at=1784183628.6075485`, `finished_at=1784184631.5288382`, `accel_seconds=992.069`, `actual_cost_usd=0.645241`, and `tokens=26618`. |
| Source and config provenance | PASS | Terminal payload matches published `git_sha=583dd8c149276f4544eb120649ba0b4952985216`, comparison `M1-sft-sampler-sweep-qwen3-1p7b`, budget `screen`, GPU `L40S`, and sampler config hash `09b7c974c5a3b49ade9447fa0619af819828bcc2da15a5703c06c6cf02bb0ec9`; local canonical rehash of `configs/m1/m1_sampler_sweep_qwen3_1p7b_v1.yaml` matches exactly. |
| Fixed data and model provenance | PASS | Sampler config still pins `Qwen/Qwen3-1.7B` revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; fixed train/dev hashes remain `c59c853cdc03c7378308c8f35baa874e0f484fa035d4297948c3f3b755afa1a6` and `69dded207ccb2a7753666752ebcbdaee0e00260bf9848817979e50427bb2cf8b`; fixed manifest hashes remain `e56e9cf2573b957b8491cf7733fa384de42908a71d6b8d2c2be581fbb402808d` and `662d21f269b7a8ea8bc70da105bd6b2b8164021e2fdc510120763aa19df800ae`. |
| SFT checkpoint provenance | PASS | Sampler config still points at `/checkpoints/runs/dftr-1784180693-f3c7ab5c/checkpoints_manifest.json`; the operator-materialized manifest attestation and independent `jq -cS . | shasum -a 256` recomputation preserve canonical compact JSON SHA-256 `2c255965575359ce8e92761befe0dd8db360b204b7385b924f4261194c0e2fb1`, protocol `m1.checkpoints.v1`, checkpoint seeds `[11,29,47]`, `474` train tokens per seed, and distinct adapter hashes across all three checkpoints. |
| Sampler grid and seeds | PASS | Terminal logs returned the fixed five grid IDs `default_t1.0_p1.0`, `cool_t0.8_p1.0`, `narrow_t1.0_p0.95`, `cool_narrow_t0.8_p0.95`, and `warm_t1.2_p0.95`, with sampling seeds `[101,202,303]`; these match the preregistered sampler grid and config. |
| Tier-1 index cardinality | PASS | Terminal logs returned `sample_count=45`, which exactly matches `3 checkpoints x 5 grid points x 3 sampling seeds`. |
| Generated-document cardinality | PASS | Inference from primary sources: terminal visible fixture count is `dev=2`, and `experiments/m1/workflow.py` writes one JSONL row per dev record for each of the 45 index rows, so the completed run necessarily produced `45 x 2 = 90` generated documents. |
| Tier-1 index pointer | PASS | Terminal logs returned exact artifact pointer `/__modal/volumes/vo-EY2fT0CaoNDuXGLZLNZcGg/runs/dftr-1784183624-2e567266/tier1_eval_index.template.json`; its bytes were not mounted locally here, so no Tier-1 analysis or artifact read was attempted in this pass. |
| Token accounting | PASS | Terminal logs reported `generated_tokens=26618`, `train_tokens=0`, and `total_tokens=26618`; terminal status reported the same total `tokens=26618`. |
| Append-only accounting | PASS | Before this pass, the ledger contained the open sampler prereg row plus the launched run row and no terminal sampler update. This pass appended exactly one `run_update` row for `dftr-1784183624-2e567266` with `status=completed`, `tokens=26618`, `accel_seconds=992.069`, `cost=0.645241`, and `metrics_ptr=/__modal/volumes/vo-EY2fT0CaoNDuXGLZLNZcGg/runs/dftr-1784183624-2e567266/tier1_eval_index.template.json`. |
| Downstream boundaries | PASS | The known Tier-1 human-bank boundary remains unchanged because current sampler cells still inline only the two fixed dev human references, and the immutable baseline/calibration/freeze path remains fail-closed. This task did not evaluate, calibrate, freeze, or use Tier 2/3, M2, or 14B routes. |
DECISION: keep. The sampler screen itself completed successfully with exact
wrapper-visible provenance and accounting, and the terminal evidence is strong
enough to checkpoint as the completed M1 sampler screen. Downstream Tier-1
analysis remains a separate bounded step that still requires the unchanged
human-bank and baseline/freeze boundaries to be respected.
NEXT: Commit and publish this exact terminal evidence before any later sampler
analysis. Subsequent tasks must not treat the completed sampler screen as
authorization to evaluate, calibrate, freeze, use Tier 2/3, M2, or 14B, or
mutate protected or fixed surfaces.

## [2026-07-16] M1 / human-calibration-proposal
HYPOTHESIS: The already-designed M1 human calibration proposal can be computed
offline from the fixed visible M0 human dev split, with pre-existing resampling
seeds and confidence, without using sampler outputs, Tier-1 reports, provider
calls, or any mutable harness surface.
SETUP: Read the governing M1 requirements, current calibration proposal config
and `experiments/m1/analysis.py`, fixed M0 human artifacts/manifests, immutable
`harness/calibration.json` read-only, latest M1 findings, and the published
sampler terminal state. No placeholders were present in
`configs/m1/m1_calibration_proposal_v1.json`. Ran
`python -m experiments.m1.analysis calibration-proposal --config
configs/m1/m1_calibration_proposal_v1.json` locally, using only
`data/artifacts/m0/dev_briefs.jsonl`, then enriched the review artifact with
fixed M0 source/manifests hashes. No infra command, provider call, sampler
artifact read, `harness eval`, Tier 2, Tier 3, M2, 14B, ledger update, or
protected/fixed-surface mutation occurred.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Review artifact | PASS | Wrote `experiments/m1/calibration_proposal_v1.json` with schema `m1.calibration_proposal.review.v1`; artifact SHA-256 is `d8cfe3bdc1825f8c03717ceb78bb79efd022d9dc1a7a3ce706039cc4da2f3c48`. |
| Fixed human records | PASS | Human split is `data/artifacts/m0/dev_briefs.jsonl` with file SHA-256 `0fabc6ffde1fbada04ad14daba880cde46e9ffafbfe6fcc0c8d969d750cb9ebb`, sample count `2`, dev split hash `69dded207ccb2a7753666752ebcbdaee0e00260bf9848817979e50427bb2cf8b`, and train split hash `c59c853cdc03c7378308c8f35baa874e0f484fa035d4297948c3f3b755afa1a6`. |
| Fixed source hashes | PASS | Source fixture `data/fixtures/fineweb_fixture.jsonl` hash is `5ded0780a3fed78e30e288fd47c3bdc093b61d0e5f406b092bb76aab58d717f3`; source manifest hash is `cf8fbcf4a184159d957acf8d53ddaf55232b1d944269cb32dcae43f443a04e42`; split-hashes file hash is `e31d3f4604e9b000b1102c96deb1487b88096624a02f3ed08d740abdf6464c60`. |
| Method and seeds | PASS | Interval method is `deterministic central quantile interval`; confidence level is `0.95`; pre-existing resampling seeds are `[404,505,606]`; subset fraction is `0.8`. |
| Point estimates | PASS | `self_bleu=0.023400584169956357`, `repeated_sentence_start_rate=0.0`, and `non_target_script_char_rate=0.0`. |
| Intervals | PASS | `self_bleu=[0.02327671389911226,0.02327671389911226]`, `repeated_sentence_start_rate=[0.0,0.0]`, `non_target_script_char_rate=[0.0,0.0]`, `paragraph_len_tokens=[3.0,33.0]`, and `sentence_len_tokens=[3.0,19.0]`. |
| Subset sensitivity | PASS | For every seed `[404,505,606]`, `ceil(2 * 0.8)` selects the full two-record visible dev split, so every subset hash is `0fabc6ffde1fbada04ad14daba880cde46e9ffafbfe6fcc0c8d969d750cb9ebb`; unique subset count is `1` and no subset-draw variation is observable in this tiny fixture. |
| Deterministic reproduction | PASS | Re-ran to a clean temporary output path and re-applied the same fixed-provenance enrichment; enriched rerun SHA-256 matched `d8cfe3bdc1825f8c03717ceb78bb79efd022d9dc1a7a3ce706039cc4da2f3c48` byte-for-byte. Raw helper rerun SHA-256 was `f3ad93f50ba3896c1847029a2d4f3dff166f8d65c88800de5b2d8f636345d105`. |
| Boundary hygiene | PASS | No `harness/`, `sources/`, fixed data/manifests/grid, ledger, sampler outputs, or model artifacts were mutated; no calibration values were transferred into `harness/calibration.json`. |
DECISION: keep. This is a versioned human-only calibration proposal for review
over the fixed visible M0 fixture. It is not a population-level estimate and
does not support any claim beyond the two visible dev records.
NEXT: Human review may transfer approved values into immutable
`harness/calibration.json`; the agent must not perform that transfer or use
these proposal values as if they were harness calibration.

## [2026-07-16] M1 / milestone-hard-boundary
HYPOTHESIS: M1 can be recorded as complete only if the independently published
tip satisfies the milestone's legal Tier-1 reporting, frozen-sampler, and
calibration-transfer requirements. If the authoritative independent verifier
returns `score=fail` with `hard_boundary=yes`, then the completed SFT,
sampler, and proposal-only calibration artifacts must be preserved as evidence
while M1 itself remains incomplete.
SETUP: Recorder-only pass from published tip
`a8482acb634317b57a591b3748fd85466d91390b`. Read `CLAUDE.md`,
`RESEARCH_CONTEXT.md`, all append-only M1 entries in `FINDINGS.md`, current
`git status`, `git log`, and `git rev-parse HEAD`, authoritative verdict
files `.swarmy/results/m1-final-boundary.txt` and
`.dispatch/tasks/m1-final-boundary-test/output.md`, the concise completed SFT
and sampler evidence already recorded in `ledger/ledger.jsonl` and prior M1
findings for runs `dftr-1784180693-f3c7ab5c` and
`dftr-1784183624-2e567266`, and the proposal-only calibration artifact
`experiments/m1/calibration_proposal_v1.json`. No infra command, no
`harness eval`, no sampler freeze, no Tier 2, no Tier 3, no M2, no 14B, no
hidden data, no duplicated references, and no protected/fixed-surface edits
were performed.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Published-tip audit target | PASS | Local `HEAD` is `a8482acb634317b57a591b3748fd85466d91390b`, matching the authoritative final-boundary verifier's `published_tip`. |
| Completed SFT evidence | PASS | Prior append-only M1 evidence already records one completed three-seed Qwen3-1.7B SFT run `dftr-1784180693-f3c7ab5c` with checkpoint-manifest provenance, `accel_seconds=36.292`, `actual_cost_usd=0.023604`, and `tokens=1422`. |
| Completed sampler evidence | PASS | Prior append-only M1 evidence already records one completed sampler run `dftr-1784183624-2e567266` with `45` cells, `90` generated documents, `accel_seconds=992.069`, `actual_cost_usd=0.645241`, and `tokens=26618`. |
| Proposal-only calibration evidence | PASS | `experiments/m1/calibration_proposal_v1.json` is the completed human-only review artifact with schema `m1.calibration_proposal.review.v1` and SHA-256 `d8cfe3bdc1825f8c03717ceb78bb79efd022d9dc1a7a3ce706039cc4da2f3c48`; it remains proposal-only evidence, not a transferred harness calibration. |
| Independent milestone verification | FAIL | `.swarmy/results/m1-final-boundary.txt` is authoritative and reports `score=fail`, `hard_boundary=yes`, and `published_tip=a8482acb634317b57a591b3748fd85466d91390b`. |
| Failed checks | FAIL | Tester failed checks exactly: `legal_tier1_reports_missing,frozen_sampler_missing,baseline_freeze_order_nonexecutable,sampler_artifacts_nonlocal,calibration_review_artifact_not_directly_reproduced`. |
| Legal Tier-1 report gate | FAIL | The authoritative tester records that current sampler rows inline `reference_completion`, immutable `harness eval` requires `4` human documents, the visible dev split has only `2` unique humans, and the sampler bytes/index are not local, so legal Tier-1 reports do not exist. |
| Freeze and baseline ordering gate | FAIL | The authoritative tester records that `harness/baseline_stats.json` is missing, `harness/calibration.json` still has null bounds, `harness/deployment_sampler.json` is unfrozen, and the checked-in `freeze_sampler()` -> `build_baseline_stats()` path is nonexecutable/circular. |
| Calibration transfer reproducibility gate | FAIL | The authoritative tester records that the checked-in calibration artifact is acceptable as a proposal-only descriptive review artifact, but it is not directly reproduced byte-for-byte by the checked-in calibration entrypoint alone. |
| Boundary compliance | PASS | No Tier-1 eval report or sampler freeze was run in this pass because bypassing the independent-human and immutable-harness gates is forbidden. |
DECISION: park M1 at a hard boundary. The SFT run, sampler run, and
proposal-only calibration artifact remain valid completed evidence, but the
milestone is incomplete because the authoritative independent verifier failed
the milestone on legal Tier-1 reporting, frozen-sampler absence,
baseline/freeze nonexecutability, remote-only sampler artifacts, and direct
calibration-artifact reproduction. Do not reinterpret this fail verdict as
permission to repair immutable boundaries inside this batch.
NEXT: Wait for the smallest lawful external unblockers only: a legal Tier-1
path that yields locally readable sampler artifacts and four independent human
references without duplication or hidden-data leakage, plus an operator-owned
immutable transfer path that makes baseline/freeze/calibration execution
lawful. Until those exist, do not run Tier-1 eval/freeze and do not start M2.

## [2026-07-16] M1 / visible-tier1-human-bank-preregistration
HYPOTHESIS: The missing independent-human gate can be resolved without reusing
training text or exposing Tier-2 data by freezing a visible, public FineWeb
bank before any sampler report is scored. A bank of 32 fingerprint-unique,
domain-distinct documents is sufficient for the immutable harness's
human-vs-human floor and is more stable than lowering its four-document rule.
SETUP: Operator-owned preparation batch after the authoritative M1 boundary.
Pinned `HuggingFaceFW/fineweb` snapshot `CC-MAIN-2024-10` at immutable dataset
revision `9bb295ddab0e05d785b879661af7260fed5140fc`. Preregistered deterministic
hash ranking under seed label `dftr-m1-tier1-visible-bank-v1`, a 512-record
eligible pool within at most 10,000 streamed rows, 40--220 word documents,
at most 2% non-Latin letters, and distinct domains. All fixed M0 train/dev
fingerprints are explicit exclusions. This batch adds the materializer,
config, and offline tests only; it does not stream the source, inspect sampler
outputs, run `harness eval`, mutate `harness/`, expose hidden data, freeze a
sampler, or start M2.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Source boundary | PREREGISTERED | Public agent-visible FineWeb crawl only; policy records `hidden_test_materialized=false`. |
| Independence | PREREGISTERED | M0 train and dev fingerprints are excluded; selected completion fingerprints must be unique. |
| Selection timing | PREREGISTERED | Config and algorithm are committed before source materialization or any Tier-1 scoring. |
| Human-floor cardinality | PREREGISTERED | Exactly 32 distinct-domain human documents, exceeding the immutable four-document minimum. |
| Output provenance | PREREGISTERED | Materializer will bind source revision, config hash, bank hash, fingerprints, domains, scan counts, and selection settings into a manifest. |
DECISION: keep as preparation only. This removes no gate until offline tests
pass, the preparation commit is published, the pinned source is materialized,
and an independent verifier confirms disjointness and reproducibility.
NEXT: Run the focused offline tests, publish this exact preparation commit,
then materialize and independently audit the bank before any Tier-1 scoring.

## [2026-07-16] M1 / visible-tier1-human-bank-materialization
HYPOTHESIS: The published visible-bank design is executable and reproducible
if the pinned FineWeb source yields exactly 32 domain-distinct eligible human
documents, the resulting bytes are disjoint from every existing M0 train/dev
fingerprint, and a clean second streaming pass reproduces both bank and
manifest byte-for-byte.
SETUP: Materialization occurred only after preparation commit
`283170f8577b61c926c30461e4298d85d6c35938` was published. Ran
`python -m data.tier1_bank --config
configs/m1/m1_tier1_human_bank_v1.json` against the preregistered immutable
FineWeb revision. The materializer scanned only until its frozen 512-record
eligible pool was filled and selected by the published seeded hash rule. Ran
offline structural/hash/disjointness checks, then repeated the complete source
stream and materialization without changing code or config. No generated
sampler completion was read, no harness metric was run, and no `harness/`,
M0 artifact, hidden data, model, ledger, compute, Tier 2/3, or M2 surface was
mutated.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Source scan | PASS | Scanned 1,470 streamed rows to collect the frozen 512-record eligible pool from `CC-MAIN-2024-10` at revision `9bb295ddab0e05d785b879661af7260fed5140fc`. |
| Bank cardinality | PASS | Exactly 32 completion-fingerprint-unique documents from 32 distinct domains; observed word-count range is 50--220 under the preregistered 40--220 filter. |
| Train/dev disjointness | PASS | Intersection with all eight fingerprints in the immutable M0 train and dev manifests is empty. |
| Provenance | PASS | Bank SHA-256 is `ebcff5bca1e6c75ab482aa831453a79986ffd700ee4a729de57fc8c496c6dc68`; manifest SHA-256 is `92a0366c313d007c5d602cc8758c148a775b94d5c927416dd668121bcf447ae1`; config SHA-256 is `090c3596853617dd4c5c0fbfa0189f177ab890994c74c3e36431c6469132265a`. |
| Deterministic reproduction | PASS | A second complete streaming/materialization pass reproduced the same bank and manifest hashes byte-for-byte. |
| Hidden-test wall | PASS | Artifact is explicitly visible Tier 1 only and contains no sealed-evaluator material or metadata. |
DECISION: keep. The independent-human data gate is resolved at the artifact
level without weakening the harness, duplicating training humans, or exposing
hidden data. This does not itself authorize scoring until the separate harness
engineering lane passes independent tests and the materialization commit is
published.
NEXT: Publish the exact bank artifacts, finish and independently verify the
external-bank/calibration/baseline/freeze harness repair, then evaluate the
already-completed sampler cells without inspecting or regenerating outputs.

## [2026-07-16] M1 / operator-sampler-exposure-incident
HYPOTHESIS: An accidental limited output exposure does not invalidate the
sampler comparison if it occurred after the data bank and decision rule were
published, is disclosed before scoring, and no human judgment or rule change
is permitted to affect the mechanical freeze decision.
SETUP: While locating the operator-materialized sampler tree, a diagnostic
command selected the alphabetically first sample JSONL and printed its first
four lines. That file contained the two documents for checkpoint seed 11,
sampler `cool_narrow_t0.8_p0.95`, sampling seed 101. The exposure occurred
after visible-bank preparation commit `283170f` and materialization commit
`2407add` were published, and after the sampler freeze rule had been
preregistered in the M1 plan. No other sample file was opened, no metric or
report had been computed, and no config, bank, calibration, gate, score,
tie-break, or sampler decision was changed.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Exposure scope | DISCLOSED | Exactly one two-row sampler cell was printed during path discovery. |
| Selection timing | PRESERVED | Visible human bank and freeze decision rule were already immutable and published. |
| Human selection influence | PROHIBITED | Final selection remains exclusively the preregistered all-gates/lowest-mean-S/tie-break algorithm; the exposed cell may not be manually favored or rejected. |
| Follow-on handling | LOCKED | Do not inspect further raw sampler completions before the complete mechanical Tier-1 report set and frozen decision artifact exist. |
DECISION: retain the comparison with this explicit caveat. The limited
post-preregistration exposure is not used as evidence and cannot change the
decision rule, but it is part of the scientific record.
NEXT: Complete harness verification, construct report paths without printing
sample contents, run the full fixed evaluation, and accept only the mechanical
freeze result.

## [2026-07-16] M1 / visible-bank-calibration-preregistration
HYPOTHESIS: Human-calibrated intervals computed from the frozen 32-document
visible FineWeb bank will be more defensible than transferring the earlier
two-document descriptive proposal, while remaining independent of model
outputs and directly reproducible by one checked-in entrypoint.
SETUP: Preparation-only batch after bank commit `2407add` and harness-boundary
commit `8922ea4`. Reused the already published confidence level 0.95,
resampling seeds `[404,505,606]`, subset fraction 0.8, and deterministic
central-quantile method. The new config binds the bank and manifest; the
analysis entrypoint now emits its schema, config/source hashes, full intervals,
point estimates, actual subset hashes and metrics, sensitivity summary, and
review limitations without manual enrichment. No calibration was computed or
transferred in this batch, and no raw sampler output, harness report, compute,
Tier 2/3, or M2 action occurred.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Human source | PREREGISTERED | Exact visible bank SHA `ebcff5bca1e6c75ab482aa831453a79986ffd700ee4a729de57fc8c496c6dc68`, 32 independent documents. |
| Procedure | PREREGISTERED | Same pre-existing 95% central-quantile method, seeds, and 80% subsets; no output-informed choice. |
| Direct reproduction | PREREGISTERED | One entrypoint must reproduce the complete review artifact byte-for-byte without post-hoc enrichment. |
| Transfer boundary | PRESERVED | Proposal remains inactive until exact SHA-validated operator transfer into immutable `harness/calibration.json`. |
DECISION: keep as preparation only. The prior two-document proposal remains
valid descriptive history but will not be activated for Tier-1 selection.
NEXT: Run focused tests, publish this preparation commit, produce and rerun the
32-human proposal, then transfer only its exact reviewed bytes.

## [2026-07-16] M1 / visible-bank-calibration-candidate-1
HYPOTHESIS: The preregistered central-quantile procedure will yield usable
human ranges for all five calibrated metrics when applied to the frozen
32-document visible bank.
SETUP: Ran the published entrypoint from commit `267c712` twice against exact
bank SHA `ebcff5bca1e6c75ab482aa831453a79986ffd700ee4a729de57fc8c496c6dc68`.
No sampler report or additional raw output was read. Review checked direct
byte reproduction, sample/subset cardinality, source hashes, point estimates,
intervals, and consistency with the preregistered principle that a
zero-repetition model is a failure rather than an optimum.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Direct reproduction | PASS | Both complete entrypoint runs produced artifact SHA-256 `25a77dd5ee2be521ad3daf235e9b70360d0478cc4c65d9a0a387a39a6bd7fef2` byte-for-byte. |
| Bank and subsets | PASS | 32 unique human documents; three distinct deterministic 26-document subsets for seeds `[404,505,606]`. |
| Self-BLEU | PASS | Human point estimate `0.0536711`; descriptive range `[0.0323373,0.0769309]`. |
| Script integrity | PASS | Human point estimate and range are exactly zero non-target-script characters. |
| Length ranges | PASS | Paragraph interval `[50,202]` tokens and sentence interval `[3,37]` tokens. |
| Repetition calibration | FAIL | Human point estimate is `1/32 = 0.03125`, but the per-document order-statistic interval is `[0,0]`; transferring it would make the observed human corpus fail its own aggregate rate and contradict the frozen requirement that zero repetition is failure. |
DECISION: discard candidate 1 for transfer while retaining it as a negative
calibration result. Direct reproducibility is fixed, but a single interval
method is not statistically appropriate for both continuous document metrics
and a rare binary per-document incidence rate.
NEXT: Before any model scoring, independently preregister and verify a
metric-specific calibration contract: retain deterministic central quantiles
for continuous metrics and use a deterministic binomial interval for the
repetition incidence rate. Produce a new candidate only after publication.

## [2026-07-16] M1 / visible-bank-calibration-candidate-2
HYPOTHESIS: The metric-specific v2 proposal and exact transfer validator will
agree across the repository's system-Python analysis runtime and the harness's
independently locked uv runtime.
SETUP: Integrated independently tested commit `1dbccdb`, reproduced proposal
SHA-256 `06e3a8ee5f4038c26161fcc43e6baa2959c1db50a9991fcfe48875afd07de420`,
then invoked the actual operator path `uv run harness
prepare-calibration-transfer` with that exact expected hash. No sampler report
or output was read and no transfer occurred after the command failed closed.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Repetition method | PASS | Wilson point estimate is `1/32 = 0.03125`; 95% interval is `[0.005537860164003122,0.15744263820012555]`, so zero correctly fails. |
| Same-runtime tests | PASS | Full harness `44/44`; focused tests `8/8`; system-Python proposal rerun was byte-identical. |
| Cross-runtime transfer | FAIL | System Python serialized `z=1.9599639845400536`, while the harness uv runtime computed `1.9599639845400534`; exact method-object equality rejected the reviewed artifact. |
| Fail-closed behavior | PASS | No immutable calibration file was changed and no model scoring began. |
DECISION: discard candidate 2 for transfer while retaining its correct
statistical result. Derived floating-point constants cannot be treated as
exact cross-runtime schema identity.
NEXT: Freeze the 95% z value as an explicit schema constant (or remove it from
exact identity and validate numerically), create a distinct v3 proposal, and
prove generation plus transfer across the two actual runtimes before any
calibration mutation.

## [2026-07-16] M1 / visible-bank-calibration-transfer-v3
HYPOTHESIS: Freezing the 95% normal critical value as the literal
`1.959963984540054` will make the metric-specific calibration proposal and
operator transfer reproducible across both actual Python runtimes while
preserving the nonzero human repetition floor.
SETUP: Integrated independent v3 harness commit `1612614`. Reproduced complete
proposal SHA-256 `db94fb4373ae81405435cc4ff28fbaf6fa132a6888dfd17ab407c6f06559f463`
under system Python, ran the real `uv run harness
prepare-calibration-transfer` command under the locked Python 3.11 harness,
reviewed the emitted object, and transferred that exact semantic object into
`harness/calibration.json`. No generated output or Tier-1 report was read and
no model scoring occurred before transfer.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Cross-runtime proposal | PASS | Two system-Python entrypoint runs reproduced v3 proposal SHA `db94fb43...f463`; the locked uv CLI accepted those exact reviewed bytes. |
| Independent harness tests | PASS | Full locked harness suite passed `45/45`; focused proposal/boundary suite passed `8/8`. |
| Repetition range | PASS | Human incidence `1/32 = 0.03125`; frozen Wilson 95% range `[0.005537860164003122,0.15744263820012558]`, so zero repetition fails as preregistered. |
| Continuous ranges | PASS | Self-BLEU `[0.032337252870941774,0.07693087958956342]`, script rate `[0,0]`, paragraph length `[50,202]`, sentence length `[3,37]`. |
| Immutable transfer | PASS | `harness/calibration.json` is frozen, semantically identical to the validator candidate, ready under harness checks, and has SHA-256 `4a71b081bbb05081a461d0968312aacd4472221fdeab9934d63202b2f8e6e039`. |
DECISION: keep. Calibration is now lawful, statistically coherent for the
observed rare incidence, directly reproducible, and bound to the frozen
32-document visible bank. V1 and v2 remain negative results.
NEXT: Publish this exact transfer before running only the default-sampler
bootstrap reports needed to propose and transfer the frozen SFT baseline.

## [2026-07-16] M1 / default-sampler-bootstrap-preregistration
HYPOTHESIS: Nine default-sampler cells can establish the SFT component and
validity baseline without circularly selecting a sampler, provided every
report is bound to the exact frozen bank, calibration, placeholder baseline,
source index, and materialized sample bytes.
SETUP: Preparation-only batch after calibration transfer commit `2afa698`.
Added a batch runner that loads the immutable dev embedder once, maps only
hash-bound local artifacts from sampler run `dftr-1784183624-2e567266`, and
rejects wrong bank/calibration/baseline provenance. Bootstrap is restricted to
`default_t1.0_p1.0`: three training seeds times three sampling seeds = nine
reports. The secondary judge is neutral for this bootstrap because it does
not enter baseline statistics; full frozen-sampler reports will use the fixed
gateway judge. No report was run in this batch and no sample content was
printed.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Source index | PREREGISTERED | SHA `3fe0cade233bdaaa6c724a525076771f7340beb9805471da55a24e3e78141763`, 45 original cells. |
| Bootstrap selection | PREREGISTERED | Exactly nine default-sampler cells; no post-result sampler choice. |
| Frozen inputs | PREREGISTERED | Human manifest ID `92a0366c...`, calibration SHA `4a71b081...`, placeholder baseline SHA `2eb736f9...`. |
| Output contract | PREREGISTERED | Every report and sample receives a SHA-bound entry in `m1.tier1_eval_index.v1`. |
DECISION: keep as preparation only. The runner must pass focused tests and be
published before loading the embedder or producing any bootstrap report.
NEXT: Test and publish this exact runner/config, then run the nine-cell
bootstrap once and build the default-sampler baseline proposal.

## [2026-07-16] M1 / default-sampler-bootstrap-results
HYPOTHESIS: The preregistered default sampler will provide stable raw SFT
component statistics and reveal whether the tiny M1 SFT baseline lies inside
the independently calibrated human diversity ranges before any alternate
sampler is compared.
SETUP: From published runner/config commit `a0e93f0`, evaluated exactly nine
default-sampler cells: checkpoint seeds `[11,29,47]` times sampling seeds
`[101,202,303]`. Loaded frozen `BAAI/bge-small-en-v1.5` once and used the exact
32-human bank, calibration SHA `4a71b081...`, placeholder baseline SHA
`2eb736f9...`, and neutral quality judge. Reports contain metrics only; no
additional raw completion was printed or inspected. No GPU/provider spend,
sampler selection, freeze, Tier 2/3, or M2 action occurred.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Report completeness | PASS | Exactly 9 reports and one index; index SHA `359c072ddc163004e5c2e35df42bef6366c34361865427a193d1b095a4e749fc`; one bank/calibration/baseline provenance tuple. |
| Raw component stability | PASS | Across nine cells: semantic MMD mean `0.00156676` (SD `0.0225790`), lexical L2 mean `0.0453387` (SD `0.0290608`), structural distance mean `0.467447` (SD `0.0852718`). |
| Semantic floor delta | DESCRIPTIVE | Mean `-0.0547374`, range `[-0.0747365,-0.0138923]`; with only two generated documents per cell and an unbiased estimator, this is not evidence of human equivalence. |
| Language integrity | PASS | `9/9` cells lie inside the frozen non-target-script interval. |
| Human-calibrated collapse gate | FAIL | `0/9` pass. Generated self-BLEU mean `0.0129224`, below human range `[0.0323373,0.0769309]`; repetition mean `0.388889`, above human range `[0.00553786,0.157443]`. |
| Validity non-inferiority | BOOTSTRAP ONLY | Recall/unsupported gates remain fail-closed `0/9` because the baseline has not yet been transferred; raw sample metrics will populate the default baseline proposal. |
| Authorship probe | DESCRIPTIVE | AUC mean `0.300347` with high cell variance; secondary only and not a promotion claim. |
DECISION: keep as the required default-sampler bootstrap and as a negative SFT
quality signal. It does not justify budget expansion, promotion, or a claim
that tuning improved human-likeness. The component statistics are still
required to standardize the complete sampler comparison.
NEXT: Publish these exact reports, compute and independently transfer the
default-sampler baseline proposal, then rerun all 45 cells with frozen
provenance and accept only the preregistered mechanical outcome.

## [2026-07-16] M1 / default-sampler-baseline-transfer
HYPOTHESIS: The nine published default-sampler reports will deterministically
produce an operator-transferable SFT baseline bound to the exact frozen human
bank and calibration, breaking the former selection/baseline circularity.
SETUP: Ran the published `baseline-stats` config twice against bootstrap index
SHA `359c072d...`; both produced proposal SHA
`8b1623c3368f9c6ec475a7a833a3cf0730807e41bd4690df1048a497a9f8c227`.
The real locked `uv run harness prepare-baseline-transfer` command accepted
the exact expected SHA. Reviewed and transferred the emitted semantic object
into `harness/baseline_stats.json`. No new evaluation, raw-output read,
provider call, compute, sampler selection, Tier 2/3, or M2 action occurred.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Direct reproduction | PASS | Two proposal runs were byte-identical at SHA `8b1623c3...c227`. |
| Frozen provenance | PASS | Default sampler only; 9 reports; human bank ID `92a0366c...`; calibration SHA `4a71b081...`. |
| Component baseline | PASS | Semantic MMD `0.00156676 +/- 0.0225790`; lexical L2 `0.0453387 +/- 0.0290608`; structural distance `0.467447 +/- 0.0852718`. |
| Validity baseline | NEGATIVE | Mean outline-fact recall `0.25`; mean unsupported-claim rate `0.854233`, confirming the tiny SFT control is weak rather than establishing good quality. |
| Immutable transfer | PASS | Harness baseline is ready and frozen; file SHA-256 `53de46c79b31a63262cc6c2329bb6acec81ae0a5e5ed77af1df2408b76f262fc`. |
DECISION: keep as the required SFT control, not as a quality success. It gives
the complete sampler comparison a fixed standardization and non-inferiority
reference without depending on the eventual selected sampler.
NEXT: Publish this exact transfer, preregister all 45 frozen-provenance reports
with the fixed gateway judge, then run the full mechanical sampler comparison.

## [2026-07-16] M1 / full-tier1-sampler-preregistration
HYPOTHESIS: At least one of the five preregistered decoding settings may reduce
the standardized distributional gap while remaining non-inferior to the weak
default SFT validity baseline and inside the human-calibrated diversity ranges.
If none passes every gate, the correct outcome is no frozen deployment sampler.
SETUP: Preparation-only batch after frozen baseline commit `4a03489`. Bound all
45 existing sampler cells to source index SHA `3fe0cade...`, bank ID
`92a0366c...`, calibration SHA `4a71b081...`, and baseline SHA `53de46c7...`.
The exact five sampler IDs, three checkpoint seeds, and three sampling seeds
remain unchanged. Full reports use the fixed gateway quality judge as a
secondary metric; the mechanical freeze rule remains all gates, then lowest
mean standardized S, with preregistered uncertainty tie-breaks. No report,
provider call, raw-output read, freeze, Tier 2/3, or M2 action occurred here.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Full factorial | PREREGISTERED | 3 checkpoints x 5 samplers x 3 sampling seeds = 45 reports. |
| Immutable inputs | PREREGISTERED | Exact source index, human bank, calibration, and baseline hashes are mandatory per report. |
| Judge role | PREREGISTERED | Fixed randomized-order gateway judge is secondary only and cannot select by itself. |
| Failure outcome | PREREGISTERED | If no setting passes all four hard gates across its reports, freeze must fail closed; do not choose the least-bad sampler. |
DECISION: keep as preparation only. Publish before any gateway judge call or
full standardized report.
NEXT: Run focused tests and publish; then execute the 45 reports once, audit
provider cost/provenance, and apply the mechanical freeze rule without manual
sample inspection.

## [2026-07-16] M1 / full-tier1-sampler-results
HYPOTHESIS: At least one preregistered sampler setting will pass every hard
Tier-1 gate across the three checkpoint seeds and three sampling seeds, making
it eligible for the deterministic lowest-mean-S freeze rule.
SETUP: Evaluated all 45 preregistered cells from published preparation commit
`fd8ee35` with the frozen 32-document human bank, calibration SHA `4a71b081...`,
baseline SHA `53de46c7...`, and fixed gateway judge. The first online attempt
stopped before reports or judge calls after a Hugging Face metadata 504; the
successful restart used the already-cached embedder in offline mode. Audited
every report hash and provenance tuple without inspecting additional raw model
output. Applied the published mechanical freeze command once after completion.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Completeness and provenance | PASS | Exactly 45 reports; index SHA `09840716aebb5357b355240512bb773ba5a4b21a02e605e7748aa0544d594206`; zero report-hash or bank/calibration/baseline provenance errors. |
| Language integrity | PASS | `45/45` reports pass. |
| Outline-fact recall | MOSTLY PASS | `44/45` reports pass; the one failure is in `warm_t1.2_p0.95`. |
| Unsupported-claim non-inferiority | FAIL | Only `18/45` reports pass; sampler pass counts are `4,2,3,4,5` in preregistered order. |
| Human-calibrated no-collapse | FAIL | `0/45` reports pass. Mean generated self-BLEU by sampler is `0.01230–0.01359`, below the human interval `[0.03234,0.07693]`; mean repetition is `0.27778–0.66667` versus human interval `[0.00554,0.15744]`. |
| Aggregate hard-gate eligibility | FAIL | Every sampler has `0/9` all-gate cells; no sampler is eligible. |
| Secondary quality judge | POSITIVE BUT NON-PROMOTING | Mean preference win rate ranges from `0.8333` to `1.0000` and mean JMQ from `1.6667` to `2.0000`; these cannot override the preregistered collapse and validity failures. |
| Mechanical freeze | FAIL CLOSED | `freeze-sampler` exited with `M1ConfigError: no sampler settings passed every hard gate`; no least-bad sampler was selected. |
| Spend after screen | PASS | Modal GPU committed remains `$0.683574 / $40`; provider spend is `$0.039521 / $100`. |
DECISION: discard all five sampler settings as deployment candidates from this
training artifact. The run is valuable evidence that the end-to-end harness,
provenance, judge, and fail-closed selection path work, but it is not evidence
that the current tuned model improved. Do not enter M2, Tier 2/3, 14B, or expand
budget from this result. The most likely limiting factor is the deliberately
minimal M0 fixture: six synthetic `.example` training documents and only 474
training tokens per seed, which was sufficient to prove plumbing but not to
support a scientific tuning claim.
NEXT: Preserve this negative M1 boundary, then preregister a cheap data-scale
recovery experiment using real, disjoint training/dev documents. Require a
small pilot to clear the same diversity and validity gates before authorizing
the full 20–30K-brief synthesis or any larger-model spend.

## [2026-07-16] M1 / realdata-pilot-source-preregistration
HYPOTHESIS: Replacing the six-document synthetic fixture with a modest,
domain-diverse real-FineWeb corpus will test whether data scale and realism,
rather than sampler choice, caused the failed M1 SFT screen.
SETUP: Preparation only. Preregistered a deterministic 320-document source
selection from immutable FineWeb revision `9bb295dd...`, split into 256 train
and 64 dev documents. Selection requires distinct domains, 80–220 words,
target-script integrity, and fixed selection/split seeds. It excludes every
M0 and visible Tier-1 bank fingerprint and also excludes their domains. The
sealed evaluator remains a different domain/time slice and is not read or
materialized. Source documents are staged outside Git and will be stored on
the constrained Modal volume; only their hash-bound manifest is published.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Source config | PREREGISTERED | SHA `6ad65d3d40c52132fa326ed64a7a1ec4a5031abc6a93eafd320958f1edf7a776`; exact dataset/config/revision/split frozen. |
| Selection cardinality | PREREGISTERED | 320 documents, 256 train / 64 dev, all domains distinct. |
| Test-wall protection | PREREGISTERED | M0 and Tier-1 visible fingerprints/domains excluded; hidden Tier-2 data remains unavailable. |
| Evidence scope | PREREGISTERED | Recovery pilot only; not promotion evidence and no detector/Tier-2 claim. |
| Scale policy | PREREGISTERED | Qwen3-1.7B is required for the eventual pilot because the protocol marks 0.6B as plumbing-only. No 4B/14B or budget expansion. |
| Verification | PASS | Seven focused source-selection/Tier-1-bank tests pass; deterministic split, exclusion, hash binding, and fail-closed domain scarcity are covered. |
DECISION: keep this exact source selection and publish it before streaming any
candidate records. Materialization does not authorize provider calls or GPU
training; those receive separate hash-bound preregistrations after the source
manifest exists.
NEXT: Publish this preparation commit, materialize twice to prove byte-level
reproducibility, upload the two source splits to the constrained volume, then
preregister capped brief-synthesis jobs.

## [2026-07-16] M1 / realdata-pilot-source-transport-amendment
HYPOTHESIS: Pinning the first immutable Parquet shard explicitly will preserve
the preregistered dataset/revision/selection semantics while avoiding the
intermittently unavailable Hugging Face dataset-listing API.
SETUP: The first two materialization attempts failed before reading or writing
any candidate record: the online attempt received HTTP 504 from dataset-info,
and offline mode correctly refused the uncached listing. The dataset's public
immutable tree identifies `data/CC-MAIN-2024-10/000_00000.parquet` at the same
already-frozen revision. Amended the source config to name only that shard and
the loader to stream its revision-qualified Parquet URL directly. Selection,
exclusions, sizes, filters, seeds, and outputs are unchanged.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Paid effects | NONE | Zero provider calls, GPU seconds, or output records before amendment. |
| Scientific selection | UNCHANGED | Same immutable dataset revision and CC snapshot; exact shard is now an additional frozen constraint. |
| Failure recovery | FAIL CLOSED | No fallback dataset, moving revision, or alternate time slice was used. |
DECISION: keep as a transport-only preregistration amendment made before any
candidate content was materialized.
NEXT: Publish the amendment, then make one exact materialization attempt and
verify reproducibility before any synthesis call.

## [2026-07-16] M1 / privileged-source-materialization-preregistration
HYPOTHESIS: Moving the exact frozen FineWeb source pull behind the constrained
gateway will recover from the local Hugging Face outage while preserving the
credential boundary and making the future full corpus path reproducible.
SETUP: Added a third fixed-code gateway task kind,
`source_materialization`. It accepts only a fully pinned source, a maximum of
5,000 records, three checkpoint-volume output URIs, one GPU field fixed at the
existing single-resource policy, and a preregistered smoke/screen budget. It
does not execute repository experiment code, reserve GPU/API spend, or expose
the Hugging Face token. The exact pilot config embeds the 40 excluded M0/Tier-1
fingerprints and domains and freezes one revision-qualified Parquet shard.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Gateway policy | PASS | New task is fail-closed on unpinned sources, unsafe outputs, and corpus sizes above 5,000. |
| Credential boundary | PASS | HF token is consumed only inside the deployed fixed-code worker and removed from its environment before selection; no research subprocess is launched. |
| Pilot config | PREREGISTERED | Canonical config hash `42b0eeec347c9078926ca9bddd4462e68af78bb24a1f5a83d88a50b5bf8e9fff`; 40/40 fingerprint and 40/40 domain exclusions exactly match the published manifests. |
| Verification | PASS | All 16 infrastructure tests pass, including deterministic/hash-bound source selection and policy rejection of non-volume outputs. |
| Budget | PASS | Source selection reserves `$0` GPU and `$0` provider spend; the Modal workspace cap remains authoritative for service overhead. |
DECISION: publish and deploy this reviewed gateway revision. This is a source
transport repair, not a model-result change and not permission to scale.
NEXT: Add the open ledger preregistration, launch one 320-document source job,
verify its manifest and reproducibility, then separately preregister synthesis.

## [2026-07-16] M1 / realdata-pilot-source-attempt-1
HYPOTHESIS: The privileged gateway can materialize the preregistered 320-record
source corpus despite the local Hugging Face listing failure.
SETUP: Deployed gateway commit `66d1895`, opened the published preregistration,
and launched source-only smoke run `dftr-1784190213-0ceedecd` at git SHA
`81d7df3`. The gateway accepted canonical config hash `42b0eeec...`, classified
the task as CPU source materialization, and reserved `$0` GPU/API spend.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Policy and launch | PASS | Exact published config accepted; single resource, 20-minute hard timeout, pinned revision/shard, and volume-only outputs enforced. |
| Source access | FAIL | The fixed worker received a Hugging Face `ReadTimeout` before selecting any record. The local API, direct pinned shard, and authenticated Modal route all timed out during the same service incident. |
| Side effects | PASS | `0` records, `0` tokens, `0` accelerator-seconds, and `$0` charged; terminal failed ledger row recorded. |
| External status | BLOCKED EXTERNALLY | Hugging Face's official status page reports some services down as of `2026-07-16 07:56 UTC`. |
DECISION: keep as a zero-cost infrastructure/outage result. Do not change the
dataset, revision, shard, selection, or exclusions to manufacture progress.
Retain the open preregistration and retry with backoff after Hub/CDN recovery.
NEXT: Monitor Hugging Face status on a longer cadence; when operational,
launch the same config unchanged, verify the volume manifest, and only then
authorize capped brief synthesis.

## [2026-07-16] M1 / source-transport-recovery-hardening
HYPOTHESIS: A bounded retry policy and longer Hub download/etag timeouts will
prevent brief transient CDN stalls from terminating the next unchanged source
job after the provider reports recovery.
SETUP: Fixed-code gateway-only change after attempt 1. Set Hub download and
etag timeouts to 60 seconds and `DownloadConfig.max_retries=5` inside the
privileged source worker. The dataset, revision, shard, selection, exclusions,
output contract, config hash, preregistration, and zero-dollar reservation are
unchanged. No job was launched while official status still reported downtime.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Scientific semantics | UNCHANGED | Transport tolerance only; no candidate content has been read. |
| Retry bound | PASS | Five retries with 60-second Hub timeouts remain inside the existing 20-minute smoke hard kill. |
| Credential handling | PASS | HF token stays in `DownloadConfig`; provider key is removed before dataset code loads. |
DECISION: keep and deploy before the next source attempt.
NEXT: Wait for operational status, then reuse the exact open preregistration
and unchanged config hash `42b0eeec...`.

## [2026-07-16] M1 / brief-synthesis-contract-hardening
HYPOTHESIS: Enforcing the disclosed brief schema, exact 25% empty-outline arm,
and resumable per-record failure handling before paid synthesis will prevent a
single malformed response from silently corrupting the real-data pilot.
SETUP: Fixed-code privileged worker change only; no provider call or source
record was available. Added deterministic rank-based empty-outline assignment
with exactly `floor(N/4)` records, strict type/value validation for every brief
field, nonempty structured outlines outside the empty arm, verbatim quotation
traceability to the source document, two bounded response attempts, per-record
failure logging, and completion only when the exact target ID set exists.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Empty-outline condition | PASS | Deterministic and exact: 16/64 dev and 64/256 train when the pilot source exists. |
| Schema validation | PASS | Prompt/use-case/style/detail/length/em-dash/outline fields fail closed on wrong types or values. |
| Grounding check | PASS | Every emitted quotation outside the empty arm must occur verbatim in the human source document. |
| Resume semantics | PASS | Existing fingerprint IDs are skipped; a run completes only when completed IDs exactly match target IDs and no record failed. |
| Verification | PASS | 22 infrastructure tests pass, including malformed target length, detail mode, quotation, and outline cases. |
DECISION: keep and deploy before any synthesis preregistration. This repairs
data validity and does not weaken or reinterpret Tier-1 model gates.
NEXT: Continue source-host backoff; after source hashes exist, bind two capped
synthesis configs to the exact train/dev source bytes and use this contract.

## [2026-07-16] M1 / pilot-artifact-validator-preparation
HYPOTHESIS: A single fail-closed operator validator can turn recovered source
and synthesized brief bytes into trustworthy training inputs without manual
document inspection or accepting provider-produced provenance fields on faith.
SETUP: Preparation only during the source-host outage. Added a validator that
recomputes every source file SHA, split hash, completion fingerprint, count,
domain uniqueness, train/dev disjointness, and exclusion intersection; then
requires exact source/brief ID equality, byte-identical preserved source
fields, the privileged brief contract, and the exact empty-outline ID set.
It emits a metadata-only validation artifact after both splits pass.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Source provenance | PASS | Dataset/config/revision/split/shard, file SHA, split hash, fingerprints, domains, labels, and counts are independently recomputed. |
| Test-wall exclusions | PASS | Any overlap with the published excluded fingerprint or domain sets fails validation. |
| Brief binding | PASS | Brief IDs must exactly equal source IDs and all source fields, including completion, must remain byte-identical. |
| Schema/arm binding | PASS | Reuses the deployed schema/grounding contract and independently checks the exact empty-outline ID set. |
| Verification | PASS | All 13 data tests and all 22 infrastructure tests pass; mutation tests reject changed completions and wrong empty-outline assignment. |
DECISION: keep as the mandatory post-download/post-synthesis gate. No source,
brief, provider call, model run, or evaluation was produced here.
NEXT: On host recovery, download the three source artifacts to the ignored
operator staging area, run this validator after capped synthesis, and publish
only its metadata result plus training config hashes.

## [2026-07-16] M1 / synthesis-input-binding-preparation
HYPOTHESIS: Binding each paid synthesis job to the exact recovered source SHA,
record count, split hash, and deployed provider model will prevent stale or
cross-split volume artifacts from consuming budget or entering training.
SETUP: Preparation only. Extended both client and server policy to require a
lowercase 64-character input SHA, bounded record count, and frozen API model.
The privileged worker now recomputes the volume input SHA before any call and
fully validates every existing output row against the current source before
resuming. Added a deterministic config builder that consumes the eventual
source manifest and emits separate 256-train and 64-dev synthesis configs with
exact hashes, counts, output URIs, model, cost caps, and provenance.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Input binding | PASS | Wrong source SHA fails before provider access; max records must be `1..50000`. |
| Model binding | PASS | Config API model must exactly match the frozen deployment model. |
| Resume safety | PASS | Existing rows must have known unique IDs, unchanged source fields, valid briefs, and correct empty-outline assignment. |
| Config generation | PASS | Train/dev configs inherit source SHA/split hash/count and exact expected empty counts (`64/256`, `16/64`). |
| Verification | PASS | All 15 data tests and 23 infrastructure tests pass. |
DECISION: keep and deploy this policy before host recovery. Proposed pilot caps
remain `$5` train plus `$2` dev inside the existing `$100` provider ceiling;
they are not reserved or authorized until real source hashes are present and
the generated configs are separately preregistered.
NEXT: Continue source backoff. On recovery, materialize and validate source,
generate exact synthesis configs, review canonical hashes, preregister, and
launch dev before train as the cheapest contract proof.

## [2026-07-16] M1 / realdata-training-boundary-preparation
HYPOTHESIS: A separate real-data protocol version can admit the validated
256/64 pilot without weakening or overwriting the original hard-coded M0
fixture contract.
SETUP: Preparation only. The existing `m1.v1` path still requires the original
six/two split hashes. Added `m1.realdata-pilot.v1`, which requires an exact
operator-published fixed-manifest SHA, schema `dftr.realdata_pilot_fixed_inputs.v1`,
cardinality `train=256/dev=64`, config/manifest path and split-hash equality,
then recomputes brief file SHAs, counts, split labels, unique fingerprints,
and fingerprint split hashes inside the GPU worker before model loading.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Legacy fixture boundary | PASS | `m1.v1` behavior and hard-coded M0 hashes are unchanged. |
| Pilot manifest binding | PASS | Manifest schema, SHA, paths, counts, and split hashes are required before record loading. |
| Worker-side data binding | PASS | Brief bytes, record counts, labels, unique IDs, and split hashes are revalidated before tokenizer/model work. |
| Model scope | UNCHANGED | Existing workflow still requires Qwen3-1.7B and training seeds `[11,29,47]`; 0.6B/4B/14B are not admitted. |
| Verification | PASS | All 14 experiment tests pass, including manifest-hash and mutated-brief rejection. |
DECISION: keep as the only training admission path for this recovery pilot.
No fixed manifest or training config can be generated until source and brief
validation artifacts exist, so this does not preregister or authorize a run.
NEXT: Resume unchanged source materialization after host recovery, then flow
its validated hashes through synthesis and this training boundary.

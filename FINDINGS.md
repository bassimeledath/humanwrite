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

## [2026-07-16] M1 / pilot-sft-config-builder-preparation
HYPOTHESIS: Generating the SFT fixed manifest and run config mechanically from
the completed validation artifact will eliminate manual hash/path transcription
between paid synthesis and GPU training.
SETUP: Preparation only. Added a builder that refuses any validation artifact
other than `dftr.realdata_pilot_validation.v1`, requires exact 256/64 counts
and source/brief split-hash equality, then emits a fixed-input manifest and a
Qwen3-1.7B three-seed SFT screen config. The fixed manifest binds volume brief
paths, file SHAs, split hashes, source-manifest SHA, validation SHA, prompt
format, token limits, and seeds. The run retains the original LoRA/training
hyperparameters so data realism/scale is the intended changed factor.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Validation dependency | PASS | Missing/wrong schema, cardinality, counts, or split hashes fail before config emission. |
| Training provenance | PASS | Generated config includes the exact fixed-manifest SHA consumed by `m1.realdata-pilot.v1`. |
| Controlled model setup | PASS | Qwen3-1.7B, seeds `[11,29,47]`, rank-64 LoRA, one epoch, LR `2e-4`, L40S screen budget. |
| Authorization | NONE | Builder emits files only; ledger preregistration and launch remain separate after real validation evidence. |
| Verification | PASS | Combined data and experiment suite passes `31/31`. |
DECISION: keep as the mandatory mechanical bridge to SFT. Do not generate or
publish a concrete config until the real source and brief validator passes.
NEXT: Continue host recovery monitoring; after dev/train synthesis validation,
generate, publish, preregister, and budget-check this exact SFT config.

## [2026-07-16] M1 / directional-sampler-preparation
HYPOTHESIS: Before spending on a complete five-sampler sweep over 64 dev
records, a default-sampler directional screen over a fixed 16-record subset
can test whether real-data SFT improved the hard-gate failure while preserving
all three training and three sampling seeds.
SETUP: Preparation only. Added a one-point default sampler grid admitted only
for `m1.realdata-pilot.v1` stage `directional_default`. The mechanical builder
validates the eventual three-seed checkpoint manifest, distinct adapter hashes,
positive train tokens, model revision, fixed dev brief SHA, and exactly 64 dev
records; it then rank-selects and publishes 16 fixed fingerprints and emits a
hash-bound sampler config. The GPU worker independently rechecks checkpoint
manifest SHA/seeds/adapters/tokens and exact subset membership/hash.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Directional design | PREREGISTERED STRUCTURE | `3` training seeds × `3` sampling seeds × `1` default sampler × `16` docs = `144` generated documents. |
| Cost containment | PASS | Avoids the premature `3×3×5×64 = 2,880`-document full sweep; full sampler selection remains conditional on directional evidence. |
| Seed evidence | PASS | Training-seed and independent sampling-seed variance are both retained. |
| Legacy/full grid | UNCHANGED | All non-directional paths still require exactly five sampler points. |
| Worker provenance | PASS | Checkpoint manifest bytes, seeds, distinct adapters, train tokens, dev membership, and subset hash are revalidated before generation. |
| Verification | PASS | Combined data and experiment suite passes `35/35`. |
DECISION: keep as the first post-SFT evidence run. It is directional screening,
not deployment sampler freeze, Tier-2 evidence, or promotion. Advance to the
five-point screen only if the default sampler shows meaningful hard-gate and
distributional improvement versus the failed tiny-data baseline.
NEXT: Continue source recovery; later generate this config mechanically from
the real validated briefs and completed checkpoint manifest.

## [2026-07-16] M1 / realdata-pilot-sft-completion
HYPOTHESIS: Training Qwen3-1.7B on the validated 256-document real-FineWeb
pilot will produce three healthy, distinct LoRA checkpoints suitable for the
fixed directional quality screen, unlike the underpowered six-document
baseline.
SETUP: Completed preregistered run `dftr-1784196107-48549e4e` from published
git SHA `23e1813` and canonical config hash `b57d06ba`. The run used one L40S,
one epoch, rank-64 LoRA, seeds `[11,29,47]`, and pinned model revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Completion | PASS | All three seeds completed with return code 0 in 146.721 accelerator-seconds for $0.095427. |
| Token accounting | PASS | Each seed processed 46,732 tokens; aggregate train tokens are exactly 140,196. |
| Checkpoint identity | PASS | Adapter SHA-256 values are `1323d1ea...`, `b2bd43f3...`, and `5d934d1b...`; all three differ. |
| Training stability | PASS | Per-seed train losses were 3.003040, 2.999729, and 3.003341 with no non-finite loss or failed seed. |
| Quality claim | PENDING | Loss and checkpoint integrity prove a valid tuning run, not writing-quality improvement. No promotion or scale-up follows without directional hard-gate evidence. |
| Directional config | PASS | Mechanical builder bound the exact checkpoint manifest SHA `bb27ef80...`, fixed dev SHA, model revision, 16-document subset hash `18a8031e...`, and 144 expected outputs; focused tests pass 10/10. |
DECISION: keep all three checkpoints and advance to the preregistered bounded
directional screen. Do not interpret training loss as a model-quality result.
NEXT: Publish and launch the fixed 144-document default-sampler evaluation,
then compare validity, collapse, claim support, and distributional metrics
against the failed tiny-data baseline before authorizing any larger sweep.

## [2026-07-16] M1 / realdata-directional-generation-completion
HYPOTHESIS: The fixed 16-document directional screen can produce a complete,
seed-balanced artifact suitable for a bounded hard-gate comparison before any
larger sampler sweep.
SETUP: Completed run `dftr-1784196454-0822114d` from immutable git SHA
`3544b3d` and canonical config hash `483d9dff`, covering three checkpoint seeds,
three sampling seeds, one default sampler, and 16 fixed dev documents.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Run completion | PASS | Return code 0 after 1,364.672 L40S-seconds for $0.887583. |
| Cell coverage | PASS | Nine JSONL files exist: 3 checkpoint seeds × 3 sampling seeds. |
| Document coverage | PASS | Exact total is 144 rows, matching the preregistered design. |
| Token accounting | PASS | Terminal wrapper accounting reports 22,759 generated tokens. |
| Quality conclusion | PENDING | Generation completeness is not evidence of improvement; Tier-1 hard-gate scoring remains required. |
DECISION: preserve the complete directional artifact and proceed only to the
bounded Tier-1 scoring boundary. Do not authorize a full sweep or scale-up yet.
NEXT: Validate record-level provenance and compute the directional Tier-1
comparison against the failed tiny-data baseline in one terminal job.

## [2026-07-16] M1 / realdata-directional-tier1-preregistration
HYPOTHESIS: The completed real-data pilot will materially improve the fixed
default sampler's validity and no-collapse profile versus the failed six-doc
baseline, justifying consideration of the larger five-sampler screen.
SETUP: Bound one terminal Tier-1 batch to source-index SHA `e4fb9d24...`, the
frozen 32-document human bank, calibration SHA `4a71b081...`, baseline SHA
`53de46c7...`, and exactly nine reports from three checkpoint seeds by three
sampling seeds. The fixed gateway judge remains secondary and non-promoting.
DECISION: publish this configuration before evaluation, then execute it once.
NEXT: Compute nine reports, audit hashes/provenance and hard gates, and stop at
the evidence decision without opening a recurring monitoring loop.

## [2026-07-16] M1 / realdata-directional-tier1-results
HYPOTHESIS: The real-data pilot will materially improve validity and
no-collapse across all nine fixed default-sampler seed cells versus the failed
six-document baseline.
SETUP: Computed all nine preregistered reports using the frozen human bank,
calibration, baseline, cached dev embedder, and secondary gateway judge. Index
SHA is `32b2cfa4...`; provider cost increased by `$0.061755`.
RESULTS:
| item | real-data pilot | tiny-data default baseline | outcome |
| --- | ---: | ---: | --- |
| All hard gates | 0/9 | 0/9 | FAIL CLOSED |
| No-collapse gate | 4/9 | 0/9 | MATERIAL IMPROVEMENT |
| Mean self-BLEU | 0.05569 | 0.01292 | MOVED INTO HUMAN RANGE ON SOME CELLS |
| Mean repetition | 0.18056 | 0.38889 | MATERIAL IMPROVEMENT |
| Outline-fact gate | 0/9 | 9/9 | SEVERE REGRESSION |
| Unsupported-claim gate | 0/9 | 3/9 | REGRESSION |
| Language-integrity gate | 9/9 | 9/9 | PRESERVED |
| Secondary preference win rate | 0.3125 | 0.8889 | REGRESSION, NON-PROMOTING |
| Mean authorship AUC | 0.64757 | 0.30035 | FARTHER FROM 0.5 |
DECISION: fail closed. The larger five-sampler sweep, Tier 2/3, 14B scale-up,
and budget expansion are not authorized. Real-data scale fixed much of the
collapse problem but created an outline/factual-control failure, so the next
experiment must target adherence rather than sampling breadth.
NEXT: Diagnose the fixed prompt/target construction and preregister a bounded
adherence recovery before spending on another generation sweep.

## [2026-07-16] M1 / realdata-pilot-source-attempt-2
HYPOTHESIS: The recovered pinned resolver plus bounded worker timeouts will
allow the unchanged 320-document source config to materialize.
SETUP: Relaunched canonical config hash `42b0eeec...` as zero-reservation run
`dftr-1784191402-9b8a3fd1` after the exact revision-qualified shard returned a
valid resolver redirect. The worker reached source processing but stopped
before reading or writing records when its URI guard called `Path.resolve()`:
Modal resolves the `/checkpoints` mount alias to `/__modal/volumes/...`, so the
subsequent comparison against literal `/checkpoints` falsely labeled every
valid volume path unsafe.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Source transport | RECOVERING | Pinned resolver responded and confirmed repository revision `9bb295dd...`; prior host-level failure moved forward. |
| Materialization | FAIL | `ValueError: unsafe volume URI` before any record selection or output. |
| Side effects | PASS | `0` records, `0` tokens, `0` accelerator-seconds, `$0` GPU/API spend; terminal ledger row recorded. |
| Root cause | CONFIRMED | Symlink-aware resolution changed the trusted mount prefix; the URI itself contained no traversal. |
| Repair | PASS | Replaced filesystem-resolution comparison with lexical `PurePosixPath` validation that rejects outside schemes, `..`, and empty paths while returning the mounted `/checkpoints/...` alias unchanged. All 28 infrastructure tests pass. |
DECISION: keep as a zero-cost wrapper negative result. Publish and deploy the
lexical guard, then retry the exact source config without changing scientific
inputs or authorization.
NEXT: Deploy the guard fix and launch attempt 3 under the same open
preregistration; verify source manifest hashes before synthesis.

## [2026-07-16] M1 / realdata-pilot-source-completion
HYPOTHESIS: With the lexical volume guard deployed, the unchanged pinned source
job will materialize a reproducible, excluded, domain-diverse 256/64 corpus.
SETUP: Attempt 3 `dftr-1784191530-7f7fbd7f` used canonical config hash
`42b0eeec...` at git SHA `5a413e5`; attempt 4
`dftr-1784191623-36875c5d` repeated it unchanged. Both were CPU-only with zero
GPU/API reservation. Downloaded the three artifacts to ignored operator
staging and ran the independent source validator without inspecting document
content.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Cardinality | PASS | Exactly 320 records: 256 train, 64 dev, 320 unique domains. |
| Selection pool | PASS | Scanned 6,023 rows and rank-selected from 1,600 eligible unique records under the frozen filters/seeds. |
| Train provenance | PASS | File SHA `3884fdf086520f1e295ecaa37ac4a866199399d659eb1f9ab2ad8886d9e2d704`; split hash `833b2c259a8e68c110d4668121213f3f98b47dac5f70f26047d69cbd2cca932e`. |
| Dev provenance | PASS | File SHA `7012e7a84e11d14b6e8e46b827de77d44bc46401946a7cb848f0587633b05ac5`; split hash `849f4e5bcbf1e78330f9d6f0e812b9d359d638ee5ed1a422e9d01c54b69e9722`. |
| Test-wall exclusions | PASS | Zero overlap with all 40 published M0/Tier-1 fingerprints and all 40 domains; sealed data remains unavailable. |
| Reproducibility | PASS | Attempts 3 and 4 emitted identical train/dev SHAs and byte-identical manifest SHA `6b264d07d18309201604ad800bbfd4b5650c5b1dc452bb1dfd5355457f7fe89b`. |
| Spend | PASS | Both completed with `0` accelerator-seconds, `0` tokens, `$0` GPU, and `$0` provider spend. |
DECISION: accept and freeze this pilot source corpus. Do not rematerialize or
change selection. Proceed to the independently capped dev brief synthesis
before committing the larger train split.
NEXT: Publish exact dev/train synthesis configs generated from the frozen
manifest; preregister both and launch the 64-record dev contract proof first.

## [2026-07-16] M1 / pilot-brief-synthesis-preregistration
HYPOTHESIS: The fixed GPT-5-mini brief worker will produce 64 valid dev briefs
with exactly 16 empty outlines inside a `$2` cap; only then should the 256-row
train synthesis reserve up to `$5`.
SETUP: Mechanically generated separate configs from manifest SHA `6b264d07...`.
Both bind exact input file SHA, split hash, count, output URI, source revision,
provider model, expected empty-outline count, 120-minute hard timeout, and the
deployed schema/grounding/resume contract. No call or reservation occurred.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Dev config | PREREGISTERED | Canonical hash `dc8fd5a1c501275737e127cbed436f1ebe85e35b9afd6206a4c190c8a5bb3829`; 64 calls target; `$2` hard reservation. |
| Train config | PREREGISTERED | Canonical hash `ea3548948c08aa8d9e46aae637161ff11d3b9c4cbe4d0851b1f939990106e543`; 256 calls target; `$5` hard reservation, conditional on dev validation. |
| Spend order | PASS | Dev launches first; train remains unlaunched until exact dev ID/schema/grounding/empty-arm validation passes. |
DECISION: publish these immutable configs. Open ledger preregistrations, then
launch dev only. This uses the existing provider cap and is not budget
expansion, model scale-up, Tier 2/3, or promotion.
NEXT: Run dev synthesis, download and validate its exact bytes; launch train
only after a clean result and cost review.

## [2026-07-16] M1 / pilot-brief-synthesis-transport-attempt-1
HYPOTHESIS: Basic JSON mode with the fixed GPT-5-mini worker will satisfy the
published brief contract across all 64 dev records inside the `$2` cap.
SETUP: Launched config hash `dc8fd5a1...` as
`dftr-1784191782-93b3f0ae` at git SHA `e48a321`. The worker used two attempts
per record, preserved only contract-valid rows, and withheld a successful
terminal state unless every target ID was present.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Early adherence | FAIL | `10/12` attempted records remained invalid after two calls each: JSON truncation/shape failures dominated; only `2` records validated. |
| Stop decision | PASS | Cancelled once ordered source IDs proved the failure was systemic rather than a sparse transient; no scientific result or train launch was allowed. |
| Spend | CONTAINED | `$0.069022` provider spend, `0` accelerator-seconds, `0` generated training/evaluation tokens. Monthly provider spend is `$0.108543/$100`. |
| Contract integrity | PASS | No invalid brief was written or admitted; the immutable source corpus, config hash, target IDs, exact 25% empty-outline assignment, and downstream validator remain unchanged. |
| Failure diagnosis | TRANSPORT, NOT DATA | The observed JSON/shape failures are consistent with GPT-5 reasoning consuming much of the old 1,800-token budget while basic JSON mode does not enforce the disclosed schema. Separately, the cancellation endpoint used a Modal flag rejected by the current SDK. |
| Repair | PASS | Added strict JSON Schema routing, provider parameter enforcement, explicit empty/non-empty outline instructions, minimal hidden reasoning, a 4,000-token completion ceiling, safe finish/error diagnostics, and current Modal cancellation semantics. Infrastructure suite passes `29/29`. |
DECISION: retain this as a low-cost transport failure, not a model/data result.
Publish and deploy the wrapper-only correction, then retry the unchanged dev
source/config. Existing validators remain the acceptance authority.
NEXT: Relaunch dev synthesis; require all 64 IDs, exact schema/grounding, and
16 empty outlines before any train synthesis.

## [2026-07-16] M1 / pilot-brief-synthesis-transport-attempt-2
HYPOTHESIS: Strict JSON Schema plus OpenRouter `require_parameters` will route
GPT-5-mini to a provider that enforces the full response format.
SETUP: Deployed the attempt-1 correction and relaunched the unchanged dev
config as `dftr-1784192398-4b31eef6` at git SHA `1a2fe65`. The run resumed
from the two valid committed rows left by attempt 1.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Routing | FAIL FAST | Every missing record returned HTTP 404 before generation because no available route satisfied the explicit `require_parameters` declaration. |
| Stop decision | PASS | Cancelled after about 24 seconds; `$0` provider/GPU cost and `0` new records/tokens. |
| Diagnostics | PASS | The new bounded error detail made the provider-routing failure explicit without exposing prompts, sources, or credentials. |
| Scientific state | UNCHANGED | The two valid attempt-1 rows remain on the checkpoint volume; no malformed output was admitted and train synthesis remains closed. |
DECISION: remove only the incompatible route-selection constraint. Preserve
strict JSON Schema, explicit outline assignment, minimal reasoning, the larger
completion ceiling, and post-response validation. Require a separate one-row,
`$0.25` transport smoke before a third dev attempt.
NEXT: Publish and run comparison
`M1-realdata-pilot-briefs-transport-smoke-v1`; relaunch dev only if its one
output independently satisfies the same contract.

## [2026-07-16] M1 / pilot-brief-synthesis-transport-smoke
HYPOTHESIS: Strict JSON Schema without the incompatible route constraint will
produce one fully grounded brief before another 64-record attempt is allowed.
SETUP: Published config hash `b175da12...` at git SHA `b0bba74` and ran
`dftr-1784192495-43eccfb8` against the first frozen dev record with a separate
output URI, 20-minute timeout, and `$0.25` cap. This artifact is transport-only
and cannot enter training or evaluation.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Terminal result | PASS | `1/1` processed, `0` failed, `$0.001711` provider spend, `0` accelerator-seconds. |
| Independent contract | PASS | Exact source ID and all preserved source fields match; generation mode, types, required fields, non-empty outline, facts, and verbatim quotations validate. |
| Artifact | PASS | SHA-256 `bd25eeeb546efb111e6bed5de0272b0c7c9448b85f21e26f297588d5c655711d`; four outline sections and eight grounded quotations. |
| Parameter audit | PASS | OpenRouter's live model metadata lists structured outputs, response format, reasoning, and max-completion tokens for `openai/gpt-5-mini`, but not temperature. Removed the unsupported temperature field before the full retry. |
DECISION: accept the wrapper transport proof. Publish/deploy the final
parameter cleanup, then resume the unchanged dev config from its two preserved
valid rows. The full 64-row independent validator remains mandatory.
NEXT: Launch dev attempt 3 under the original `$2` cap; do not launch train
until exact dev validation passes.

## [2026-07-16] M1 / pilot-brief-synthesis-dev-bulk-pass
HYPOTHESIS: The strict-schema worker will preserve a contract-valid majority,
allowing deterministic resume to isolate any remaining grounding failures.
SETUP: Ran the unchanged 64-row dev config as
`dftr-1784192598-f958d49d` at git SHA `e8b651d`, resuming two valid rows from
attempt 1. The worker retried every missing record twice, committed after 50
new valid rows, and failed closed on incomplete ID coverage.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Durable progress | PASS | `53` new rows plus `2` preserved rows = `55/64` contract-valid briefs; partial SHA `8e45f9bf86e3182a3f690d5a3d50ea682f8e71d32179060d06b91dd6f39e8895`. |
| Independent partial audit | PASS | 55 unique known IDs, exact preserved source fields, generation mode, schema/types, facts/outline shape, and every emitted quotation passed; all 16 deterministic empty-outline IDs are already present. |
| Rejections | FAIL CLOSED | Eight non-empty-outline records altered at least one quotation instead of copying a contiguous source substring; one additional record ended with provider `content_filter`. Exactly nine source IDs remain absent. |
| Spend | CONTAINED | `$0.104784` provider spend, `0` accelerator-seconds; cumulative known provider spend is about `$0.215`, far below the `$100` cap. |
| Train gate | CLOSED | Partial bytes cannot enter training; no train synthesis or GPU run launched. |
DECISION: preserve the 55 validated rows. Strengthen only the prompt wording to
require byte-for-byte contiguous quotations and explicitly allow an empty
quotation list; keep the validator unchanged. Publish/deploy, then resume the
same config so only nine missing IDs are called.
NEXT: Require terminal 64/64 coverage and a full independent dev validation
before train synthesis.

## [2026-07-16] M1 / pilot-brief-synthesis-dev-missing-row-pass
HYPOTHESIS: Explicit byte-for-byte quotation instructions will recover the
eight grounding failures while preserving all 55 previously validated rows.
SETUP: Published/deployed the prompt-only correction and resumed the unchanged
dev config as `dftr-1784193252-fe023827` at git SHA `1020bf8`.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Grounding recovery | PASS | All eight quotation failures produced contract-valid rows; no prior row changed. |
| Remaining failure | ISOLATED | The sole remaining ID, a public community-college governing-board campaign page, returned provider `content_filter` on both attempts. |
| Independent partial audit | PASS | Exactly `63/64` unique known IDs validate, including all 16 exact empty-outline IDs; partial SHA `5a9721163eb3850946e6bb7e70cc65766cef39bb8103a99b91e48c2b08d5a738`. |
| Spend | CONTAINED | `$0.013267` provider spend and `0` accelerator-seconds. |
DECISION: preserve 63 rows. Add a retry-only neutral archival instruction
after `content_filter`: prohibit persuasion, advocacy, voter targeting, calls
to action, and imitated campaign copy; require a neutral factual user prompt
and empty quotations. The frozen source and output validator stay unchanged.
NEXT: Resume the one missing ID, then run the full 64-row dev validator.

## [2026-07-16] M1 / pilot-brief-synthesis-dev-content-filter-boundary
HYPOTHESIS: A neutral non-persuasive GPT-5-mini retry will recover the sole
campaign-page record without changing source or contract.
SETUP: Resumed 63 validated rows as `dftr-1784193442-8e4e57b7` at git SHA
`46a7145`; only the missing fingerprint was eligible for a call.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| GPT-5-mini policy boundary | FAIL | Both the normal and explicit neutral archival prompts ended with `content_filter`; no response body was admitted. |
| Side effects | PASS | `0` processed, `0` provider cost, `0` accelerator-seconds; the 63-row checkpoint remains byte-valid. |
| Source diagnosis | BENIGN POLITICAL | The 122-word public page is a contact/campaign page for a community-college governing-board candidate. The incompatibility is political-persuasion policy routing, not malformed or unsafe source bytes. |
DECISION: preregister a one-record recovery config that preserves GPT-5-mini as
primary and invokes pinned `anthropic/claude-haiku-4.5` only after an explicit
`content_filter`. The fallback receives the neutral archival prompt, cannot
produce persuasion/calls to action, uses empty quotations, remains under a
`$0.25` cap, and must pass the unchanged validator.
NEXT: Publish/deploy the allowlisted fallback, recover the one row, then run
the full 64-row independent dev validation before train synthesis.

## [2026-07-16] M1 / pilot-brief-synthesis-dev-completion
HYPOTHESIS: The pinned neutral fallback will recover only the one
GPT-5-mini-filtered row, yielding a complete hash-bound dev artifact that
passes the independent source/brief validator.
SETUP: Published recovery config hash `214c1659...` and ran
`dftr-1784193575-4cd14c56` at git SHA `d1f1a4c`. The worker validated all 63
existing rows, observed the primary content filter, invoked allowlisted
`anthropic/claude-haiku-4.5` once with the neutral prompt, and committed one
new row.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Recovery run | PASS | `1/1` missing record processed, `0` failed, `$0.001930`, `0` accelerator-seconds. |
| Full dev cardinality | PASS | Exactly `64` unique source/brief IDs with all preserved source fields and generation mode. |
| Brief contract | PASS | Every row passes schema/type/grounding validation; every quotation is a verbatim source substring. |
| Empty-outline arm | PASS | Exactly `16/64 = 0.25`, matching the deterministic ID set. |
| Immutable provenance | PASS | Source SHA `7012e7a8...`, split hash `849f4e5b...`, final dev brief SHA `524c224ab8215e6a696b0353c0acbae0daacdafd39190a9fd8bbc1c05bf1ef7f`. |
| Total recovery spend | PASS | All dev synthesis, transport smokes, and retries cost `$0.190714`; total monthly provider spend is `$0.230235/$100`. GPU commitment remains `$0.683574/$40`. |
DECISION: accept and freeze the 64-row dev brief artifact. The preregistered
dev gate is satisfied; launch the exact 256-row train synthesis under its `$5`
cap. No SFT GPU run is authorized until the combined 256/64 validator passes.
NEXT: Run train synthesis, resume only contract-valid rows if needed, then
publish the full pilot validation and mechanical three-seed SFT config.

## [2026-07-16] M1 / pilot-brief-synthesis-train-bulk-pass
HYPOTHESIS: The dev-hardened strict-schema worker will synthesize a
contract-valid 256-row train artifact, with resume isolating any sparse
quotation failures.
SETUP: Launched canonical train config hash `ea354894...` as
`dftr-1784193698-9f68c860` at git SHA `bc95fb1`, under the preregistered `$5`
cap. The worker committed every 50 valid rows and retried each rejection twice.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Durable progress | PASS | `248/256` unique train briefs committed; partial SHA `08d57fa74629212123178d99600e4915977c3977a95816c05f15e290a00fb0bd`. |
| Independent partial audit | PASS | All 248 rows preserve exact source fields/IDs, generation mode, schema/types, and verbatim quotation grounding. All 64 deterministic empty-outline IDs are already present. |
| Rejections | FAIL CLOSED | Exactly eight non-empty-outline rows remained absent after altering at least one quotation; no malformed row was written. |
| Reliability | PASS | Two slow provider cases resolved within the fixed timeout/retry policy; checkpoints 50, 100, 150, and 200 remained durable throughout. |
| Spend | CONTAINED | `$0.382337` provider spend, `0` accelerator-seconds; total monthly provider spend is `$0.612572/$100`. |
| GPU gate | CLOSED | No partial training artifact or SFT launch is permitted. |
DECISION: accept the 248-row checkpoint as valid partial progress. Resume the
same immutable config once so only the eight absent IDs are called; use a
separate preregistered recovery only if any remain.
NEXT: Complete 256/256, run the combined source/brief validator, and only then
publish the mechanical three-seed SFT config.

## [2026-07-16] M1 / pilot-brief-synthesis-train-missing-row-pass
HYPOTHESIS: An unchanged stochastic resume will recover the eight quotation
failures without revisiting 248 valid rows.
SETUP: Relaunched the canonical train config as
`dftr-1784195557-df9c869d` at git SHA `6ef4c35`; the worker first validated all
248 existing rows and called only the eight missing IDs.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Recovery | PARTIAL PASS | Five rows recovered and committed; exactly three IDs still altered at least one quotation after two attempts. |
| Checkpoint | PASS | `253/256` rows are now present; no existing row changed and no invalid row was admitted. |
| Spend | CONTAINED | `$0.023358` provider spend, `0` accelerator-seconds. |
DECISION: stop stochastic retries. Preregister a smoke-only recovery bounded
to at most three missing IDs. Preserve each generated section and
`supported_facts`, but deterministically replace its optional `quotations`
array with `[]` before the unchanged brief validator. The gateway must reject
the mode unless the checkpoint proves no more than three IDs are missing.
NEXT: Publish/deploy the bounded recovery, complete 256/256, and run the full
combined validator.

## [2026-07-16] M1 / realdata-pilot-validation-and-sft-preregistration
HYPOTHESIS: The bounded quote-free recovery will complete the final three
train rows, after which a mechanically generated three-seed Qwen3-1.7B SFT
screen can test whether real data fixes the tiny-data collapse failure.
SETUP: Published recovery config hash `edbf5be4...` and ran
`dftr-1784195793-aeb8eca5` at git SHA `f43d784`. The gateway independently
verified that exactly three IDs were missing before allowing optional
quotation arrays to be emptied. Downloaded terminal train bytes and ran the
combined source/brief validator, then generated fixed inputs and SFT config
only from that validation artifact.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Recovery | PASS | `3/3` processed, `0` failed, `$0.005067`, `0` accelerator-seconds. Sections and supported facts were preserved; only optional quotation arrays were deterministically cleared. |
| Combined cardinality | PASS | Exactly 256 train and 64 dev briefs, unique and source-matched. |
| Contract and arms | PASS | Every row passes schema/grounding/source preservation; exactly 64/256 train and 16/64 dev outlines are empty. |
| Brief provenance | PASS | Train SHA `498294a865db2b8c7d9466aad562e187c0475b84051c59453c4a030f2041d1d8`; dev SHA `524c224ab8215e6a696b0353c0acbae0daacdafd39190a9fd8bbc1c05bf1ef7f`. |
| Validation provenance | PASS | Artifact SHA `ffe015c599c6c245d6df9b52829eb22811b9d31fb34912741d0aa270d6a81ba5`; fixed-manifest SHA `3c8776b6321eefebce6f4b616042c6146ead6ff6f0695eb64a9be4e2810ec46b`. |
| SFT config | PREREGISTERED | Canonical hash `b57d06bafdddb9f546b8df21e7d61c4f729b8766fd647336b6f5abc1d46adcad`; Qwen3-1.7B revision `70d244cc...`; seeds `[11,29,47]`; one epoch; L40S screen cap. |
| Verification | PASS | Data, experiment, infrastructure, and harness suites pass `112/112`; stale placeholder-era harness assertion was updated to verify the now-frozen calibration/baseline hash binding. |
| Budget | PASS | Total provider spend `$0.640997/$100`; GPU commitment remains `$0.683574/$40` before SFT reservation. |
DECISION: freeze the validated 256/64 brief artifacts and publish the exact
SFT inputs/config. This is the first model-training run capable of producing
directional recovery evidence; it remains a 1.7B screen, not 14B scale-up,
Tier 2/3, or promotion.
NEXT: Launch the three-seed SFT screen, verify checkpoint manifest/provenance,
then run the fixed 16-document directional default-sampler evaluation.

## [2026-07-16] M1 / adherence-recovery-model-size-screen
HYPOTHESIS: Repairing the synthesis meta-prompts and conditioning both train
and generation on the full structured brief will restore factual control; a
single-seed 1.7B/4B comparison determines whether model size merits a
three-seed confirmation.
SETUP: Replaced only `user_prompt` in the frozen 256/64 corpus and proved all
other values unchanged. Enforced `dft.full-brief.v1` rendering of use case,
style, detail mode, length, em-dash policy, and outline in both SFT and
sampling. Trained seed 11 for Qwen3-1.7B and Qwen3-4B, then generated the same
16 dev records at sampling seeds 101, 202, and 303.
RESULTS:
| item | 1.7B | 4B |
| --- | ---: | ---: |
| Mean outline-fact recall | 0.80296 | 0.81037 |
| Mean unsupported rate, non-empty outlines | 0.02602 | 0.01955 |
| Outline / unsupported / language gates | 3/3 each | 3/3 each |
| No-collapse gates | 1/3 | 2/3 |
| Mean self-BLEU | 0.04272 | 0.04795 |
| Directional decision | FAIL | PASS |

REVIEWER AUDIT: The primary conditioning diagnosis is confirmed by code and
the approximately 1,600-fold recall recovery from the failed pilot. Two
evaluator caveats remain. First, the preference judge uses a prompt-unmatched
external human bank and is secondary. Second, the positive calibrated lower
bound makes zero repeated sentence starts fail `no-collapse`; this penalized
1.7B seed 303 despite no observed repetition. The frozen rule is preserved
for this decision, but must be redesigned before promotion.
DECISION: advance Qwen3-4B to the preregistered three-seed confirmation. Do
not authorize 14B, Tier 2, Tier 3, or promotion yet.
NEXT: Train Qwen3-4B seeds 11/29/47 on the identical corrected corpus, repeat
the 144-output directional screen, and require the factual and collapse gates
to hold across adapters before any budget expansion.

## [2026-07-16] M1 / adherence-recovery-4b-three-seed-confirmation
HYPOTHESIS: The corrected Qwen3-4B result will reproduce across training seeds
11, 29, and 47, with all nine factual/language gates and at least six of nine
no-collapse gates passing.
SETUP: Trained three adapters on the identical frozen 256-record full-brief
corpus, then generated the same 16 dev records under sampling seeds 101, 202,
and 303. The evaluator retained the exact calibration, baseline, external
human bank, and gateway judge hashes used in the bounded model-size screen.
RESULTS:
| item | result | gate |
| --- | ---: | ---: |
| Outline-fact gate | 9/9 | 9/9 |
| Unsupported-claim gate | 9/9 | 9/9 |
| Language-integrity gate | 9/9 | 9/9 |
| No-collapse gate | 8/9 | >= 6/9 |
| Mean outline-fact recall | 0.78963 | >= 0.40 directional threshold |
| Mean unsupported rate, non-empty outlines | 0.03083 | <= 0.60 directional threshold |
| Mean self-BLEU | 0.04505 | human interval 0.03234-0.07693 |
| Generated evidence | 144 docs / 20,809 tokens | exact |
| Compute | 1,966.832 L40S-sec / $1.279227 | under cap |

REVIEWER AUDIT: Factual recovery is deterministic and consistent across
adapters. Quality preference remains secondary because its human comparison
bank is prompt-unmatched. The only no-collapse failure was checkpoint seed 11
/ sampling seed 202 at repeated-start rate 0.1875, above the frozen 0.15744
upper bound. The separate lower-bound problem remains relevant to future
promotion design, but did not determine this pass.
DECISION: pass the directional three-seed confirmation. This is enough
evidence to justify the next bounded evaluation stage, but not an automatic
14B scale-up. Preserve 4B as the current winning size and repair the discrete
repetition gate before treating it as a promotion criterion.
NEXT: Publish the result, then choose between a bounded 4B sealed evaluation
and the next M2 experimental arm. Require explicit evidence before any 14B
budget expansion.

## [2026-07-16] M1 / prospective-repetition-evaluator-v2-audit
HYPOTHESIS: Recasting repeated sentence starts as an upper-bound-only collapse
signal will remove the reviewer's counterintuitive penalty for zero repetition
without weakening the existing high-repetition rejection or changing any other
metric.
SETUP: Preserved all v1 reports, preregistered the change, and rescored the
same immutable 144 outputs from source-index SHA `0b4fd3e4...` with calibration
v2 SHA `b84783c3...`. Self-BLEU remains two-sided; the repetition upper bound
remains exactly `0.15744263820012558`.
RESULTS:
| item | result | audit |
| --- | ---: | --- |
| Outline / unsupported / language gates | 9/9 each | identical to v1 |
| No-collapse gate | 8/9 | identical for this 4B evidence |
| Mean self-BLEU / repetition | 0.04505 / 0.12500 | identical to v1 |
| Deterministic report fields | 9/9 identical | excludes calibration metadata and secondary judge |
| Secondary quality preference | 1.00000 | stochastic rerun; not a decision input |

REVIEWER AUDIT: The evaluator defect was real, but it did not inflate this
4B confirmation: its only failing cell remains seed 11 / sampling 202 at
repetition `0.1875`, above the unchanged upper bound. The corrected rule now
also behaves sensibly on the audited zero-repetition case while preserving
the high-repetition failure. Unknown bound modes fail closed.
DECISION: accept evaluator v2 prospectively and retain the 4B directional
pass. Preserve v1 byte-for-byte. This audit authorizes one bounded sealed 4B
submission, not an automatic promotion or scale-up.
NEXT: Publish the v2 audit, preregister the exact sealed comparison, and use
the aggregate hidden-split verdict to select M2 work and the 14B evidence gate.

## [2026-07-16] M2 / sealed-4b-seed29-promotion-check
HYPOTHESIS: The seed-29 Qwen3-4B adapter selected by the frozen Tier-1 rule
will confirm human-distribution similarity on the independent hidden domain
and time slice; only a sealed confirm authorizes 14B or Tier 3.
SETUP: Merged the exact adapter SHA `a34c1423...` into immutable base revision
`1cfa9a7...`, bound the default `temperature=1.0/top_p=1.0` deployment
sampler, and submitted checkpoint hash `0f437f62bc1cca0c`. The evaluator used
private revision `cc9f748`, 128 hidden examples, an independent heavyweight
embedder, fresh authorship probe, deterministic sampling, and aggregate-only
output.
RESULTS:
| item | result | gate |
| --- | ---: | --- |
| Semantic MMD | 0.059529 | FAIL |
| Delta vs human floor | +0.029529 | FAIL |
| Aggregate S | 0.575422 | report |
| Authorship AUC | 0.726914 | FAIL |
| Authorship AUC 95% CI | [0.626166, 0.828487] | excludes 0.5 |
| Sealed verdict | REJECT | confirm required |

EVALUATOR AUDIT: Initial requests exposed no scores: four failed artifact URI
or stale-volume preflight and one reached sequential generation but returned
`sealed scoring unavailable` after 960 seconds. The operator fixed only
plumbing by refreshing volumes, batching eight prompts, seeding sampling, and
binding the already frozen sampler. Ten private tests and 125 public tests
pass. An append-only one-slot credit records the no-result retry. No hidden
text, embedding, per-item score, metric, threshold, or embedder was exposed or
changed. Total sealed attempt cost was `$1.594056`; the successful scored run
cost `$0.526296`.
DECISION: fail closed. The 4B recipe has strong visible factual control but
does not generalize to human-likeness on the hidden split. Do not launch 14B
and do not spend Tier-3 detector calls on this checkpoint.
NEXT: A new preregistered research cycle must improve semantic distribution
and authorship indistinguishability at 4B before scale. Candidate directions
are broader and longer human data, a genuinely independent training reward,
or staged adversarial/distribution objectives; none may reuse the sealed
embedder or hidden split.
## [2026-07-17] M2 / measurement-v2-visible-source-preregistration
HYPOTHESIS: A separately pinned FineWeb shard can provide 192 unique,
distinct-domain, public human documents for three disjoint measurement-v2
panels without exposing candidate outputs or reusing the DFT training corpus.
SETUP: Frozen CC-MAIN-2024-10 revision
`9bb295ddab0e05d785b879661af7260fed5140fc`, shard
`000_00001.parquet`, deterministic selection/split seeds, 80-220-word English
eligibility, 1,600-record candidate pool, 50,000-row scan limit, and a 128/64
floor/eval split. The earlier SFT corpus used shard `000_00000.parquet`.
RESULTS: Pending privileged fixed-code materialization. No model output,
measurement endpoint, or hidden data has been opened.
DECISION: Preregister and launch only the bounded public-source job. Reject the
bundle if it does not reproduce exactly, lacks 192 unique documents/domains,
or overlaps the published training fingerprints.
NEXT: Download the three source artifacts read-only, independently verify
hashes/cardinality/non-overlap, then synthesize briefs for only the 64 frozen
human-eval prompts before any A0 or A64 output is generated.

## [2026-07-17] M2 / measurement-v2-visible-source-result
HYPOTHESIS: The separately pinned shard will produce a usable public panel
pool without overlap against existing training or Tier-1 human records.
SETUP: Fixed-code CPU materialization run `dftr-1784273680-482885ef` using the
preregistered 192-record config; downloaded artifacts were checked against the
gateway and source-manifest hashes before inspection.
RESULTS:
| item | result | gate |
| --- | ---: | --- |
| Human records | 192 | exact |
| Unique fingerprints / domains | 192 / 192 | exact |
| Floor / eval records | 128 / 64 | exact |
| Training fingerprint overlap | 0 | required |
| Tier-1 fingerprint overlap | 0 | required |
| Cross-panel fingerprint overlap | 0 | required |
| Provider/GPU cost | $0 | bounded CPU route |
DECISION: Accept the public source pool. Freeze it before candidate generation.
NEXT: Synthesize and validate the 64 full briefs, then let the independent
operator materialize bandwidth, power, calibration, selection, and signatures.

## [2026-07-17] M2 / measurement-v2-prompt-briefs-preregistration
HYPOTHESIS: The frozen brief-synthesis contract can turn the 64 human-eval
documents into complete, prompt-matched briefs for the public v2 comparison
within a $1 API cap, while preserving each original human completion as the
reference and opening no candidate outputs.
SETUP: Exact input SHA `942551d9...`, source manifest SHA `84c5a7e0...`, 64
records, deterministic 25% empty-outline assignment, GPT-5-mini primary and
the already allowlisted neutral fallback. This step cannot inspect A0/A64.
RESULTS: Pending bounded synthesis and exact semantic validation.
DECISION: Preregister the prompt panel. Reject partial or hash-inconsistent
output and repair only through a separately recorded recovery config.
NEXT: Download read-only, validate all 64 rows against their source records,
then freeze the measurement protocol before model generation.

## [2026-07-17] M2 / measurement-v2-prompt-briefs-partial-result
HYPOTHESIS: Resumable fixed-code synthesis will preserve accepted rows and
isolate any schema/quotation failure rather than silently admitting it.
SETUP: Initial run `dftr-1784273798-046bcd91` and unchanged resume run
`dftr-1784274016-25b847dd`; both loaded and revalidated existing output before
new calls. Combined API spend was `$0.090966`.
RESULTS: 63 of 64 unique source IDs validate and preserve every frozen source
field. Exactly one source fingerprint, `dc919575...`, is missing; there are no
duplicates or unknown IDs. The second run failed closed with one rejected row.
DECISION: Preserve the 63 accepted bytes. Do not rerun them. Preregister a
one-record quote-free recovery, the same bounded mechanism used in M1, with a
strict `max_missing_records=1` gate and $0.10 cap.
NEXT: Recover and validate only the missing row, then hash-freeze all 64 briefs.

## [2026-07-17] M2 / measurement-v2-prompt-repair-preregistration
HYPOTHESIS: The prompt-only recovery contract can replace all 64 synthesis-meta
requests with document-specific standalone writing prompts while leaving every
reference, outline, quotation, style, and provenance field unchanged.
SETUP: Quote recovery run `dftr-1784274810-a0159737` completed the exact 64-row
brief artifact for `$0.001566`; full validation passed with SHA
`888c26189a...`, 16/64 empty outlines, and the exact source split hash. The
separate standalone-prompt validator then rejected 64/64 prompts because they
echoed the synthesis instructions, reproducing the earlier M1 conditioning
defect before any A0/A64 output existed.
RESULTS: Prompt repair pending under the frozen 64-record, $1 contract. All
non-prompt fields are immutable and the output URI is new.
DECISION: Do not freeze the defective prompt panel. Preregister and run the
same prompt-only repair mechanism that restored M1 adherence.
NEXT: Require 64 unique natural requests and exact non-prompt equality, then
use only the repaired SHA in the signed measurement protocol.

## [2026-07-17] M2 / measurement-v2-power-and-decision-freeze
HYPOTHESIS: The frozen n=64 public panels can resolve a scientifically material
A64-versus-A0 effect without using candidate output to choose thresholds.
SETUP: Before A0/A64 generation, embedded the 192 frozen humans with immutable
`BAAI/bge-small-en-v1.5` revision `5c38ec7c...`. Two thousand no-replacement
64v64 permutations of the 128 floor embeddings gave MMD standard deviation
`0.0006077874` and absolute 95th percentile `0.00116319`; the fixed floor MMD
was `-0.00011304`. The 64-document eval panel had 12 repeated-start events
(`0.1875`). Balanced n=64 AUC null SE was conservatively frozen at `0.052`.
RESULTS: The prospective 1,000-trial simulation with MIEs MMD `0.0018`, AUC
separability improvement `0.15`, and repetition-boundary distance `0.20`
produced type-I `0.045`, MMD power `0.897`, AUC power `0.904`, repetition power
`0.899`, and coverage `0.946`; every target passes. Decision thresholds are
candidate-minus-control MMD <= `-0.0018` with paired p <= `0.05`, authorship
separability improvement >= `0.15`, repetition margin `0.15`, and prompt-linked
quality win rate >= `0.45`, plus all four exact hard gates.
DECISION: Freeze these inputs before any model output. Do not weaken them after
seeing A0 or A64. If the signed v2 pipeline later disagrees with this preview,
fail closed and investigate implementation identity rather than substituting
new assumptions.
NEXT: Complete prompt repair, generate the exact one-seed A0 matched control,
and freeze/sign the protocol before A64 generation.

## [2026-07-17] M2 / measurement-v2-prompt-repair-result
HYPOTHESIS: Prompt-only repair will produce 64 natural standalone requests
without mutating any scientific reference or brief field.
SETUP: Run `dftr-1784275019-436dd5cf`, exact preregistered 64-row input SHA
`888c2618...`, GPT-5-mini, new output URI, and $1 cap.
RESULTS: Completed 64/64 with zero failures for `$0.023396`. The repaired
artifact SHA is `b5c8b665...`; all 64 prompts are unique, every standalone-
prompt semantic guard passes, and validation proves only `user_prompt`
changed. Prompt lengths are 331-852 characters. All completions, fingerprints,
outlines, quotations, styles, and provenance remain exact.
DECISION: Accept and freeze only the repaired brief artifact for measurement
v2. The un-repaired SHA remains a documented negative data artifact.
NEXT: Bind repaired SHA `b5c8b665...` into the A0 generation config, then
freeze/sign the real protocol before generating A64.

## [2026-07-17] M2 / first-genuine-dft-a64-versus-a0-result
HYPOTHESIS: At 4B and 64 generated tokens, adding the frozen score-function
MMD term (`mmd_coefficient=0.1`) will improve held-out human-distribution
similarity relative to an exposure-matched zero-MMD continuation from the
same seed-11 SFT adapter.
SETUP: A0 training run `dftr-1784279399-17837c4a` and A64 run
`dftr-1784283279-cbace20d` used the same method-contract SHA
`57fd684468f2...`, eight steps, the same 256-record training-only corpus,
Qwen3-4B revision, seeds, optimizer, rollout budget, KL/SFT anchors, and LoRA
initialization. Adapter-native generation runs `dftr-1784282330-d16f81d5`
(A0) and `dftr-1784283920-38506a78` (A64) each produced 64 prompt-aligned
documents and 4,096 sampled tokens under generation-contract SHA
`1554d88b...` and decoding SHA `85acea23...`. Before A64 existed, the
independent operator froze n=64 power, human panels, bandwidths, thresholds,
seed rules, and a signed protocol. All runs have hash-bound manifests and
signed wrapper receipts.
RESULTS:

| endpoint | A0 control | A64 candidate | frozen decision |
| --- | ---: | ---: | --- |
| Unbiased MMD2 to human eval | -0.0045134 | -0.0045804 | treatment delta `-0.0000670`, fail (`<= -0.0018` required) |
| Paired permutation p | — | 0.739626 | fail (`<= 0.05` required) |
| Authorship separability | 0.093701 | 0.103271 | worsened by 0.00957; fail (`>= 0.15` improvement required) |
| Repeated-start rate | 0.015625 | 0.03125 | candidate passes human non-inferiority |
| Outline-fact recall, diagnostic | 0.25979 | 0.26804 | small +0.00825 movement |
| Unsupported proxy, nonempty outlines | 0.05325 | 0.04762 | small favorable movement |

Training was stable: eight A64 steps completed, MMD rewards were finite and
nonconstant, KL stayed below `0.00118`, maximum gradient norm was `0.587`, and
the rollout uniqueness/collapse sentinels did not trip. The signed report SHA
is `facfd43c...`; the independently signed blind attestation SHA is
`2c03689f...`, with all 13
qualification groups passing and historical inventory verified. The durable
copies live under
`/checkpoints/measurement-v2/results/m2-a0-a64-seed11-v1/`.
DECISION: This is the first genuine, measurement-valid negative DFT result.
The direct 64-token score-function MMD treatment was trainable but produced no
material paired public-v2 effect. Apply the prospective stop rule: do not run
seeds 29/47, 128 tokens, a paid quality judge, sealed evaluation, 14B, or
Tier 3 for A64. Quality judging cannot repair failed distribution and
authorship intersections, so skipping it saves provider spend without changing
the decision.
NEXT: Retain A0 as the current matched control and preregister one cheaper,
lower-variance mechanism (teacher-forced distribution moments is the leading
candidate) as a new arm with its own coefficient and stop rules. Do not tune
A64 after observing this endpoint. Require a fresh single-seed 64-token public
effect before confirmations or scale-up.

## [2026-07-17] M2 / A64-reviewer-audit-and-uncertainty-correction
HYPOTHESIS: Independent review of the reviewer and score artifacts will find
no sign, threshold, cardinality, seed, or provenance error capable of changing
the A64 stop decision.
SETUP: A read-only reviewer checked the signed protocol/report, training logs,
both generation manifests and receipts, ledger identities, seed grid, and
attestation. It specifically audited the evaluator rather than duplicating the
scientific vote.
RESULTS: The A64 stability, MMD sign, frozen thresholds, n=64 cardinality,
training/sampling seeds, output hashes, receipts, and stop interpretation all
passed. The reviewer found one non-decision defect: bootstrap copies were
assigned occurrence-specific group IDs, allowing identical text to cross
authorship-probe folds. This inflated the original interval to roughly
`0.79-0.92` despite point AUC near `0.40`. The implementation now preserves
source group IDs and stratifies disjoint classes; a leakage regression and 25
focused tests pass. Recomputing from the same immutable bytes gives A0 AUC
`0.4063`, corrected interval `[0.3249, 0.6966]`, and A64 AUC `0.3967`,
corrected interval `[0.3351, 0.6588]`. Point separabilities remain exactly
`0.09370` and `0.10327`, so the preregistered authorship failure is unchanged.
The correction is stored as a post-hoc, non-decision diagnostic rather than
silently rewriting the old code-hash-bound report. The reviewer also noted
that the old attestation copied its source manifest signature; the operator
now signs the attestation itself with the independent blind-test key, and
direct Ed25519 verification passes.
DECISION: Preserve the A64 negative result. Do not cite the superseded original
authorship interval. Future protocols must bind the corrected metric code and
independently signed attestation path.
NEXT: Before spending on another objective, quantify how little A64 moved the
policy (48 of 64 sampled outputs were byte-identical) and preregister the next
lower-variance mechanism against the same matched control.

## [2026-07-17] M2 / A64-mechanism-diagnostic
HYPOTHESIS: The failed public effect came from insufficient or noisy policy
movement rather than training divergence or factual collapse.
SETUP: Compare A0/A64 step traces and exact matched generation bytes without
opening any new endpoint, provider call, or GPU job.
RESULTS: A64's score-function estimator variance averaged `17.98`, ranged from
`0.44` to `76.78`, and the MMD policy-loss contribution changed sign three
times across eight steps. Mean advantage standard deviation was only `0.047`.
Despite a 3.2x larger mean gradient norm than A0, A64's KL remained at or below
`0.00118`. In matched seed-101 generation, 48/64 documents (`75%`) were
byte-identical. The 16 changed documents had mean word-sequence similarity
`0.584` and only `2.125` mean absolute words of length difference.
DECISION: The arm was stable but mostly failed to displace the A0 policy. Do
not interpret the negative result as evidence that 4B is too small, and do not
repair it by spending on longer outputs or confirmation seeds. A new arm must
reduce estimator variance or provide a differentiable distribution signal,
and it needs a fresh prospective measurement panel because the current one has
been opened.
NEXT: The cheapest defensible choices are (a) a larger effective-rollout MMD
estimator under a matched token budget, or (b) teacher-forced distribution
moment matching. Freeze the next mechanism and fresh panel before any model
output; reserve 14B and Tier 3 for a genuine 4B effect.

## [2026-07-17] M2 / independent-review-reconciliation
HYPOTHESIS: Independent reviews of the A64 result will preserve the exact
recipe failure while identifying whether the broader DFT interpretation and
measurement apparatus require correction.
SETUP: Reconciled two external read-only reviews against repository code and
recomputed their material claims from the frozen embeddings, prompt/human
manifests, outputs, training data, and decision code. No new model output,
provider call, hidden evaluation, or detector result was opened.
RESULTS: All 64 prompt sources exactly equal the old 64-document human
distribution panel. Their matched cross-kernel mean is `0.739`, versus
`0.518-0.519` unmatched, shifting absolute MMD by approximately `-0.0069`.
Excluding matched cross-pairs yields A0 MMD2 `+0.002391` and A64 `+0.002339`;
the treatment contrast remains negligible at `-0.000052`, so the exact recipe
still fails. The audit also confirmed an impossible realized authorship gate,
a power simulation that did not execute the scored intersection rule, a
wrong-tail absolute MMD diagnostic, token-versus-word target semantics,
post-EOS reward pollution, unexpected non-Latin letters in `6/64` A0 and
`7/64` A64 outputs, and materially unclean training targets. The earlier
"sign-oscillating gradient" interpretation is withdrawn: logged scalar
surrogate variance and loss signs are not parameter-gradient diagnostics.
DECISION: Narrow the result to a failed delivered A64 recipe, not a general
negative for score-function MMD, DFT, or 4B. Retire the opened panel for model
decisions. Rebuild an unpaired, positive-control-qualified instrument and run
a frozen-policy group-size/gradient audit before selecting the next treatment.
NEXT: Implement the fresh-panel contract and training-semantic repairs in a
new prospective cycle. If the larger-group score-function gradient is coherent
and style-sensitive, test it directly; otherwise use a one-round shared-rollout
reward-weighted SFT comparison. Prepare a clean 256/1024 data ladder in
parallel. Compute scale is authorized, but 14B remains gated on a real 4B
effect.

## [2026-07-17] M2 / frozen-estimator-audit-preregistration
HYPOTHESIS: If the current score-function MMD mechanism is usable at Qwen3-4B,
increasing the nested on-policy group from K=4 to K=32 under EOS-aware,
length-matched 64-token support will produce a coherent gradient estimate:
K32 split-half CountSketch cosine at least `0.50` and gradient-norm coefficient
of variation at most `1.0`.
SETUP: Frozen adapter `dftr-1784216516-91130dd3/seed-11`; no optimizer update;
16 independently seeded nested K=`4,8,16,32` groups; one fixed 256-dimensional
whole-LoRA gradient CountSketch; prompts explicitly request 64 tokens; EOS is
included as the final scored action and all post-EOS padding is masked. Compute
both full-human and human-truncated-to-64-token reward supports with separately
human-derived bandwidths. Config hash is
`186744e14b0f428dc0aed79ab3a7d270751ff41cc9912c28cf37fe303f9443ca`;
contract hash is
`ae86a9148122672564d24f3ae377a2dfaa633185bcc551fb84bc91ea05732e80`.
RESULTS: Pending. The implementation, runner dispatch, client guard, and Modal
gateway guard pass 101 focused tests. Maximum generated exposure is 32,768
tokens and the two-hour L40S reservation is below $5.
DECISION: Launch this diagnostic before any K32 training. A passing
length-matched K32 audit permits a direct global-K32 score-function screen. A
failure routes the next arm to a lower-variance differentiable or
reward-weighted SFT objective rather than spending more on the same estimator.
NEXT: Deploy the gateway at the committed audit code, launch once, and archive
the result before selecting a treatment.

## [2026-07-17] M2 / frozen-estimator-audit-v1-launch-failure
HYPOTHESIS: The preregistered diagnostic will execute without updating model
parameters and return group-size gradient summaries.
SETUP: Launched commit `450baabe75fd0366dc7f85ee844f524f8eede3cc`,
config hash `186744e1`, as run `dftr-1784302659-54f7383f` on one L40S.
RESULTS: Failed before the first audit row or result artifact. A differentiable
K-way log-probability forward retained whole-model activations and attempted an
additional 2.11 GiB allocation with only 1.63 GiB free. Runtime was 52.747
accelerator-seconds, actual cost `$0.034307`, and token accounting was zero.
The worker trace is archived at the run artifact path.
DECISION: This is an implementation OOM, not evidence about the estimator.
Replace the K-way differentiable forward with exact per-sequence loss/gradient
accumulation divided by K, as required by the reviewed topology. Preserve the
nested rollouts, rewards, advantages, and pass thresholds.
NEXT: The repaired v2 config adds `logprob_microbatch_size: 1` and has config
hash `67c5b0b96321bd0f27e2d78ed7edf3e34ce294d74edec52cd3f8905484f1f682`.
An explicit tiny-model test proves single-sequence accumulation equals the
full-group loss and parameter gradient. Commit, deploy, and relaunch as a new
comparison identity.

## [2026-07-17] M2 / frozen-estimator-audit-v2-result
HYPOTHESIS: Length-matched K32 score-function semantic MMD will produce a
stable enough whole-LoRA gradient to justify a direct training screen.
SETUP: Run `dftr-1784303078-239a1732`; 16 nested replicates at K=`4,8,16,32`;
full-human and 64-token-human supports; EOS-aware 64-token sampling; exact
per-sequence gradient accumulation; fixed 256-dimensional CountSketch; no
optimizer updates. Result SHA is `74a7ccd9`.
RESULTS: The repaired run completed in 871.678 accelerator-seconds for
`$0.566940`. K32 length-matched gradient-norm CV passed (`0.394 <= 1.0`), but
split-half cosine failed (`-0.01385 < 0.50`). Full-human K32 cosine was also
near zero (`0.04074`). Increasing K reduced advantage variance, gradient-norm
CV, and mean gradient norm, but did not stabilize direction. Length matching
shifted K32 mean absolute MMD2 from `+0.00694` to `-0.00174` without repairing
coherence. All 512 unique rollouts used the full 64-token horizon. Independent
recomputation verified all 128 result rows, artifact SHA, and reported cosines.
DECISION: Fail the prospective gate. Do not run global-K32 score-function
training. Larger rollout batches reduce scalar variability but do not provide
a reproducible parameter direction in this reward representation.
NEXT: Build the lower-variance matched 4B screen: ordinary SFT control,
teacher-forced token-distribution moments, and one-round shared-rollout
MMD-witness-weighted SFT. Use cleaned data and a fresh unpaired two-embedder
instrument. Keep 14B, sealed evaluation, and Tier 3 gated on a real 4B effect.

## [2026-07-17] M2 / lower-variance-data-boundary
HYPOTHESIS: A fresh, historically disjoint FineWeb pool plus Qwen3-32B
line-selection cleaning will remove the raw-page artifacts that weakened the
old 256-document human target while preserving original prose bytes.
SETUP: Added a privileged cleaning contract in which Qwen3-32B returns only
strictly increasing original line numbers. Fixed code reconstructs the text
and rejects rewrites, reordered lines, out-of-range selections, and documents
outside 80-220 words. Source materialization uses fresh shard
`000_00002.parquet`, 2,200 unique domains, and wrapper-resolved exclusions from
the old 256 training rows and 192 public measurement rows. Planned raw split is
1,400 training candidates and 800 evaluation candidates, oversized for clean
targets of 1,024 and 640.
RESULTS: Cleaning, policy, source, and gateway code passes 57 focused tests.
No source or provider job has launched yet.
DECISION: This faithful data boundary precedes the token-moment and
MMD-witness-weighted training arms. Evaluation and training pools remain
domain/fingerprint disjoint by construction.
NEXT: Commit and deploy the gateway, materialize the raw pool, verify overlap
and content quality, then launch bounded Qwen3-32B cleaning.

## [2026-07-17] M2 / parallel-lower-variance-foundations
HYPOTHESIS: The lower-variance training screen, fresh evaluator, and clean-pool
qualification can be implemented independently while Qwen3-32B cleaning runs.
SETUP: Three isolated workstreams added new modules only: differentiable
teacher-forced token moments plus one-round MMD witness weights; measurement-v3
with unpaired panels, two independent embedders, correct MMD tails, token L2,
human-calibrated margins, and exact-rule power; and deterministic provenance
qualification plus 128/256/128/128 evaluation partitioning.
RESULTS: Combined local verification passes 117 tests (`100` experiment/data
and `17` measurement). Qwen cleaning has 550 committed accepted training
documents and 194 logged rejections, a `73.9%` acceptance rate versus `73.1%`
required to reach 1,024 from 1,400 candidates.
DECISION: Integrate these cores now rather than waiting for cleaning. Training
runner/config wiring still remains; the pure objectives and decision metrics
are implemented and independently testable.
NEXT: Commit the foundations, finish cleaning/qualification, then bind one
matched three-arm runner and freeze the measurement-v3 panel before outputs.

## [2026-07-18] M2 / scale-ladder-source-validation-and-clean-launch
HYPOTHESIS: The completed preregistered 27,000-document nested raw source
pool is sufficient to advance the bounded 4K/16K scale ladder only if its
returned counts and SHA-256 manifests validate against the frozen source
contract, after which the next safe handoff is strict Qwen3-32B cleaning of
the 26,000-train and 1,000-scale-dev partitions. This preserves the
independent scale-development panel and does not outrun the user-unapproved
46,080 cell.
SETUP: Scheduled 90-minute safety audit on Saturday, July 18, 2026. Read-only
validation used the completed gateway record for
`dftr-1784357254-035e8d3c`, its worker log, and the frozen source config
`configs/m2/m2_scale_ladder_source_pool_16k_v1.yaml`. The audit then added
the missing downstream cleaner configs
`configs/m2/m2_scale_ladder_clean_train_16k_v1.yaml` and
`configs/m2/m2_scale_ladder_clean_dev_640_v1.yaml`, preregistered
comparisons `M2-scale-ladder-clean-train-16k-v1` and
`M2-scale-ladder-clean-dev-640-v1`, and launched them as
`dftr-1784358360-4f83b039` and `dftr-1784358360-a40e0584`. Config hashes are
`ad131251512f9b754a5be7be8f6d791cb17f2fb3eefc1497c82847ad328e94df` for the
16,384-train cleaner and
`1d7b8c9a9fd51760ae9f8cd249513217441defd8faed445b86805fb630851dec` for the
640-document scale-development cleaner.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Source-pool terminal validation | PASS | Run `dftr-1784357254-035e8d3c` completed with `selected=27000`, `train=26000`, `dev=1000`; returned `train_sha256=1d346936dd3bb0e12d60b578d8cbcf2af32f3c02a093fcee5f23309e4b99b942` and `dev_sha256=a952f68933c301e967abeb0cf6cd2dcd3079e956e67ba3467ca18ac426194c94`. |
| Safety-audit handoff check | FAIL | The repo had the source-materialization config only; no checked-in scale-ladder cleaning continuation existed, while `progress/status.json` still reported source selection as the active state after terminal completion. |
| Fixed continuation repair | PASS | Added pinned cleaner configs using the returned SHA-256s and the existing frozen 80-220-word Qwen3-32B cleaning contract. |
| Next safe async launches | PASS | Cleaner runs `dftr-1784358360-4f83b039` (target `16,384/26,000`, API reserve `$12`) and `dftr-1784358360-a40e0584` (target `640/1,000`, API reserve `$3`) are running. |
| Budget boundary after launch | PASS | Gateway budget reports Modal committed `$16.75756/$100` and API spend/reserve `$31.598985/$100`, leaving `$83.24244` Modal and `$68.401015` API. |
DECISION: Keep the bounded 4K/16K ladder active. The completed raw pool is
valid, the missed handoff is repaired, and the correct next wake condition is
cleaner completion rather than the already-handled source-selection run. Do
not launch brief synthesis, 4,096 training, or any 46,080 artifact until the
clean outputs are validated prospectively.
NEXT: When either cleaner reaches a terminal state, verify acceptance counts,
exact-source provenance, and disjointness; then freeze the scale-development
panel shape and launch only the next data-preparation step supported by those
validated clean artifacts.

## [2026-07-18] M2 / scale-ladder-dev-freeze-launch
HYPOTHESIS: The completed 640-document scale-development cleaner can be
validated prospectively against the raw 1,000-document dev pool, shown
disjoint from the 26,000-document raw train pool, and frozen into a
deterministic 128/256/128/128 bundle without waiting for the 16,384-train
cleaner to finish. If that succeeds, the next safe async handoff is prompt
brief synthesis over only the frozen prompt-source slice.
SETUP: Terminal-transition continuation on Saturday, July 18, 2026 after
`dftr-1784358360-a40e0584` reached `completed`. Read-only validation used the
gateway status/log endpoints plus the fixed document-cleaning worker contract.
That confirmed terminal acceptance counts and spend for the 640-document
scale-dev cleaner while the 16,384-train cleaner `dftr-1784358360-4f83b039`
remained active. The repo then added a new frozen validation workflow at
commit `b6a8602d970032c5e3df23cc58520f106ee77ea1`:
`data/materialize_scale_ladder_dev_panels.py`,
`experiments/m2/scale_ladder_dev_panel.py`,
`configs/m2/m2_scale_ladder_freeze_dev_panel_v1.yaml`, and focused tests. The
new comparison `M2-scale-ladder-freeze-dev-panel-v1` was preregistered and
launched as smoke run `dftr-1784360648-7505beeb` with config hash
`6eff35fe4e03810ff70f9e34794f681d7f3282fe4599569d6eb2abd8aaed86ac`.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Scale-dev cleaner terminal validation | PASS | Run `dftr-1784358360-a40e0584` finished at `2026-07-18T06:37:26Z` with `records_processed=640`, `records_failed=342`, and `actual_api_cost_usd=$0.239342`. The fixed worker can only emit `completed` after 640 accepted rows are written and committed. |
| Scale-dev artifact evidence standard | FAIL | The current document-cleaning route records counts and logs, but not a hash-bound output manifest analogous to source materialization. This left the finished 640-document artifact insufficiently frozen for the next handoff even though the terminal counters were sound. |
| Fixed validation/freeze workflow | PASS | Added a gateway-compatible smoke experiment that reads the clean dev artifact plus its raw train/dev parents from the checkpoint volume, verifies exact ordered-line provenance and train/dev disjointness, and writes a deterministic prompt/reference/floor bundle. Focused test `data/tests/test_scale_ladder_dev_panels.py` passed. |
| Next safe async launch | PASS | Run `dftr-1784360648-7505beeb` is running with workflow step `freeze_scale_dev_panel`; it is the correct next wake target alongside the still-running 16K cleaner. |
| 16K cleaner live progress snapshot | PASS | At `2026-07-18T07:44:47Z`, run `dftr-1784358360-4f83b039` remained active with the latest worker log showing `processed=922 total_completed=922 api_cost_usd=0.366849`. |
| Budget boundary after freeze launch | PASS | Gateway budget reports Modal committed `$17.53804/$100` and API spend `$28.838326/$100`, leaving `$82.46196` Modal and `$71.161674` API. |
DECISION: Keep the bounded 4K/16K ladder active. The completed 640-document
scale-dev cleaner is terminally sound, but the prospective scientific artifact
still needed an explicit freeze step; that gap is now repaired via a launched
validator rather than by assuming success from counters alone.
NEXT: When `dftr-1784360648-7505beeb` completes, extract the frozen
`prompt_sources` hash/path from its manifest/logs and launch the faithful
two-provider prompt-brief synthesis job. Separately, when
`dftr-1784358360-4f83b039` reaches a terminal state, run the analogous clean
train qualification before any 4,096-document training launch.

## [2026-07-18] M2 / scale-ladder-dev-freeze-complete
HYPOTHESIS: The dedicated scale-dev freeze validator can complete quickly
enough to replace the finished 640-document cleaner as the monitored handoff,
materializing a deterministic prompt/reference/floor bundle before the 16K
cleaner finishes.
SETUP: Same-day follow-up after launching smoke run
`dftr-1784360648-7505beeb` from commit
`b6a8602d970032c5e3df23cc58520f106ee77ea1`. Evidence was limited to the
approved gateway `status`, `logs`, and `budget` surfaces plus the known fixed
config `configs/m2/m2_scale_ladder_freeze_dev_panel_v1.yaml`; no direct Modal
volume access or unapproved data route was used.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Scale-dev freeze validator completion | PASS | Run `dftr-1784360648-7505beeb` completed at `2026-07-18T07:44:56Z` with `return_code=0`, `accel_seconds=18.443`, `actual_cost_usd=$0.011995`, and workflow step `freeze_scale_dev_panel`. |
| Active monitor set reduction | PASS | With the smoke validator terminally complete, the only remaining active asynchronous run is the 16,384-train cleaner `dftr-1784358360-4f83b039`. |
| Prompt-source handoff surface | FAIL | The approved gateway status surface exposes only the validator's generic experiment metadata (`artifact_dir`, return code, cost) and the gateway log surface is empty for this run, so the frozen `prompt_sources` SHA/path needed for `brief_synthesis` is not retrievable through the approved surfaces in this turn. |
| 16K cleaner live snapshot after validator completion | PASS | Gateway status still shows `dftr-1784358360-4f83b039` running; the latest earlier worker log remains `processed=922 total_completed=922 api_cost_usd=0.366849`, and the updated monthly budget is Modal committed `$16.769555/$100`, API spend `$28.838326/$100`. |
DECISION: Keep autonomy enabled for the still-running 16K cleaner, but treat
the missing prompt-source metadata surfacing as a real pipeline observability
gap. Do not guess the prompt-source hash or bypass the approved gateway/data
route to launch brief synthesis.
NEXT: When the 16K cleaner reaches a terminal state, validate it with the same
strictness. In parallel scientific planning, the next data-preparation action
remains prompt-brief synthesis from the frozen scale-dev prompt-source slice,
but that launch now depends on a surfaced hash/path from the completed freeze
artifact rather than on any further cleaning.

## [2026-07-18] M2 / scale-ladder-handoff-surface-repair
HYPOTHESIS: If the only remaining live job is a healthy 16,384-document train
cleaner, the scheduled 90-minute audit should not launch speculative new work.
The recoverable intervention is to repair the sanctioned gateway handoff
surface so future terminal runs expose immutable manifest or output handles
through approved status fields, reducing the risk of another stalled
continuation after valid compute completes.
SETUP: Scheduled 90-minute safety audit on Saturday, July 18, 2026. Read-only
live validation used the coordinator's keychain-backed gateway access plus the
approved `status`, `logs`, and `budget` surfaces for the only monitored run
`dftr-1784358360-4f83b039`. The repo then inspected the gateway, local
backend, and shared volume-path code, implemented a fixed metadata-surfacing
repair in `infra/backend/modal_app.py`, `infra/backend/local_backend.py`,
`infra/backend/local_worker.py`, and `infra/backend/volume_paths.py`, added
focused tests, and redeployed the Modal gateway with
`uv run --project infra modal deploy -m infra.backend.modal_app`. Targeted
verification used `PYTHONPATH=. uv run --project infra pytest
infra/tests/test_volume_paths.py infra/tests/test_local_backend.py`.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| 16K cleaner health check | PASS | The only live remote run remained `dftr-1784358360-4f83b039`; its latest approved worker-log snapshot reached `processed=2626 total_completed=2626 api_cost_usd=1.011754 concurrency=128`, so the ladder is making real progress rather than silently stalling. |
| Live budget boundary | PASS | Gateway budget remained inside the frozen caps at Modal committed `$16.769555/$100` and OpenRouter spend `$28.838326/$100`, leaving `$83.230445` Modal and `$71.161674` API. |
| Existing handoff surface | FAIL | Generic experiment completions surfaced only `artifact_dir`, and fixed-code cleaning/source/brief routes surfaced counts without immutable artifact pointers or SHA-bound output handles, recreating the same observability class that blocked prompt-brief synthesis after the completed scale-dev freeze run. |
| Fixed sanctioned metadata surface | PASS | Terminal run updates now surface a small approved handoff payload: `metrics_ptr` plus `run_manifest_sha256` for run manifests, scale-dev `panel_bundle`/`prompt_sources` pointers when present, and `output_uri`/`output_sha256` or source-manifest URIs for fixed-code data routes. |
| Local verification | PASS | Targeted tests passed `8/8` under the repo-managed infra environment. |
| Production deployment | PASS | The repaired gateway was deployed successfully to `https://bassimfaizal--humanwrite-gpu-gateway-gateway.modal.run` during this audit. |
| Additional async launch need | PASS | None. The live cleaner is healthy and remains the only correct wake target. Launching anything else now would be speculative. |
DECISION: Keep autonomy enabled with only the 16K cleaner monitored. The
pipeline is healthy, but the sanctioned handoff surface needed a real repair
and now has one. No new scientific job is justified until the cleaner reaches
its next terminal transition.
NEXT: When `dftr-1784358360-4f83b039` turns terminal, inspect its approved
status surface first for `output_uri` and `output_sha256`. If those fields are
present, use them for the next clean-train qualification handoff. If they are
absent, treat that as a non-retroactive deployment limitation for this
already-running job and launch only an explicit qualification/freeze validator
from the known frozen config before any 4,096-document training step.

## [2026-07-18] M2 / scale-ladder-train-prefix-handoff-wireup
HYPOTHESIS: With the only live remote job still the healthy 16,384-document
train cleaner, the highest-value safety intervention is not another launch but
an explicit, preregistered train-side qualification/freeze workflow. If that
workflow exists before the cleaner turns terminal, the next continuation can
freeze immutable 4,096- and 16,384-document clean-train prefixes
prospectively instead of discovering another missing handoff after valid
compute completes.
SETUP: Scheduled 90-minute safety audit on Saturday, July 18, 2026. No new
remote compute, provider calls, candidate outputs, or evaluation panels were
opened. The repo inspected the existing lower-variance qualification code and
the newly repaired status-surface contract, then added
`data/materialize_scale_ladder_train_prefixes.py`,
`experiments/m2/scale_ladder_train_prefixes.py`, and the pinned smoke config
`configs/m2/m2_scale_ladder_freeze_train_prefixes_v1.yaml`
(`train_prefix_contract_sha256=5cd6e8a942fc7057ae462cebc06d9b7e2e89fca44fe7197cd7c75f0b1ce4627a`).
The shared sanctioned metadata surface in `infra/backend/volume_paths.py` was
extended so completed train-prefix validators expose bundle and nested-prefix
SHA/path fields. Focused local verification used
`PYTHONPATH=. uv run --project infra pytest data/tests/test_scale_ladder_train_prefixes.py experiments/tests/test_m2_scale_ladder_train_prefixes.py infra/tests/test_volume_paths.py`.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| 16K cleaner live-state check | PASS | No new terminal transition occurred during this audit slice; the only monitored run remains `dftr-1784358360-4f83b039`, already validated as healthy in the prior same-day audit. |
| Existing train-side handoff | FAIL | The repo still lacked a checked-in, exact-source qualification/freeze step for the 16,384-document clean-train artifact, so a terminal cleaner could still dead-end before any 4,096-document training launch. |
| Fixed train-prefix workflow | PASS | The new workflow validates the completed clean-train artifact against the raw 26,000-document train pool and source manifest, enforces the unopened-candidate boundary plus raw-dev disjointness, and writes immutable `clean-train-4096.jsonl` and `clean-train-16384.jsonl` prefixes with manifests and a bundle. |
| Approved metadata surfacing | PASS | Completed train-prefix validators now surface `train_prefix_bundle`, `clean_train_4096`, and `clean_train_16384` SHA-bound pointers through the same approved status-manifest path used for scale-dev handoffs. |
| Local verification | PASS | Focused tests passed `11/11` with the new data materializer, exact config contract, and metadata-surface coverage. |
| Additional async launch need | PASS | None. Launching the new validator now would be speculative because the 16K cleaner has not yet reached a terminal state. |
DECISION: Keep the bounded 4K/16K ladder active with no new remote launch.
The live cleaner is still the only wake target, but its train-side consumer is
no longer missing. This does not unblock scale-dev prompt-brief synthesis:
the already-completed scale-dev freeze run still lacks retrievable
`prompt_sources` SHA/path on approved surfaces, so that separate blocker
remains real.
NEXT: When `dftr-1784358360-4f83b039` reaches a terminal state, use its
approved `output_uri`/`output_sha256` if present; otherwise launch
`configs/m2/m2_scale_ladder_freeze_train_prefixes_v1.yaml` as the explicit
qualification/freeze step before any 4,096-document brief-synthesis or
training launch.

## [2026-07-18] M2 / scale-ladder-prompt-brief-handoff-recovery
HYPOTHESIS: The completed scale-dev freeze artifact should be sufficient to
start faithful two-provider prompt-brief synthesis immediately, but only if
the approved gateway status surface can expose the frozen `prompt_sources`
path and SHA without bypassing the sanctioned data route. If that handoff can
be recovered, the 128-brief preparation job can run safely in parallel with
the still-active 16,384-document clean-train ladder.
SETUP: Scheduled 90-minute safety audit on Saturday, July 18, 2026. Live
inspection first confirmed that train cleaner `dftr-1784358360-4f83b039`
remained active and advancing while completed scale-dev freeze run
`dftr-1784360648-7505beeb` still lacked surfaced prompt-source metadata on
the approved status surface. The repo then repaired the gateway a second time:
`infra/backend/modal_app.py` now mounts the checkpoint volume on the web
gateway, reloads it during `/status`, and backfills missing sanctioned
artifact metadata from the stable `/checkpoints/runs/<run_id>` alias via
`infra/backend/volume_paths.py`. Focused verification used
`PYTHONPATH=. uv run --project infra pytest infra/tests/test_volume_paths.py
infra/tests/test_local_backend.py` and passed `10/10`; the gateway was
redeployed to the existing production URL. After the repaired status surface
returned the frozen `prompt_sources` fields, the repo added pinned config
`configs/m2/m2_scale_ladder_eval_prompt_briefs_v1.yaml`, preregistered
comparison `M2-scale-ladder-eval-prompt-briefs-v1`, and launched promo brief
run `dftr-1784376478-461b4d1b` at git SHA
`754c79ceba311222ba14bac7e910625bd9d96574` with config hash
`b10ef5c537f253804f0f7280f427f889a9e130cffe8e47031597573ecdac0f6e`.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Retroactive scale-dev handoff surface | PASS | Approved `status` for `dftr-1784360648-7505beeb` now surfaces `metrics_ptr`, `panel_bundle_path`, `panel_bundle_sha256`, `prompt_sources_path=modal-volume://humanwrite-checkpoints/data/m2-scale-ladder-v1/scale-dev-panels/prompt_sources.jsonl`, and `prompt_sources_sha256=2eba6d6e18131e5571ba19b7c7c9cd7f6e5643d60e08df26ad32a252e9a8f569`. |
| Root cause of the stale blocker | PASS | The prior repair wrote terminal metadata correctly for future runs, but the web gateway itself could not read historical artifacts because it lacked the checkpoint mount and the stored `artifact_dir` used a resolved Modal volume path rather than the stable mount alias. |
| Local verification after repair | PASS | Focused infra tests passed `10/10` after the checkpoint-mounted status backfill and mount-alias fallback were added. |
| Next safe async launch | PASS | Prompt-brief synthesis run `dftr-1784376478-461b4d1b` is running under the frozen two-provider lower-variance protocol with `$1.0` API reserve, using only the surfaced 128-row frozen `prompt_sources` artifact. |
| 16K cleaner health snapshot | PASS | The still-monitored cleaner `dftr-1784358360-4f83b039` remained healthy and advanced to `processed=8300 total_completed=8300 api_cost_usd=3.180233 concurrency=128` in the latest approved worker-log snapshot. |
| Budget boundary after launch | PASS | Gateway budget reports Modal committed `$16.769555/$100` and OpenRouter spend/reserve `$29.838326/$100`, leaving `$83.230445` Modal and `$70.161674` API. |
DECISION: Keep autonomy enabled. This audit found a real missed handoff,
repaired it without weakening any quality gate, and launched the correct next
data-preparation job while the 16K cleaner continues independently. The 46K
cell remains blocked by the preregistered 16K gate, and Measurement-v4
remains blocked by the Nemotron manipulation check miss.
NEXT: Wake on the next terminal transition from either
`dftr-1784376478-461b4d1b` or `dftr-1784358360-4f83b039`. If the brief job
finishes first, validate the 128 prompt briefs before any candidate
generation. If the cleaner finishes first, use surfaced `output_uri` and
`output_sha256` if present; otherwise launch the pinned train-prefix freeze
validator before any 4,096-document training or brief-synthesis step.

## [2026-07-18] M2 / scale-ladder-prompt-brief-terminal-repair
HYPOTHESIS: The failed scale-ladder prompt-brief run is recoverable without a
new protocol, panel, or side channel if the committed partial artifact is
still canonical. Because the fixed-code brief worker validates any existing
rows before emitting new ones, rerunning the same frozen 128-source config and
output URI should safely recheck the 125 committed briefs and attempt only the
three missing identities.
SETUP: Terminal-transition audit on Saturday, July 18, 2026. The repo used the
sanctioned gateway credential pattern already recorded in `.operator/autonomy/`
to inspect terminal state for failed brief run `dftr-1784376478-461b4d1b` and
the still-running clean-train run `dftr-1784358360-4f83b039`. Gateway status
showed the prompt-brief artifact remained at
`modal-volume://humanwrite-checkpoints/data/m2-scale-ladder-v1/scale-dev-panels/prompt_briefs-128.jsonl`
with `output_sha256=8860fdb48ee2db3563f5516727732b47ebec46462beb32e7a5636233676c3b76`,
`records_processed=125`, `records_failed=3`, and `actual_api_cost_usd=0.115273`.
Approved logs identified the three failing fingerprints and validator causes.
No code change was required. The repo then resubmitted the same pinned config
`configs/m2/m2_scale_ladder_eval_prompt_briefs_v1.yaml` under the same open
comparison `M2-scale-ladder-eval-prompt-briefs-v1`, producing resume run
`dftr-1784377434-a4c6dae1` at git SHA
`897582552789560fa04342857be9ce301a20d673`. An immediate follow-up status
check confirmed the retry is running rather than failing on existing-row
validation. The 16K cleaner was rechecked once and remained healthy at
`processed=8696 total_completed=8696 api_cost_usd=3.340495`.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Terminal prompt-brief failure classified | PASS | The failure was ordinary provider/validator behavior, not a stale handoff or panel-corruption event: `01cf7bcac18b7c0f289b0d042d2c9553bfc07c31c029290caed23616adbf94de` and `75e5a5aeec5d46cfe5d88516beb45edf93cac37aacec8efb507128c892c4b990` returned Qwen metadata with the wrong fingerprint; `b92f8d3fe2009484c193138800854bd9e53552cd82369b0107765ea69b9367b1` failed `user_prompt lacks source-grounded topic terms`. |
| Partial artifact preserved | PASS | Status still binds the 125-row partial output to the sanctioned volume URI and SHA, so the fixed-code worker can treat it as the sole committed state. |
| Bounded repair route | PASS | Relaunching the same 128-source config and output URI means the worker must validate the existing 125 rows against the full source set before it can skip them, and it can only generate the three missing identities. |
| Resume launch | PASS | Promo resume run `dftr-1784377434-a4c6dae1` launched successfully with the same config hash `b10ef5c537f253804f0f7280f427f889a9e130cffe8e47031597573ecdac0f6e` and entered `running` state immediately. |
| 16K cleaner health recheck | PASS | The independent clean-train run `dftr-1784358360-4f83b039` remained active and advanced from the earlier 8,300-row snapshot to `8,696` accepted rows at `3.340495` dollars cumulative API spend. |
| Budget boundary after repair | PASS | Gateway budget now reports Modal committed `$16.769555/$100` and OpenRouter spend `$28.953598/$100`, leaving ample room inside both user-approved caps. |
DECISION: Keep autonomy enabled. The missed work here was a recoverable data-
preparation failure, and the sanctioned repair is now in flight. No code or
protocol mutation was justified because the generic worker already supports
safe resume semantics over a committed partial artifact. The 46K cell remains
blocked by the preregistered 16K gate, and Measurement-v4 remains blocked by
the Nemotron manipulation-check miss.
NEXT: Wake on the next terminal transition from
`dftr-1784377434-a4c6dae1` or `dftr-1784358360-4f83b039`. If the resume run
finishes first, validate whether all 128 prompt briefs are now present before
any 4K/16K candidate generation. If the cleaner finishes first, launch the
pinned train-prefix freeze validator against the surfaced clean-train output
before any 4,096-document training job.

## [2026-07-18] M2 / scale-ladder-prompt-brief-resume-complete
HYPOTHESIS: If the partial 125-row prompt-brief artifact was canonical, the
bounded resume launch should terminate successfully after validating the
committed rows and filling only the three missing identities, producing a
whole 128-brief panel without any new protocol or panel mutation.
SETUP: Immediate follow-up status audit after launch of resume run
`dftr-1784377434-a4c6dae1`. The repo queried the sanctioned gateway once more
to confirm whether the retry still needed monitoring before finalizing
`progress/autonomy.json`. No new compute was launched beyond the already-
running clean-train ladder.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Resume terminal result | PASS | Run `dftr-1784377434-a4c6dae1` reached `completed` at `2026-07-18T12:24:35Z` with `records_processed=3`, `records_failed=0`, and `actual_api_cost_usd=0.00468`. |
| Whole prompt-brief artifact restored | PASS | Completed status surfaces the same sanctioned output URI and an updated `output_sha256=24d55ae6695930b07030fbe6c4d88b47cb24fee2bfa217d42968ce9703f278f0`, which implies the worker revalidated the existing 125 rows and closed the remaining three identities. |
| Candidate-blind boundary preserved | PASS | The repair reused the frozen 128-source config and existing output path; no candidate output was generated or opened, and no panel membership changed. |
| Remaining active remote work | PASS | After prompt-brief completion, the only active asynchronous run is still the 16,384-document cleaner `dftr-1784358360-4f83b039`. |
DECISION: Keep autonomy enabled, but reduce monitoring to the 16K cleaner
only. Scale-dev prompt-brief preparation is complete. The next sanctioned step
is still blocked on the clean-train artifact finishing and then being frozen
into immutable 4,096/16,384 prefixes before any 4K training launch.
NEXT: Wake only on terminal transition from `dftr-1784358360-4f83b039`. When
it completes, validate the clean-train artifact and launch the pinned
train-prefix freeze workflow before any downstream 4K/16K candidate training.

## [2026-07-18] M2 / scale-ladder-live-status-repair
HYPOTHESIS: The scheduled 90-minute audit should see the same cumulative
progress that approved worker logs already show for the live 16K cleaner. If
the sanctioned `/status` surface is only exposing the most recent incremental
`api_cost` event instead of cumulative progress, then the autonomy monitor is
stale even though the underlying run is healthy. A narrow gateway repair that
parses the existing worker progress log should restore accurate live counts
without opening any candidate outputs or touching the scientific protocol.
SETUP: Scheduled 90-minute safety audit on Saturday, July 18, 2026. Direct
inspection of the sanctioned gateway showed an inconsistency for active
clean-train run `dftr-1784358360-4f83b039`: `/status` returned only
`cost_usd=0.025067` with no cumulative `actual_api_cost_usd` or
`records_processed`, while `/logs` simultaneously showed
`processed=10510 total_completed=10510 api_cost_usd=4.085475`. The repo
patched the gateway prospectively by adding pure parser
`infra/backend/status_progress.py` and wiring `infra/backend/modal_app.py` to
enrich running `brief_synthesis` and `document_cleaning` states from their own
progress logs. Focused verification used
`PYTHONPATH=. uv run --project infra pytest infra/tests/test_status_progress.py
infra/tests/test_volume_paths.py infra/tests/test_autonomy_coordinator.py`
and passed `21/21`. The Modal gateway was then redeployed to the existing
production URL and the same live cleaner was rechecked through the sanctioned
status and logs routes.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Root cause classified | PASS | Running API jobs wrote cumulative progress only to worker logs, while `/status` exposed the latest incremental `api_cost` event. The autonomy coordinator therefore saw a stale delta-only view for healthy long-running cleaners. |
| Prospective live-status repair | PASS | Running `brief_synthesis` and `document_cleaning` states are now enriched from their existing log snapshots, surfacing cumulative `records_processed`, `records_failed`, and `actual_api_cost_usd` without changing terminal accounting or protocol semantics. |
| Focused verification | PASS | Focused infra tests passed `21/21` before redeploy. |
| Live post-deploy validation | PASS | Re-querying `dftr-1784358360-4f83b039` after deploy returned `records_processed=10748`, `records_failed=6148`, and `actual_api_cost_usd=4.182616` on `/status`, matching the latest approved `/logs` snapshot `processed=10748 total_completed=10748 api_cost_usd=4.182616 concurrency=128`. |
| Cleaner health during repair | PASS | The 16K cleaner remained productively active throughout the audit; no cancellation, resume, or speculative downstream launch was required. |
| Budget boundary after repair | PASS | Official gateway budget remained within cap at Modal committed `$16.769555/$100` and OpenRouter spend `$28.958278/$100`. |
DECISION: Keep autonomy enabled and keep monitoring only
`dftr-1784358360-4f83b039`. This audit found and repaired a real stale monitor
surface, not a scientific or data-pipeline failure. No new async launch is
justified until the clean-train run reaches a terminal state, because the next
sanctioned step is still the pinned train-prefix freeze workflow.
NEXT: Wake only on terminal transition from `dftr-1784358360-4f83b039`. When
it completes, validate the surfaced clean-train artifact and launch the pinned
4,096/16,384 train-prefix freeze step before any downstream 4K training or
candidate generation.

## [2026-07-18] M2 / scale-ladder-clean-complete-and-prefix-freeze-launch
HYPOTHESIS: The bounded in-place cleaning resume can complete the frozen
16,384-document target, after which the already-tested qualification workflow
can freeze immutable nested 4,096/16,384 corpora without changing data-quality
rules or opening candidate outputs.
SETUP: Sanctioned gateway status and logs confirmed resume run
`dftr-1784387469-9e5b1936` completed at exactly 16,384 total accepted rows.
The existing train-prefix config was then append-only preregistered under
comparison `M2-scale-ladder-freeze-train-prefixes-v1` and launched unchanged
as smoke run `dftr-1784403212-e147e7cb`.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Clean-train cardinality | PASS | Initial run contributed 12,648 accepted rows and the resume contributed 3,736, reaching exactly 16,384. |
| Clean artifact identity | PASS | Completed output SHA-256 is `ee8d6f35e11e4e7dbf4b7d11c1c5532b7426d46c42b0ac41867382127ef79626`. |
| Quality-rule stability | PASS | The same Qwen3-32B ordered-line cleaner and frozen 80-220-word bounds remained in force. |
| Prefix-freeze launch | PASS | `dftr-1784403212-e147e7cb` entered running state under the pinned smoke protocol. |
| Candidate boundary | PASS | No tuned candidate output or evaluation result was generated or opened. |
DECISION: Continue the bounded 4K/16K ladder. Source expansion is unnecessary
because the exact target completed through safe retry.
NEXT: Validate the prefix-freeze terminal artifact, synthesize faithful
training briefs, and launch the matched 4B SFT and MMD-witness fine-tuning
arms after their immutable input hashes are available.

## [2026-07-18] M2 / scale-ladder-clean-train-timeout-resume
HYPOTHESIS: The 16K clean-train timeout is recoverable without changing the
comparison, protocol, output path, or quality gate because the fixed-code
cleaner commits accepted rows incrementally and, on startup, skips any source
fingerprints already present in the canonical output artifact. Relaunching
the same config against the same output URI should therefore preserve the
prospective 4K/16K ladder while continuing from the 12,648 committed accepts.
SETUP: Terminal-transition continuation on Saturday, July 18, 2026 after the
coordinator observed failed run `dftr-1784358360-4f83b039` for comparison
`M2-scale-ladder-clean-train-16k-v1`. The repo read `CLAUDE.md`,
`RESEARCH_CONTEXT.md`, `progress/autonomy.json`, `progress/status.json`,
`FINDINGS.md`, and recent git history, then queried the sanctioned gateway
status and logs through the same keychain-backed credential pattern already
used by the local coordinator. The frozen config remained
`configs/m2/m2_scale_ladder_clean_train_16k_v1.yaml` with output URI
`modal-volume://humanwrite-checkpoints/data/m2-scale-ladder-v1/clean-train-16384.jsonl`.
The worker implementation in `infra/backend/modal_app.py` was inspected to
verify that existing committed rows are re-read by `source_fingerprint`, then
skipped before new work begins. No code or protocol change was made. After
confirming budget headroom, the same frozen config was relaunched through the
approved wrapper with `uv run --project infra ./infra/gpu submit ...`,
producing resume run `dftr-1784387469-9e5b1936` at git SHA
`b42bd7d63ef712ca784c2f397ce111fc129f37d2`.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Terminal failure classification | PASS | Run `dftr-1784358360-4f83b039` ended `failed` with `worker_result_error=FunctionTimeoutError`, not with an input-hash, policy, or quality-contract failure. Terminal status showed `records_processed=12648`, `records_failed=7192`, and `actual_api_cost_usd=4.919519`. |
| Partial artifact resumability | PASS | The fixed-code cleaner appends accepted rows to the canonical output path, commits the checkpoint volume after each batch, and rebuilds its completed set from existing `source_fingerprint` values before resuming. This makes an in-place bounded resume scientifically cleaner than forking a new output artifact. |
| Budget boundary before relaunch | PASS | Approved budget check before resume showed Modal committed `$16.769555/$100` and OpenRouter spend/reserve `$21.877799/$100`, leaving `$83.230445` Modal and `$78.122201` API. |
| Resume launch | PASS | Promo resume run `dftr-1784387469-9e5b1936` launched successfully under the same config hash `ad131251512f9b754a5be7be8f6d791cb17f2fb3eefc1497c82847ad328e94df` and entered `running` state immediately. |
| Budget boundary after relaunch | PASS | Post-launch budget remained inside the user caps at Modal committed `$16.769555/$100` and OpenRouter spend/reserve `$33.877799/$100`, leaving `$83.230445` Modal and `$66.122201` API. |
| Additional async launch need | PASS | None. The resumed 16K cleaner is now the only active remote job, and the 46K cell remains blocked by the preregistered 16K gate. |
DECISION: Keep autonomy enabled and retarget monitoring to the new resume run
only. The scientific pipeline did not need a new method or relaxed gate; it
needed a bounded continuation of the same frozen cleaning comparison. The
already-prepared train-prefix freeze step remains the next sanctioned handoff
after this cleaner reaches a terminal state.
NEXT: Wake only on terminal transition from `dftr-1784387469-9e5b1936`. If it
completes with 16,384 accepted rows, validate the surfaced clean-train
artifact and launch the pinned train-prefix freezer before any 4K training
run. If it fails again, inspect whether the remaining gap is due to timeout,
insufficient remaining valid candidates, or a new contract failure before
considering any further relaunch.

## [2026-07-18] M2 / scale-ladder-4k-terminal-brief-recovery-audit
HYPOTHESIS: The 4K brief pipeline did not need a new scientific branch; it
needed the continuation state repaired to follow the actual bounded retries.
If the narrow safe-excerpt recovery path works as intended, the frozen
4,096-document brief artifact should advance materially beyond 4,078 valid
rows without weakening provenance or opening candidate outputs, leaving only a
small final retry set if the remaining misses are ordinary provider or
validator failures.
SETUP: Scheduled 90-minute safety audit on Saturday, July 18, 2026 after the
deterministic coordinator surfaced failed retry `dftr-1784418273-0bb9e6ef`.
The repo read `CLAUDE.md`, `RESEARCH_CONTEXT.md`, `progress/autonomy.json`,
`progress/status.json`, `FINDINGS.md`, and recent git history, then queried
the sanctioned gateway status for `dftr-1784418273-0bb9e6ef`,
`dftr-1784418784-df1cd806`, and `dftr-1784419062-7c7e2b47` through the same
keychain-backed status path used by the local coordinator. Evidence remained
limited to approved status and budget surfaces plus the append-only ledger and
git history; no direct checkpoint-volume access or unapproved provider route
was used.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Terminal failure for first final retry | PASS | Run `dftr-1784418273-0bb9e6ef` was genuinely terminal, not stale: it finished `failed` after recovering `records_processed=4`, leaving `records_failed=18`, with `actual_api_cost_usd=$0.029285` and output SHA `40802d4fd03836bed2a94487e797b70de4a6c0570d6a5fee30af2ce89eca7c31`. |
| Safe-excerpt recovery value | PASS | Run `dftr-1784418784-df1cd806` used the narrowed fallback path and recovered `records_processed=12` additional identities while leaving only `records_failed=6`, at `actual_api_cost_usd=$0.038267` and output SHA `ae8ace39980fa587c9cacd9ac8b1508bb494bfc607bae813830c15b4dc26a83b`. |
| Current 4K brief cardinality | PASS | Starting from 4,024 validated rows, then adding 50, 4, and 12 across the successive bounded retries yields 4,090 validated training briefs, so the exact 4,096-row contract is now six identities short rather than twenty-two. |
| Active remote work after audit | PASS | Run `dftr-1784419062-7c7e2b47` is the only active remote job. It is a same-comparison, same-config final retry over the remaining six identities under the unchanged output URI `modal-volume://humanwrite-checkpoints/data/m2-scale-ladder-v1/train-briefs-4096.jsonl`. |
| Continuity state integrity | FAIL | `progress/autonomy.json` and `progress/status.json` were still pointing at failed run `dftr-1784418273-0bb9e6ef` even though the ledger and git history had already advanced through two later relaunches. This was a real missed handoff / stale monitor target, not a scientific failure. |
| Budget boundary during audit | PASS | Approved budget remained within the user caps at Modal committed `$16.778820/$100` and OpenRouter spend `$43.475634/$100`, leaving `$83.221180` Modal and `$56.524366` API. |
DECISION: Keep autonomy enabled, append the missing terminal evidence to the
ledger, and retarget monitoring to `dftr-1784419062-7c7e2b47` only. Do not
launch the 4K baseline witness, matched 4B SFT control, or matched 4B
MMD-witness arm until the exact 4,096-row brief artifact is complete and
validated. This audit repaired continuity and evidence capture; it did not
justify weakening the brief-quality gate.
NEXT: Wake only on terminal transition from `dftr-1784419062-7c7e2b47`. If it
completes with all 4,096 validated briefs, validate the surfaced output SHA
and immediately launch the pinned 4K baseline-witness and matched fine-tuning
handoff. If it fails again, inspect whether the remaining six are persistent
provider/validator misses and decide whether the current faithful-brief method
family has reached a defensible negative conclusion at the 4K data cell
rather than reopening panels or relaxing quality rules.

## [2026-07-18] M2 / scale-ladder-4k-briefs-complete-pause
HYPOTHESIS: The last six missing 4K brief identities can close under the same
frozen protocol, but whether to proceed into GPU witness generation and matched
4B fine-tuning is no longer an autonomous decision if the user has explicitly
paused the pipeline for independent review.
SETUP: Follow-up validation after the continuity-repair commit. The sanctioned
gateway status for `dftr-1784419062-7c7e2b47` was rechecked through the
keychain-backed coordinator path. At the same time, `progress/autonomy.json`
changed outside this turn to `enabled=false`, `generation=21`, objective
`PAUSED by user before GPU witness generation or fine-tuning, pending
independent review of hardcoded contracts, implementation, and sampled 4K
data.`
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Final 4K retry completion | PASS | Run `dftr-1784419062-7c7e2b47` completed with `records_processed=6`, `records_failed=0`, `actual_api_cost_usd=$0.007393`, and output SHA `0c35745e5a352a63fef17bee246e2c1822cf54a609e0c41e05327754db135d47`. |
| Exact 4K brief contract | PASS | The canonical 4K brief artifact is now whole at 4,096 validated rows under the unchanged output URI `modal-volume://humanwrite-checkpoints/data/m2-scale-ladder-v1/train-briefs-4096.jsonl`. |
| Remaining remote work | PASS | None. No asynchronous run remains active after the final retry completed. |
| User-authority gate | PASS | `progress/autonomy.json` now explicitly pauses the pipeline before witness generation or fine-tuning, so launching the next GPU jobs would override a direct user stop signal. |
| Budget boundary at park | PASS | Approved budget remained inside cap at Modal committed `$16.778820/$100` and OpenRouter spend `$28.483027/$100`. |
DECISION: Preserve the user pause and park the pipeline. Do not launch the 4K
baseline witness, matched 4B SFT control, or matched 4B MMD-witness arm until
the user re-enables autonomy after the requested independent review.
NEXT: Wait for user authority. The next scientific action, once explicitly
re-enabled, is to validate the completed 4K brief artifact and then launch the
pinned baseline-witness plus matched 4B training handoff from the frozen 4,096
row corpus.

## [2026-07-18] M2 / 1K confirmation outcome backfill
HYPOTHESIS: The matched 1,024-row MMD-witness arm could displace the initial
adapter and improve independently measured proximity to human writing over a
matched SFT continuation.
SETUP: Matched Qwen3-4B SFT and MMD-witness arms, 1,024 training briefs, two
complete epochs, 128-token teacher-forced horizon, followed by a fresh blinded
128-prompt evaluation with independent BGE and Nemotron representations and
randomized human-style/overall-quality comparisons.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Mechanical training | PASS | Both arms completed 1,024 steps, 2,048 examples, and 256,466 completion tokens; combined H100 cost was `$2.36196`. |
| Witness-gradient gate | PASS | First-64 median witness-to-uniform gradient ratio was `0.096835`, inside `[0.05, 0.30]`; all-step median was `0.080187`. |
| Human-style direction | INCONCLUSIVE | MMD-witness won `54.7%` of human-style comparisons and `53.1%` overall, but the differences were not significant. |
| Independent distribution direction | INCONCLUSIVE | Candidate-minus-control improvement was `0.000032` under BGE MMD and `0.000461` under Nemotron MMD; neither cleared promotion. |
| Lexical metric | FAIL | Token 1-gram L2 worsened by `0.000858`. |
| Policy displacement | FAIL | Treatment and control were byte-identical on `78%` of evaluation outputs. |
DECISION: The 1K result is not a promotion. It provides weak favorable human
preference and embedding directions, but insufficient displacement and a
lexical regression. It justifies one bounded, cleaner 4K scale test—not a claim
that the method works.
NEXT: Use the independently reviewed 4K pipeline with exact token semantics,
matched support, a bounded timing gate, and a fresh evaluation panel.

## [2026-07-18] M2 / independent-4K-review-repair-and-restart
HYPOTHESIS: Repairing the review-confirmed token-unit, runtime, support, and
witness-durability defects can make the 4K experiment mechanically interpretable
without changing its scientific treatment after outcomes are known.
SETUP: Independent report `/Users/bassime/Downloads/feedback-claude-humanwrite.md`
was read in full before any GPU witness or tuning launch. Its findings were
reconciled in
`research_reviews/feedback_claude_reconciliation_2026-07-18.md`. Code now
normalizes all target lengths with the pinned Qwen tokenizer, verifies runtime
versions and adapter files, exposes batch/prompt bounds in config, matches the
training representation to 128-token witness support, supports either-arm
resume, preserves partial witness diagnostics, and adds a bounded 64-step H100
MMD-witness timing gate.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Reviewer verdict | ACCEPT | `FIX-THEN-RUN`; no original 4K GPU config was launched. |
| Focused repaired-path tests | PASS | `21/21` passed after the final optimization and timing-smoke changes. |
| Broader selected regression | QUALIFIED PASS | `65/66` passed; the sole failure is a pre-existing source-inspection assertion that rejects the already-committed restricted worker's deliberate final volume commit and is unrelated to this repair. |
| Scientific result | PENDING | No 4K candidate exists yet. The next jobs are input normalization, baseline witness, timing smoke, and—only if gates pass—matched fine-tuning. |
DECISION: User authority is restored by the explicit instruction to proceed.
Launch only the repaired, preregistered chain; do not bypass the timing or
empty-witness gates.
NEXT: Run exact-token normalization, generate the 4K baseline witness, run the
64-step MMD-witness timing smoke, then launch matched full arms concurrently if
the gate passes.

## [2026-07-19] M2 / scheduled-4k-witness-safety-audit
HYPOTHESIS: The scheduled 90-minute audit should either find a real stale
handoff/silent failure worth repairing or confirm that the only active 4K
baseline-witness job is healthy enough to leave alone. If any observability
surface itself is broken, that defect should be repaired prospectively without
changing the scientific treatment or launching extra jobs.
SETUP: Read `CLAUDE.md`, `RESEARCH_CONTEXT.md`, `progress/autonomy.json`,
`progress/status.json`, the append-only `FINDINGS.md` record, and recent git
history. Verified the live gateway state for
`dftr-1784429664-02939bdc` through the same sanctioned path the coordinator
uses: fixed gateway URL plus macOS Keychain token `humanwrite-gateway-token`.
Inspected the witness config and the Modal gateway wrapper/logging code before
making any change.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Active 4K witness job age | PASS | At `2026-07-19T03:14:38Z`, run `dftr-1784429664-02939bdc` was only `20.22` minutes old against a `90` minute timeout, so the `running` state was not stale. |
| Live run status | PASS | Gateway status still reported `status=running`, `comparison=M2-scale-ladder-witness-4096-v1`, `workflow_step=generate_scale_ladder_witness`, `gpu=L40S`, and no terminal transition to validate yet. |
| Budget headroom | PASS | Sanctioned budget check reported Modal committed `$20.301228/$100` and OpenRouter spend `$28.483027/$100`; no cap was pressured by this audit. |
| Log-surface integrity | FAIL | The experiment wrapper streamed stdout to `/tmp/<run_id>.worker.log`, but the sanctioned `/logs/{run_id}` route read `/state/logs/<run_id>.log`. Running experiment logs were therefore guaranteed blank even when the job was healthy, weakening scheduled-audit visibility. |
| Recoverable repair | PASS | The gateway wrapper now writes experiment logs to `/checkpoints/runs/<run_id>/worker.log`, periodically commits the checkpoint volume during long-running training, and reloads the checkpoint volume before serving `/logs/{run_id}`. Direct import/assert verification for the shared worker-log path passed; `python3 -m py_compile` also passed. `pytest` could not be run on this machine because the module is not installed. |
DECISION: Keep the live 4K witness run active and do not launch any additional
remote job during this audit. The pipeline itself is healthy enough to
continue, but the sanctioned experiment-log surface needed a real wrapper fix,
which is now applied prospectively.
NEXT: Wait for the terminal transition of `dftr-1784429664-02939bdc`. On
completion, validate the witness manifest/artifacts and launch the 64-step H100
MMD timing smoke only if the witness is complete and non-empty.

## [2026-07-19] M2 / M2-scale-ladder-4b-4096-timing-smoke-v1
HYPOTHESIS: If the completed 4K baseline witness is truly whole, non-empty,
and bound to the exact normalized 4,096-brief corpus, then the next safe step
is the preregistered 64-step H100 MMD-witness timing smoke. That smoke should
consume the same initial adapter, 128-token witness contract, and exact 4K
anchor without opening candidate evaluation outputs. The full 4K SFT and
MMD-witness arms should remain gated behind this bounded timing result.
SETUP: Validated the completed witness through the sanctioned gateway path used
by the autonomy coordinator: fixed gateway URL plus macOS Keychain token
`humanwrite-gateway-token`. The completed status for
`dftr-1784429664-02939bdc` exposed
`metrics_ptr=modal-volume://humanwrite-checkpoints/runs/dftr-1784429664-02939bdc/run_manifest.json`
with `run_manifest_sha256=fbd94205993e1d8bd74b89792e8e4d9ae7a1da6eefaa8bcbd0e172ba0fd721a5`.
Sanctioned worker logs ended with the full witness manifest JSON, including
`documents=4096`, `status=completed`, `sampling_seed=41001`,
`output_sha256=99d8b0ae93fb29ca7d692edd6c025e4310fcd0bf84188c37827e89d8a00eedae`,
and `generation_contract_sha256=52ad0ca17e804709f2287396b41710a7d0bfa717a3a74f98f04fcf748ed38a1c`.
Used that manifest to materialize:
`configs/m2/m2_scale_ladder_4b_4096_sft_v1.yaml`
(`sha256=88c7898f26c35cd2803b599a0c188e818c00cdbd6155869b8cc6d6fc98022d99`),
`configs/m2/m2_scale_ladder_4b_4096_mmd_witness_v1.yaml`
(`sha256=58ed3d678827e3a7b521625603ca824f01831f8b1b8a329771785fab723e5ad5`),
and the smoke config
`configs/m2/m2_scale_ladder_4b_4096_timing_smoke_v1.yaml`
(`sha256=1d2d130082a2c1255e107b2e50d78c6f13cc28cac490be59fbc182148162e05b`).
These bindings had to use the existing two-arm confirmation contract rather
than the older three-arm lower-variance screen contract, because the 4K ladder
is gated to matched SFT versus MMD-witness only.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Witness terminal status | PASS | Sanctioned status showed `dftr-1784429664-02939bdc` completed with `return_code=0`, `accel_seconds=3871.837`, and `actual_cost_usd=$2.518243`. |
| Witness artifact integrity | PASS | Sanctioned logs ended with a completed `dftr.m2.lower_variance_baseline_witness.v2` manifest bound to the exact 4K normalized briefs, `documents=4096`, and non-empty output SHA `99d8b0ae93fb29ca7d692edd6c025e4310fcd0bf84188c37827e89d8a00eedae`. |
| Local experiment ledger | PASS | Appended a local `ledger update` row for the completed witness before opening the next run. |
| Bound 4K arm configs | PASS | Materialized exact 4K SFT and MMD-witness configs plus the derived 64-step timing-smoke config, all hash-stable and witness-bound. |
| Timing-smoke preregistration | PASS | Added preregistration `M2-scale-ladder-4b-4096-timing-smoke-v1` before launch. |
| Next safe async launch | PASS | Submitted smoke run `dftr-1784434111-716a0ed8` at `2026-07-19T04:08:31Z`; sanctioned status immediately reported `status=running`, `gpu=H100`, `budget_class=smoke`, and `reserved_cost_usd=1.57968`. |
| Budget headroom after launch | PASS | Post-launch sanctioned budget showed Modal committed `$20.886991/$100` and OpenRouter spend `$28.483027/$100`, leaving the bounded cycle inside both caps. |
DECISION: Keep the 4K scale-ladder active. The witness handoff validated
cleanly enough to spend the bounded 64-step H100 smoke, and the full 4K arms
remain properly gated behind that result. No candidate-writing evaluation was
opened in this turn.
NEXT: Wait for terminal transition of `dftr-1784434111-716a0ed8`. If it
completes within the smoke budget and does not expose a contract/runtime
failure, preregister and launch the matched full 4K SFT and MMD-witness arms
from the newly materialized configs. If it fails or overruns, record the defect
and do not open the full 4K arms.

## [2026-07-19] M2 / M2-scale-ladder-4b-4096-timing-smoke-v1-terminal-audit
HYPOTHESIS: The terminal failure of `dftr-1784434111-716a0ed8` should be
treated as a real scientific negative only if the sanctioned worker log shows
the MMD-witness trainer actually reached the frozen training path. If the job
failed before step 1 because of a wrapper/runtime handoff defect, the correct
action is to repair the defect, preserve the failed run as invalid evidence,
and relaunch a fresh preregistered smoke without weakening any quality gate.
SETUP: Read the sanctioned status and worker log for
`dftr-1784434111-716a0ed8` through the same fixed gateway URL and Keychain
token the autonomy coordinator uses. Inspected the exact traceback against the
current repo code after the earlier log-surface repair. Updated the lower
variance runner so the sanctioned wrapper-owned `worker.log` does not trip the
artifact-dir emptiness guard, and realigned the 4K scale-ladder materializers
to the already-recorded `L40S`-only execution policy. The repaired 4K config
hashes are `configs/m2/m2_scale_ladder_4b_4096_sft_v1.yaml`
`sha256=c859da292b8a81cc8f00bcaf3055d6a698c66e44e28f2b933109feeff6efa7bd`,
`configs/m2/m2_scale_ladder_4b_4096_mmd_witness_v1.yaml`
`sha256=5382aaa5df25a138b675f532697dffc03ec1b55a3891661077b8596302ee232e`,
and fresh smoke config
`configs/m2/m2_scale_ladder_4b_4096_timing_smoke_l40s_v1.yaml`
`sha256=e2dd1da0d70b7cf57fcbfe7a7bf78518d77d598dc2a846ee014bb97215b2525a`.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Terminal run validity | FAIL | Sanctioned status shows `dftr-1784434111-716a0ed8` failed on Sunday, July 19, 2026 after only `14.154` accelerator-seconds, `tokens=0`, and `actual_cost_usd=$0.018632`, so the run never became a scientific timing result. |
| Root cause | FAIL | Sanctioned worker log ends with `M2ConfigError: lower-variance checkpoint directory already contains artifacts`. The earlier log-surface repair now writes `/checkpoints/runs/<run_id>/worker.log`, and `run_lower_variance()` still called `_require_no_existing_files()` on that same directory before training. |
| Scientific interpretation | PASS | Because the failure happened before step 1 and before any completion tokens were generated, it is an infrastructure/handoff defect, not evidence against MMD-witness on the 4K ladder. |
| Recoverable code repair | PASS | `_require_no_existing_files()` now tolerates only the sanctioned precreated `worker.log`, while still failing closed on any other preexisting artifact or symlink. `python3 -m py_compile` passed for the repaired training/materializer modules. |
| Ladder backend alignment | PASS | The 4K scale-ladder timing and full-arm configs are now aligned with the existing `L40S`-only execution policy recorded earlier in `progress/status.json`, instead of reopening the stale `H100` path. |
| Local verification gap | WARN | Focused `pytest` was not runnable in this shell because the main environment lacks `pytest`, `torch`, and `yaml`, while the checked-in infra venv lacks the full training stack. The sanctioned worker traceback supplied the decisive failure evidence. |
DECISION: Discard `M2-scale-ladder-4b-4096-timing-smoke-v1` as invalid due to a
wrapper-owned artifact collision and do not use it for any scientific gate. A
fresh `L40S` smoke is the next safe step because it preserves the same 64-step
two-arm MMD timing purpose while matching the active ladder runtime policy and
the repaired wrapper/trainer contract.
NEXT: Commit the repair, preregister
`M2-scale-ladder-4b-4096-timing-smoke-l40s-v1`, launch it through the
sanctioned wrapper, and monitor only that new run. Launch the matched 4K SFT
and MMD-witness full arms only if the fresh smoke completes without a contract
or runtime failure.

## [2026-07-19] M2 / M2-scale-ladder-4b-4096-timing-smoke-l40s-v1-launch-audit
HYPOTHESIS: Once the wrapper-log collision is repaired and the 4K ladder is
realigned to `L40S`, the fresh smoke should either begin normal training or
fail with a new concrete transport/runtime defect that can be independently
validated before any full-arm launch. A launch failure before Python startup is
not a model result.
SETUP: Preregistered
`M2-scale-ladder-4b-4096-timing-smoke-l40s-v1`, submitted
`configs/m2/m2_scale_ladder_4b_4096_timing_smoke_l40s_v1.yaml` through the
sanctioned wrapper, then inspected the sanctioned status and worker log for
`dftr-1784435621-a7cf08e3`.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Fresh smoke launch | PASS | Wrapper accepted the repaired `L40S` config and created run `dftr-1784435621-a7cf08e3` with `reserved_cost_usd=$0.78048` under the same capped smoke budget. |
| Terminal validity | FAIL | Sanctioned status shows the run failed after only `1.334` accelerator-seconds, `tokens=0`, and `actual_cost_usd=$0.000868`, so it never became a usable timing result. |
| Root cause | FAIL | Sanctioned worker log contains `[dftr] wrapper failure: CalledProcessError ... git checkout --detach 2ff023680785feb444fd217c6b39de922cc327f8 ... exit status 128`. The remote worker could not check out the new repair commit because it had not yet been pushed to origin. |
| Scientific interpretation | PASS | This is a wrapper transport failure before experiment startup, not evidence about MMD-witness or the 4K ladder. |
DECISION: Keep the repaired `L40S` smoke comparison open but discard run
`dftr-1784435621-a7cf08e3` as invalid. The next safe action is operational:
push the committed repair SHA to origin, then relaunch the same config so the
worker can resolve the requested checkout.
NEXT: Push branch `swarmy/humanwrite-next-cycle`, relaunch
`M2-scale-ladder-4b-4096-timing-smoke-l40s-v1`, and monitor only the fresh run
ID from that pushed commit. Do not open the matched 4K full arms until the
relaunch survives startup and produces a real timing signal.

## [2026-07-19] M2 / M2-scale-ladder-4b-4096-timing-smoke-l40s-v1-relaunch
HYPOTHESIS: If the only remaining defect is that the requested repair commit
was not yet on origin, then pushing the branch and relaunching the exact same
`L40S` smoke config should clear startup and return the pipeline to a real
timing run. No full 4K arm should launch until that live smoke reaches a
meaningful terminal state.
SETUP: Committed the repaired ladder state and pushed branch
`swarmy/humanwrite-next-cycle` to origin at commit `c6af24b`. Relaunched the
same config `configs/m2/m2_scale_ladder_4b_4096_timing_smoke_l40s_v1.yaml`
through the sanctioned wrapper. Validated the new launch status for
`dftr-1784435737-0a9af38e` and refreshed the sanctioned budget snapshot.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Publish repair commit | PASS | Git push advanced `origin/swarmy/humanwrite-next-cycle` from `fcf2602` to `c6af24b`, making the requested checkout SHA remotely resolvable. |
| Relaunch acceptance | PASS | Wrapper accepted the same `L40S` smoke comparison and created run `dftr-1784435737-0a9af38e` with `config_hash=1c520241d8a4c6c7653a37ac82daee33a295a463bd231ace467b7927042bf3fa`, `git_sha=c6af24b4cf4cccb47f82e4f5f09638e36de87193`, and `reserved_cost_usd=$0.78048`. |
| Immediate live status | PASS | At `2026-07-19T04:36:14Z`, sanctioned status still reported `status=running`, `gpu=L40S`, `workflow_step=train_lower_variance`, and no terminal transition yet. |
| Budget headroom after relaunch | PASS | Sanctioned budget reported Modal committed `$20.107291/$100` and OpenRouter spend `$28.483027/$100`, keeping the bounded cycle safely inside both caps. |
DECISION: Keep the 4K ladder active and monitor only
`dftr-1784435737-0a9af38e`. The fresh relaunch has cleared the known wrapper
transport failures, so the next continuation should validate its genuine
terminal outcome before any matched full-arm launch.
NEXT: Wait for terminal transition of `dftr-1784435737-0a9af38e`. Launch the
matched 4K SFT and MMD-witness full arms only if this smoke completes without a
contract/runtime failure and produces a real timing result.

## [2026-07-19] M2 / M2-scale-ladder-4b-4096-v1-launch
HYPOTHESIS: If the repaired `L40S` timing smoke completes with a real
step-64 checkpoint artifact, intact matched-exposure contract, and bounded
cost, then the next safe work is to open the preregistered matched full 4K SFT
and MMD-witness runs under the same exact 4,096 normalized briefs and frozen
128-token witness contract. This remains a training-only advance; any claim
about “more human” behavior still requires independent held-out evaluation in a
non-training representation.
SETUP: Read sanctioned status, logs, and budget for
`dftr-1784435737-0a9af38e` via the fixed gateway URL plus macOS Keychain token
`humanwrite-gateway-token`. The completed terminal state exposed
`metrics_ptr=modal-volume://humanwrite-checkpoints/runs/dftr-1784435737-0a9af38e/run_manifest.json`
with `run_manifest_sha256=9be7ffd10f0f9a10ee62adcb02b0c78bac8168190f2c15e5a64df04c774186f8`,
`return_code=0`, `accel_seconds=219.304`, and `actual_cost_usd=$0.142635`.
Sanctioned worker logs ended with a completed
`dftr.m2.lower_variance_confirmation_result.v2` payload showing
`executed_arm=MMD_WITNESS`, `steps=64`, `optimizer_examples=128`,
`teacher_forced_completion_tokens=15837`, and a saved step-64 adapter under
`/__modal/volumes/vo-EY2fT0CaoNDuXGLZLNZcGg/runs/dftr-1784435737-0a9af38e/MMD_WITNESS`.
Locally appended the missing ledger terminal update for that smoke, validated
`configs/m2/m2_scale_ladder_4b_4096_sft_v1.yaml`,
`configs/m2/m2_scale_ladder_4b_4096_mmd_witness_v1.yaml`, and
`configs/m2/m2_scale_ladder_4b_4096_timing_smoke_l40s_v1.yaml` with
`validate_lower_variance_config`, preregistered
`M2-scale-ladder-4b-4096-v1`, and launched the matched full SFT/MMD runs
through the sanctioned wrapper.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| L40S timing gate terminal validity | PASS | Sanctioned status showed `dftr-1784435737-0a9af38e` completed on Sunday, July 19, 2026 with `return_code=0`, `accel_seconds=219.304`, `tokens=0`, and `actual_cost_usd=$0.142635`, so the smoke became a real timing result rather than another startup-class failure. |
| Smoke artifact integrity | PASS | Sanctioned logs ended with a completed `dftr.m2.lower_variance_confirmation_result.v2` record carrying the exact 4K anchor SHA `723ebf559a4139c49454f5898a0e51120cdf424bd3cd12e39466c6758d25217b`, witness SHA `99d8b0ae93fb29ca7d692edd6c025e4310fcd0bf84188c37827e89d8a00eedae`, matched-exposure contract SHA `b09f69d1e090f994775e48149983084224a1f0fd3218394207f97a83a41246b6`, and a saved step-64 adapter manifest. |
| Missed handoff repair | PASS | The local ledger now contains the completed `run_update` row for `dftr-1784435737-0a9af38e` before any full-arm launch, closing the stale-monitor gap that triggered this continuation. |
| Full 4K comparison preregistration | PASS | Added preregistration `M2-scale-ladder-4b-4096-v1` with matched SFT/MMD-witness arms only, preserving the frozen two-arm contract and keeping held-out evaluation out of this launch turn. |
| Full-run local config validation | PASS | `validate_lower_variance_config` accepted the exact 4K SFT, MMD-witness, and smoke configs unchanged immediately before launch. |
| Next safe async launch | PASS | Sanctioned wrapper accepted SFT run `dftr-1784436708-b6637838` with `config_hash=11ff3703772708db0a511246ee931384f501f386d6a9826a9b935ecb6a0c586b` and MMD-witness run `dftr-1784436736-c94a4507` with `config_hash=db7f457c2cc30f9705c54568ba099d041ce8470fb0f76d7f35786be0593a305a`; both launched on `L40S` under `budget_class=screen` with `reserved_cost_usd=$4.68288` each and immediate `status=running`. |
| Budget headroom after full launches | PASS | Post-launch sanctioned budget reported Modal committed `$28.835206/$100` and OpenRouter spend `$28.483027/$100`, leaving the authorized 4K/16K ladder comfortably inside both caps and well short of the disallowed 46K cell. |
| Quality-gate discipline | PASS | No held-out candidate evaluation, sealed submission, or Tier 3 detector use was opened in this turn; this was strictly the gated training handoff after the smoke passed. |
DECISION: Keep `M2-scale-ladder-4b-4096-v1` active. The repaired `L40S`
timing gate passed with bounded cost and real checkpoint artifacts, so opening
the matched full 4K SFT and MMD-witness arms is justified within the existing
authorization. There is still no scientific answer about human-likeness until
these runs finish and are scored on the prospective held-out screen.
NEXT: Wait for terminal transitions of `dftr-1784436708-b6637838` and
`dftr-1784436736-c94a4507`. On completion, validate both checkpoint/run
manifests and launch the next prospective held-out sampling/evaluation step
only if the terminal artifacts are sound and the hard validity gates remain
intact.

## [2026-07-19] M2 / M2-scale-ladder-4b-4096-generation-v1
HYPOTHESIS: If the matched 4K SFT and MMD-witness confirmation runs completed
cleanly, then the next scientifically valid step is prospective held-out
generation on the frozen 128-prompt scale-dev panel. That step should remain
fully sanctioned and byte-bound only if the gateway exposes completed
checkpoint identity rather than forcing ad hoc operator reconstruction.
SETUP: Read sanctioned status for completed training runs
`dftr-1784436708-b6637838` and `dftr-1784436736-c94a4507`, completed prompt
brief run `dftr-1784377434-a4c6dae1`, and current budget from the fixed
gateway URL plus macOS Keychain token `humanwrite-gateway-token`. The audit
verified terminal artifacts and found the real pipeline gap: status exposed the
top-level run manifest but not the lower-variance checkpoint identity needed
for a sanctioned held-out generation launch. Repaired that surface by adding
checkpoint manifest and adapter SHA handoff metadata, added the frozen
`dftr.m2.lower_variance_generation.v1` worker/policy path plus focused tests,
redeployed the gateway, materialized
`configs/m2/m2_scale_ladder_4b_4096_sft_generation_v1.yaml` and
`configs/m2/m2_scale_ladder_4b_4096_mmd_witness_generation_v1.yaml` from live
status payloads, committed/pushed bridge commit `f09dc19`, preregistered
comparison `M2-scale-ladder-4b-4096-generation-v1`, and launched both held-out
generation arms through the sanctioned wrapper only after the pushed SHA was on
origin.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| 4K SFT terminal artifact validity | PASS | Sanctioned status confirmed `dftr-1784436708-b6637838` completed on Sunday, July 19, 2026 at `2026-07-19T05:15:25.305529Z` with `return_code=0`, `actual_cost_usd=$0.906923`, run-manifest SHA `0c544452ccf382aa042903080cf97d5af9dc27d65250147ab414aa38196c2702`, checkpoint manifest SHA `863e7d731858d997e0519b2f032461b4e7c24dc565f2358ca8302a58882ab337`, and adapter SHA `e71c14bc1ac7bd49c266a8ec489004ef3173a8d7f08c9b8ce9d4875f45335c1f`. |
| 4K MMD-witness terminal artifact validity | PASS | Sanctioned status confirmed `dftr-1784436736-c94a4507` completed on Sunday, July 19, 2026 at `2026-07-19T05:22:22.589968Z` with `return_code=0`, `actual_cost_usd=$1.166656`, run-manifest SHA `434d0f5400d2608f2c8f5e64789f17cd4567d292e07d0b76502a5acee0f438ae`, checkpoint manifest SHA `7b12529a9ef92a8ffb1d8d76071114d2eaeec95a03f9a455873b79574fc7608b`, and adapter SHA `7e071cd770b4e328162318144523f1ccbb7d735bb3b57e72804859f2999919d6`. |
| Missed handoff diagnosis | PASS | The blocked step was real and recoverable: before this patch, completed lower-variance status did not expose checkpoint identity for deterministic held-out generation, leaving the pipeline healthy in compute but incomplete in orchestration. |
| Sanctioned handoff repair | PASS | Live gateway status now surfaces `arm_checkpoint_dir_path`, `arm_checkpoint_manifest_sha256`, `arm_adapter_model_sha256`, and `arm_method_contract_sha256`, which is sufficient to materialize frozen held-out generation configs without manual checkpoint reconstruction. |
| Frozen held-out generation contract | PASS | Commit `f09dc19` added the frozen `generate_lower_variance` worker/policy path with exact 128-token raw-policy categorical sampling over the frozen 128-prompt panel, plus focused tests; direct verification passed `23/23` targeted tests in the managed harness environment. A broader focused suite also hit one unrelated pre-existing failure in `experiments/tests/test_m2_generate_dft.py::test_receipt_signing_secret_is_isolated_from_training_worker` against unchanged `modal_app.py`. |
| Prospective panel preservation | PASS | The held-out generation configs bind only the completed 4K checkpoints and prompt-brief artifact `modal-volume://humanwrite-checkpoints/data/m2-scale-ladder-v1/scale-dev-panels/prompt_briefs-128.jsonl` with SHA `24d55ae6695930b07030fbe6c4d88b47cb24fee2bfa217d42968ce9703f278f0`; no non-training scoring, sealed evaluation, or Tier 3 detector usage was opened in this turn. |
| Next safe async launch | PASS | Sanctioned wrapper accepted SFT-generation run `dftr-1784439725-1082f9d4` and MMD-witness-generation run `dftr-1784439725-6b6ec394`, both on `L40S` under `budget_class=screen`, each reserving `$4.68288`, with immediate `status=running` and `workflow_step=generate_lower_variance`. |
| Budget headroom after held-out launch | PASS | Post-launch sanctioned budget reported Modal committed `$30.908785/$100` and OpenRouter spend `$28.483027/$100`, leaving the authorized 4K/16K ladder within both hard caps and still short of the unapproved 46K cell. |
DECISION: Keep `M2-scale-ladder-4b-4096-generation-v1` active. The correct
next step was not more training or repeated polling; it was repairing the
missing sanctioned checkpoint-to-generation bridge, validating the completed
4K artifacts, and launching the prospective held-out generation pair. There is
still no scientific answer about human-likeness until these generation runs
complete and are scored on a non-training surface.
NEXT: Wait for terminal transitions of `dftr-1784439725-1082f9d4` and
`dftr-1784439725-6b6ec394`. On completion, validate both generation manifests
and only then open the next prospective non-training scoring step.

## [2026-07-19] M2 / M2-scale-ladder-4b-4096-generation-v1-relaunch
HYPOTHESIS: The failed held-out generation pair should be recoverable without
changing the scientific comparison if the terminal failures were startup-class
orchestration issues rather than panel, checkpoint, or sampling-contract
violations. In that case the correct action is to repair the fixed worker
handoff, preserve the same prospective 128-prompt panel and completed 4K
checkpoints, and relaunch the same preregistered comparison.
SETUP: Read sanctioned status and worker logs for failed SFT-generation run
`dftr-1784439725-1082f9d4` and failed MMD-witness-generation run
`dftr-1784439725-6b6ec394` using the fixed gateway URL plus macOS Keychain
token `humanwrite-gateway-token`. Recorded both failed terminal states in the
local append-only ledger. The SFT-generation worker failed after startup
because `generate_lower_variance()` still required an empty wrapper checkpoint
directory even though the sanctioned wrapper now precreates `worker.log`; the
MMD-witness generation run failed separately during wrapper `git clone`
startup with exit status `128`. Repaired the worker by allowing only the
wrapper-owned `worker.log` in the checkpoint directory, added a focused
regression test, verified `PYTHONPATH=. uv run --project harness pytest
experiments/tests/test_m2_generate_lower_variance.py -q` passed `13/13`,
committed and pushed fix `87a0681`, then relaunched the same two held-out
generation configs under the still-open comparison
`M2-scale-ladder-4b-4096-generation-v1`.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Failed generation diagnosis | PASS | Sanctioned status/logs showed `dftr-1784439725-1082f9d4` failed after `260.644` accelerator-seconds and `$0.169523` with `GenerationConfigError: generate_lower_variance requires an empty wrapper checkpoint directory`, while `dftr-1784439725-6b6ec394` failed after `157.356` accelerator-seconds and `$0.102344` during wrapper `git clone` startup. |
| Scientific artifact preservation | PASS | Neither failure implicated the frozen 128-prompt panel, the completed 4K checkpoints, or the exact held-out sampling contract; this was orchestration/runtime failure, not evidence about human-likeness. |
| Worker-log handoff repair | PASS | Commit `87a0681` now allows the sanctioned wrapper-owned `worker.log` and still fails closed on any other preexisting artifact in the generation output directory. |
| Focused regression coverage | PASS | `experiments/tests/test_m2_generate_lower_variance.py` passed `13/13`, including the new worker-log startup regression. |
| Relaunch acceptance | PASS | Sanctioned wrapper accepted SFT-generation relaunch `dftr-1784440682-c648bc57` and MMD-witness-generation relaunch `dftr-1784440682-36bfe848`, both under the unchanged comparison `M2-scale-ladder-4b-4096-generation-v1`, on `L40S`, with `reserved_cost_usd=$4.68288` each. |
| Immediate live status | PASS | Immediate sanctioned status shows both relaunched runs in `running` state under `workflow_step=generate_lower_variance` at pushed SHA `87a06818e1eada1dad01ab3d32352fefd93d978e`. |
| Budget headroom after relaunch | PASS | Sanctioned budget reported Modal committed `$31.180652/$100` and OpenRouter spend `$28.483027/$100`, preserving the bounded 4K/16K authorization and remaining well short of the unapproved 46K cell. |
| Quality-gate discipline | PASS | No held-out scoring, sealed submission, or Tier 3 detector use was opened in this repair turn. |
DECISION: Keep `M2-scale-ladder-4b-4096-generation-v1` active with the fresh
relaunch IDs only. The right response to this terminal transition was to
repair the startup regression, record the failed runs as negative operational
evidence, and relaunch the same prospective held-out generation step without
weakening any panel or quality gate.
NEXT: Wait for terminal transitions of `dftr-1784440682-c648bc57` and
`dftr-1784440682-36bfe848`. On completion, validate both generation manifests
and only then open prospective non-training scoring.

## [2026-07-19] M2 / M2-scale-ladder-4b-4096-generation-v1-validation
HYPOTHESIS: If the relaunched held-out 4K SFT and MMD-witness generation runs
completed cleanly, then the correct next action is to validate their exact
output bytes and restore any missing sanctioned handoff metadata required for a
future non-training scoring step. That validation should not open scoring
itself unless a frozen scale-ladder scoring protocol already exists for this
exact 128-prompt panel.
SETUP: Scheduled terminal-transition audit on Sunday, July 19, 2026 using the
fixed gateway URL plus macOS Keychain token `humanwrite-gateway-token`.
Sanctioned status confirmed SFT-generation run `dftr-1784440682-c648bc57`
completed at `2026-07-19T06:03:26.731823Z` and MMD-witness-generation run
`dftr-1784440682-36bfe848` completed at `2026-07-19T06:04:08.622314Z`, both
under `workflow_step=generate_lower_variance`. The audit then identified a real
handoff gap in the deployed gateway: completed `generate_lower_variance` runs
still surfaced only `metrics_ptr` and `run_manifest_sha256`, not the scorer-
ready `output_uri`/`output_sha256` or signed wrapper receipt already available
for the older `generate_dft` path. Repaired that gap in
`infra/backend/modal_app.py` and `infra/backend/volume_paths.py`, added focused
tests in `infra/tests/test_volume_paths.py` and
`experiments/tests/test_m2_generate_lower_variance.py`, verified
`PYTHONPATH=. uv run --project infra pytest infra/tests/test_volume_paths.py -q`
passed `13/13` and
`PYTHONPATH=. uv run --project harness pytest experiments/tests/test_m2_generate_lower_variance.py -q`
passed `14/14`, then redeployed the Modal gateway. After deployment, sanctioned
status for the same completed run IDs surfaced `output_uri`, `output_sha256`,
`wrapper_receipt_path`, and `wrapper_receipt_sha256`. The exact output,
manifest, and wrapper-receipt files were then downloaded from the sanctioned
`humanwrite-checkpoints` volume into `.operator/scale_ladder/4k_generation_v1/`
and SHA-verified locally. The frozen scale-dev panel bundle and prompt-source
artifacts were also materialized locally from
`data/m2-scale-ladder-v1/scale-dev-panels/`.
RESULTS:
| item | status | notes |
| --- | --- | --- |
| Relaunched SFT-generation completion | PASS | Sanctioned status showed `dftr-1784440682-c648bc57` completed after `314.483` accelerator-seconds for `$0.20454`, with `tokens=16384`, manifest SHA `9c8d70a2d6822177254ec865746e9691987df1891c5def5bccf3a0de88af0588`, output SHA `bf91a2418b681e892885f15f5cd05030534cab8c373e66fb1de89f7c8535bdac`, and wrapper receipt SHA `460936b2e5ffdf0e8f2e160676fcee4cfda2503fd8d4a9841bf45ed6b7dd0d3a`. |
| Relaunched MMD-witness-generation completion | PASS | Sanctioned status showed `dftr-1784440682-36bfe848` completed after `358.666` accelerator-seconds for `$0.233277`, with `tokens=16384`, manifest SHA `d8a1f9ba3bdc84ec872917f5055732f2d8a1ec3e16ed94641ade16213f594ad0`, output SHA `abedf64c30740b16d28aebaf8ad57d3241b9f91ff96d3c61f8d6e90257588efb`, and wrapper receipt SHA `aab40754cae57609b9b65b5adc11435aa08fd783269113d1342d803e1d9ff666`. |
| Deployed handoff repair | PASS | The production gateway on Sunday, July 19, 2026 now treats `generate_lower_variance` like the existing generation path for wrapper-receipt finalization and sanctioned output surfacing, exposing `output_uri`, `output_sha256`, and `wrapper_receipt_sha256` for already-completed runs. |
| Local byte validation | PASS | Downloaded local copies of both completed outputs, run manifests, and wrapper receipts matched the sanctioned SHA-256 values exactly under `.operator/scale_ladder/4k_generation_v1/`. |
| Frozen panel recovery | PASS | The completed scale-dev freeze run still binds `panel_bundle_sha256=658b769d5de40fb70e01912ec5c936faba8dbf6f0805056f54ec5bd1151798c8` and `prompt_sources_sha256=2eba6d6e18131e5571ba19b7c7c9cd7f6e5643d60e08df26ad32a252e9a8f569`; those artifacts were materialized locally under `.operator/scale_ladder/scale_dev_panels/` for the next scoring step. |
| Immediate budget state | PASS | Post-validation sanctioned budget on Sunday, July 19, 2026 reported Modal committed `$22.252709/$100` and OpenRouter spend `$28.483027/$100`, leaving the authorized 4K/16K ladder well inside both hard caps and still short of the unapproved 46K cell. |
| Next-step readiness | BLOCKED | There is still no checked-in, frozen non-training scoring protocol or config bound specifically to the scale-ladder `scale-dev-panels` bundle and these exact 4K generation artifacts, so opening scoring in this turn would require inventing a new evaluation route rather than executing an approved one. |
DECISION: Keep the scientific M2 ladder alive, but park this terminal-driven
autonomy cycle. The 4K held-out generation pair is now complete and locally
validated, and the missing sanctioned handoff metadata for this workflow has
been repaired and deployed. The next work is a fresh local implementation and
preregistration step for scale-ladder non-training scoring, not more polling or
another remote launch.
NEXT: When work resumes, bind a frozen scale-ladder scoring protocol to
`data/m2-scale-ladder-v1/scale-dev-panels/panel_bundle.json`, the locally
validated SFT/MMD output artifacts, and independent non-training embeddings
before opening any scoring or the authorized 16K follow-up.

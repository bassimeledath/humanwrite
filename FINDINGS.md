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

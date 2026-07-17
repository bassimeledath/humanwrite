# Post-v3 route to the first genuine 4B DFT result

Date: 2026-07-17 (America/Los_Angeles)

Scope: public repository evidence only. This review did not inspect hidden
data, private evaluator source, secrets, or remote checkpoint contents. It did
not preregister, deploy, launch, submit, or spend.

## Executive decision

Fidelity v3 failed a deliberately stronger claim than the DFT program needs.
It showed that the original bfloat16 merged SFT checkpoint is **not an exact
behavioral substitute** for the seed-29 adapter. That breaks the inference
from the historical sealed merged-checkpoint score to the visible adapter, but
it does not test DFT and does not block training or evaluating adapters
natively.

The shortest valid route is therefore:

1. finish the operator-owned measurement-v2 materialization and end-to-end
   scorer;
2. implement one unmistakably distributional 64-token objective at 4B;
3. compare the resulting adapter natively with an exposure-matched zero-objective
   adapter under the frozen public v2 protocol; and
4. qualify deployment only after a public treatment effect exists.

Exact adapter/merge string equality should remain a useful diagnostic, not a
universal prerequisite for DFT research. If sealed deployment requires a
merged model, the merged bytes must be treated as a distinct deployment
candidate and visibly evaluated as such. The preferred alternative is a
versioned sealed loader that evaluates the immutable base plus hash-bound PEFT
adapter directly.

## What fidelity v3 established

The completed result in `experiments/m2/fidelity_v3_result.json` is internally
specific:

- adapter and merged prompt token IDs and attention masks matched exactly;
- greedy output matched on 13 of 16 visible prompts;
- three prompt fingerprints differed;
- teacher-forced top-1 agreement was `0.9990234375` (1023/1024 reference
  positions);
- mean absolute logit difference was `0.04251855`, versus the diagnostic
  threshold `0.002`;
- maximum absolute logit difference was `1.1015625`, versus `0.05`; and
- archive reproduction and prospective sampled pairs were not attempted after
  the deterministic gate failed.

The failure is therefore not explained by the four merged-only tokenizer
configuration fields: v3 loaded both tokenizers independently and proved exact
prompt tokenization before diagnostics. It is a real adapter-versus-bfloat16-
merge numerical/behavioral difference. It also is not evidence about any DFT
arm; the tested checkpoint is SFT.

### What is blocked and what is not

| Claim or action | Consequence of v3 |
| --- | --- |
| Treat the historical merged score as the exact seed-29 adapter score | Blocked |
| Reuse the original bfloat16 merge as if it were lossless | Blocked |
| Deploy that same merged SFT checkpoint again | No value; its sealed rejection already stands |
| Train or publicly evaluate a DFT adapter | Not blocked |
| Compare DFT and control adapters under the same public protocol | Not blocked |
| Require exact merge string equality before learning whether DFT works | Overblocking |
| Submit a future merged DFT artifact without scoring those exact bytes publicly | Blocked |

## What the prior sealed SFT result still means

The aggregate in `experiments/m2/sealed_4b_seed29_v1.json` remains a valid
historical rejection of one exact deployed artifact: checkpoint hash
`0f437f62bc1cca0c`. It scored semantic MMD `0.059529`, delta versus floor
`+0.029529`, and authorship AUC `0.726914` with 95% interval
`[0.626166, 0.828487]`; both frozen absolute gates rejected.

V3 narrows, rather than erases, that evidence:

- **Still informative:** the merged artifact was loadable and measurably poor
  under that hidden window and deployment contract. It should not be
  resubmitted, and it remains a useful negative deployment benchmark.
- **Not established:** the native seed-29 SFT adapter would have received the
  same hidden score. The merge changed behavior materially enough that this
  counterfactual is unknown.
- **Not a DFT conclusion:** no arm A-E was trained or scored. The result cannot
  show that distribution fine-tuning fails, succeeds, or needs a larger model.
- **Not a future matched control:** its visible measurement protocol is
  quarantined by measurement v2, its artifact form is now known to differ from
  the adapter, and a future DFT treatment needs a same-design current SFT
  control. The historical score may be reported as context only.
- **No tuning oracle:** the hidden numbers must not be used to choose the
  training embedder, objective coefficient, visible panel, merge method, or
  stopping threshold.

## Shortest genuine 4B DFT experiment

### Scientific question

On the current frozen 256-record full-brief training corpus, does adding a
direct distribution objective improve held-out human-distribution similarity
relative to an otherwise identical continuation from the same SFT adapter?

This isolates the objective before paying for a larger corpus. Data scale is a
high-value later arm, but it is not logically required to produce the first
causal 4B DFT result.

### Treatment and control

Use Qwen3-4B at frozen revision
`1cfa9a7208912126459214e8b04321603b3df60c`, starting from one existing SFT
adapter chosen by a fixed rule (seed 11 is the simplest non-adaptive choice).
Create two new adapters with identical initialization, rollout prompts, token
budget, batches, optimizer steps, RNG schedule, KL reference, CE/SFT anchor,
and LoRA capacity:

- **A64:** direct score-function MMD at 64 generated tokens, using an unbiased
  batch estimator, leave-one-out control variate, KL-to-starting-SFT penalty,
  and CE/SFT anchor;
- **A0 control:** the same continuation and rollout exposure with the MMD
  coefficient exactly zero.

The treatment is genuinely distributional because a nonzero corpus-level MMD
term contributes to the policy gradient. Merely selecting high-scoring samples
and running ordinary SFT would be a useful arm, but it would not be the
shortest unambiguous test of the stated reconstruction hypothesis.

The training representation must be public and frozen but different from the
measurement-v2 embedder. Human targets come only from the training split, never
from measurement panels or hidden data. Log per-step reward components,
leave-one-out baseline, gradient variance, effective rollout count, KL drift,
CE loss, generated tokens, and collapse sentinels.

### Precompute tests and stop rules

Before the 4B screen, require:

1. an explicit-loop small-batch MMD reward/gradient oracle;
2. a test that each sample's reward excludes its own baseline contribution;
3. zero-MMD coefficient equivalence to the A0 control objective;
4. deterministic rollout and resume tests;
5. no evaluation-v2 or sealed representation imported into training; and
6. a short 0.6B/1.7B plumbing smoke that produces finite gradients and a
   resumable adapter, carrying no scientific claim.

Stop A64 immediately on non-finite loss/gradients, effective sample collapse,
KL breach, persistent gradient-variance ratio above 4x the A0 reference,
repetition/degeneracy, or failure to preserve full-brief factual gates. Do not
repair coefficients after inspecting the locked v2 endpoint; a new coefficient
is a new preregistered arm.

### Public evaluation

Generate A64 and A0 adapters directly, not merged models, under one canonical
full-brief generation contract. Measurement v2 must use identical prompt IDs,
document count, 64-token cap, decoding, training/sampling seed grid, and human
panels. Candidate and control rows are aligned by prompt before scoring.

The first result is valid only if the prospective power artifact says the
chosen `n` can resolve the preregistered minimally important effects. Promotion
is the intersection of:

- a favorable paired treatment-minus-control effect on the frozen primary
  distribution endpoints;
- no reliance on the representation used for training;
- factuality, brief-adherence, validity, and collapse gate passes; and
- a trusted signed v2 report with exact candidate/control output bindings.

If power fails, the result is `inconclusive`, not negative. If A64 is stable
but has no favorable paired effect, stop this objective before seeds 29/47 or
128 tokens. If it passes, confirm at seeds 29/47, then advance the same frozen
method to 128 tokens.

## Deployment after a public DFT effect

### Preferred: sealed adapter-native loading

A new private evaluator version should accept a public artifact contract
containing:

- immutable base model ID and revision;
- base serialization identity;
- PEFT adapter directory URI and complete signed file map;
- adapter model SHA-256 and adapter-config semantics;
- tokenizer directory identity;
- exact generation-contract hash; and
- combined deployment identity computed from all of the above.

The loader should instantiate the base in the same dtype used for native public
evaluation, attach the adapter without `merge_and_unload`, run a public canary
load/tokenization test, and preserve the evaluator's aggregate-only hidden
wall. This removes merge rounding as a scientific confound. It requires a new
versioned deployment contract and private tests, not a silent change to the
historical evaluator.

For a causal sealed statement, submit matched SFT and DFT adapters under the
same evaluator revision/window and artifact form. If quota permits only one
absolute candidate, a confirm can gate safety but cannot estimate a DFT
treatment effect.

### Fallback: merge-required deployment

If the serving stack requires merged weights, do not demand byte-identical or
string-identical behavior. A floating-point merge can be a distinct but valid
deployment implementation. Instead:

1. freeze one deterministic merge algorithm, load dtype, accumulation dtype,
   output dtype, PEFT/Transformers versions, and complete source/output maps;
2. merge both matched SFT and DFT adapters through the same path;
3. prove exact tokenizer/prompt ID/mask identity;
4. report teacher-forced numeric diagnostics as integrity sentinels, with
   thresholds frozen from a control merge before DFT results are opened;
5. generate and score the **actual merged bytes** under measurement v2;
6. require preregistered paired non-inferiority of merged versus native adapter
   on public primary effects and every hard gate; and
7. seal the exact hash that passed public evaluation.

This tests the property deployment actually needs: the merge preserves the
scientific decision and safety/adherence margins. It does not pretend finite-
precision arithmetic is exact. If SFT and DFT interact differently with the
merge, report that as a deployment failure rather than averaging it away.

## Measurement-v2 materialization inventory

The merged measurement-v2 code is a strong validator and metric library, not a
turnkey materializer or end-to-end scorer. Its final independent report passes
the implementation while explicitly leaving the checked-in candidate
unqualified. The current files show the remaining boundary precisely.

| Dependency | Current state | Required materialization |
| --- | --- | --- |
| Historical inventory/quarantine | Complete and verified | Preserve byte-for-byte |
| Human panel manifest/content | `unmaterialized`, zero IDs | At least `3n` unique eligible public humans, disjoint panels, per-row text/fingerprint, eligibility basis and attestation |
| Prompt panel/full brief | Empty IDs and placeholder text | `n` unique held-out prompt IDs, exact full-brief bytes, original prompt-matched human references, split/fingerprint bindings |
| Bandwidths | Empty | Human-floor-only embeddings, preprocessing/embedder hashes, frozen values and value hash |
| Power plan | `not_run` | Prospective null/alternative generators, minimally important effects, >=1000 trials/scenario, type-I/power/coverage/multiplicity results |
| Calibration | `unmaterialized` | Same-n human/control calibration, no v1 baseline transfer |
| Selection policy | Fixed-seed shape but null seed | Endpoint-independent fixed seed or all-seed rule, frozen before output scoring |
| Matched SFT baseline | Empty seed grid/output map | Exact prompt x training-seed x sampling-seed SFT output grid under the same generation contract |
| V2 scorer/report builder | Metrics and validators only | Operator-owned orchestration that embeds, scores, builds gate evidence, signs the candidate/control-bound report, and reproduces deterministically |
| Trusted keys | `{}` | Ed25519 public trust record in the artifact root; private signing key outside the repository and research-agent environment |
| Blind qualification | 13 groups `not_run` | Independent aggregate signed manifest bound to evaluator commit, dependency lock, protocol, runtime and fixture hash |
| Hard-gate evidence | No real artifacts | Four distinct exact-schema, byte-bound factuality/adherence/validity/collapse evidence files |
| Frozen protocol/attestation | Candidate only | Complete hashes, seeds, approval, trusted signature, inventory recheck and operator attestation |

Two practical facts keep this bounded:

- The pinned public FineWeb source previously yielded a 512-row eligible pool
  after scanning only 1,470 rows. It can likely supply disjoint public panels,
  but only 32 selected rows were retained, so the larger pool must be
  re-materialized under a new prospective selection config.
- The earlier 64-brief dev synthesis, including retries, cost `$0.190714`.
  A new 64-128 prompt panel is therefore small provider spend if the same
  contract is used, but its source slice and selection must be frozen first.

## Bounded autonomous sequence and cost

Costs are planning ranges extrapolated from recorded runs, not provider
quotes. Every paid step remains separately preregistered and capped.

| Stage | Autonomous deliverable | External cost estimate | Hard cap / stop |
| --- | --- | ---: | --- |
| 0. Operator engineering | Panel/prompt materializers, power runner, v2 scorer/report builder, signer interface, end-to-end synthetic fixture | `$0` | No paid action until full synthetic bundle validates |
| 1. Public source freeze | Re-materialize a buffered disjoint FineWeb pool; freeze 3 panels plus prompt/reference candidates | `$0-$0.10` CPU | Stop if eligibility/disjointness or deterministic reproduction fails |
| 2. Prompt brief synthesis | Synthesize and validate 64 prompts first; retain capacity to extend to 128 if power requires | `$0.20-$0.50` expected | Provider cap `$1`; no candidate generation yet |
| 3. Bandwidth/power/protocol | Embed human panels, freeze bandwidths, run >=1000-trial scenarios, choose n in `{64,96,128}`, sign and independently qualify protocol | `$0-$0.50` | Stop if required n >128 or blind group fails |
| 4. Matched A0 baseline | Generate adapter-native 64-token A0 rows on the exact v2 grid and produce four hard-gate artifacts | `$0.25-$0.75` | GPU cap `$1`; fail closed on any missing cell |
| 5. DFT plumbing | CPU gradient tests plus one small-model resumability smoke | `$0.05-$0.20` | GPU cap `$0.50`; no scientific interpretation |
| 6. First 4B A64 screen | Train A64 and exposure-matched A0 continuation, generate adapter-native outputs, produce signed v2 comparison | `$1.5-$3.5` | Combined GPU cap `$5`; stop on training or v2 gate |

Expected incremental cost to the first genuine, measurement-valid single-seed
4B DFT result is roughly **$2-$5**, with a conservative combined hard cap of
**$7**. If A64 passes and expands to three training seeds and 128-token
confirmation, the native postmortem's **$6-$12** whole-arm range remains the
more realistic total bound. No sealed or Tier-3 cost belongs in this phase.

The stages are dependency-ordered but engineering can overlap safely: the DFT
objective unit tests may proceed while the independent operator materializes
measurement inputs, provided neither lane opens candidate outputs before the
protocol is frozen. Signing-key custody and blind qualification remain separate
from the research-agent lane.

## Decision points

1. **Measurement power cannot qualify n<=128:** stop and redesign the visible
   study; do not substitute the old v1 reports.
2. **A64 training unstable:** stop the direct score-function path and retain a
   clean negative engineering result; teacher-forced moments or reward-weighted
   SFT becomes the cheaper next mechanism.
3. **Stable A64, no paired v2 improvement:** stop before more seeds or longer
   outputs. This is the first genuine negative DFT result.
4. **Single-seed v2 improvement:** confirm seeds 29/47 at 64 tokens, then 128;
   do not select the best seed by AUC/MMD.
5. **Three-seed/128-token public pass:** qualify adapter-native sealed loading,
   or qualify a matched merge deployment by non-inferiority on the actual
   merged bytes.
6. **Sealed matched DFT/control confirm:** only then consider 14B or human-
   triggered Tier 3.

## Actions that would overclaim or waste budget

- Re-running fidelity with looser exact-string thresholds before any DFT arm
  exists.
- Treating the old merged SFT sealed score as the native adapter baseline.
- Merging a future DFT adapter before learning whether its native treatment
  effect exists.
- Using the sealed MMD/AUC values to select visible panels, rewards, or merge
  thresholds.
- Calling a teacher-forced or reward-weighted SFT arm “DFT” without a frozen
  distribution-level training term.
- Generating candidate outputs before measurement-v2 panels, power, seed rule,
  and protocol signature are frozen.
- Expanding to 14B, 256+ tokens, GAIL, or Tier 3 before the bounded 4B A64/A0
  contrast passes.

## Recommended program status

> The selected 4B SFT merge failed both the historical sealed gates and exact
> adapter/merge fidelity. Those results reject that deployment artifact but do
> not test DFT. Measurement-v2 validation is ready for operator materialization.
> The next scientific milestone is a native-adapter, single-seed 4B
> score-function MMD treatment versus an exposure-matched zero-MMD control at
> 64 tokens, under a signed prospective v2 protocol. Deployment equivalence is
> a downstream non-inferiority problem, not a prerequisite for learning whether
> DFT works.

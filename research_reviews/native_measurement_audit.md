# Native audit: visible measurement repair contract

Date: 2026-07-16  
Branch audited: `swarmy/humanwrite-next-cycle`  
Scope: visible local artifacts and code only. No sealed examples, sealed implementation, paid calls, training, generation, or evaluator edits were used.

## Executive verdict

The visible Tier-1 evaluator is not reliable enough for another promotion decision in its current form. The strongest defects are statistical construction errors, not merely small confidence intervals:

1. `human_floor_mmd` bootstraps each fixed half with replacement. On the frozen 32-document human bank this reports a floor of `0.0563041` with interval `[0.0381321, 0.0898469]`; 1,000 deterministic disjoint 16-vs-16 partitions without replacement instead give mean `-0.0000268` and central partition-sensitivity interval `[-0.0039262, 0.0051193]`. The current resamples contain only 10.44 and 10.30 unique documents on average in nominal groups of 16. Duplicate rows appear as off-diagonal within-group pairs in the unbiased U-statistic and create the large positive floor.
2. The MMD candidate, floor, and baseline are not measured with a common design. Candidate cells use 16 generated documents versus all 32 humans; the floor uses nominal 16-vs-16 groups; the frozen baseline used two generated documents versus 32 humans. In addition, the median-heuristic kernel bandwidth is recomputed from each compared pair, so the quantities subtracted in `semantic_mmd_delta_vs_human_floor` are not even guaranteed to use the same kernel.
3. The quality judge compares a candidate response with an unrelated human-bank document while giving both the candidate prompt. All nine corrected 4B cells score a perfect `1.0`; this is evidence that the comparator is invalid for prompt-conditioned quality, not that the model beats humans.
4. At 16 documents, repeated-start rate moves only in increments of `1/16 = 0.0625`. Under the current upper threshold `0.15744`, 0, 1, or 2 positive documents pass and 3 fail. Wilson 95% intervals are `[0,0.19361]` for 0/16, `[0.01112,0.28329]` for 1/16, and `[0.03498,0.36023]` for 2/16. The point gate therefore makes a promotion decision at a resolution that cannot establish the calibrated rate.
5. Seed 29 was explicitly chosen for the smallest absolute deviation of visible mean AUC from 0.5. The nine corrected 4B per-cell AUC intervals have mean width `0.3393`, all reuse the same 32-human bank, and the three sampling cells for a checkpoint are not independent replications. Selecting the luckiest of three training seeds on this endpoint creates optimizer's-curse bias.

Historical reports remain valid records of what evaluator v1/v1.1 computed, but they must be marked non-comparable and ineligible for future promotion. They must not be rewritten.

## Verified defects and exact surfaces

| Severity | Defect | Evidence | Consequence |
| --- | --- | --- | --- |
| Critical | Replacement-biased human floor | `harness/src/harness/metrics/distribution.py:103-122` fixes one 16/16 split, then samples each half with replacement at lines 116-118. Local BGE reproduction is `0.0563041` versus approximately zero for disjoint partitions. | The reported negative candidate-vs-floor deltas near `-0.058` are mostly an artifact of an inflated floor. Negative unbiased MMD itself is ordinary estimator noise near zero, not “better than human.” |
| Critical | Data-dependent kernels differ across comparisons | `_mmd_from_embeddings` derives bandwidths from the pooled inputs at `distribution.py:65-78`; `evaluate` calls candidate-vs-human and human-floor separately at `harness/src/harness/cli.py:672-674`. | Candidate MMD and human-floor MMD are on comparison-specific kernel scales; subtracting them is not a controlled delta. |
| Critical | Candidate/floor/baseline cardinality mismatch | `cli.py:672-673` uses all loaded humans, currently 32. Bootstrap sample files contain 2 documents per cell; corrected 4B files contain 16. `harness/baseline_stats.json.sample_count=9` counts reports, not documents. | MMD variance, self-BLEU reference count, repetition resolution, histograms, lexical estimates, and AUC power differ. Standardized `S` is apples-to-oranges. In a direct check, the same baseline two-document cell's MMD across matched two-human panels had a 95% sensitivity range of about `[-0.0341,0.0228]`. |
| Critical | Prompt-unmatched quality comparator | `cli.py:664` cycles independent humans by row number; `cli.py:699` sends that unrelated text and the generated prompt to `quality_preference`. The human-bank contract proves distributional disjointness, not prompt linkage. | Preference/JMQ measures whether generated text answers its own prompt better than an unrelated web page. Perfect `1.0` scores are expected and non-informative. |
| High | Human self-BLEU calibration is at the wrong statistic level and cardinality | `cli.py:754-757` computes 32 per-document BLEUs, each against 31 references, then takes an order-statistic interval at lines 765-782. Candidate `validity.self_bleu` is one corpus mean over 16 documents, each against 15 references (`validity.py:119-128`). | The frozen human range `[0.03234,0.07693]` is not a sampling interval for the candidate statistic. Self-BLEU increases with the number of available references, so the gate is structurally cardinality-dependent. |
| High | Repetition point gate is under-resolved | `validity.py:164-186` produces a document incidence proportion; the corrected upper-only policy is applied at `validity.py:95-107`. Candidate n=16 permits only three passing values below 0.15744. | One document flips the gate, while every passing n=16 count still has a wide interval. The v2 upper-only change fixed the erroneous positive lower bound but not power. |
| High | Adaptive seed selection on noisy AUC | `configs/m2/m2_sealed_4b_seed29_v1.json:13-17` records selection by mean AUC proximity to 0.5. `quality.py:148-190` fits a new probe per cell. | The selected visible AUC is optimistically biased and expected to regress on a new bank. The sealed seed-29 failure remains a valid rejection of that artifact; it is not evidence that this selection rule was valid. |
| High | AUC interval omits probe-fitting and fold uncertainty | `quality.py:125-145` bootstraps already-computed OOF scores; it does not refit vectorizer/classifier or vary folds. Each candidate also gets a different fitted probe. | The interval is conditional on one feature vocabulary, fit, and split. Cross-arm AUC differences and seed rankings are more uncertain than reported. |
| High | Pseudoreplication across cells | Reports reuse the same prompt IDs and human bank across sampling and training seeds; downstream code averages reports and uses ordinary report-level standard errors in `experiments/m1/analysis.py:172+`. | Nine reports are not nine independent datasets. Uncertainty must cluster by prompt and model-training seed, with sampling seed nested inside checkpoint. |
| Medium | Baseline schema hides effective sample size | `experiments/m1/analysis.py:112-169` records only number of selected reports as `sample_count`; transfer at `cli.py:926+` preserves that ambiguity. | A baseline can pass provenance checks while using the wrong number of documents, prompts, humans, or seed structure. |
| Medium | Validity calibration mixes document and corpus statistics | Language/repetition calibration is built from per-document values while evaluation gates a corpus aggregate. Empty-outline unsupported claims also need a separately reported stratum. | Some gates can be overconfident or uninterpretable even when their point estimates are useful diagnostics. |

The current overlap-based outline and unsupported metrics remain useful continuity diagnostics, but they are not semantic entailment tests. This memo does not propose modifying them to resemble the sealed evaluator.

## Operator-owned visible measurement v2

### Ownership boundary

The research agent must not write the replacement evaluator or its acceptance tests. An operator-appointed measurement maintainer implements v2 in a private repository or protected branch. The research agent receives only:

- the public input/output contract;
- frozen protocol, bank, prompt-panel, and dependency hashes;
- aggregate v2 reports after the protocol is frozen; and
- pass/fail reasons that do not expose operator-only test fixtures.

The maintainer must not inspect sealed evaluator source, hidden examples, hidden per-item metrics, or tune v2 against sealed outcomes. V2 is a correction of visible statistical validity, not an imitation of Tier 2.

### Frozen design

For each comparison, choose a document count `n` before generating outputs. `n=16` is forbidden for promotion; the default minimum is `n=64`, and `n=128` is preferred. A prospective power artifact may require a larger n.

Create an operator-owned visible human pool of at least `3n` unique documents, disjoint by fingerprint and domain from every training, development, prompt, and prior Tier-1 record. Freeze three non-overlapping hash-selected panels:

- `H_eval`, n documents, used for generated-vs-human corpus metrics;
- `H_floor_a`, n documents; and
- `H_floor_b`, n documents.

All text preprocessing, truncation, length strata, embedder ID and immutable revision, embedding normalization flag, and panel hashes are frozen before candidates are scored. Human selection should be stratified on visible domain and length policy fixed independently of candidate outputs. It must not use sealed data or sealed scores.

### Corrected MMD contract

1. Compute embeddings once under the frozen visible embedder.
2. Estimate the base bandwidth only from `H_floor_a ∪ H_floor_b`, never from generated outputs. Freeze the exact bandwidth array and its byte hash. Candidate, control, and floor all use that array.
3. Compute unbiased MMD-squared for `G_n` versus `H_eval` and for `H_floor_a` versus `H_floor_b`, always n-versus-n with no replacement and no duplicate IDs.
4. Report negative unbiased estimates unchanged, with the explicit interpretation “sampling variation around zero.” Do not clamp them and do not call them super-human.
5. Report a 10,000-draw deterministic label-permutation p-value for the absolute generated-vs-human null using the fixed kernel. For candidate-versus-control claims, use prompt-clustered paired swaps between candidate and matched-control outputs, with both compared to the same `H_eval` panel.
6. A deterministic list of additional disjoint human partitions may be used to report panel sensitivity. It must be labeled a partition-sensitivity interval, not a bootstrap confidence interval. Replacement resampling is prohibited for the human-floor U-statistic.
7. `semantic_mmd_delta_vs_human_floor` may be emitted only when both terms share n, preprocessing, embedder, bandwidths, and disjoint panel provenance. The report must carry these hashes.

### Matched control and cardinality contract

Every treatment is evaluated beside a current SFT control on the same:

- prompt IDs and full-brief bytes;
- number of documents;
- output-token cap and decoding policy;
- training and sampling seed grid;
- human panel, metric code, and feature definitions; and
- checkpoint-selection rule.

Lexical L2, structural distance, self-BLEU, repetition, and AUC are computed at the same candidate/control/human cardinality. Old two-document bootstrap statistics cannot standardize new results. Prefer raw paired treatment-minus-control effects. If composite `S` is retained, v2 must derive its center and scale from the matched control under the same n and clustered seed design; it must not read `harness/baseline_stats.json`.

Every v2 report schema must expose at least `documents_per_cell`, `human_documents_per_panel`, prompt-panel hash, all human-panel hashes, training seeds, sampling seeds, effective prompt-cluster count, bandwidth hash, code commit, dependency-lock hash, and power-plan hash. `sample_count` alone is not accepted.

### Prompt-conditioned quality

The independent human distribution bank remains valid for corpus metrics but is banned from pairwise preference. Quality/JMQ is computed only when each generated row has a cryptographically bound prompt-matched human reference from the same held-out source record. The evaluator must verify one-to-one prompt ID, brief hash, reference fingerprint, and split provenance. If a matched reference is absent, quality is `not_measured`; modulo/cyclic pairing is forbidden.

Pair order stays deterministically randomized and the judge remains secondary. The report includes paired win/loss/tie counts and a prompt-clustered interval. No quality-judge result may override factual, distributional, or power failures.

### Repetition and self-BLEU

- Repetition remains an upper-tail harm metric; zero repetition is never a failure.
- Use a prespecified one-sided non-inferiority comparison against a same-n human panel, with an explicit margin and an exact/Newcombe interval for the rate difference. A point estimate alone cannot promote.
- If the planned n cannot distinguish the margin with at least 80% power, repetition is `underpowered`, which is a non-promotion result. At n=16 even 0/16 has Wilson upper bound 0.19361; this design is not decision-grade. N=64 is the minimum screen size, not an automatic guarantee of power.
- Calibrate self-BLEU as a corpus statistic on same-n human panels: each document has n-1 references and the panel mean is the unit. Do not compare candidate n=16 to per-document values computed with 31 human references.

### Authorship AUC and checkpoint selection

Use equal human/generated counts and a locked probe family. Uncertainty must refit the complete vectorizer-classifier pipeline under repeated, grouped cross-fitting; bootstrapping only fixed OOF scores is insufficient. Report symmetric separability `abs(AUC - 0.5)` alongside signed AUC and its interval. Prompt/domain clusters, fold seeds, and fit count are recorded.

Training seed is fixed before visible endpoint evaluation (for example, the first preregistered seed or a hash-selected seed), or all preregistered seeds are evaluated and the treatment conclusion is aggregated. If a single checkpoint must be exported, choose it by a prespecified endpoint-independent training statistic or deterministic seed rule. A manifest that selects by MMD, AUC, JMQ, lexical/structural score, or any visible promotion endpoint is invalid.

### Prospective power and uncertainty gate

Before candidate outputs are opened, the operator freezes a simulation-based power report using only visible human/control pilot data and prespecified minimally important effects. It must demonstrate:

- empirical type-I error no greater than 0.05 for MMD and paired treatment tests;
- at least 80% power for the declared MMD effect and AUC-separability improvement;
- at least 80% power for the repetition non-inferiority margin;
- interval coverage between 93% and 97% in blind synthetic trials; and
- the required n, number of prompt clusters, seed grid, and multiplicity rule.

Failure of any power requirement yields `inconclusive/underpowered`, never pass. Promotion is an intersection rule: both primary distribution endpoints and all hard safety/adherence gates must pass. Sampling-seed repeats do not substitute for more prompt clusters.

## Blind test contract

The operator-only test pack contains synthetic text/embedding fixtures and public visible IDs, not sealed data. An independent tester receives the candidate evaluator commit without implementation commentary and must pass all items below.

1. **Disjoint-floor sentinel:** every MMD floor draw has unique IDs within and across its two groups. A fixture modeled on the current 32-row bank must reject replacement resampling. On identical-distribution synthetic clouds, empirical mean is near zero and type-I error is within the frozen tolerance.
2. **Exact small-matrix oracle:** an independently written reference computes MMD by explicit loops for small matrices. Vectorized v2 must match to `1e-12`, including a negative unbiased result.
3. **Kernel-freeze test:** candidate changes cannot alter the bandwidth array. Candidate, matched control, and both floor panels expose the same bandwidth hash.
4. **Cardinality fail-closed test:** 16-vs-32, 2-vs-32, missing document counts, duplicate IDs, and candidate/control prompt-set mismatch are rejected before metrics run.
5. **Matched-control test:** changing prompt order is harmless after ID alignment; dropping, duplicating, or replacing one prompt fails. Training/sampling seed nesting is preserved in the report.
6. **Self-BLEU cardinality test:** human and candidate panels use exactly n documents and n-1 references per document. A fixture proves that the old 32-reference calibration cannot be applied to n=16.
7. **Repetition-resolution test:** n=16 returns `underpowered`, not pass. Zero events never fail for being below a lower bound. High repetition still fails. Power status and rate-difference interval are deterministic.
8. **Prompt-match test:** quality rejects an unrelated human, cyclic pairing, duplicate reference, wrong brief hash, and wrong split. A properly bound prompt/reference pair passes regardless of row order.
9. **AUC refit test:** an instrumented estimator proves the full pipeline is refit for every uncertainty replicate and varies the frozen fold seeds. A small-sample fixture must be labeled underpowered. The report distinguishes signed AUC from distance to 0.5.
10. **Selection-firewall test:** checkpoint manifests referencing AUC, MMD, JMQ, `S`, or other promotion metrics are rejected. Deterministic preselected seed and all-seed aggregation pass.
11. **Cluster/power test:** duplicated sampling seeds for the same prompts do not increase effective prompt n. Known-null simulations meet type-I and coverage limits; known alternatives meet the frozen power requirement.
12. **Historical immutability test:** the SHA-256 inventory of every existing calibration, baseline, report, index, and sealed aggregate remains byte-identical after v2 installation.
13. **No-sealed-imitation attestation:** dependency and source scans show no sealed evaluator code, hidden fixture, hidden prompt, private embedder identifier, or hidden per-item output entered the visible repair.

The blind tester returns only a signed test manifest containing evaluator commit, lock hash, fixture-pack hash, test names, pass/fail, runtime versions, and timestamp. The research agent does not receive fixture contents.

## Proposed implementation surfaces

No code was changed in this audit. The operator maintainer should expect a new versioned implementation rather than silent edits to v1:

- `harness/src/harness/metrics/distribution.py`: fixed-bandwidth MMD, disjoint panels, permutation inference, and explicit cardinality contract.
- `harness/src/harness/metrics/quality.py`: prompt-linked comparison and full-pipeline grouped AUC uncertainty.
- `harness/src/harness/metrics/validity.py`: same-n corpus self-BLEU calibration and uncertainty-aware repetition result.
- `harness/src/harness/cli.py`: `EvalReportV2`, panel/protocol hashes, matched-control inputs, power status, and fail-closed schema validation. Keep v1 command/report reader available only for reproduction.
- `experiments/m1/analysis.py`: raw paired treatment/control effects and clustered aggregation; no old-baseline z scoring.
- `harness/tests/test_distribution_v2.py`, `test_quality_v2.py`, `test_validity_v2.py`, and `test_cli_v2.py`: public invariant tests. Operator-only blind tests live outside the research checkout.
- New operator-frozen artifacts: `measurement_protocol_v2.json`, `human_panels_v2.manifest.json`, `bandwidths_v2.json`, `power_plan_v2.json`, `calibration_v2_measurement.json`, and `matched_sft_baseline_v2.json`. Names are illustrative; schemas and hashes are mandatory.

## Migration and attestation path

1. **Inventory:** operator hashes all current `harness/calibration*.json`, `harness/baseline_stats.json`, Tier-1 reports/indexes/summaries, selection configs, and sealed aggregate records. Commit the inventory before implementation.
2. **Quarantine semantics:** publish a metadata-only interpretation manifest listing v1 artifacts as `historical_reproducibility_only`, with reasons `replacement_biased_floor`, `cardinality_mismatch`, and where applicable `prompt_unmatched_quality`. Do not edit or delete the originals.
3. **Independent implementation:** measurement maintainer implements v2 outside the research agent's branch. A second operator reviews statistical formulas and code without sealed access.
4. **Blind qualification:** run the operator-only test pack. Freeze evaluator commit, dependency lock, public API schema, test attestation, and no-sealed-imitation attestation.
5. **Freeze visible inputs:** materialize and hash the >=3n human panels, matched prompt panel, bandwidth array, seed policy, preprocessing, and power plan before any candidate evaluation.
6. **Matched baseline:** generate or reuse a current SFT control only if its raw outputs exactly match v2 prompt IDs, n, sampler, token cap, and seed grid. Otherwise regenerate it. Never transfer the old two-document baseline.
7. **Shadow replay:** score preserved candidate/control outputs under v2 only when they satisfy the exact new contract. Label results `post_hoc_shadow`, never promotion evidence. This quantifies how conclusions changed without rewriting history.
8. **Prospective use:** preregister the first v2 treatment/control comparison, then open candidate outputs once. Research agents may optimize training, but cannot edit v2, see blind fixtures, choose human panels, or choose the best seed from endpoint results.
9. **Sealed boundary:** the prior seed-29 sealed rejection remains unchanged. V2 neither appeals nor reverse-engineers it. A future sealed submission requires a prospective visible-v2 pass and the existing independent sealed contract.

## Acceptance criteria before research resumes

Research may begin the next objective experiment only when all of the following are true:

- the historical hash inventory passes with zero changed bytes;
- an operator-owned v2 commit and dependency lock are frozen;
- all 13 blind-test groups pass and the signed attestation is present;
- no-sealed-imitation attestation passes;
- at least 3n disjoint visible humans and n prompt-matched references are frozen, with n justified by the power plan and n >= 64;
- candidate/control/floor all use equal n and identical fixed bandwidths;
- MMD null type-I, interval coverage, and target-effect power meet the stated bounds;
- repetition and AUC power meet at least 80%, or those metrics are explicitly non-promoting;
- matched quality linkage is proven or quality is omitted;
- checkpoint selection is endpoint-independent and frozen before scoring;
- a matched current SFT baseline exists under the exact v2 design; and
- report v2 exposes all sample-size, panel, code, seed, bandwidth, and power hashes and fails closed on any mismatch.

Until then, visible metrics may be used for debugging only. They should not authorize 14B, Tier 3, a sealed submission, or a claim that one objective improves human-likeness.

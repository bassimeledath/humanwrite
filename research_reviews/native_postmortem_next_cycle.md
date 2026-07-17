# Independent postmortem and next-cycle plan

Date: 2026-07-16
Scope: visible repository evidence only. No private evaluator source, hidden examples, hidden per-item scores, secrets, Modal hidden volumes, training, deployment, submission, provider calls, or scientific-artifact edits were used.

## Executive verdict

**Fact.** The project has built a credible, provenance-conscious experimental pipeline and a corrected Qwen3-4B **SFT** checkpoint that is strong on the visible structured-brief adherence proxies. Across three training seeds and three sampling seeds, the corrected SFT produced 144 documents; all nine cells passed the visible outline, unsupported-claim, and language gates, and eight of nine passed the revised no-collapse gate.

**Fact.** The only scored sealed submission was the selected seed-29 SFT checkpoint, not a distribution-fine-tuned arm. It failed the sealed semantic gate (`MMD=0.059529`, delta versus floor `+0.029529`) and the sealed authorship gate (`AUC=0.726914`, 95% CI `[0.626166, 0.828487]`).

**Inference.** The sealed result is decision-grade for one narrow action: reject that exact checkpoint under that evaluator window and keep 14B and Tier 3 closed. The AUC interval is far enough from 0.5 that plumbing noise is not a persuasive explanation for the rejection.

**Inference.** The sealed result is not decision-grade for the broader dashboard statement that "the 4B recipe does not generalize." Only one training seed was submitted, no matched hidden SFT control or DFT treatment was submitted in the same window, semantic uncertainty was not returned, and the visible checkpoint-selection statistic was noisy and reused for selection. The supported statement is: **the selected seed-29 4B SFT artifact failed the preregistered absolute sealed gates.**

**Fact.** No DFT arm A-E has yet produced scientific evidence. Under the milestone definitions in `CLAUDE.md`, M2 (64/128-token objective experiments) has not scientifically started. Calling the current state "M2 blocked" conflates a pre-M2 SFT promotion probe with a failed DFT experiment. A more accurate state is: **M1 SFT baseline established; its first sealed promotion probe rejected; M2 objective experiments unrun.**

## What actually succeeded

### Scientific successes

1. **Conditioning repair produced a reproducible adherence gain.** The first real-data pilot sent only `user_prompt` to the model; use case, style, detail mode, target length, em-dash policy, and outline were omitted. After prompt repair and `dft.full-brief.v1` rendering at both train and generation time, mean visible outline recall rose to about 0.79-0.81 and the non-empty-outline unsupported proxy fell to about 0.02-0.03. The three-seed 4B repetition is evidence that full-brief conditioning matters for this task.

2. **Moving from the six-document fixture to 256 real FineWeb documents reduced obvious collapse.** The initial tiny-data SFT had zero of nine no-collapse passes; the first 256-document pilot improved to four of nine before the conditioning repair. This is directional evidence that the plumbing fixture was scientifically underpowered.

3. **The 4B size screen beat the matched 1.7B screen on the preregistered directional decision.** With the repaired corpus and the same seed-11 evaluation, 4B passed two of three no-collapse cells versus one of three for 1.7B while both retained factual-proxy control. This justifies using 4B as the next-cycle bridge size. It does not prove a general 4B advantage outside this screen.

4. **The corrected 4B SFT is stable across training seeds on visible proxies.** Losses were close (`0.6972`, `0.7009`, `0.6960`), all factual/language gates passed in all nine cells, and eight no-collapse cells passed. Training-seed instability is therefore not the leading explanation for the visible result.

### Operational successes

1. Data selection, source/domain disjointness, fixed hashes, prompt-only repair validation, three-seed checkpoint identity, and aggregate-only sealed handling were recorded with unusually good provenance.
2. The calibration pipeline caught and preserved two real negative results: the inappropriate order-statistic interval for rare binary repetition, and cross-runtime drift in a derived floating-point constant.
3. Sealed pre-result failures were separated from the scored run, and the quota credit was append-only and limited to a run that returned no aggregate metrics.

These are infrastructure successes, not evidence that the model is human-like.

## What failed, with operational failures separated from results

| Event | Type | Correct interpretation |
| --- | --- | --- |
| M0 terminal accounting omission | Operational | Accounting defect; no scientific conclusion. |
| Hugging Face timeouts, resolver and mount-alias URI failures | Operational | Source-transport failures before data materialization; no model conclusion. |
| Provider JSON/quotation/content-filter failures | Operational/data-production | Contained by validation; useful reliability evidence, not model-quality evidence. |
| Six-document, 474-token SFT collapse | Scientific but deliberately low external validity | Valid negative result for the plumbing-scale artifact only. |
| 256-document pilot with only `user_prompt` conditioning | Experimental implementation failure plus observed result | It improved diversity but failed adherence. It should not be used to infer that real-data SFT intrinsically harms adherence because most structured inputs were absent. |
| Repetition lower bound that made zero repetition fail | Evaluator design failure | Real defect. The prospective upper-only correction did not change the final 4B 8/9 result. |
| Sealed URI/snapshot failures and 960-second sequential timeout | Operational | Returned no metrics and must not be counted as sealed scientific failures or replications. |
| Final sealed seed-29 rejection | Scientific result | Valid rejection of the exact selected SFT artifact under the hidden window; not a DFT result and not a treatment-effect estimate. |

## Visible-evaluator and reviewer audit

Several visible oddities materially limit the interpretation of the M1 pass.

### 1. The visible baseline and candidate were not measured at the same sample size

**Fact.** `harness/baseline_stats.json` was estimated from nine cells containing only two generated documents per cell. The corrected 4B candidate reports used sixteen generated documents per cell. The semantic MMD is an unbiased U-statistic whose variance depends strongly on sample count; self-BLEU, repetition incidence, structural histograms, and fresh-probe AUC also depend on corpus cardinality.

**Inference.** Standardizing the 16-document candidate with mean/SD from two-document cells makes `S` an apples-to-oranges statistic. The candidate's mean visible `S=-1.166` is directionally interesting, but its magnitude is not calibrated evidence of a 1.17-standard-deviation improvement. A next-cycle baseline must be regenerated on the exact same prompt set, number of outputs, lengths, and sampling seeds as every treatment.

### 2. Negative visible MMD was treated too optimistically

**Fact.** All nine corrected 4B MMD estimates were negative (mean about `-0.00171`), which is mathematically allowed for the unbiased MMD-squared estimator. The visible human-floor estimates were positive because their bootstrap resampled with replacement, adding duplicate-driven bias and using a different sampling construction.

**Inference.** A negative point estimate is evidence of estimator noise near zero, not evidence that generated text is "more human than human." The very favorable visible delta (about `-0.058`) should have triggered a power/estimator-compatibility audit before hidden submission.

### 3. Tier-1 humans were distribution references, not prompt-matched answers

**Fact.** The 32-document visible bank was selected by crawl, length, script, fingerprint, and domain rules. It was not matched to the 16 generated prompts. The harness cycles these unrelated humans as pairwise comparators while sending the generated prompt to the quality judge.

**Fact.** The resulting quality preference became `1.0` in all nine corrected 4B reports. The project correctly excluded it from the decision.

**Inference.** The perfect judge score is a reviewer red flag, not a quality result: an unrelated human web document is predictably a poor answer to a generated brief. The same bank can remain a corpus-level distribution reference, but it cannot support prompt-conditioned quality preference. A separate prompt-matched human target is required for that metric.

### 4. The validity gates were easy and proxy-sensitive

**Fact.** The frozen baseline was weak: recall `0.25` and unsupported rate `0.8542`, so the formal non-inferiority gates were only recall `>=0.20` and unsupported `<=0.9042`. The unsupported metric labels sentences through token overlap with supplied facts; it can reward close copying and is not an entailment or source-verification test. Empty-outline documents also require separate handling because no fact can support a nontrivial sentence under this proxy.

**Inference.** Passing nine of nine gates does not by itself establish factual quality. The observed corrected values (recall near 0.79 and non-empty-outline unsupported proxy near 0.03) are much stronger than the gate, but they remain surface-overlap measurements. Next-cycle controls should add a locked, independently implemented entailment/source-span audit on a sample while keeping the frozen proxy for continuity.

### 5. Seed 29 was selected on a noisy, reused statistic

**Fact.** Seed 29 was selected because its mean visible AUC was closest to 0.5 (`0.4915`), using three 16-generated-document cells against the same 32-human bank. Individual cell CIs were wide; the same visible bank and probe family had already been used throughout development.

**Inference.** This is an optimizer's-curse setup. Selecting the most human-looking of three noisy seeds makes regression toward worse hidden AUC likely. The hidden `0.7269` does not require a mysterious reversal. Future checkpoint selection should be independent of the promotion endpoint: choose the median seed by a prespecified training statistic, or evaluate all three seeds on a second locked visible bank and aggregate without choosing the best.

### 6. The deployment-sampler record is internally inconsistent

**Fact.** The checked-in `harness/deployment_sampler.json` remains `frozen=false` with null fields. The sealed artifact instead embeds a local sampler object (`temperature=1`, `top_p=1`, `max_new_tokens=384`) and the sealed record calls this "already frozen." The public harness's checkpoint-generation path additionally requires a prompt bank, prompt format, seed, batch size, and input limit that are absent from that embedded object. The public path also restricts `prompt_format` to `{user_prompt}`, while the successful visible 4B workflow used the full structured brief.

**Inference.** This does not prove the sealed score is wrong; the private service may own hidden prompt rendering. It does mean the repository lacks one canonical, reviewable deployment contract that proves visible and sealed generation differ only in data and evaluator. Before another sealed submission, freeze one versioned schema covering full-brief rendering, tokenizer/chat template, seed policy, batching, input truncation, output length, and decoding.

## Is the sealed rejection decision-grade?

### Yes, for the exact checkpoint

- The final request returned a contract-valid aggregate result after the evaluator repair.
- The recorded repair changed batching, volume refresh, and deterministic sampling, not weights, hidden data, metrics, thresholds, or embedder.
- The authorship AUC interval excludes 0.5 by a wide margin.
- The preregistered rule required a sealed confirm, so blocking 14B and Tier 3 is the correct fail-closed decision.

### No, for broader causal claims

- Only seed 29 was scored; the hidden training-seed distribution is unknown.
- There is no same-window matched SFT-versus-DFT contrast; indeed, the candidate itself is SFT.
- There is no unadapted 4B or alternative SFT hidden control, so the direction of the SFT treatment effect is unknown.
- The aggregate API returns no semantic-MMD CI or paired uncertainty, so the precision of `+0.029529` is not auditable from visible evidence.
- The checkpoint was selected on the same noisy Tier-1 AUC family that motivated promotion.
- Several pre-result evaluator fixes and the sampler-contract inconsistency warrant a reproducibility check before treating the exact point estimates as permanent benchmarks, although they do not justify overriding the reject verdict.

The correct conclusion is narrower than the current narrative: **the selected repaired 4B SFT checkpoint failed both absolute hidden gates. That result closes scale-up of this checkpoint, not the 4B research program.**

## Ranked explanations for Tier-1 versus Tier-2 divergence

1. **Tier-1 estimator noise plus adaptive seed selection — high confidence.** Sixteen generated documents per cell and 32 reused humans yield wide AUC intervals and noisy unbiased MMD. Selecting seed 29 for AUC proximity to 0.5 compounds optimism. **Disconfirming evidence:** all three seeds and all sampling seeds showed similar visible distribution scores, so the effect is not a single-cell accident.

2. **Visible reference-bank and hidden domain/time mismatch — medium-high confidence.** The visible bank is a small, heterogeneous 2024-10 FineWeb sample chosen without quality or prompt matching; the hidden split is intentionally different in time/domain and uses a heavyweight embedder. With only 256 training documents, representation coverage is thin. **Disconfirming evidence:** training/dev were domain-distinct and the base instruct model has broad pretraining, so pure topic memorization is not the whole story.

3. **SFT learned conditional copying/adherence rather than a human web distribution — medium confidence.** Cross-entropy on 256 prompt-document pairs strongly rewards reproducing outline wording. The visible factual proxies also reward lexical overlap. Nothing in this training objective explicitly reduces corpus-level semantic MMD or authorship separability. **Disconfirming evidence:** visible lexical, structural, self-BLEU, and AUC metrics all moved in the desired direction, so SFT did alter more than factual recall; the question is whether that movement was robust.

4. **Prompt-rendering, length, or truncation transport mismatch — medium-low confidence but audit-critical.** Visible training/generation used a full brief with 1,024 input tokens and 384 new tokens, while the checked-in deployment sampler is unfrozen and the sealed artifact does not expose the full rendering contract. **Disconfirming evidence:** the final evaluator explicitly reports the frozen default decoding values and completed 128 deterministic generations; no direct evidence shows the hidden prompts were malformed.

5. **Sealed implementation artifact — low confidence.** The evaluator needed operational repair immediately before the score. **Disconfirming evidence:** pre-result failures returned no metrics; the final run passed public/private tests; both independent gates failed; and the AUC miss is large. Treat this as a reason to reproduce later, not as a reason to dismiss the result now.

**Speculation.** The most likely joint explanation is mundane: a small SFT corpus plus a low-powered, adaptively reused visible bank created an optimistic Tier-1 picture, while the larger independent split measured easily separable SFT prose. No evidence currently requires a hidden-evaluator defect or an unusual training-seed pathology.

## Next-cycle experimental arms at 4B

Cost ranges below are planning estimates, not quotes. They extrapolate from the observed three-seed 4B SFT cost (`$0.321`), 144-output generation/evaluation cost (`$1.279`), and pilot synthesis spend. They include expected GPU generation/evaluation and, for the data arm, provider synthesis; they exclude sunk costs and must remain subject to the existing wrappers and caps.

Every arm uses Qwen3-4B at the pinned revision, full-brief rendering, training seeds `[11,29,47]`, sampling seeds `[101,202,303]`, a matched SFT control on identical examples and update budget, and 64 then 128 output tokens. A 256-token stage is unlocked only after the 128-token gate; 512/1024 remain a later curriculum, not part of this bounded cycle.

| Arm | Mechanism and control | Estimated bounded cost | Falsification / stopping rule |
| --- | --- | ---: | --- |
| **S0: data-scale SFT control** | Compare (a) 256 docs x 1 epoch, (b) 256 docs repeated to match exposure, and (c) 1,024 disjoint docs x 1 epoch. This separates data diversity from extra optimization. | `$4-8` including ~768 new briefs/dev references and two-length evaluation | Stop if 1,024-doc SFT does not beat both controls on matched-sample semantic MMD and AUC distance from 0.5 at seed 11, or if recall drops >0.05 / unsupported rises >0.05. Confirm seeds 29/47 only after the seed-11 screen. |
| **E: teacher-forced moment matching** | Add a small loss matching expected unigram/bigram and pooled hidden-state moments under teacher forcing to moments from the human completions, with CE/KL anchor. Training features are frozen and explicitly distinct from BGE and the sealed representation. Control is S0 with identical data, steps, and LoRA capacity. | `$3-6` | Stop if moment loss falls without improvement in either non-reward semantic MMD or fresh-probe AUC, if CE/KL drift exceeds the preregistered bound, or if either validity margin fails at 64 tokens. |
| **D: n-gram residual/unlikelihood control** | Penalize only n-grams overrepresented in S0 rollouts and upweight underrepresented human n-grams; no semantic reward. This tests whether the hidden authorship gap is mostly lexical. | `$3-6` | Stop if lexical L2 improves but semantic MMD and AUC do not improve on two of three seeds, or if repetition/self-BLEU leaves the human interval. This arm cannot win promotion on lexical metrics alone. |
| **C: reward-weighted SFT** | Generate a fixed small rollout set per prompt; score with an independent, frozen composite of non-BGE semantic/style moments plus hard validity; imitate top-weighted rollouts. Matched rollout and token budget versus S0. | `$5-10` | Stop after 64 tokens if effective sample size collapses, reward becomes highly correlated with length/copying, or held-out non-reward metrics fail to improve. Advance to 128 only if all three seed-11 sampling cells pass validity/collapse. |
| **A: score-function MMD** | Direct 64-token MMD optimization with an independent training embedder, leave-one-out baseline, KL to S0, and SFT anchor; then 128 tokens. This is the first genuinely distributional policy-gradient arm. | `$6-12` | Stop on non-finite gradients, gradient-variance ratio >4x the preregistered SFT reference for two checkpoints, KL breach, collapse, or no paired Tier-1 gain by the halfway budget. Never evaluate it solely with its reward embedder. |

Whole-sequence and segment-level GAIL should be deferred until at least one cheaper arm produces a stable 128-token gain. The repository's own prior-art note says adversarial training destabilized at long lengths; starting B1/B2 now would add discriminator failure modes before the visible measurement problem is repaired.

## Exact first two experiments

### Experiment 1: `4B-S0-scale-exposure-64-128-v1`

1. Materialize a new public FineWeb train/dev slice with `1,024/256` documents, disjoint by fingerprint, domain, and crawl slice from all current train/dev/Tier-1 material. Freeze the selection and brief-synthesis prompt before calls.
2. Build three matched Qwen3-4B SFT cells: current 256 x 1 epoch; current 256 repeated to the same completion-token exposure as the 1,024 corpus; and 1,024 x 1 epoch. Use seed 11 first.
3. Evaluate each on a newly locked 128-prompt visible holdout, with exactly 128 outputs per cell at 64 and 128 tokens and identical sampling seed. Use a second sampling seed only for cells that pass the first screen.
4. Primary contrast: 1,024 x 1 epoch versus 256 exposure-matched. Report paired bootstrap CIs for raw semantic MMD difference, lexical/structural differences, and AUC distance from 0.5. Do not standardize against the old two-document baseline.
5. Promotion to seeds 29 and 47 requires: semantic MMD lower for the 1,024 arm; AUC distance from 0.5 no worse; all absolute validity/collapse gates pass; and no effect attributable only to length. If not, stop and retain the data-scale negative result.

This experiment comes first because the current sealed failure cannot distinguish "SFT objective is insufficient" from "256 documents are insufficient," and every later objective needs a matched, credible SFT control.

### Experiment 2: `4B-E-teacher-moments-64-128-v1`

1. Start from the winning matched S0 setup, not from the adaptively selected seed-29 checkpoint.
2. Add teacher-forced moment loss with three preregistered coefficients (`0`, low, high) at seed 11. Keep examples, token exposure, LoRA rank, optimizer, and CE steps identical.
3. Evaluate all coefficients on the same locked 128 prompts and cardinality as Experiment 1. Select the coefficient using a composite that excludes the training representation and requires hard-gate success; do not select on AUC alone.
4. Confirm only the selected coefficient and coefficient zero at seeds 29/47. Require improvement versus coefficient zero in every training seed on raw matched-sample `S`, improvement in at least two seeds on both semantic MMD and AUC distance from 0.5, and no validity/collapse regression.
5. Stop the entire moment-matching direction if the training moment loss improves while both independent semantic MMD and AUC remain flat. That result would falsify teacher-forced moment transfer and justify moving to on-policy Arm C or A.

## Prospective gates

### Before any new sealed submission

All conditions are required:

1. One canonical deployment-generation contract is checked in and frozen: full-brief serializer, chat template, tokenizer revision/hash, prompt-bank schema, input/output limits, sampler, seed policy, and batching. Visible replay must reproduce a fixed checksum set before remote submission.
2. A new locked visible holdout has at least 128 prompts and 128 independent humans, with distribution references separate from prompt-matched quality references. Baseline and candidate use identical cardinality, prompts, lengths, and sampling seeds.
3. The candidate beats its matched 4B SFT control on all three training seeds. For raw `S`, the candidate must be lower in each seed and the pooled paired 95% bootstrap CI must exclude zero. Semantic MMD and authorship AUC must each improve in at least two seeds, and no result may rely only on the representation used for training.
4. Every seed passes absolute outline, unsupported, language, self-BLEU, repetition, and length gates. The overlap proxies remain reported for continuity, but a locked entailment/source-span audit must show no >0.05 degradation.
5. Checkpoint selection is endpoint-independent. Prefer the median training seed under a prespecified non-evaluation statistic, or submit a prespecified set of seeds. Do not choose the seed closest to visible AUC 0.5.
6. The sealed comparison is preregistered as **matched SFT versus DFT**, ideally in the same window. A single absolute candidate submission may gate safety, but it cannot establish a DFT treatment effect.

### Before 14B

1. A genuine 4B DFT arm, not SFT, passes the visible gates above and receives sealed confirmation against a matched 4B SFT control on both semantic and authorship gates.
2. The benefit appears outside the training representation and persists at 128 and 256 tokens without rising gradient variance, KL drift, or collapse.
3. Only then run a one-seed 14B bridge at the same 128-token method/control and matched token budget. Expand to three 14B seeds or longer lengths only if the treatment effect does not reverse.
4. A 14B SFT-only run is not authorized merely because a 4B SFT seed failed; scaling without a validated objective would confound size with method and spend.

### Before GPTZero or Pangram

1. Tier 3 remains human-triggered and one-shot after a final frozen candidate passes Tier 2. Detector model/version, date, sample count, prompts, and decision rule are preregistered before scores are seen.
2. Run the required four-way comparison `{SFT, best DFT} x {raw, production wrapper}` plus fresh human references. Preserve randomized/blinded labels.
3. Treat detector outputs as secondary robustness evidence, never as training reward or an iteration oracle. Do not repair the model after seeing per-example detector outputs and resubmit to the same exam.
4. Pair detector results with blinded human preference/adherence evaluation. Passing GPTZero or Pangram alone cannot support "human indistinguishability."

## Recommended status rewrite

Replace "Independent sealed evaluator rejected the winning 4B checkpoint" with:

> The selected seed-29 Qwen3-4B full-brief SFT checkpoint passed the bounded visible adherence screen but failed both absolute sealed gates. This blocks scale-up of that checkpoint. No DFT arm has yet been tested; the next step is a matched 4B SFT/data control followed by bounded 64/128-token objective experiments.

That wording preserves the legitimate fail-closed decision without overstating what the experiment identified.

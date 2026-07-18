# Measurement v3 lower-variance 4B report

## Executive verdict

No treatment produced a decision-grade improvement over matched SFT, and this cycle did not produce a model that can yet be described as writing like a human.

- **FACT:** Both lower-variance treatments moved BGE MMD, Nemotron MMD, and token unigram L2 in the desired direction.
- **FACT:** Token-moment passed the heavyweight Nemotron MMD endpoint, but failed the BGE endpoint and was judged significantly worse on overall writing quality.
- **FACT:** MMD-witness showed the safer quality pattern (54.7% human-style wins and 53.1% overall-quality wins), but neither result was significant and it missed the preregistered automatic effect thresholds.
- **FACT:** Neither treatment passed the full automatic intersection rule. No automatic or quality winner was selected.
- **INFERENCE:** MMD-witness is the only current method worth one stronger 4B dose-response test. Token-moment should be retired in its current form because its distributional gain did not transfer to writing quality.

## Experiment scale

- Model: Qwen3-4B, starting from the same seed-11 SFT adapter.
- Training: 1,024 cleaned and structured briefs, exactly one exposure per arm, batch size 2, and 512 optimizer steps.
- Arms: matched SFT, token-distribution moment matching, and MMD-witness-weighted SFT.
- Evaluation: a fresh prospective panel of 128 prompts, 256 independent semantic references, two independent 128-document human floors, BGE-small and Llama-Embed-Nemotron-8B, token L2, validity checks, and 1,024 blinded pairwise judge calls.
- Screening horizon: 64 generated tokens with normal EOS stopping.

## Automatic results

Lower is better for every numeric endpoint in this table. Deltas are treatment minus SFT.

| Endpoint | SFT | Token-moment | Token delta | Token p | MMD-witness | MMD delta | MMD p |
|---|---:|---:|---:|---:|---:|---:|---:|
| BGE MMD² | 0.000883 | 0.000795 | -0.000088 | 0.1155 | 0.000851 | -0.000032 | 0.3045 |
| Nemotron MMD² | 0.004552 | 0.003898 | -0.000654 | 0.0254 | 0.004092 | -0.000461 | 0.0888 |
| Token unigram L2 | 0.016791 | 0.016400 | -0.000392 | — | 0.015933 | -0.000858 | — |

Additional validity and displacement findings:

| Endpoint | SFT | Token-moment | MMD-witness |
|---|---:|---:|---:|
| Unexpected non-Latin rows | 17.19% | 17.19% | 15.63% |
| Unique outputs | 100% | 100% | 100% |
| Byte-identical to SFT | — | 76.56% | 78.13% |

The frozen meaningful MMD boundary was -0.0006 in both embedding families with one-sided p <= 0.05. Token-moment cleared that rule only for Nemotron. Both treatments failed the human-calibrated unexpected-script equivalence gate, and every arm failed absolute hard validity because the underlying 4B policy still emits multilingual artifacts too often.

## Blinded writing judge

The judge was GPT-5.4-mini with frozen side randomization, four separate rubrics, and 128 comparisons per treatment/rubric. The primary human-style rule required at least 55% wins and one-sided p <= 0.05, with no significant overall-quality regression.

| Treatment | Human style | p (better) | Overall quality | p (better) | Creativity | Depth | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Token-moment | 51.56% | 0.3955 | 42.19% | 0.9685 | 46.88% | 48.44% | Fail; overall quality significantly worse (one-sided worse p=0.0463) |
| MMD-witness | 54.69% | 0.1655 | 53.13% | 0.2681 | 53.13% | 50.00% | Fail; positive trend but not significant |

Because most treatment outputs were byte-identical to SFT, the preregistered all-prompt comparison is diluted by forced judgments between identical strings. An exploratory changed-output-only view is therefore mechanistically useful but not a promotion endpoint:

| Treatment | Changed prompts | Human-style wins | Overall-quality wins |
|---|---:|---:|---:|
| Token-moment | 30 | 33.3% | 40.0% |
| MMD-witness | 28 | 57.1% | 53.6% |

This exploratory result strengthens the decision to retire the current token-moment loss and retain only MMD-witness for a bounded follow-up.

## What succeeded

- The full clean-data, matched-training, fresh-panel, dual-embedder, and blinded-judge pipeline ran end to end.
- All three arms completed the same 1,024-example exposure and produced hash-bound artifacts.
- The lower-variance objectives affected the intended distribution metrics rather than returning another exact null.
- MMD-witness improved every automatic distribution endpoint without a detected quality regression.

## What failed

- Neither method produced the preregistered cross-representation MMD effect.
- The model moved too little: only 22-23% of treatment outputs differed from SFT.
- The 4B policy still emits unexpected non-Latin fragments at an unacceptable rate.
- Token-moment optimized a proxy while making judged writing worse.
- MMD-witness's human-style advantage is currently only a weak trend, not reliable evidence.
- A 64-token screen cannot establish long-form or rewriting product quality.

## Recommended next decision

Run one bounded MMD-witness-only 4B dose-response cycle before considering 14B:

1. Keep matched SFT and remove token-moment.
2. Increase MMD-witness treatment strength and/or exposure prospectively until expected byte identity is below 60%, with KL and validity stop rules.
3. Evaluate at 128 tokens with EOS-aware generation, repaired token-length semantics, and a new untouched panel.
4. Require improvement in both embedders, token L2 non-inferiority, no unexpected-script regression, and significant blinded human-style wins.
5. Scale to 14B only if that 4B confirmation passes. Do not run sealed evaluation or GPTZero/Pangram before then.

If the stronger MMD-witness arm again improves proxies but not blinded writing quality, this reconstruction should be stopped: the undisclosed Rosmine method is materially different from the methods tested here.

## Evidence artifacts

- Automatic decision: `.operator/lower_variance/measurement-v3-automatic-decision.json`
- Quality summary: `.operator/lower_variance/measurement-v3-quality-summary.json`
- Candidate outputs and embeddings: `.operator/lower_variance/measurement-v3-candidates/candidate-outputs-v1/`
- Judge results: `.operator/lower_variance/measurement-v3-judge/quality-judge-v1/`
- Frozen protocol: `.operator/lower_variance/measurement-v3-protocol-v1/measurement_protocol_v3.json`
- Original judge contract: `configs/m2/m2_measurement_v3_judge_contract_v1.json`
- Documented output-ceiling amendment used by the completed judge: `configs/m2/m2_measurement_v3_judge_contract_v3.json`

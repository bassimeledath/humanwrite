# Scale-ladder 4K result

Date: 2026-07-19

Machine-readable result:
`research_reviews/scale_ladder_4k_score_2026-07-19.json`, SHA-256
`0e2327b98d0de954768302a7dde8e85c8e9123ed93675a1b77bec371fa91cd58`.

## Bottom line

The MMD-witness treatment moved every headline proximity/quality point estimate
in the favorable direction, but the effects were small and statistically
inconclusive. The frozen 4K gate nevertheless stops the ladder because the
treatment produced a real Unicode replacement-character defect. The control
also produced one such defect, so this is not evidence that MMD uniquely caused
the problem; it is evidence that the current model/sampler combination does not
meet the preregistered hard-validity standard.

## Results

| Endpoint | SFT | MMD-witness | Treatment change | Interpretation |
| --- | ---: | ---: | ---: | --- |
| BGE MMD² | 0.002126 | 0.002049 | -0.000077 (-3.6%) | Favorable, paired p=0.3434 |
| Nemotron MMD² | 0.000841 | 0.000791 | -0.000050 (-6.0%) | Favorable, paired p=0.3301 |
| Token unigram L2 | 0.013137 | 0.012551 | -0.000586 (-4.5%) | Favorable |
| Human-style judge | 50% null | 70/128 (54.7%) | +4.7 pp | Favorable trend, one-sided improvement p=0.1655 |
| Overall-quality judge | 50% null | 66/128 (51.6%) | +1.6 pp | Neutral/favorable, one-sided improvement p=0.3955 |
| Byte identity | — | 68.0% identical | 32.0% displaced | Better than the prior 78%, still weak |
| Mean repeated-bigram rate | 0.04188 | 0.04518 | +0.00330 | Slightly worse |
| Unexpected non-Latin rows | 6/128 (4.7%) | 11/128 (8.6%) | +3.9 pp | Worse, though below the 15% absolute cap |
| Replacement-character rows | 1/128 | 1/128 | equal | Both arms fail the frozen zero-tolerance rule |

The blinded judge cost was `$0.199926` for 256 comparisons. Both embedding
families used their pinned independent revisions and human-floor-only bandwidth
selection. The training-only Qwen hidden-state representation was excluded.

## Epistemic interpretation

- **FACT:** Both independent MMD point estimates, token L2, human-style win
  rate, and overall-quality win rate favor MMD-witness.
- **FACT:** Neither MMD effect nor the human-style result is statistically
  decisive at n=128.
- **FACT:** Multilingual corruption is more frequent in the treatment, and one
  treatment output contains the Unicode replacement character.
- **FACT:** The same replacement defect also appears once in SFT, on a different
  prompt. Inspection shows genuine malformed/multilingual model output, not a
  scorer parsing bug.
- **INFERENCE:** Scaling from 1K to 4K may be increasing displacement while
  preserving a weak favorable writing trend, but it has not demonstrated a
  reliable human-writing advantage.
- **INFERENCE:** The replacement-character stop is a shared base-policy/sampler
  limitation rather than clear treatment-specific harm. It still cannot be
  waived after seeing the result without invalidating the frozen gate.

## Decision

`STOP` under the frozen July 17 4K rule. Do not automatically launch 16K from
this panel.

Scientifically defensible next paths are:

1. retire this exact scale ladder because it failed its hard-validity gate; or
2. preregister a separate decoder-validity experiment that applies an identical,
   model-agnostic byte-safe sampling policy to both arms, re-generates a new
   untouched panel, and tests whether the favorable point estimates persist.

The second path is not permission to filter bad outputs after generation or to
re-score this opened panel. It must be a new comparison with a new panel.

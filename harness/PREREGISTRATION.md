# Preregistration (fill in before M2; changes are human-only + timestamped)

## Primary endpoint
S = w1*z(semantic_MMD) + w2*z(lexical_L2) + w3*z(structural_dist)
- z() standardizes each component against the SFT-baseline mean/std recorded
  at M1 freeze (harness/baseline_stats.json).
- The M1 baseline is bootstrapped from the preregistered default sampler
  `default_t1.0_p1.0`, before sampler selection. This avoids selecting a
  sampler with scores standardized by statistics that depend on that same
  selection. The frozen baseline is then used to rerun every sampler report.
- Report S AND raw components AND the delta vs the human-vs-human floor
  (Delta_MMD = MMD(gen,H1) - MMD(H2,H1)) for the semantic term.
- Weights: w1=0.50 w2=0.25 w3=0.25   (frozen 2026-07-15)

## Semantic MMD
- Embedder (dev): BAAI/bge-small-en-v1.5 (frozen 2026-07-15)
- Kernel: multi-bandwidth RBF, scales [0.25,0.5,1,2,4] x median heuristic
- Unbiased U-statistic; exclude self-similarities.

## Lexical L2
- Normalized unigram + selected bigram/trigram frequency vector, L2 distance.
- Feature set frozen at M1 (hashed n-grams ok).

## Structural distance
- Paragraph-length dist, sentence-length dist, sentence-opening-template
  dist. Distance metric: Jensen-Shannon distance (frozen 2026-07-15)
- Paragraph token-count bins: [0,20), [20,50), [50,100), [100,200),
  [200,400), [400,+inf). Sentence token-count bins: [0,5), [5,10),
  [10,20), [20,30), [30,50), [50,+inf). (frozen 2026-07-15)
- Sentence-opening templates use the case-folded first lexical word:
  pronoun={i,we,you,he,she,they,it}; article={a,an,the};
  conjunction={and,but,or,so,yet}; question={what,when,where,who,why,how};
  number=first word is numeric; other=everything else. (frozen 2026-07-15)

## Hard validity gates (all vs SFT baseline, absolute non-inferiority margin delta=0.05; frozen 2026-07-15)
- outline_fact_recall >= baseline - delta
- unsupported_claim_rate <= baseline + delta
- language_integrity: non-target-script char rate within calibration range
- no_collapse: self-BLEU and repetition within calibration ranges
- repeated_sentence_start_rate is the fraction of documents containing at
  least three consecutive sentences whose first lexical word is identical,
  case-insensitive (Rosmine-disclosed definition; ~0.174 in its human corpus).

## Secondary (report, do not gate on point values)
- quality_pref: judge win rate vs human, non-inferiority framing, order
  randomized, self-preference caveat noted
- JMQ: Rosmine-exact (2x win rate), comparability only
- authorship_AUC: fresh probe, equivalence test around 0.5, report CI
- diversity/repetition/length: inside calibration.json intervals

## Cross-representation rule
eval() must reject: (a) semantic scoring of an MMD checkpoint with the
embedder id recorded in its train config; (b) any use of a GAIL checkpoint's
own discriminator as an evaluator.

## M1 freeze order

1. An operator provisions and freezes an independently selected visible human
   bank of at least four unique, non-training, non-hidden-test documents.
2. The operator reviews the human-only calibration proposal and transfers its
   exact intervals into `calibration.json`; no sampler outputs enter this step.
3. Tier 1 produces bootstrap reports for only `default_t1.0_p1.0`. Raw metrics
   are usable for a baseline proposal while unavailable baseline gates remain
   fail-closed.
4. The operator reviews and transfers that default-sampler proposal into
   `baseline_stats.json`.
5. Every sampler cell is rerun against the exact frozen human-bank,
   calibration, and baseline hashes. Only these reports may feed sampler
   freeze; the selected sampler is then transferred into
   `deployment_sampler.json`.

# RESEARCH_CONTEXT: what we know about DFT / Deft, and how we know it

Status legend: [V] verified against a primary source snapshot in sources/,
[R] reported by a secondary analysis, verify before load-bearing use,
[H] hypothesis / reconstruction.

## The target

Rosmine's "Fixing LLM writing with Distribution Fine Tuning" (blog,
2026-05-18) and the productized Deft API (deftwriting.com). The core training
algorithm is proprietary and undisclosed. Everything below is disclosed
material or reconstruction.

## Disclosed training setup (blog)

- [V] Base models: Qwen3 instruct family (base models failed on instruction
  following). Demo model: 14B. A 4B DFT beat a 14B SFT superbaseline on MMD.
- [V] Data: 185K-sample cleaned subset of FineWeb; 2,000 held-out samples for
  eval (first 400 for JMQ).
- [V] Training used a sequence of LoRAs (cites LoRA + ReLoRA). [V] 13 LoRAs
  total, confirmed against the archived author X post in sources/.
- [V] Results reported at a single fixed sampler setting.
- [V] All training on a local 6x RTX 6000 Ada server.
- [V] Appendix 2 input schema: user prompt, use case + style (extracted by
  Qwen3-32B), outline with facts/quotes (GPT-5-mini), target length in
  tokens, em-dash permission flag; 25% of samples have empty outlines.
  Author note: schema is interface, not the algorithm.
- [V] Author-reported prior art result: TextGAIL beat SFT at 64-token outputs
  and destabilized at 1,024 tokens.

## Disclosed evaluation (blog)

- [V] MMD over document embeddings (Llama-embed-nemotron-8B, Gaussian RBF
  kernel), chosen because it tests whether two sample sets share a
  distribution.
- [V] JMQ: judge model (GPT5.4-mini), randomized order, JMQ = 2x model win
  rate, optimum = 50% win rate (matching, not maxing).
- [V] Token-frequency L2 distance; self-BLEU for diversity; overused-token
  analysis (em dash, "Elara Voss" etc.); repeated-sentence-start analysis
  (human corpus: ~17.4% of docs legitimately have 3+ same-start sentences).
- [V] 100/100 outputs scored human-written by Pangram (held-out exam framing).

## Deployment architecture (Deft API docs, archived in sources/)

- [V] Billing taxonomy reveals a 4-stage pipeline: preprocessing via
  OpenRouter (prompt tokens = input, completions = thinking), Qwen3-14B
  final-text generation (= output), chunk-judge + post-processing (= thinking).
- [V] Legacy generationMode literally named `prompt_optimizer`.
- [V] detailMode strict/creative controls whether OUTLINE details may be
  invented -> planner-renderer split.
- [V] "Trained to produce whole samples of text from the web"; fragments and
  continuations degrade. No temperature control. No streaming. Inputs to
  ~150K tokens accepted (Qwen3-14B context tops out ~128K, so long inputs
  must be chunked/compressed). No billing line for discarded drafts ->
  probably judge-and-repair, not best-of-n.
- [H] Chunk-judge operates on input chunks (relevance/compression), output
  chunks (adherence verification), or both. Undetermined.

## Reconstruction hypotheses (the thing we are testing)

- [H-A] Direct score-function MMD fine-tuning. Feasibility anchor: kCGM
  (arXiv 2606.19496): unbiased MMD estimator, score-function gradients,
  leave-one-out control variate (ablated as critical), KL regularization.
  Caveat: validated on molecules/proteins/DNA with small models; long-text
  variance is unproven.
- [H-B] Adversarial imitation (TextGAIL-lineage): sequence-level contrastive
  discriminator reward + policy-gradient update, TTUR stabilization.
  B1 = whole-sequence (as published: ONE sparse reward per sequence).
  B2 = segment/prefix-level reward (a distinct, unpublished variant).
- [H] Lexical mechanism: additive kernel (embedding RBF + linear n-gram
  kernel) or explicit unlikelihood on overrepresented n-grams. Explains both
  headline metrics with one objective.
- [H] Training data = (synthetic brief -> full FineWeb doc); the brief
  absorbs content variance so the objective only has to shape style.
- [H] Fixed sampler is part of the trained artifact:
  system = weights + prompt schema + sampler (+ wrapper).

## Source archive checklist (sources/) -- human task, agent verifies presence

- [x] rosmine.ai DFT blog post, full snapshot incl. all appendices
      (site 404s intermittently; do not rely on live fetch)
- [x] rosmine.ai "$48K GPU server" post
- [x] Deft API docs page (deftwriting.com)
- [x] Rosmine X post on 13 LoRAs (x.com/rosmine/status/2057226528950305083)
      + any schema clarification posts
- [x] kCGM paper (arXiv 2606.19496) PDF
- [x] TextGAIL paper (arXiv 2004.13796) PDF
- [x] GAIL (Ho & Ermon 2016), scalable IRL (Wulfmeier 2024), TTUR/FID
      (Heusel 2017), unlikelihood (Welleck 2019) PDFs
- [x] Pangram humanizer-audit blog post (context on detector robustness)

Agent rule: any [R] item you rely on for a design decision must first be
upgraded to [V] by checking the snapshot, or flagged in FINDINGS.md as an
assumption.

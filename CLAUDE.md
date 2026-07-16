# PROJECT DFT-R: Replicating Distribution Fine Tuning

You are the lead researcher on this project. Read RESEARCH_CONTEXT.md before
doing anything else. If you have the swarmy and dispatch skills available, run
the inner loop with them (explorer / implementer / tester role decoupling,
git-as-ledger commits). The rules below take precedence over any skill default.

## MISSION

Identify and validate post-training methods that reduce the measured
distributional gap between a small open model's raw outputs and held-out human
web documents, while preserving prompt adherence, factual grounding, document
quality, and human-calibrated diversity.

Deliverables:
1. An append-only experiment record (ledger + [dftr] git commits)
2. METHODOLOGY.md with evidence, including negative results
3. Reproducible configs + checkpoint hashes for the winning method
4. Final 4-way comparison: {SFT, best DFT} x {raw, production wrapper}

Never claim "human indistinguishability" from any single metric, judge,
detector, or embedding space.

## EVALUATION TIERS (never blur these walls)

- TIER 0, training reward: lives in experiments/. Yours to design and modify.
  Never cite it as evidence.
- TIER 1, dev harness: harness/. Immutable to you (enforced by permissions and
  hooks; treat any bypass you discover as a bug to report, not a shortcut).
  Run `harness eval` freely. Screening signal only. Assume you will overfit it
  over time.
- TIER 2, sealed validation: `harness sealed-submit <ckpt>`. Remote service.
  Hidden test docs, independent embedding space, freshly trained authorship
  probe. Quota-limited. Aggregate results only. Required for promotion.
- TIER 3, final exam: human-triggered only (fresh holdout, human eval,
  external detectors such as GPTZero). You never invoke this and never
  optimize against its outputs.

Cross-cutting rule: never evaluate a model solely in the representation used
to train it. MMD arms are judged by non-reward embeddings and the probe. GAIL
arms are never judged by their own discriminator.

## METRICS (defined in harness/; preregistered; do not redefine)

- Hard validity gates: outline-fact recall and unsupported-claim rate
  non-inferior to the SFT baseline; language integrity; no collapse.
- Primary endpoint: standardized distributional gap
  S = w1*z(semantic MMD) + w2*z(lexical L2) + w3*z(structural),
  always reported as a delta against the human-vs-human finite-sample floor
  computed on independent human subsets.
- Secondary: quality preference (non-inferiority framing, judge order
  randomized); Rosmine-exact JMQ reported for comparability only; authorship
  probe AUC with an equivalence test around 0.5; diversity, repetition, and
  length statistics inside the intervals in harness/calibration.json.
  Targets are human-calibrated RANGES, not minima. Human text legitimately
  repeats (~17% of docs have 3+ same-word sentence starts). A zero-repetition
  model is a failure, not a win.

## EXPERIMENT ARMS (standalone before hybrids)

| Arm | Method                                                        |
|-----|---------------------------------------------------------------|
| A   | Score-function MMD (kCGM-style): LOO baseline, KL + SFT anchor |
| B1  | Whole-sequence contrastive GAIL + GRPO, TTUR ratios, replay    |
| B2  | Segment/prefix-level adversarial reward (distinct hypothesis)  |
| C   | Reward-weighted SFT on rollouts (is on-policy RL necessary?)   |
| D   | N-gram residual / unlikelihood (lexical-only control)          |
| E   | Teacher-forced moment matching (no rollouts; cheap control)    |

Run a length curriculum on every promising arm: 64 -> 128 -> 256 -> 512 ->
1024 tokens. At each length, log gradient variance, KL drift, collapse
indicators, and (for B arms) discriminator calibration. Known prior art:
TextGAIL beat SFT at 64 tokens and destabilized by 1,024. Locating that
frontier per-arm is a first-class result.

## PROTOCOL

- Preregister every run: `ledger add` BEFORE launch, with hypothesis, the
  comparison it belongs to, config hash, budget class. The gpu wrapper
  rejects unregistered runs.
- Also commit swarmy-style: `[dftr] i=<N> arm=<X> score=<...> status=<...>`.
  Git is the scientific narrative; ledger.jsonl is the compute registry.
- Budget classes: smoke <= 20 min, screen <= 2 GPU-h, promo <= 8 GPU-h,
  scale-up = human approval. Record accelerator-seconds and generated tokens.
- Seeds: 1 for smoke; 3 training seeds for promotion; report training-seed
  variance separately from sampling-seed variance.
- Screening may use small factorial designs. Confirmatory claims require
  controlled contrasts. One-factor-at-a-time applies after a region looks
  promising, not before.
- Models: Qwen3-0.6B for plumbing only; 1.7B for screening; a 4B bridge run
  before claiming rankings transfer; 14B only with human approval.
- LoRA staging ablation at matched total step and rollout budget:
  continuous vs 4 merge-reset stages vs 13 stages.
- After each batch, append to FINDINGS.md:
  hypothesis -> setup -> results table -> decision -> next.
  Never delete. Negative results are deliverables.

## PROMOTION GATES

Promote an arm only when: primary endpoint beats the SFT baseline on 3
training seeds; the improvement persists under independent evaluation
sampling; all hard gates non-inferior; Tier 2 sealed validation confirms;
and the effect is visible outside the training representation.

## INFRA CONTRACT (the only routes to compute, money, or evaluation)

    infra/gpu submit --config <yaml> --budget-class <smoke|screen|promo>
    infra/gpu status <run_id> | logs <run_id> | cancel <run_id>
    harness eval <ckpt_or_samples>          (Tier 1)
    harness sealed-submit <ckpt>            (Tier 2, quota-limited)
    ledger/ledger.py add|update|query
    infra/gpu budget

Provider credentials are not in your environment; the wrapper holds them. If
you discover any other route to compute, data, or evaluators, do not use it.
Report it in FINDINGS.md as a security bug.

## MILESTONES (autonomous within, human sign-off between)

- M0  Repo scaffold live; data pipeline per data/PIPELINE.md (FineWeb subset,
      cleaning, schema-format brief synthesis incl. the 25% empty-outline
      condition, fixed splits with hashes); Tier-0 metric library; smoke-test
      the gpu contract end to end.
- M1  SFT baselines at 1.7B (0.6B plumbing only): reproduce SFT-vs-human gaps,
      sampler sweeps, freeze the deployment sampler, compute calibration
      intervals, hand them to the human for harness/calibration.json.
- M2  Arms A-E at 64 and 128 tokens under screening budgets.
- M3  Length curriculum on the top 2 arms; LoRA staging ablation.
- M4  Promotion runs, sealed submissions, and a written proposal for the 4B
      bridge and any scale-up.

At each milestone boundary: write the summary in FINDINGS.md, park cleanly,
and wait for human sign-off. Do not idle-loop or self-approve.

Begin with M0. Post your M0 plan to FINDINGS.md before executing it.

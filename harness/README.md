# harness/ -- TIER 1 development evaluator (IMMUTABLE TO THE RESEARCH AGENT)

Vendor this read-only. The research agent RUNS it, never edits it. Ideally a
SEPARATE Claude session (or you) implements the metric bodies so the agent
that optimizes the score never authored the scoreboard.

## What lives here vs. the sealed evaluator

harness/ (Tier 1)                     sealed_evaluator/ (Tier 2, separate repo)
- small/fast embedder                 - heavyweight independent embedder
- dev human reference bank            - hidden test docs (never near agent)
- surface + distribution + quality    - fresh authorship probe per window
- run as often as you like            - quota-limited, aggregate-only
- SCREENING signal, will be overfit   - PROMOTION signal

Cross-representation rule is enforced here: `harness eval` refuses to score an
MMD-trained checkpoint with the same embedder id recorded in its training
config, and refuses to use a GAIL checkpoint's own discriminator.

## CLI

    harness eval <checkpoint_dir | samples.jsonl> [--report out.json]
    harness sealed-submit <checkpoint_dir>          # proxies to Tier 2
    harness calibrate <human_split.jsonl>           # M1: build calibration.json

`eval` consumes pre-generated JSONL and never calls a generation API. Records
accept the canonical `data/PIPELINE.md` `completion` field (plus
`generated_completion`, `generated_text`, or `output` for generated samples).
Human references must either be paired as `reference_completion` /
`human_completion`, or supplied as a canonical JSONL bank through
`HARNESS_HUMAN_REFERENCE`. Checkpoint directories must include a train config
and may include `samples.jsonl` (or `eval_samples.jsonl` /
`generations.jsonl`). Without samples, generation fails closed until
`deployment_sampler.json` is human-frozen; it then uses canonical
`user_prompt` records and a local transformers/PEFT checkpoint.

The secondary preference judge remains neutral unless injected directly or
both `HARNESS_JUDGE_URL` and `HARNESS_JUDGE_TOKEN` are set. The minimal remote
adapter sends only `{prompt,candidate_a,candidate_b}` and accepts only an
aggregate `{winner}` response. The authorship probe requires no service: each
evaluation freshly trains a deterministic character n-gram logistic model and
reports stratified out-of-fold AUC and CI.

`sealed-submit` requires `SEALED_EVAL_URL` and `SEALED_EVAL_TOKEN`. It sends the
bearer token only in the authorization header and rejects any response outside
the aggregate-only Tier 2 contract. `SEALED_ARTIFACT_URI` may override the
artifact URI recorded in the checkpoint config.

## Metric definitions are preregistered

PREREGISTRATION.md pins metric math, kernel bandwidths, weights w1/w2/w3, and
non-inferiority margins BEFORE the agent starts. Changing them is a human
action recorded there with a timestamp and rationale.

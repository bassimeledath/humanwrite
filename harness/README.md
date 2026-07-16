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
`HARNESS_HUMAN_REFERENCE`. Tier 1 requires at least four unique held-out human
documents. When `HARNESS_HUMAN_REFERENCE` is set it intentionally overrides
inline references and requires `HARNESS_HUMAN_REFERENCE_MANIFEST` (or the
default `<bank-stem>.manifest.json`) with the operator-owned
`dftr.tier1_human_bank.manifest.v1` contract:

```json
{
  "artifact_schema": "dftr.tier1_human_bank.manifest.v1",
  "bank_path": "<visible canonical JSONL>",
  "bank_sha256": "<sha256 of exact JSONL bytes>",
  "config_path": "<preregistered selection config>",
  "config_sha256": "<sha256 of selection config>",
  "counts": {"bank_size": 32, "unique_domain_count": 32},
  "domains": ["<in bank-row order>"],
  "fingerprints": ["<in bank-row order>"],
  "policy": {"agent_visible": true, "hidden_test_materialized": false, "purpose": "Independent visible Tier-1 distribution bank; never training data"},
  "selection": {"bank_size": 32, "seed_label": "<frozen seed label>"},
  "source": {"dataset_id": "<public source>", "dataset_config": "<slice>", "revision": "<immutable revision>"}
}
```

Every bank row needs `completion`, `fingerprint`, `domain`, pinned
`source_config`/`source_revision`, and `split=tier1_visible_human`; train/test rows,
duplicates, manifest mismatches, and overlap with sampled fingerprints or IDs
are rejected. The current fixed M0 fixture has only two dev humans, so it does
not satisfy this contract and must not be padded with train rows or duplicates.

Checkpoint directories must include a train config
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

### Operator-only M1 transfers

The harness never copies review artifacts automatically. It validates exact
operator-reviewed source bytes and emits the only accepted target shape:

```bash
harness prepare-calibration-transfer experiments/m1/calibration_proposal_visible_bank_v2.json \
  --expected-sha256 <reviewed-proposal-sha256> > /tmp/calibration.candidate.json
harness prepare-baseline-transfer experiments/m1/baseline_stats_v1.json \
  --expected-sha256 <reviewed-proposal-sha256> > /tmp/baseline.candidate.json
```

Only `m1.calibration_proposal.review.v2` is transferable. It must carry exact
per-metric `interval_methods`, repetition `{successes,trials}`, the matching
point estimate, and the matching Wilson interval. A v1 proposal or a v2
proposal that relabels the old `[0,0]` repetition interval is rejected.

The operator reviews the candidate, removes no provenance fields, replaces
the corresponding immutable harness JSON with exactly that object, and
commits its SHA-256. The calibration mapping is exact: each accepted proposal
`intervals.<metric>.{low,high}` becomes the identically named harness interval;
`interval_methods` and repetition counts are copied into
`harness.calibration.v2`, and no percentile is inferred. Repetition uses the
two-sided 95% Wilson score interval over document-level incidence; self-BLEU,
script rate, and length use deterministic central order-statistic intervals.
The two-human M0 calibration proposal remains a
descriptive, operator-reviewed limitation and does not satisfy the independent
four-human Tier-1 reference-bank contract.

The non-circular order is: independent human bank -> calibration transfer ->
default-sampler bootstrap reports -> baseline proposal/transfer -> rerun all
sampler cells with frozen hashes -> sampler freeze. Missing/unfrozen/null
calibration or baseline artifacts keep gates fail-closed.

## Metric definitions are preregistered

PREREGISTRATION.md pins metric math, kernel bandwidths, weights w1/w2/w3, and
non-inferiority margins BEFORE the agent starts. Changing them is a human
action recorded there with a timestamp and rationale.

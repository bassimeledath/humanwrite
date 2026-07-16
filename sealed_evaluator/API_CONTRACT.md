# Sealed evaluator (TIER 2) -- build this in a SEPARATE PRIVATE REPO

The research agent must NEVER see this repo, its data, or its embedder. It
holds the hidden test split, an independent heavyweight embedder, and a
freshly trained authorship probe. It exposes one HTTP endpoint. The harness
`sealed-submit` command is its only client.

## Why separate

A read-only directory + hooks are best-effort (Anthropic's own hooks docs say
so). The only real boundary is: the test data and the promotion embedder do
not exist anywhere the agent can reach. Deploy this as its own service
(Modal endpoint, small VM, whatever) with its own secrets.

## Independence requirements

- Embedder: different family/size from the Tier 1 dev embedder AND from any
  embedder any arm uses as a training reward. This is the whole point.
- Test docs: a FineWeb split with a different domain/time slice than train/dev.
  Never shipped out of the service.
- Authorship probe: retrained per submission window on {held-out human} vs
  {recent submitted generations}, so it tracks the current frontier of
  attempts rather than a stale one.

## Endpoint

    POST /submit
    Auth: bearer token (service-issued, not in agent env)
    Body: {
      "checkpoint_hash": "<16-hex from harness._ckpt_hash>",
      "artifact_uri":    "s3://.../ckpt or modal-volume path",
      "arm":             "A|B1|B2|C|D|E|SFT",
      "train_embedder_id": "<so the service can assert independence>",
      "comparison_id":   "<must match a preregistered promotion comparison>"
    }

    200 -> {
      "window_id": "...",
      "quota_remaining": <int>,
      "primary": {
        "semantic_mmd": <float>,
        "semantic_mmd_delta_vs_floor": <float>,
        "S": <float>
      },
      "authorship_auc": <float>, "authorship_auc_ci": [lo, hi],
      "gates": {"...": true|false},
      "verdict": "confirm" | "reject" | "inconclusive",
      "aggregate_only": true
    }

    429 -> quota exhausted for the window
    409 -> train_embedder_id collides with the sealed embedder (independence
           violation) -> rejected without scoring

## Anti-overfit controls (adaptive-overfitting defense)

- Weekly submission quota per comparison_id (e.g. <= 5). Every consumed
  submission is logged with its hash.
- Aggregate output only. No per-example scores, no gradients, no nearest
  human neighbors -- those leak the test set one query at a time.
- Optional: add calibrated noise to reported scores near the decision
  threshold, or rotate a fresh test subsample per window, to blunt
  hill-climbing on the hidden set.
- Reject resubmission of an identical checkpoint_hash within a window.

## What it never does

- Never returns text, embeddings, or the test docs.
- Never accepts a request to change metric definitions.
- Never runs external commercial detectors (that is Tier 3, human-only).

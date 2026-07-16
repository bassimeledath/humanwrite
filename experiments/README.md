# experiments/ -- TIER 0 agent workspace

Everything here is yours: training loops, reward code (discriminators, MMD
coefficients, unlikelihood penalties), rollout scripts, analysis notebooks.

Rules:
- Reward code may share math with harness/ but is a separate copy. Improving
  your reward is progress; improving your harness score via reward-metric
  coupling is Goodharting -- that is why promotion requires Tier 2.
- Structure: experiments/<comparison_id>/<run_id>/ with config.yaml, code,
  and a pointer to ledger entries. Commit everything; diffs are docs.
- Rollout generation at the FROZEN deployment sampler unless the experiment
  is explicitly about sampling.

Offline M0 entrypoints:
- `python -m experiments.runner --config configs/m0_offline_smoke.yaml --run-id <run_id>`
- `python -m experiments.verify_m0`

# infra/gpu -- the ONLY route to compute or money

A thin CLI the agent calls. It holds provider credentials (Modal recommended);
the agent's environment has none. Recommended backend: Modal, because managed
teardown removes the "agent left an H100 running overnight" failure mode. Keep
the reaper anyway; defense in depth.

## Commands

    infra/gpu submit --config <yaml> --budget-class <smoke|screen|promo>
    infra/gpu status <run_id>
    infra/gpu logs   <run_id> [--tail N]
    infra/gpu cancel <run_id>
    infra/gpu budget            # remaining $ across GPU + API pools

## submit MUST enforce, before spending a cent

1. Preregistration: config.run.comparison_id exists in the ledger with an
   OPEN preregistration entry. Reject otherwise. (This is what makes "register
   before launch" real instead of an honor-system rule in CLAUDE.md.)
2. Budget class limits (hard-kill wall time + max GPUs):
   smoke  <= 20 min,   1 GPU
   screen <= 2 GPU-h,  1 GPU
   promo  <= 8 GPU-h,  1 GPU
   scale-up: NOT launchable by the agent. Requires a human-set env flag
   (DFTR_ALLOW_SCALEUP=1 with an approved comparison_id).
3. Global spend cap: if budget() would go negative for the run's worst case,
   reject.
4. Single-GPU rule: reject multi-GPU requests unless human-flagged.

## submit side effects

- Create the ledger run entry (status=launched) BEFORE the job starts, with
  config hash, git SHA, budget class, requested wall/accel budget.
- Attach immutable container image digest + data split hashes to the entry.
- On completion: upload checkpoint to the artifact store, write final metrics
  pointer, record actual accelerator-seconds, generated tokens, and cost --
  EVEN on failure (failed runs cost money and are data).
- Notify (email/Slack) on every promo-class launch.

## The reaper (separate cron, independent of the agent and the wrapper)

Runs every ~5 min. Kills any job exceeding its budget-class wall clock or the
global cap, regardless of what the agent or wrapper believe. Notifies on every
kill. An agent-controlled cleanup path is not trusted to stop runaway spend.

## Credentials

Modal token / cloud keys live in the wrapper's deployment environment only.
Never printed to logs, never passed through config.yaml, never in the agent's
shell. If a run needs the HF token (gated embedder / rate limits), the wrapper
injects it into the job env; the agent never sees it.

# ledger/ -- compute + preregistration registry

Two jobs:
1. PREREGISTRATION: every comparison must be registered (hypothesis, arms,
   the contrast it belongs to) BEFORE any run. `infra/gpu submit` refuses
   configs whose comparison_id has no open preregistration. This is the
   enforcement that makes the CLAUDE.md protocol real.
2. RUN REGISTRY: one row per launched run (config hash, git SHA, budget
   class, seed, actual accel-seconds, tokens, cost, status, metrics pointer).

Git commits (swarmy-style `[dftr] ...`) are the scientific narrative;
ledger.jsonl is the machine registry the wrapper and reaper read. Keep both.
Back up ledger.jsonl on every write.

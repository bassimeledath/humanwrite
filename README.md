# DFT-R: replicating Distribution Fine Tuning

Research scaffold for autonomously exploring post-training methods that match
an LLM's output distribution to human web text (replicating Rosmine's DFT).
The autonomous researcher is Claude Code; its constitution is CLAUDE.md.

## Layout

    CLAUDE.md            agent constitution: mission, tiers, protocol
    RESEARCH_CONTEXT.md  verified facts, hypotheses, source checklist
    FINDINGS.md          append-only research narrative (agent-written)
    sources/             archived primary sources (human-populated)
    harness/             TIER 1 dev evaluation package (immutable to agent)
    sealed_evaluator/    TIER 2 public API contract only
    infra/               constrained Modal gateway, GPU client, and reaper
    ledger/              append-only compute/run registry and CLI
    data/                data pipeline spec (M0)
    configs/             experiment config examples
    experiments/         TIER 0 agent workspace (reward code, training code)

## Human setup status

Completed in this repository:

1. Every source in RESEARCH_CONTEXT.md is archived with SHA-256 checksums.
2. The Tier 1 harness is implemented and preregistered. Calibration remains
   intentionally unset until M1 runs `harness calibrate`.
3. The sealed evaluator is implemented in the separate private repository
   `bassimeledath/humanwrite-sealed-evaluator`; only its API contract is here.
4. `infra/gpu`, the constrained Modal gateway, separate reaper, checkpoint
   volume wiring, fixed-code brief synthesis, and fixed-code quality judge are
   implemented with internal $200 monthly GPU and $100 monthly API caps.

Remaining human-only deployment steps:

1. Rotate any credentials ever pasted into chat, authenticate Modal locally,
   and create the secrets listed in `infra/backend/README.md`.
2. Set provider-side hard caps of $200 in Modal and $100 in OpenRouter, with
   automatic top-ups disabled where applicable.
3. Configure `DFTR_ALERT_WEBHOOK_URL`, then deploy the gateway, reaper, and
   private sealed evaluator. Export only their capability URLs/tokens to the
   research-agent environment.
4. Obtain a GPTZero key (and optionally Pangram) for human-only Tier 3.
5. Decide your milestone-review cadence, then kick off Claude Code with:
   "Read CLAUDE.md and RESEARCH_CONTEXT.md, propose your M0 plan, then
   execute."

## Security model, honestly stated

.claude/settings.json denies edits to harness/, sources/, and the ledger
spec, and a PreToolUse hook guards Bash. Per Anthropic's own docs these
filters are best-effort, so they are DEFENSE IN DEPTH, not the boundary.
The boundary is environmental: no provider credentials, no test data, and no
sealed-evaluator code exist in the agent's environment. Anything the agent
can reach, assume it will eventually optimize against.

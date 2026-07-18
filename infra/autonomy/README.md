# Humanwrite autonomy coordinator

`coordinator.py` is a deterministic, event-driven bridge between asynchronous
Modal jobs and bounded Codex continuation turns.

- It polls the constrained gateway every ten minutes using ordinary HTTP.
- Polling consumes no model tokens and never invokes a provider model.
- It invokes Codex once when a monitored run first becomes terminal.
- A separate 90-minute LaunchAgent invokes one real Codex safety audit when no
  continuation has run during the preceding hour. This catches missed
  handoffs, stale monitor targets, and silent failures.
- Background continuations use GPT-5.4 with high reasoning because the current
  unattended Codex CLI rejects GPT-5.6; interactive research model selection
  is unaffected.
- A persisted transition signature prevents repeated invocations.
- At most eight Codex continuations may start in any rolling 24-hour period
  while the bounded 4K/16K ladder is active.
- Each continuation must replace the monitored run list and increment the
  generation in `progress/autonomy.json` before exiting.
- Tier 3 detectors remain human-triggered and are outside this coordinator.

Runtime logs and locks live in the gitignored `.operator/autonomy/` directory.
The macOS LaunchAgents are installed as `com.humanwrite.autonomy` for the
zero-token ten-minute watcher and `com.humanwrite.autonomy-audit` for the
model-backed 90-minute safety audit.

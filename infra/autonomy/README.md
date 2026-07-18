# Humanwrite autonomy coordinator

`coordinator.py` is a deterministic, event-driven bridge between asynchronous
Modal jobs and bounded Codex continuation turns.

- It polls the constrained gateway every ten minutes using ordinary HTTP.
- Polling consumes no model tokens and never invokes a provider model.
- It invokes Codex once when a monitored run first becomes terminal.
- A persisted transition signature prevents repeated invocations.
- At most eight Codex continuations may start in any rolling 24-hour period
  while the bounded 4K/16K ladder is active.
- Each continuation must replace the monitored run list and increment the
  generation in `progress/autonomy.json` before exiting.
- Tier 3 detectors remain human-triggered and are outside this coordinator.

Runtime logs and locks live in the gitignored `.operator/autonomy/` directory.
The macOS LaunchAgent is installed as `com.humanwrite.autonomy`.

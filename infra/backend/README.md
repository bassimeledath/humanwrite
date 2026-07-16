# Constrained Modal backend

This backend is the credential boundary behind `infra/gpu`. The research
shell receives only a capability token for these budgeted routes. Modal,
Hugging Face, OpenRouter, and notification credentials stay in Modal secrets.

## Security properties

- Both client and server enforce preregistration, budget class, timeout,
  single-GPU, command allowlist, and 14B approval rules.
- The gateway serializes submissions and reserves worst-case cost before
  spawning a job.
- GPU and API pools each have an internal monthly cap of $100. The Modal
  dashboard workspace/environment budget is the authoritative hard cap.
- The worker uses the HF token only to populate the model cache, removes all
  provider keys, and starts experiment code with a small allowlisted env.
- The independent reaper checks every five minutes and can cancel a Modal
  function call even if the gateway or job is unhealthy.
- Alerts are emitted for every promo launch and reaper kill when
  `DFTR_ALERT_WEBHOOK_URL` is configured.

## One-time human deployment

Authenticate with a newly rotated token, never one pasted into chat:

```sh
modal token new --profile=bassimfaizal
modal profile activate bassimfaizal
```

Create the following secrets interactively or in the Modal dashboard:

- `humanwrite-gateway-auth`: `DFTR_GPU_GATEWAY_TOKEN`
- `humanwrite-provider-secrets`: `HF_TOKEN`, `OPENROUTER_API_KEY`, and a
  frozen `DFTR_OPENROUTER_MODEL` slug for brief synthesis plus a frozen
  `DFTR_JUDGE_MODEL` slug. The preregistered defaults are
  `openai/gpt-5-mini` for synthesis and `openai/gpt-5.4-mini` for the judge;
  changing either is a human-reviewed configuration change.
- `humanwrite-alerts`: `DFTR_ALERT_WEBHOOK_URL`
- `humanwrite-reaper-auth`: a distinct operational token or marker

Then deploy both apps:

```sh
modal deploy -m infra.backend.modal_app
modal deploy -m infra.backend.reaper
```

Copy the gateway URL from the deploy output and set only these two values in
the Claude Code launch environment:

```sh
export DFTR_GPU_GATEWAY_URL='https://...modal.run'
export DFTR_GPU_GATEWAY_TOKEN='...limited gateway token...'
export HARNESS_JUDGE_URL="$DFTR_GPU_GATEWAY_URL/judge"
export HARNESS_JUDGE_TOKEN="$DFTR_GPU_GATEWAY_TOKEN"
```

In the Modal dashboard, set the workspace monthly budget to $100 before any
run. In OpenRouter, set the provider credit limit to $100 and disable automatic
top-ups. The internal counters are defense in depth and do not replace those
provider-side hard limits.

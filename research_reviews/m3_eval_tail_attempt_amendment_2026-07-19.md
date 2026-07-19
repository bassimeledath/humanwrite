# M3 fresh-evaluation tail attempt amendment

Status: frozen before candidate-model generation or any SFT14/HUMANWRITE14
training or evaluation output exists.

## Trigger

Five complete executions of the frozen four-attempt constructor left 25 of the
224 prospectively selected evaluation identities without an accepted rewrite.
The accepted artifact remained hash-valid and the failures were produced by
the intended factual, protected-literal, normalized-identity, and surface-
divergence gates. The fifth execution accepted only two additional identities,
showing that restarting four-attempt workers discards useful per-record failure
feedback while producing diminishing returns.

## Bounded repair

For only the still-missing identities, version 2 raises `max_attempts` from 4
to 12 in one worker execution. Each record therefore carries the exact prior
validation error into later recovery prompts rather than losing it at a worker
boundary. Everything else is unchanged:

- the 224 prospectively selected identities and 640-row input SHA;
- the identity's originally assigned Gemini or Claude generator;
- independent Qwen3-32B verification;
- semantic similarity at least 0.90;
- normalized nonidentity and multi-provider surface similarity below 0.95;
- exact protected-literal, language, length, factual-support, and no-added-fact
  gates;
- output URI, tokenizer revision, concurrency, and USD 6 reservation.

No panel row is replaced, no provider is swapped, no accepted row is
regenerated, and no acceptance threshold changes. The gateway permits 12
attempts only for the exact arm name
`cross-provider-public-eval-input-tail-recovery-v2`; all other rewrite jobs
remain fixed at four attempts.

## Stop and next action

If the bounded 12-attempt pass remains partial, preserve the accepted artifact
and investigate the exact persistent failures before authorizing any further
constructor change. Do not select replacement evaluation identities based on
these outcomes.

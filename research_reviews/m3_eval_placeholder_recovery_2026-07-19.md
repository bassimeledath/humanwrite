# M3 evaluation placeholder recovery

Status: frozen after literal-inventory v3 ended at 221/224 and before launch.

The three remaining identities have observed, specific failure modes: two have
long quotation/title literals that the assigned Claude generator still altered
despite explicit byte-level inventory, and one repeatedly violated the frozen
token-length gate. No identity, provider, or acceptance threshold will change.

For only these missing identities, v4 deterministically replaces protected
literal spans in the generator-visible target with unique placeholder tokens.
The assigned generator must preserve each placeholder exactly; the worker then
restores the original literal bytes before independent Qwen verification and the
unchanged semantic, factual, literal, language, length, nonidentity, and
surface-divergence validators. The prompt also states the measured target token
count and exact validator-accepted range. Restoration cannot make a failing row
pass any gate except by preserving text that the scientific contract already
requires verbatim. No rejected candidate is selected or edited by outcome.

This is a single 12-attempt pass. If it remains partial, stop and inspect the
new exact failure rather than replaying or replacing panel identities.

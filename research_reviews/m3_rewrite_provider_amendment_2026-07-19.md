# M3 rewrite-source provider amendment

Status: frozen before accepting or generating any replacement-provider row.

The 128-record construction smoke produced 42 valid Claude Haiku 4.5 source
rows independently verified by Qwen3-32B. All 54 identities assigned to
Qwen3-32B as the source generator failed after the transport defects were
repaired. The observed terminal failures were substantive: source/target
length violations, completions shorter than the frozen 32-token floor,
missing protected numeric literals, or provider length exits. No Qwen-source
row entered the accepted artifact.

For the remaining 54 frozen identities only, replace Qwen3-32B's source-
generation role with `google/gemini-3.1-flash-lite`. Retain Qwen3-32B as the
independent verifier. The already accepted Claude/Qwen rows remain valid and
their deterministic assignments are unchanged. Generator templates, maximum
attempts, semantic threshold, protected-literal rules, language rules,
token-length bounds, source identities, output path, and downstream training
protocol remain unchanged.

This is a prospective task-construction amendment, not endpoint-driven model
selection: zero Qwen-generated rows passed into the corpus, and neither
training nor evaluation has begun. The amended smoke must still reach all 96
valid rows before the Qwen3-14B H100 mechanical smoke may launch.

## Zero-acceptance tail repair

Two resumed passes accepted 50 of the 54 replacement-slot records. Four
Gemini-assigned records returned the human target byte-for-byte on every
generation attempt, despite recovery prompts explicitly requiring a
meaning-preserving alternate draft. The deterministic validator rejected every
candidate as a no-op.

Before model training or evaluation, these four zero-acceptance fingerprints
were prospectively routed to the already-approved Claude Haiku generator.
Qwen3-32B remains the independent verifier, and every schema, literal-
preservation, language, token-length, semantic-similarity, and factual-
equivalence gate remains frozen. The 92 accepted rows are preserved byte-for-
byte. This repair changes only which approved generator constructs the four
missing rows; it does not select model-training outcomes.

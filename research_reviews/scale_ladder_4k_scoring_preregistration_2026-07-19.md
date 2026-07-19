# Scale-ladder 4K scoring binding

Status: frozen after candidate generation but before any automatic or judge
metric was computed. One output row was inspected only to verify schema. The
decision rule is inherited verbatim from the candidate-blind July 17 scale
ladder preregistration and is not selected from these outputs.

## Bound comparison

- Control: the completed 4,096-document matched SFT checkpoint, held-out output
  SHA `bf91a2418b681e892885f15f5cd05030534cab8c373e66fb1de89f7c8535bdac`.
- Treatment: the completed 4,096-document MMD-witness checkpoint, held-out
  output SHA `abedf64c30740b16d28aebaf8ad57d3241b9f91ff96d3c61f8d6e90257588efb`.
- Panel: the already-frozen 128-prompt scale-development panel, 256 unpaired
  distribution references, and two disjoint 128-document human floors.
- Complete machine-readable bindings and seeds:
  `configs/m2/m2_scale_ladder_4k_scoring_contract_v1.json`.

## Endpoints

1. BGE-small MMD-squared treatment-minus-control with a 9,999-draw paired
   prompt-swap test.
2. Llama-Embed-Nemotron-8B MMD-squared under the same design.
3. Qwen3-4B-tokenizer unigram L2 against the 256 human references.
4. Empty, replacement-character, unexpected-script, uniqueness, repetition,
   and byte-identity diagnostics.
5. Exactly 128 blinded GPT-5.4-mini comparisons each for human style and
   overall quality, with SHA-derived side randomization.

The training-only Qwen hidden-state representation is excluded from evaluation.
Human floors alone select the kernel bandwidths for each independent embedding
family.

## Frozen 4K decision

This is the diagnostic 4K gate already specified on July 17. Unlock the 16K
cell unless any of the following occurs:

1. non-finite values or artifact mismatch;
2. the MMD-witness output fails the frozen hard-validity rule;
3. overall judged quality is significantly worse at one-sided `p <= 0.05`; or
4. both embedding-family MMD effects move in the wrong direction and the
   human-style win rate is below 50%.

Opening this panel does not authorize hyperparameter, coefficient, sampler, or
checkpoint selection. The 16K cell has its own stronger safety-and-signal gate
before any 46K work.

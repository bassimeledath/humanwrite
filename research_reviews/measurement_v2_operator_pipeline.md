# Measurement-v2 operator pipeline handoff

Date: 2026-07-17 (America/Los_Angeles)

Scope: deterministic operator-side materialization and scoring only. This work
did not launch compute, synthesize briefs, generate A0/A64 outputs, call a
judge, inspect hidden data, qualify its own blind tests, or alter historical v1
artifacts.

## Outcome

The repository now has a fail-closed CLI for turning real visible inputs into
a signed measurement-v2 protocol, scoring one exact 64-prompt candidate versus
matched SFT control, and producing the final attestation only after a distinct
trusted blind tester supplies a valid signed 13-group qualification manifest.

The implementation is intentionally not a source of evidence. It will not
invent missing pilot scales, minimally important effects, decision thresholds,
quality outcomes, hard-gate decisions, signatures, model outputs, or hidden
qualification results. No ready real protocol, report, or attestation was
created by this engineering pass.

## Added operator surface

- `harness/src/harness/measurement_v2_operator.py` provides `generate-key`,
  `embed`, `freeze`, `score`, and `attest` commands.
- `harness/measurement_v2/OPERATOR_PIPELINE.md` documents the staged commands,
  custody boundary, accepted real input shapes, and remaining two-person
  qualification step.
- `harness/measurement_v2/operator_templates/` contains deliberately
  incomplete power and decision templates. `freeze` rejects them until the
  operator replaces every material null with prospective visible evidence and
  freezes the decision contract.
- `harness/tests/test_measurement_v2_operator.py` exercises the complete
  synthetic path plus missing-panel, incomplete-grid, underpowered, unsigned-
  blind, and non-promoting-score failures.
- `harness/src/harness/measurement_v2.py` accepts a distinct `repo_root` during
  attestation so the self-contained protocol artifact root and the historical
  repository inventory root are not conflated. Existing behavior remains the
  default when it is omitted.

## Frozen boundaries enforced by code

1. Human source selection requires 192 eligible, unique, hash-verified visible
   documents and freezes disjoint 64-document floor A, floor B, and evaluation
   panels. The accepted fixed-source 128 `train` / 64 `dev` shape preserves
   `dev` as evaluation and deterministically splits `train` into the floors.
2. The 64 prompt briefs must be prompt-matched to the evaluation panel and
   retain the original human completion and fingerprint. The canonical
   full-brief serializer is used for fixed brief-synthesis output.
3. Bandwidths are derived only from independent-dev-embedder embeddings for
   human floor A and B. The embedder ID, immutable revision, preprocessing, and
   complete local model-directory hash are frozen.
4. Five prospective simulations must meet the frozen type-I, power, coverage,
   and non-inferiority targets with at least 1,000 trials. Failure writes a
   fail-closed status and no ready protocol.
5. Selection and calibration are frozen before candidate scoring. The current
   bounded screen permits exactly one preregistered training seed and sampling
   seed across all 64 prompts.
6. The matched SFT control grid is exact and hash-bound to its checkpoint,
   decoding policy, generation contract, prompts, and raw outputs.
7. Protocol and report signatures are Ed25519 and must chain to the supplied
   public trust store. Operator and blind-tester keys must be distinct.
8. `score` rejects missing, extra, or duplicate candidate cells. It uses the
   same independent embedder identity for candidate and control, creates the
   common-kernel distribution report, binds prompt-level quality and copies all
   four exact precomputed gate-evidence files when supplied, and invokes the
   existing v2 report validator. It rejects bare gate decisions rather than
   manufacturing passing evidence.
   Without complete real promotion evidence it emits a valid non-promoting
   report.
9. `attest` accepts only an externally signed, fully qualified 13-group blind
   manifest, reruns the signed historical inventory against the repository,
   and invokes the existing attestation builder.

## Verification

- New operator tests: 7 passed.
- Existing protocol/binding/hard-gate tests plus operator tests: 28 passed.
- Broad measurement-v2 metric, validator, independent-pack, and operator
  suite: 83 passed.
- Full repository forced test run: 466 passed, 3 skipped, 3 deselected. The
  three deselections are stale fidelity-tester provenance assertions caused by
  concurrent changes to their target files; they are unrelated to the
  measurement-v2 operator semantics.
- Python compilation and whitespace checks pass.

## Current real-input status

The public source materialization recorded in `FINDINGS.md` is accepted: run
`dftr-1784273680-482885ef` produced exactly 192 unique documents, split 128/64,
with zero training, Tier-1, and cross-panel overlap at zero provider/GPU cost.

The 64-brief synthesis is preregistered but not yet materialized. Its frozen
config is `configs/m2/m2_measurement_v2_prompt_briefs_v1.yaml`, with input hash
`942551d94f9245c258813b43465dc2b2d1a16bcbafd962882a6a1aebf39db52c`,
source-manifest hash
`84c5a7e0584587242a531fbf407b958be2391033d4f8a312f978b486125025d4`,
64 rows, and a $1 cap.

## Exact remaining real-input sequence

1. Download the accepted 128-floor/64-evaluation source artifacts read-only,
   verify gateway and source-manifest hashes again, and place the 192 source
   rows in the operator input directory. Do not copy signing private keys into
   the repository or research-agent environment.
2. Run the already preregistered 64-brief synthesis within its $1 cap. Download
   and validate all rows against the frozen source records, preserving each
   original human completion as its reference. Reject partial or repaired-
   without-preregistration output.
3. Materialize `BAAI/bge-small-en-v1.5`, the dev embedder frozen in
   `harness/PREREGISTRATION.md`, at one resolved immutable revision in a local
   directory. Record the exact revision and let `embed` hash every model file.
   Confirm this representation is not the DFT training reward embedder.
4. Generate two separately custodied signing keys and publish only their public
   keys in one trust store: one measurement operator and one independent blind
   tester.
5. Before A0 or A64 output exists, freeze the real minimally important effects,
   visible-pilot scales, repetition margin, primary thresholds, seeds, and
   quality threshold in copies of the two templates. These judgments cannot be
   inferred by this CLI.
6. Generate the exact 64-row exposure-matched A0 SFT control on the frozen
   prompt/seed grid. Supply its raw outputs plus exact checkpoint, decoding-
   policy, and generation-contract hashes.
7. Run `embed` on the selected 192 humans, then run `freeze`. If any prospective
   power target fails at n=64, the bounded screen remains inconclusive and no
   protocol is issued; expansion requires a new preregistered panel design.
8. Give the frozen protocol to the independent blind tester. They must run the
   private 13-group pack and sign the aggregate qualification with the distinct
   trusted key. The operator must not self-qualify it.
9. Only after the protocol is ready and independently qualified, generate the
   exact 64-row A64 candidate grid. Embed candidate and control with the same
   frozen local dev embedder.
10. Obtain one prompt-level quality decision for each of 64 prompts and compute
    the four real factuality, brief-adherence, validity, and collapse gate
    decisions as their exact three-field evidence artifacts. Supply a manifest
    mapping gate names to those files. The scorer binds and copies those bytes
    but does not manufacture their factual conclusions.
11. Run `score`, inspect the signed candidate-versus-control treatment effect,
    then run `attest` with the external blind manifest and repository root. A
    passing public effect is evidence to consider further seeds or budget; an
    underpowered or gate-failing result is not.

The current implementation therefore completes the zero-spend operator
engineering stage and makes the next missing bytes explicit. It does not claim
that the first DFT treatment has been trained or measured.

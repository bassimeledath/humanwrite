# Measurement-v2 operator pipeline

This pipeline materializes and scores prospective visible measurement evidence.
It does not make the checked-in placeholder candidate ready, synthesize human
documents, run a model, call a judge, qualify its own blind tests, or infer
missing decisions.

All commands fail closed with exit code 2. Use a new empty artifact directory
for each freeze attempt. Keep private keys outside the repository and outside
the research agent's environment.

## Required real input shapes

- `humans.jsonl`: exactly 192 recommended (at least 192 accepted). The CLI
  accepts the normalized `{document_id,text,eligible,eligibility_basis,
  exclusion_flags}` shape or the fixed source-materializer
  `{fingerprint,completion,source_revision,split}` shape after verifying the
  content fingerprint. A 128 `train` / 64 `dev` source preserves dev as
  `human_eval` and deterministically divides train into the two floor panels.
  Text bytes must be globally unique and visibly disjoint from
  training/dev/hidden data.
- `prompt_briefs.jsonl`: exactly 64 rows. The normalized shape has unique
  `prompt_id`, `full_brief`, prompt-matched `reference_text`, its SHA-256 in
  `reference_fingerprint`, and `split: quality_visible_human`. The fixed brief
  synthesis shape is also accepted: `fingerprint`/`completion` bind the
  reference and the eight full-brief fields are rendered through the canonical
  serializer. Prompt IDs must exactly equal the 64 human-eval IDs.
- `matched_control_raw.jsonl`: exactly 64 rows with the same prompt IDs and one
  frozen `{training_seed, sampling_seed}` cell plus nonempty `text`.
- human embedding bundle: produce from the selected 192 rows with the pinned
  local independent dev embedder. With a source larger than 192, reproduce the
  documented hash ranking first and embed exactly the selected IDs.
- power assumptions and decision contract: copy the templates and replace all
  nulls from prospective visible pilot evidence before candidate outputs exist.

## Stage 1: two-person trust store

```bash
python -m harness.measurement_v2_operator generate-key \
  --private-key /private/operator-key.json \
  --trusted-keys /private/trusted-keys.json \
  --key-id measurement-operator-v1

python -m harness.measurement_v2_operator generate-key \
  --private-key /private/blind-tester-key.json \
  --trusted-keys /private/trusted-keys.json \
  --key-id measurement-blind-tester-v1
```

The second command appends a distinct public key. Never copy either private
file into the artifact root.

## Stage 2: independent visible embeddings

The model must already exist at a local immutable directory. This command never
downloads a model and hashes the full regular-file tree.

```bash
python -m harness.measurement_v2_operator embed \
  --input-jsonl /operator/inputs/humans-selected-192.jsonl \
  --output /operator/inputs/human-embeddings.json \
  --model-path /operator/models/dev-embedder-at-frozen-revision \
  --model-id BAAI/bge-small-en-v1.5 \
  --model-revision REPLACE_WITH_IMMUTABLE_REVISION \
  --id-field document_id \
  --text-field text
```

## Stage 3: freeze protocol inputs

```bash
python -m harness.measurement_v2_operator freeze \
  --artifact-root /operator/evidence/measurement-v2-run-001 \
  --human-source /operator/inputs/humans-selected-192.jsonl \
  --prompt-briefs /operator/inputs/prompt-briefs-64.jsonl \
  --control-outputs /operator/inputs/matched-control-raw-64.jsonl \
  --human-embeddings /operator/inputs/human-embeddings.json \
  --power-assumptions /operator/inputs/power-assumptions.json \
  --decision-contract /operator/inputs/decision-contract.json \
  --dependency-lock harness/uv.lock \
  --metric-code harness/src/harness/measurement_v2.py \
  --private-key /private/operator-key.json \
  --trusted-keys /private/trusted-keys.json \
  --historical-inventory harness/measurement_v2/historical_v1_inventory.json \
  --repo-root . \
  --control-checkpoint-sha256 REPLACE \
  --decoding-policy-sha256 REPLACE \
  --generation-contract-sha256 REPLACE \
  --operator independent-measurement-operator \
  --reviewed-at 2026-07-17T00:00:00Z
```

Freeze writes a signed ready protocol only when the historical inventory,
panels, human-only bandwidths, exact control grid, calibration, selection rule,
and all five prospective power targets validate. It writes an unsigned
`blind_test_manifest_v2.candidate.json`; that file cannot attest anything.

## Stage 4: independent blind qualification

A separate tester runs the 13-group private synthetic pack, binds its fixture
hash, runtime, evaluator code hash, dependency lock and exact protocol hash,
sets every group to `pass`, asserts no sealed imitation, and signs with the
blind-tester private key. The operator pipeline only ingests this manifest; it
does not generate or self-sign a qualified result.

## Stage 5: score one exact candidate grid

Generate the 64 candidate outputs through the approved compute wrapper. Build a
second embedding JSONL source containing IDs `candidate:<prompt_id>` and
`control:<prompt_id>`, then run `embed` with the same model directory and
metadata used for humans.

Optional promotion inputs are:

- prompt-level quality JSONL with exactly one `winner` in
  `{candidate,human,tie}` for every prompt; and
- a JSON manifest with exactly the four gate names mapped to paths of their
  already-computed evidence files. Each evidence file must have the exact
  frozen three-field schema and a `pass` decision. The scorer copies and binds
  those bytes; it never creates a passing gate from a bare decision.

Without both, the report is valid but non-promoting.

```bash
python -m harness.measurement_v2_operator score \
  --artifact-root /operator/evidence/measurement-v2-run-001 \
  --candidate-outputs /operator/inputs/candidate-raw-64.jsonl \
  --score-embeddings /operator/inputs/candidate-control-embeddings.json \
  --candidate-checkpoint-sha256 REPLACE \
  --private-key /private/operator-key.json \
  --quality-results /operator/inputs/quality-winners-64.jsonl \
  --hard-gate-results /operator/inputs/hard-gates.json
```

## Stage 6: final attestation

```bash
python -m harness.measurement_v2_operator attest \
  --artifact-root /operator/evidence/measurement-v2-run-001 \
  --blind-manifest /operator/inputs/blind-qualified-signed.json \
  --repo-root . \
  --operator independent-measurement-operator \
  --attested-at 2026-07-17T00:00:00Z
```

The attestation reruns the signed historical inventory against `--repo-root`,
revalidates every protocol-bound byte, verifies the independent blind signature,
and rejects missing or duplicated blind groups.

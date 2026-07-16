# Offline M0 data pipeline

This package materializes the M0 local fixture corpus into deterministic
FineWeb-compatible brief records, split manifests, and split hashes without
network access or hidden-test completions.

## Entrypoint

```sh
python -m data.pipeline \
  --input data/fixtures/fineweb_fixture.jsonl \
  --output-dir data/artifacts/m0
```

## Outputs

- `cleaned_records.jsonl`: original + cleaned text with deterministic
  fingerprints.
- `brief_records_all.jsonl`: canonical schema-format records.
- `train_briefs.jsonl` and `dev_briefs.jsonl`: agent-visible splits only.
- `train_manifest.json`, `dev_manifest.json`, `split_hashes.json`: fixed
  manifests and hashes used by configs and ledger entries.
- `hidden_test_boundary.json`: metadata-only sealed-evaluator boundary;
  no completions or outlines are emitted for hidden test data.

## Determinism contract

- Deduplication happens on cleaned-text fingerprint before any augmentation.
- Split assignment and empty-outline selection are stable hash rankings over
  fingerprints.
- The checked-in fixture contains 8 unique records, so the exact 25% empty
  outline rule resolves to exactly 2 records.


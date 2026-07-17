# Prospective tokenizer-aware fidelity replay v3

The v2 run `dftr-1784271369-f403028f` failed closed before generation because
the adapter and original merged artifact do not have byte-identical
`tokenizer_config.json` files. This is a metadata difference, not evidence of
tokenization equivalence, so v2 remains failed and unchanged.

Replay v3 is a new prospective protocol and comparison:

- protocol: `dftr.adapter_merge_replay.v3`;
- comparison: `M2-adapter-merge-fidelity-replay-v3`;
- config: `configs/m2/m2_adapter_merge_fidelity_replay_v3.yaml`;
- tokenizer identity manifest:
  `configs/m2/manifests/m2_adapter_merge_tokenizer_identity_v3.json`.

## Frozen tokenizer identity

The manifest binds the adapter and original merge paths and their complete
seven-file tokenizer surfaces. Six files are exact serialization-byte matches:
`added_tokens.json`, `chat_template.jinja`, `merges.txt`,
`special_tokens_map.json`, `tokenizer.json`, and `vocab.json`.

The only permitted byte difference is `tokenizer_config.json`:

- adapter SHA-256:
  `443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0`;
- original merged SHA-256:
  `a32ee532e3437966f2b52bb0fe0e7c525234dc1034814718b0467d8104a09371`.

The exact JSON difference is four merged-only fields:

```json
{
  "max_length": 384,
  "stride": 0,
  "truncation_side": "right",
  "truncation_strategy": "longest_first"
}
```

There are no adapter-only or changed-value fields. Any additional difference,
field value change, hash substitution, reordered/expanded exception list, or
semantic-equivalence claim fails closed.

## Runtime authority

V3 does not infer behavioral equivalence from this metadata declaration. Before
diagnostics, the runtime still:

1. loads the adapter and merged tokenizers independently;
2. requires identical loaded chat-template text;
3. independently renders and tokenizes all frozen prompts under the canonical
   generation contract; and
4. requires exact equality of prompt attestations, token IDs, attention masks,
   lengths, truncation flags, and their hashes.

Only after that exact gate passes can deterministic diagnostics, archive replay,
or prospective adapter/merge comparisons run. A runtime tokenization mismatch
raises an error and produces no scientific interpretation.

## Launch binding

The worker independently hashes and parses both the v3 config and tokenizer
manifest. At artifact verification it also parses both real
`tokenizer_config.json` files and recomputes the adapter-only, merged-only, and
changed-value maps; the observed JSON difference must equal the manifest
exactly. Backend policy requires the exact v3 protocol, comparison, canonical
config hash, both v2 snapshot-identity bindings, both tokenizer-config hashes,
the one-file exception, exact shared-file identity, and the pre-diagnostic
runtime attestation. The local GPU client uses the same canonical backend launch
contract. V1 and v2 paths and byte identities are unchanged.

No v3 preregistration was created. No deployment, launch, provider call, or
spend was performed.

## Verification

The v3-only fail-closed pack covers exact config/manifest acceptance, runtime
JSON-difference recomputation, the four
JSON additions, protocol/comparison downgrades, both tokenizer-config hash
substitutions, exception-list expansion/removal, weakened shared-file identity,
weakened runtime attestation, manifest substitution, and preservation of v1/v2
bytes. It passes `15 passed`.

The forced focused replay/policy suite passes `203 passed, 2 deselected in
3.86s`. The forced repository-wide suite passes `388 passed, 2 deselected in
7.68s`. The only deselections are tester-owned assertions that fidelity
implementation files remain unchanged relative to their prior v2 target
commits; those scope assertions necessarily become stale when v3 is added. No
semantic test is deselected.

The frozen file identities after implementation are:

- v1 config SHA-256:
  `8015afd23f7d21953e0e7f0f1045db824a87377ec38de4c7c478b7455570ef4c`;
- v2 config SHA-256:
  `a5f0504dfdcfd12cfda5081e068919a603a395c7155a725bd9e7c13016ba1d8c`;
- v2 snapshot manifest SHA-256:
  `602cb05fed6fe3a0ecc1e37bc811ae5bb255c2624b57b051ae0744c7a0973b2c`;
- v3 config SHA-256:
  `71ac41a8cbf8eaa0fc4346e3c87cfa7c6e7ea196eeeb8797d0dba819a3d4405b`;
- v3 tokenizer manifest SHA-256:
  `54891d4320ee45db4f4ad08124c22b1696410b70210e63f0da5239e3958a7712`.

# Independent 4K preflight review reconciliation

Source review: `/Users/bassime/Downloads/feedback-claude-humanwrite.md`

Date: 2026-07-18

## Verdict

The review's `FIX-THEN-RUN` verdict is accepted. The original 4K configs were
not launched. Three launch-blocking defects were repaired before any witness
generation or fine-tuning: token-length semantics, runtime/compute feasibility,
and unverified hardcoded witness-generation/runtime assumptions.

## Finding disposition

| Review finding | Disposition | Resolution |
| --- | --- | --- |
| F1 provider-estimated target lengths | ACCEPT | Added a tokenizer-pinned normalization workflow. Every one of the 4,096 rows must bind `target_length` to the exact Qwen tokenizer count of its completion; the normalized artifact is new and hash-bound. |
| F2 L40S/120 minute feasibility and arm-specific resume | ACCEPT | Full arms use one H100 for at most 120 minutes. Resume materialization now supports either confirmation arm. A 64-step H100 MMD-witness timing smoke precedes the full arms. |
| F3 six safe-excerpt semantic mismatches | MODIFY | Exact length semantics are repaired. The six rows remain provenance-labeled and represent 0.15% of the corpus; the reviewer explicitly did not classify them as independently launch-blocking after length repair. Their residual limitation is recorded rather than silently treated as faithful full-text recovery. |
| F4 runtime not verified | ACCEPT | Worker dependencies are pinned and the witness workflow verifies observed torch, transformers, and PEFT versions before loading the model. |
| F5 representation support mismatch | ACCEPT | The prospective training representation is frozen to 128 tokens, matching the baseline witness generation horizon. |
| F6 128-token curriculum caveat | ACCEPT AS LIMITATION | This 4K cell tests a 128-token curriculum, not full-document imitation. No broader claim will be made from it. |
| F7 all-or-nothing witness and empty outputs | ACCEPT | Witness generation writes an incremental partial artifact, records all empty indices, and only publishes the canonical output/manifest after all 4,096 rows succeed. |
| F8 hardcoded batch/prompt bounds | ACCEPT | Generation batch size 8 and prompt maximum 1,024 are explicit, validated config fields and are recorded in the manifest. |
| F9 missing outcome reconciliation | ACCEPT | Prior 1,024-arm outcomes are backfilled in `FINDINGS.md`; each new comparison is preregistered before launch. |
| F10 serializer golden bytes | ACCEPT | Added a byte-exact serializer regression test. |
| F11 three awkward prompts | ACCEPT AS NONBLOCKING | Retained for the frozen 4K corpus; stricter prompt-quality validation is a future data revision, not an unregistered mutation of this cell. |
| F13 minibatch-relative witness weighting | ACCEPT AS METHOD LIMITATION | The behavior is explicit and unchanged so the 4K cell remains comparable to the 1,024 cell. Any global-weighting revision must be a separate comparison. |

## Timing-smoke modification

The reviewer proposed a 64-step smoke using the exact full config. A complete
4,096-row epoch requires 2,048 batch-of-two steps, so the frozen epoch scheduler
correctly rejects a 64-step partial epoch. The timing-only smoke therefore uses
the already-supported seeded sample-without-replacement schedule. The full SFT
and MMD-witness arms retain the complete one-epoch schedule.

The training loop was also made cheaper without changing the optimized loss:
gradient probes are skipped for structurally zero objective components and the
already-populated total gradient supplies the pre-clip total norm. This removes
redundant backward traversals while preserving the scientific logs.

## Launch gates

1. Exact-token normalization completes with 4,096 unique rows and no validation
   errors.
2. The pinned 4K baseline witness completes with no empty generations.
3. The 64-step MMD-witness timing smoke passes its existing first-64 gradient
   ratio gate and projects the full arm inside the 120-minute timeout with
   reasonable headroom.
4. Only then may matched full SFT and MMD-witness arms launch concurrently.

These repairs establish mechanical validity. They do not presuppose that the
MMD-witness method will improve human-writing proximity; that remains the
purpose of the matched tuning and fresh evaluation.

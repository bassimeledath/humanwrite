# M3 4K API rewrite recovery amendment

Status: frozen while the original v2 constructor remains active and before any
4K corpus materialization or training outcome exists. This config may launch
only after the current v2 pass ends partial.

An audit of 126 recent intended rejections found 39 normalized identities, 35
protected-literal mismatches, 16 token-length failures, 12 invalid non-noop
texts, 10 verifier failures, and 7 surface-similarity failures. The completed
evaluation constructor independently showed that explicit literal inventories
recovered 12 of 15 persistent hard cases without weakening validation.

For only the missing 4K rewrite identities, recovery v3 therefore increases the
within-worker feedback chain from 4 to 12 attempts, lists every deterministic
protected literal byte-for-byte, and states the measured target token count and
accepted validator range. It preserves the original identities, assigned Gemini
or Claude generator, independent Qwen verifier, output URI, semantic threshold,
surface-divergence requirement, and every factual, language, literal, length,
nonidentity, and validity gate. Existing accepted rows are immutable and skipped.

This recovery is a construction-efficiency repair, not a promotion result. It
must not launch concurrently with the original v2 worker.

# M3 evaluation literal-inventory contingency

Status: implemented and tested prospectively; launch is conditional on the
terminal outcome of final exact replay `dftr-1784477641-e4bb544b`.

Across the prior tail passes, persistent failures were concentrated in exact
protected-literal mismatches, with a smaller number of normalized identities or
invalid texts. The generator prompt required literal preservation by category
but did not enumerate the actual values extracted by the deterministic validator.

If and only if the capped exact replay remains partial, v3 may retry only the
missing identities while listing each validator-derived protected literal in the
prompt and requiring byte-for-byte inclusion. It retains the original identity,
assigned Gemini or Claude generator, independent Qwen verifier, output URI,
12-attempt bound, semantic threshold, surface-divergence rule, token limits, and
every hard acceptance gate. It does not post-process output, replace rows, swap
providers, or relax validation. Further stochastic v2 replay is forbidden.

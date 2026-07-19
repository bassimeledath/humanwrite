# M3 baseline-draft resume amendment

The first 14B baseline-draft construction run (`dftr-1784470064-bf4ab7d8`)
committed 333 unique, schema-valid records, then aborted when one of four sampled
candidates contained the Unicode replacement character. The original 120-minute
screen timeout was also too short for the observed throughput: the job reached
333 of 819 records in roughly 90 minutes.

Before any corpus materialization, training, or outcome evaluation, protocol v2
makes two operational repairs while preserving the frozen model, source records,
sampling parameters, four-candidate requirement, output URI, and deterministic
seed schedule:

1. Invalid empty or replacement-character candidates are skipped, with at most
   12 deterministic generation attempts to obtain the same four valid candidates.
2. The resumable job moves to the promo budget class with a 300-minute timeout.

Existing rows remain valid and are skipped by fingerprint. This amendment does
not select candidates using downstream metrics or alter the scientific training
arms. If four valid candidates cannot be obtained within 12 attempts, the run
still fails closed.

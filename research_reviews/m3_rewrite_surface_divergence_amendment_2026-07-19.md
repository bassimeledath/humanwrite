# M3 rewrite-construction surface-divergence amendment

Status: frozen after inspecting a partial construction artifact, before any 4K
corpus was assembled, before either 14B training arm, and before the fresh
evaluation panel or candidate outputs existed.

## Observed construct failure

Run `dftr-1784470057-7ab3e04f` produced 615 accepted API rewrite rows before
the audit. Among the 537 `multi_provider_ai` rows, 231 were identical to their
human target after whitespace normalization. The normalized character
SequenceMatcher distribution was strongly bimodal: median `0.99893`, with only
135/537 below `0.95`. An inspected accepted row differed primarily by joining
paragraphs and list lines. This satisfies the old byte-nonidentity rule but is
not a meaningful AI-paraphrase input and would silently inflate no-op training
exposure beyond the frozen 5% no-op stratum.

The independent base-Qwen constructor did not show this defect: its first 123
records contained four candidates each, mean 3.65 unique normalized candidates,
zero empty/replacement-character rows, and median candidate-target surface
similarity `0.650`.

## Prospective repair

The defective API run was cancelled at 552 log-confirmed accepted rows; none of
its rows may enter training. Version 2 uses a new output URI and protocol. It:

1. rejects every non-noop input identical to the target after whitespace and
   case normalization;
2. rejects `multi_provider_ai` inputs with normalized character
   SequenceMatcher similarity at or above `0.95`;
3. explicitly instructs generators to recast sentence structure, paragraph
   flow, transitions, and wording, and forbids formatting-, whitespace-, or
   punctuation-only variants; and
4. leaves all factual, literal, language, token-length, provider-balance,
   independent-Qwen verification, and semantic-similarity gates unchanged.

The `controlled_light_edit` stratum is exempt from the `0.95` ceiling because
its prospectively defined purpose is a restrained edit, but it must still
differ after normalization. The dedicated `already_human_noop` stratum remains
the only identity path.

This is a construct-validity repair, not outcome-driven method selection: no
training, model output, writing-quality metric, or promotion endpoint existed
when the rule was frozen. The old artifact remains provenance evidence but is
excluded from all scientific corpora.

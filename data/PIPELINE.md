# data/ -- pipeline spec (M0 deliverable)

Goal: turn a FineWeb subset into (structured brief -> full human document)
training pairs matching Rosmine's disclosed schema, with fixed, hashed splits.
Start at 20-30K docs, not the full 185K.

## Steps

1. PULL FineWeb subset (HF `datasets`, streaming). No HF token strictly
   required (public), but keep one in env for rate limits / gated embedders.
2. CLEAN. Rosmine notes many FineWeb samples have unsuitable spans. Strip
   boilerplate, nav junk, truncated docs, non-target-script noise. Record the
   original + cleaned text and a fingerprint per doc.
3. DEDUPE + SPLIT FIRST, before any augmentation, so no doc leaks across
   train/dev/test. Persist: fineweb id, url/domain, fingerprint, split.
   Hash each split (sha256 of sorted fingerprints) -> these hashes go in every
   experiment config and ledger entry.
4. BRIEF SYNTHESIS (the (brief -> doc) target). Per doc, an LLM produces:
   - use_case, style, style_kind        (schema fields)
   - outline: sections, each with supported_facts + quotations
   - target_length (tokens)
   - em_dashes_allowed flag
   - user_prompt (reverse-prompt: the request this doc would answer)
   Reproduce the 25% EMPTY-OUTLINE condition (blog Appendix 2).
   Keep strict vs creative distinguishable: for strict examples every outline
   fact must trace to a source span.
5. EMIT canonical JSONL records (schema below).

## Record schema

    {
      "fineweb_id": "...", "domain": "...", "fingerprint": "...",
      "split": "train|dev|test",
      "generation_mode": "generate|rewrite",
      "use_case": "company_blog|news|essay|...",
      "style_kind": "professional|...", "style": "direct, restrained",
      "detail_mode": "strict|creative",
      "target_length": 1200, "em_dashes_allowed": false,
      "user_prompt": "...",
      "outline": [ {"section": "...", "supported_facts": ["..."],
                    "quotations": ["..."]} ],     // may be [] for 25%
      "completion": "<the cleaned human document>"
    }

## Split discipline (the test wall starts here)

- train/dev live in the agent's environment.
- TEST is NOT built here into the agent's tree. The sealed evaluator owns a
  separate, differently-sliced test split (different domain/time). Do not
  emit test completions anywhere the agent can read.

## Cost note

Brief synthesis is ~1 LLM call/doc = 20-50K calls. Provision an
Anthropic/OpenRouter key with a spend cap; batch it; the wrapper (not the
agent) holds the key.

"""Create a compact, deterministic packet for an independent pre-training audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess


FINAL_RECOVERY_IDS = {
    "0baba5ef45bb01df12300da4eff210a59d6fc96ff3dc2873141ce5fbd0e5e61b",
    "10d3ee2bfe5aed97b07e76eccf80401dd2e5f938ec0274f613cb39845e3c9298",
    "256cd03f0fbda71a11ad04df2427a51d93a8e895a3eeacd88f27873ab77bedf8",
    "2d06e1099d955493a452907e621a3be511b5c7f45ab211aa363dd4dfeb4ffe4c",
    "369ac80343911be0472c903b1fc309bf0314ebec6a69d297b8c75f45f6e8025b",
    "3769e24971f71359659a3fa4f067da38d984f5b2ecfcfc7994c9254e5633d622",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    bundle = args.bundle_dir.resolve()
    brief_path = bundle / "train-briefs-current.jsonl"
    source_path = bundle / "clean-train-4096.jsonl"
    briefs = read_jsonl(brief_path)
    sources = read_jsonl(source_path)
    if len(briefs) != 4096 or len(sources) != 4096:
        raise ValueError("review bundle requires exact 4,096-row artifacts")
    source_by_id = {str(row["fingerprint"]): row for row in sources}
    if len(source_by_id) != 4096:
        raise ValueError("source fingerprints are not unique")

    selected: dict[str, set[str]] = {}
    def choose(row: dict, reason: str) -> None:
        selected.setdefault(str(row["fingerprint"]), set()).add(reason)

    rng = random.Random(20260718)
    for row in rng.sample(briefs, 20):
        choose(row, "deterministic_random")
    for row in briefs:
        if str(row["fingerprint"]) in FINAL_RECOVERY_IDS:
            choose(row, "final_safe_excerpt_recovery")
    for row in sorted(briefs, key=lambda item: int(item["target_length"]))[:3]:
        choose(row, "lowest_target_length")
    for row in sorted(briefs, key=lambda item: int(item["target_length"]), reverse=True)[:3]:
        choose(row, "highest_target_length")
    for row in sorted(briefs, key=lambda item: len(item["outline"]), reverse=True)[:3]:
        choose(row, "largest_outline")

    sample_rows = []
    for brief in briefs:
        fingerprint = str(brief["fingerprint"])
        if fingerprint not in selected:
            continue
        sample_rows.append({
            "selection_reasons": sorted(selected[fingerprint]),
            "brief": brief,
            "clean_source": source_by_id[fingerprint],
        })
    sample_path = bundle / "review-samples.jsonl"
    sample_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in sample_rows),
        encoding="utf-8",
    )

    exact_fact_failures = 0
    source_completion_mismatches = 0
    empty_outlines = 0
    ratios = []
    for brief in briefs:
        source = source_by_id.get(str(brief["fingerprint"]))
        if source is None or brief["completion"] != source["completion"]:
            source_completion_mismatches += 1
        if not brief["outline"]:
            empty_outlines += 1
        for section in brief["outline"]:
            for fact in section["supported_facts"] + section["quotations"]:
                if fact not in brief["completion"]:
                    exact_fact_failures += 1
        ratios.append(float(brief["target_length"]) / max(1, int(brief["word_count"])))
    summary = {
        "brief_rows": len(briefs),
        "source_rows": len(sources),
        "brief_sha256": sha256(brief_path),
        "source_sha256": sha256(source_path),
        "sample_rows": len(sample_rows),
        "final_recovery_rows_found": sum(str(row["fingerprint"]) in FINAL_RECOVERY_IDS for row in briefs),
        "source_completion_mismatches": source_completion_mismatches,
        "exact_fact_substring_failures": exact_fact_failures,
        "empty_outline_rows": empty_outlines,
        "target_tokens_per_word": {
            "min": min(ratios),
            "mean": sum(ratios) / len(ratios),
            "max": max(ratios),
        },
    }
    (bundle / "mechanical-audit-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    prompt = f"""# Independent pre-training audit request

Act as an adversarial ML systems and experimental-design reviewer. Do not defend the current implementation. The sole product objective is to determine whether this project can fine-tune Qwen3-4B to write measurably closer to high-quality human prose than its matched SFT control.

Fine-tuning is deliberately PAUSED. No baseline-witness GPU job and no 4B training arm has launched. Audit checkout `{repo}` at commit `{commit}` plus the downloaded artifacts in `{bundle}`.

## Required code review

Read these files completely and trace every hardcoded value to either a frozen scientific contract, an artifact property, or an unjustified assumption:

- `infra/backend/modal_app.py` (worker image versions and final safe-excerpt brief recovery)
- `data/lower_variance_briefs.py` (brief schemas and validators)
- `experiments/m2/scale_ladder_witness.py` (new 4K baseline generation)
- `experiments/m2/materialize_scale_ladder_witness_config.py`
- `experiments/m2/materialize_scale_ladder_training_configs.py`
- `experiments/m2/lower_variance_train.py`
- `experiments/m2/representation.py`
- `experiments/runner.py`
- `infra/backend/policy.py` and `infra/gpu`
- `configs/m2/m2_confirmation_4b_sft_v1.yaml` (template inherited by the 4K materializer)
- `research_reviews/independent_reviews_reconciliation_2026-07-17.md`
- `FINDINGS.md` from line 2374 onward

At minimum, verify model/revision, source adapter and hashes, 4,096 cardinality, 128-token horizon, EOS behavior, prompt serialization, tokenizer semantics, seed behavior, batching, one-epoch arithmetic (2,048 steps x batch 2), checkpoint cadence, LR/weight decay/clipping, representation model/layer/pooling/max tokens, MMD bandwidths and temperature, exact runtime package enforcement, Modal timeout, resume behavior, and whether SFT and MMD arms are genuinely matched.

Look specifically for duplicated constants that can drift, configuration fields that are validated but ignored, code constants that bypass configuration, declared runtime versions that differ from the worker image, wrong hash/cardinality bindings, leakage between train/dev/eval, post-EOS or target-length-unit errors, missing checkpoint/resume behavior, and failure modes that could silently produce a plausible but scientifically invalid run.

## Required data review

- Full cleaned sources: `{bundle / 'clean-train-4096.jsonl'}`
- Full assembled briefs: `{bundle / 'train-briefs-current.jsonl'}`
- Deterministic random/extreme/recovery sample: `{sample_path}`
- Mechanical checks: `{bundle / 'mechanical-audit-summary.json'}`

The six rows tagged `final_safe_excerpt_recovery` deserve special scrutiny: their provider prompts used a short verbatim excerpt after full-page content filters. Confirm that the final brief remains grounded in the full completion and assess whether target length, style, use case, or topic conditioning became biased. Also sample the rest independently; do not rely only on the supplied sample.

## Output format

1. Executive verdict: BLOCK / FIX-THEN-RUN / READY.
2. Findings ranked by severity, each labeled FACT / INFERENCE / SPECULATION with exact file:line or fingerprint evidence.
3. A table of every material hardcoded value: location, actual consumer, intended source of truth, whether checked, and recommended repair.
4. Data-quality results, including at least 30 independently sampled rows plus all six recovery rows.
5. Recompute the expected training exposure and rough time/cost; identify any timeout or checkpoint gap.
6. Exact patches/tests required before launch.
7. A short preflight checklist whose every item can be mechanically verified.

Do not recommend launching merely because unit tests pass. The verdict should depend on whether the intended experiment is actually encoded and whether failures would be loud rather than silent.
"""
    (bundle / "INDEPENDENT_REVIEW_PROMPT.md").write_text(prompt, encoding="utf-8")


if __name__ == "__main__":
    main()

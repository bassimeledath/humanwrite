"""Automatic, detector-free scoring for the fresh M3 rewrite panel."""

from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np

from data.m3_eval_panel import EVAL_PANEL_PROTOCOL
from data.rewrite_tasks import contains_unexpected_non_latin
from experiments.m1.contracts import write_json


class M3AutomaticScoreError(ValueError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bound(path: Path, sha256: str) -> list[dict[str, Any]]:
    if not re.fullmatch(r"[0-9a-f]{64}", sha256) or not path.is_file() or file_sha256(path) != sha256:
        raise M3AutomaticScoreError(f"artifact hash mismatch: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalized_similarity(left: str, right: str) -> float:
    normalize = lambda text: re.sub(r"\s+", " ", text).strip().casefold()
    return SequenceMatcher(None, normalize(left), normalize(right), autojunk=False).ratio()


def hard_content_preserved(panel_row: dict[str, Any], output: str) -> bool:
    return (
        bool(output.strip())
        and "�" not in output
        and all(str(literal) in output for literal in panel_row.get("protected_literals") or [])
    )


def _ngrams(tokens: list[int], order: int) -> Iterable[tuple[int, ...]]:
    return zip(*(tokens[offset:] for offset in range(order)))


def ngram_distribution(tokenized: list[list[int]], order: int) -> dict[tuple[int, ...], float]:
    counts: Counter[tuple[int, ...]] = Counter()
    for tokens in tokenized:
        counts.update(_ngrams(tokens, order))
    total = sum(counts.values())
    if total <= 0:
        raise M3AutomaticScoreError("empty n-gram distribution")
    return {key: value / total for key, value in counts.items()}


def distribution_l2(left: dict[Any, float], right: dict[Any, float]) -> float:
    return math.sqrt(sum((left.get(key, 0.0) - right.get(key, 0.0)) ** 2 for key in left.keys() | right.keys()))


def human_split_scale(tokenized: list[list[int]], order: int, seed: int = 6201) -> dict[str, float]:
    if len(tokenized) != 256:
        raise M3AutomaticScoreError("human split scale requires 256 references")
    rng = np.random.default_rng(seed + order)
    values = []
    for _ in range(256):
        permutation = rng.permutation(256)
        left = ngram_distribution([tokenized[index] for index in permutation[:128]], order)
        right = ngram_distribution([tokenized[index] for index in permutation[128:]], order)
        values.append(distribution_l2(left, right))
    return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1))}


def proportion_interval(values: list[bool], seed: int = 6201) -> dict[str, float]:
    if not values:
        raise M3AutomaticScoreError("proportion interval received no values")
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed + len(values))
    indices = rng.integers(0, len(array), size=(10_000, len(array)))
    estimates = array[indices].mean(axis=1)
    return {
        "point": float(array.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "n": len(values),
    }


def align_outputs(panel: list[dict[str, Any]], outputs: list[dict[str, Any]], arm: str) -> list[str]:
    if len(outputs) != 256 or len({row.get("fingerprint") for row in outputs}) != 256:
        raise M3AutomaticScoreError(f"{arm} output cardinality mismatch")
    by_id = {str(row["fingerprint"]): row for row in outputs}
    result = []
    for panel_row in panel:
        row = by_id.get(str(panel_row["fingerprint"]))
        if row is None or row.get("arm") != arm or not str(row.get("output") or "").strip():
            raise M3AutomaticScoreError(f"{arm} output identity mismatch")
        result.append(str(row["output"]))
    return result


def score(
    panel: list[dict[str, Any]],
    base_outputs: list[dict[str, Any]],
    sft_outputs: list[dict[str, Any]],
    treatment_outputs: list[dict[str, Any]],
    tokenizer: Any,
) -> dict[str, Any]:
    if (
        len(panel) != 256
        or len({row.get("fingerprint") for row in panel}) != 256
        or any(row.get("artifact_schema") != EVAL_PANEL_PROTOCOL for row in panel)
    ):
        raise M3AutomaticScoreError("fresh panel invariants failed")
    aligned = {
        "BASE": align_outputs(panel, base_outputs, "BASE"),
        "SFT14": align_outputs(panel, sft_outputs, "SFT14"),
        "HUMANWRITE14": align_outputs(panel, treatment_outputs, "HUMANWRITE14"),
    }
    references = [str(row["human_reference"]) for row in panel]
    inputs = [str(row["input_text"]) for row in panel]
    ai_indices = [index for index, row in enumerate(panel) if row["category"] != "already_human_noop"]
    noop_indices = [index for index, row in enumerate(panel) if row["category"] == "already_human_noop"]
    arms: dict[str, Any] = {}
    for arm, texts in aligned.items():
        preserved = [hard_content_preserved(row, text) for row, text in zip(panel, texts)]
        similarities = [normalized_similarity(source, text) for source, text in zip(inputs, texts)]
        meaningful = [
            preserved[index]
            and similarities[index] < 0.98
            and re.sub(r"\s+", "", inputs[index]) != re.sub(r"\s+", "", texts[index])
            for index in ai_indices
        ]
        restraint = [preserved[index] and similarities[index] >= 0.90 for index in noop_indices]
        arms[arm] = {
            "hard_content_preservation": proportion_interval(preserved),
            "meaningful_edit_ai_inputs": proportion_interval(meaningful),
            "acceptable_noop_restraint": proportion_interval(restraint),
            "unexpected_non_latin": proportion_interval(
                [contains_unexpected_non_latin(text) for text in texts]
            ),
            "replacement_character_count": sum("�" in text for text in texts),
            "mean_input_output_similarity": float(np.mean(similarities)),
            "byte_identity_count": sum(left == right for left, right in zip(inputs, texts)),
        }
    tokenized = {
        "HUMAN": [tokenizer.encode(text, add_special_tokens=False) for text in references],
        **{
            arm: [tokenizer.encode(text, add_special_tokens=False) for text in texts]
            for arm, texts in aligned.items()
        },
    }
    lexical = {}
    for order in (1, 2, 3):
        human_distribution = ngram_distribution(tokenized["HUMAN"], order)
        scale = human_split_scale(tokenized["HUMAN"], order)
        distances = {
            arm: distribution_l2(ngram_distribution(tokenized[arm], order), human_distribution)
            for arm in aligned
        }
        lexical[f"token_{order}gram_l2"] = {
            "human_split": scale,
            "distance": distances,
            "treatment_minus_sft": distances["HUMANWRITE14"] - distances["SFT14"],
        }
    return {
        "artifact_schema": "humanwrite.m3.rewrite_automatic_score.v1",
        "records": 256,
        "ai_styled_records": len(ai_indices),
        "noop_records": len(noop_indices),
        "arms": arms,
        "lexical": lexical,
    }


def main() -> int:
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    for name in ("panel", "base", "sft", "treatment"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panel = load_bound(args.panel, args.panel_sha256)
    base = load_bound(args.base, args.base_sha256)
    sft = load_bound(args.sft, args.sft_sha256)
    treatment = load_bound(args.treatment, args.treatment_sha256)
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-14B",
        revision="40c069824f4251a91eefaf281ebe4c544efd3e18",
        trust_remote_code=True,
    )
    result = score(panel, base, sft, treatment, tokenizer)
    result["input_hashes"] = {
        "panel": args.panel_sha256,
        "base": args.base_sha256,
        "sft": args.sft_sha256,
        "treatment": args.treatment_sha256,
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Training-only metrics for Tier 0 experimentation.

These utilities intentionally live outside `harness/` and must never be cited
as evaluation evidence. They exist only to support training-time reward and
diagnostic calculations.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from math import exp, log
from typing import Any

import numpy as np


TRAINING_ONLY_NOTICE = "Tier 0 only: do not use these metrics as evaluation evidence."
WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
RBF_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in WORD_RE.findall(str(text))]


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_RE.split(str(text)) if part.strip()]


def _embed(texts: list[str], embedder: Any) -> np.ndarray:
    if hasattr(embedder, "encode"):
        try:
            values = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        except TypeError:
            values = embedder.encode(texts)
    elif callable(embedder):
        values = embedder(texts)
    else:
        raise TypeError("embedder must be callable or expose encode()")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != len(texts):
        raise ValueError("embedder returned an invalid shape")
    if not np.isfinite(array).all():
        raise ValueError("embedder returned non-finite values")
    return array


def _sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.maximum(
        np.sum(x * x, axis=1)[:, None]
        + np.sum(y * y, axis=1)[None, :]
        - 2.0 * x @ y.T,
        0.0,
    )


def _mmd_from_embeddings(x: np.ndarray, y: np.ndarray, scales: tuple[float, ...]) -> float:
    if len(x) < 2 or len(y) < 2:
        raise ValueError("unbiased MMD requires at least two samples per group")
    pooled = np.concatenate([x, y], axis=0)
    distances = _sq_dists(pooled, pooled)
    positive = distances[distances > 0]
    median = float(np.median(positive)) if positive.size else 1.0
    bandwidths = [max(median * scale * scale, np.finfo(float).eps) for scale in scales]

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        dists = _sq_dists(a, b)
        return sum(np.exp(-dists / (2.0 * bw)) for bw in bandwidths) / len(bandwidths)

    kxx = kernel(x, x)
    kyy = kernel(y, y)
    kxy = kernel(x, y)
    xx = (kxx.sum() - np.trace(kxx)) / (len(x) * (len(x) - 1))
    yy = (kyy.sum() - np.trace(kyy)) / (len(y) * (len(y) - 1))
    return float(xx + yy - 2.0 * kxy.mean())


def semantic_mmd(gen_texts: list[str], human_texts: list[str], embedder: Any) -> float:
    x = _embed(list(gen_texts), embedder)
    y = _embed(list(human_texts), embedder)
    if x.shape[1] != y.shape[1]:
        raise ValueError("generated and human embeddings have different dimensions")
    return _mmd_from_embeddings(x, y, RBF_SCALES)


def lexical_l2(gen_texts: list[str], human_texts: list[str], hash_dim: int = 2048) -> float:
    if hash_dim < 1:
        raise ValueError("hash_dim must be positive")

    def vectorize(texts: list[str]) -> np.ndarray:
        counts = np.zeros(hash_dim, dtype=np.float64)
        for text in texts:
            words = _tokens(text)
            for size in (1, 2, 3):
                for start in range(len(words) - size + 1):
                    feature = " ".join(words[start : start + size])
                    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                    counts[int.from_bytes(digest, "big") % hash_dim] += 1
        return counts / counts.sum() if counts.sum() else counts

    return float(np.linalg.norm(vectorize(gen_texts) - vectorize(human_texts)))


def _js_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / a.sum() if a.sum() else np.full_like(a, 1.0 / len(a))
    b = b / b.sum() if b.sum() else np.full_like(b, 1.0 / len(b))
    mid = (a + b) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_a = np.where(a > 0, a * np.log2(a / mid), 0.0).sum()
        kl_b = np.where(b > 0, b * np.log2(b / mid), 0.0).sum()
    return float(np.sqrt(max((kl_a + kl_b) / 2.0, 0.0)))


def _bucket_counts(values: list[int], edges: list[int]) -> np.ndarray:
    return np.bincount(np.digitize(values, edges), minlength=len(edges) + 1).astype(float)


def structural_distance(gen_texts: list[str], human_texts: list[str]) -> float:
    def structure(texts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        paragraph_lengths: list[int] = []
        sentence_lengths: list[int] = []
        openings = Counter()
        for text in texts:
            paragraphs = [part for part in re.split(r"\n\s*\n", str(text)) if part.strip()]
            paragraph_lengths.extend(len(_tokens(part)) for part in paragraphs)
            for sentence in _sentences(text):
                words = _tokens(sentence)
                if not words:
                    continue
                sentence_lengths.append(len(words))
                first = words[0]
                if first in {"i", "we", "you", "he", "she", "they", "it"}:
                    openings["pronoun"] += 1
                elif first in {"a", "an", "the"}:
                    openings["article"] += 1
                elif first in {"and", "but", "or", "so", "yet"}:
                    openings["conjunction"] += 1
                elif first in {"what", "when", "where", "who", "why", "how"}:
                    openings["question"] += 1
                elif first.isdigit():
                    openings["number"] += 1
                else:
                    openings["other"] += 1
        return (
            _bucket_counts(paragraph_lengths, [20, 50, 100, 200, 400]),
            _bucket_counts(sentence_lengths, [5, 10, 20, 30, 50]),
            np.array(
                [openings[key] for key in ("pronoun", "article", "conjunction", "question", "number", "other")],
                dtype=float,
            ),
        )

    generated = structure(list(gen_texts))
    human = structure(list(human_texts))
    return float(np.mean([_js_distance(a, b) for a, b in zip(generated, human)]))


def _facts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        facts: list[str] = []
        for key in ("supported_facts", "facts", "quotations"):
            facts.extend(_facts(value.get(key)))
        return facts
    facts = []
    for item in value:
        facts.extend(_facts(item))
    return facts


def _overlap(source: str, target: str) -> float:
    source_tokens = set(_tokens(source))
    target_tokens = set(_tokens(target))
    if not source_tokens:
        return 1.0
    return len(source_tokens & target_tokens) / len(source_tokens)


def outline_fact_recall(gen_texts: list[str], outlines: list[Any]) -> float:
    if len(gen_texts) != len(outlines):
        raise ValueError("generated texts and outlines must have equal length")
    required = [(text, fact) for text, outline in zip(gen_texts, outlines) for fact in _facts(outline)]
    if not required:
        return 1.0
    return sum(_overlap(fact, text) >= 0.8 for text, fact in required) / len(required)


def unsupported_claim_rate(gen_texts: list[str], outlines: list[Any]) -> float:
    if len(gen_texts) != len(outlines):
        raise ValueError("generated texts and outlines must have equal length")
    unsupported = 0
    total = 0
    for text, outline in zip(gen_texts, outlines):
        facts = _facts(outline)
        for sentence in _sentences(text):
            if len(_tokens(sentence)) < 4:
                continue
            total += 1
            supported = any(max(_overlap(sentence, fact), _overlap(fact, sentence)) >= 0.6 for fact in facts)
            unsupported += not supported
    return unsupported / total if total else 0.0


def non_target_script_char_rate(text: str) -> float:
    letters = [char for char in str(text) if char.isalpha()]
    if not letters:
        return 0.0
    latin = 0
    for char in letters:
        try:
            latin += "LATIN" in unicodedata.name(char)
        except ValueError:
            latin += 0
    return 1.0 - latin / len(letters)


def self_bleu(texts: list[str], max_order: int = 4) -> float:
    items = list(texts)
    if len(items) < 2:
        return 0.0
    return float(np.mean([sentence_bleu(items[index], items[:index] + items[index + 1 :], max_order=max_order) for index in range(len(items))]))


def sentence_bleu(hypothesis: str, references: list[str], max_order: int = 4) -> float:
    hypothesis_tokens = _tokens(hypothesis)
    references_tokens = [_tokens(reference) for reference in references]
    if not hypothesis_tokens or not references_tokens:
        return 0.0
    precisions = []
    effective_order = min(max_order, len(hypothesis_tokens))
    for order in range(1, effective_order + 1):
        hyp = Counter(tuple(hypothesis_tokens[i : i + order]) for i in range(len(hypothesis_tokens) - order + 1))
        max_ref = Counter()
        for tokens in references_tokens:
            ref = Counter(tuple(tokens[i : i + order]) for i in range(len(tokens) - order + 1))
            for ngram, count in ref.items():
                max_ref[ngram] = max(max_ref[ngram], count)
        clipped = sum(min(count, max_ref[ngram]) for ngram, count in hyp.items())
        total = sum(hyp.values())
        precisions.append((clipped + 1.0) / (total + 1.0))
    reference_length = min(
        (len(tokens) for tokens in references_tokens),
        key=lambda length: (abs(length - len(hypothesis_tokens)), length),
    )
    brevity = 1.0 if len(hypothesis_tokens) > reference_length else exp(1.0 - reference_length / len(hypothesis_tokens))
    return brevity * exp(sum(log(value) for value in precisions) / effective_order)


def repeated_sentence_start_rate(texts: list[str], run_length: int = 3) -> float:
    items = list(texts)
    if not items:
        return 0.0
    positives = 0
    for text in items:
        first_words = []
        for sentence in _sentences(text):
            tokens = _tokens(sentence)
            if tokens:
                first_words.append(tokens[0])
        positives += any(len(set(first_words[index : index + run_length])) == 1 for index in range(len(first_words) - run_length + 1))
    return positives / len(items)


def length_stats(texts: list[str], targets: list[int] | None = None) -> dict[str, float]:
    lengths = [len(_tokens(text)) for text in texts]
    result = {
        "mean_tokens": float(np.mean(lengths)) if lengths else 0.0,
        "median_tokens": float(np.median(lengths)) if lengths else 0.0,
        "min_tokens": float(min(lengths)) if lengths else 0.0,
        "max_tokens": float(max(lengths)) if lengths else 0.0,
    }
    if targets is not None:
        if len(lengths) != len(targets):
            raise ValueError("targets must match text count")
        errors = [abs(length - target) for length, target in zip(lengths, targets)]
        result["mean_abs_target_error"] = float(np.mean(errors)) if errors else 0.0
    return result


def collapse_diagnostics(texts: list[str], non_target_limit: float = 0.15) -> dict[str, float | bool]:
    bleu = self_bleu(texts)
    repetition = repeated_sentence_start_rate(texts)
    script_rate = float(np.mean([non_target_script_char_rate(text) for text in texts])) if texts else 0.0
    return {
        "self_bleu": bleu,
        "repeated_sentence_start_rate": repetition,
        "non_target_script_char_rate": script_rate,
        "pass": bleu <= 0.95 and repetition <= 0.9 and script_rate <= non_target_limit,
    }


def distribution_gap(
    gen_texts: list[str],
    human_texts: list[str],
    embedder: Any,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, float]:
    semantic = semantic_mmd(gen_texts, human_texts, embedder)
    lexical = lexical_l2(gen_texts, human_texts)
    structural = structural_distance(gen_texts, human_texts)
    weighted = (
        weights[0] * semantic
        + weights[1] * lexical
        + weights[2] * structural
    )
    return {
        "semantic_mmd": semantic,
        "lexical_l2": lexical,
        "structural_distance": structural,
        "weighted_gap": float(weighted),
    }


def batch_diagnostics(
    gen_texts: list[str],
    human_texts: list[str],
    outlines: list[Any],
    embedder: Any,
    targets: list[int] | None = None,
) -> dict[str, Any]:
    distribution = distribution_gap(gen_texts, human_texts, embedder)
    collapse = collapse_diagnostics(gen_texts)
    return {
        "notice": TRAINING_ONLY_NOTICE,
        "distribution": distribution,
        "validity": {
            "outline_fact_recall": outline_fact_recall(gen_texts, outlines),
            "unsupported_claim_rate": unsupported_claim_rate(gen_texts, outlines),
        },
        "diversity": {
            "self_bleu": self_bleu(gen_texts),
            "repeated_sentence_start_rate": repeated_sentence_start_rate(gen_texts),
        },
        "length": length_stats(gen_texts, targets=targets),
        "collapse": collapse,
    }


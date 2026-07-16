"""Frozen distributional metrics used by the Tier 1 harness."""
from __future__ import annotations

import hashlib
import re
from collections import Counter

import numpy as np


_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_BANDWIDTHS = (0.25, 0.5, 1, 2, 4)


def _embed(texts, embedder):
    texts = list(texts)
    if not texts:
        raise ValueError("at least one text is required")
    if hasattr(embedder, "encode"):
        try:
            values = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        except TypeError:
            values = embedder.encode(texts)
    elif callable(embedder):
        values = embedder(texts)
    elif isinstance(embedder, str):
        # Lazy import keeps offline tests and non-semantic CLI commands free of
        # model initialization and network access.
        from sentence_transformers import SentenceTransformer

        values = SentenceTransformer(embedder).encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
    else:
        raise TypeError("embedder must be an id, callable, or expose encode()")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != len(texts):
        raise ValueError("embedder returned an invalid shape")
    if not np.isfinite(array).all():
        raise ValueError("embedder returned non-finite values")
    return array


def _resolve_embedder(embedder):
    if not isinstance(embedder, str):
        return embedder
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(embedder)


def _sq_dists(x, y):
    return np.maximum(
        np.sum(x * x, axis=1)[:, None]
        + np.sum(y * y, axis=1)[None, :]
        - 2.0 * x @ y.T,
        0.0,
    )


def _mmd_from_embeddings(x, y, bandwidth_scales=_BANDWIDTHS):
    if len(x) < 2 or len(y) < 2:
        raise ValueError("unbiased MMD requires at least two samples per group")
    pooled = np.concatenate([x, y], axis=0)
    distances = _sq_dists(pooled, pooled)
    positive = distances[distances > 0]
    # Distances are squared, so the usual median-heuristic sigma scaled by c
    # has variance median_squared_distance * c^2.
    median_squared_distance = float(np.median(positive)) if positive.size else 1.0
    bandwidths = [
        max(median_squared_distance * float(scale) ** 2, np.finfo(float).eps)
        for scale in bandwidth_scales
    ]

    def kernel(a, b):
        dists = _sq_dists(a, b)
        return sum(np.exp(-dists / (2.0 * bw)) for bw in bandwidths) / len(bandwidths)

    kxx, kyy, kxy = kernel(x, x), kernel(y, y), kernel(x, y)
    xx = (kxx.sum() - np.trace(kxx)) / (len(x) * (len(x) - 1))
    yy = (kyy.sum() - np.trace(kyy)) / (len(y) * (len(y) - 1))
    return float(xx + yy - 2.0 * kxy.mean())


def semantic_mmd(gen_texts, human_texts, embedder_id, bandwidth_scales=_BANDWIDTHS):
    """Return unbiased multi-bandwidth RBF MMD squared.

    ``embedder_id`` can be a SentenceTransformer id, an injected callable, or
    an object exposing ``encode``. The latter two forms make tests fully
    offline and let deployments own model lifecycle.
    """
    if not bandwidth_scales or any(float(scale) <= 0 for scale in bandwidth_scales):
        raise ValueError("bandwidth scales must be positive")
    embedder = _resolve_embedder(embedder_id)
    generated = _embed(gen_texts, embedder)
    human = _embed(human_texts, embedder)
    if generated.shape[1] != human.shape[1]:
        raise ValueError("generated and human embeddings have different dimensions")
    return _mmd_from_embeddings(generated, human, bandwidth_scales)


def human_floor_mmd(human_texts, embedder_id, n_boot=50):
    """Return ``(mean, (ci_low, ci_high))`` for independent human halves."""
    embeddings = _embed(human_texts, _resolve_embedder(embedder_id))
    if len(embeddings) < 4:
        raise ValueError("human floor requires at least four documents")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    rng = np.random.default_rng(0)
    shuffled = embeddings[rng.permutation(len(embeddings))]
    midpoint = len(shuffled) // 2
    left, right = shuffled[:midpoint], shuffled[midpoint : midpoint * 2]
    values = []
    for _ in range(n_boot):
        x = left[rng.integers(0, len(left), len(left))]
        y = right[rng.integers(0, len(right), len(right))]
        values.append(_mmd_from_embeddings(x, y))
    return float(np.mean(values)), (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


def lexical_l2(gen_texts, human_texts, ngram_feature_spec):
    """L2 distance between normalized frozen or deterministically hashed n-grams."""
    spec = ngram_feature_spec or {}
    if isinstance(spec, int):
        spec = {"hash_dim": spec}
    minimum, maximum = tuple(spec.get("ngram_range", (1, 3)))
    if minimum < 1 or maximum < minimum:
        raise ValueError("invalid ngram_range")
    explicit = spec.get("features")
    dimension = int(spec.get("hash_dim", 4096))
    if explicit is None and dimension < 1:
        raise ValueError("hash_dim must be positive")
    feature_index = {str(value).casefold(): i for i, value in enumerate(explicit or [])}

    def vector(texts):
        size = len(feature_index) if explicit is not None else dimension
        counts = np.zeros(size, dtype=np.float64)
        for text in texts:
            words = [word.casefold() for word in _WORD_RE.findall(str(text))]
            for n in range(minimum, maximum + 1):
                for start in range(len(words) - n + 1):
                    feature = " ".join(words[start : start + n])
                    if explicit is not None:
                        index = feature_index.get(feature)
                        if index is None:
                            continue
                    else:
                        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
                        index = int.from_bytes(digest, "big") % dimension
                    counts[index] += 1
        return counts / counts.sum() if counts.sum() else counts

    return float(np.linalg.norm(vector(gen_texts) - vector(human_texts)))


def _js_distance(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a = a / a.sum() if a.sum() else np.full_like(a, 1.0 / len(a))
    b = b / b.sum() if b.sum() else np.full_like(b, 1.0 / len(b))
    middle = (a + b) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_a = np.where(a > 0, a * np.log2(a / middle), 0.0).sum()
        kl_b = np.where(b > 0, b * np.log2(b / middle), 0.0).sum()
    return float(np.sqrt(max((kl_a + kl_b) / 2.0, 0.0)))


def _bucket_counts(values, edges):
    return np.bincount(np.digitize(values, edges), minlength=len(edges) + 1).astype(float)


def _structure(texts):
    paragraph_lengths, sentence_lengths = [], []
    openings = Counter()
    for text in texts:
        paragraphs = [part for part in re.split(r"\n\s*\n", str(text)) if part.strip()]
        paragraph_lengths.extend(len(_WORD_RE.findall(part)) for part in paragraphs)
        sentences = [part.strip() for part in _SENTENCE_RE.split(str(text)) if part.strip()]
        for sentence in sentences:
            words = [word.casefold() for word in _WORD_RE.findall(sentence)]
            if not words:
                continue
            sentence_lengths.append(len(words))
            first = words[0]
            if first in {"i", "we", "you", "he", "she", "they", "it"}:
                template = "pronoun"
            elif first in {"a", "an", "the"}:
                template = "article"
            elif first in {"and", "but", "or", "so", "yet"}:
                template = "conjunction"
            elif first in {"what", "when", "where", "who", "why", "how"}:
                template = "question"
            elif first.isdigit():
                template = "number"
            else:
                template = "other"
            openings[template] += 1
    return (
        _bucket_counts(paragraph_lengths, [20, 50, 100, 200, 400]),
        _bucket_counts(sentence_lengths, [5, 10, 20, 30, 50]),
        np.array(
            [openings[key] for key in ("pronoun", "article", "conjunction", "question", "number", "other")],
            dtype=float,
        ),
    )


def structural_distance(gen_texts, human_texts):
    """Mean Jensen-Shannon distance over the three frozen structures."""
    generated, human = _structure(gen_texts), _structure(human_texts)
    return float(np.mean([_js_distance(a, b) for a, b in zip(generated, human)]))

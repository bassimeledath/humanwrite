"""Pure M3 HUMANWRITE14 moment and witness objective helpers."""

from __future__ import annotations

import math
import re
from typing import Iterable

import numpy as np


MOMENT_COEFFICIENTS = (0.01, 0.03, 0.1, 0.3, 1.0)
SURFACE_FEATURE_NAMES = (
    "token_count",
    "sentence_count",
    "paragraph_count",
    "sentence_length_mean",
    "sentence_length_std",
    "type_token_ratio",
    "comma_rate",
    "semicolon_rate",
    "colon_rate",
    "parenthesis_rate",
    "emdash_rate",
    "newline_rate",
)
WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?])(?:[\"'”’)]*)\s+")


class M3ObjectiveError(ValueError):
    pass


def select_frequent_tokens(
    sequences: Iterable[Iterable[int]], *, special_ids: set[int], count: int = 256
) -> tuple[list[int], np.ndarray]:
    frequencies: dict[int, int] = {}
    total = 0
    for sequence in sequences:
        for token in sequence:
            value = int(token)
            if value in special_ids:
                continue
            frequencies[value] = frequencies.get(value, 0) + 1
            total += 1
    if total <= 0 or len(frequencies) < count:
        raise M3ObjectiveError("insufficient target tokens for the frozen frequent-token set")
    selected = sorted(frequencies, key=lambda token: (-frequencies[token], token))[:count]
    probability = np.asarray([frequencies[token] / total for token in selected], dtype=np.float64)
    if np.any(probability <= 0) or not np.all(np.isfinite(probability)):
        raise M3ObjectiveError("frequent-token probabilities are invalid")
    return selected, probability


def token_moment_loss(logits, labels, selected_ids, target_frequency):
    """Match teacher-forced probability first/second moments without full softmax."""
    import torch

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise M3ObjectiveError("moment logits/labels shape mismatch")
    selected = torch.as_tensor(selected_ids, dtype=torch.long, device=logits.device)
    target = torch.as_tensor(target_frequency, dtype=logits.dtype, device=logits.device)
    if selected.ndim != 1 or target.shape != selected.shape or selected.numel() != 256:
        raise M3ObjectiveError("moment objective requires exactly 256 frozen tokens")
    mask = labels.ne(-100)
    if not bool(mask.any().item()):
        raise M3ObjectiveError("moment objective received no target positions")
    flat = logits[mask]
    selected_logits = flat.index_select(-1, selected)
    selected_probabilities = torch.exp(selected_logits - torch.logsumexp(flat, dim=-1, keepdim=True))
    first = selected_probabilities.mean(dim=0)
    second = selected_probabilities.square().mean(dim=0)
    weights = target.clamp_min(torch.finfo(target.dtype).eps).rsqrt()
    weights = weights / weights.mean()
    return (weights * ((first - target).square() + (second - target).square())).mean()


def calibrate_moment_coefficient(
    ce_gradient_norms: Iterable[float],
    raw_moment_gradient_norms: Iterable[float],
    *,
    target_ratio: float = 0.20,
) -> float:
    ce = np.asarray(list(ce_gradient_norms), dtype=np.float64)
    moment = np.asarray(list(raw_moment_gradient_norms), dtype=np.float64)
    if (
        ce.size != 32
        or moment.size != 32
        or np.any(~np.isfinite(ce))
        or np.any(~np.isfinite(moment))
        or np.any(ce <= 0)
        or np.any(moment <= 0)
        or not 0 < target_ratio < 1
    ):
        raise M3ObjectiveError("moment calibration requires 32 positive finite norm pairs")
    ce_median, moment_median = float(np.median(ce)), float(np.median(moment))
    target = target_ratio * ce_median
    return min(
        MOMENT_COEFFICIENTS,
        key=lambda coefficient: (
            abs(math.log((coefficient * moment_median) / target)),
            coefficient,
        ),
    )


def surface_features(text: str) -> np.ndarray:
    words = WORD_RE.findall(text)
    token_count = max(len(words), 1)
    sentences = [part for part in SENTENCE_RE.split(text.strip()) if part.strip()]
    sentence_lengths = [len(WORD_RE.findall(sentence)) for sentence in sentences] or [token_count]
    paragraphs = [part for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]
    values = np.asarray(
        [
            len(words),
            len(sentences),
            len(paragraphs),
            float(np.mean(sentence_lengths)),
            float(np.std(sentence_lengths)),
            len({word.casefold() for word in words}) / token_count,
            text.count(",") / token_count,
            text.count(";") / token_count,
            text.count(":") / token_count,
            (text.count("(") + text.count(")")) / token_count,
            text.count("—") / token_count,
            text.count("\n") / token_count,
        ],
        dtype=np.float64,
    )
    if values.shape != (12,) or np.any(~np.isfinite(values)):
        raise M3ObjectiveError("surface feature extraction failed")
    return values


def witness_weights(
    all_human_residuals: np.ndarray,
    subset_human_residuals: np.ndarray,
    subset_policy_residuals: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    all_human = np.asarray(all_human_residuals, dtype=np.float64)
    subset_human = np.asarray(subset_human_residuals, dtype=np.float64)
    subset_policy = np.asarray(subset_policy_residuals, dtype=np.float64)
    if (
        all_human.ndim != 2
        or subset_human.ndim != 2
        or subset_policy.shape != subset_human.shape
        or all_human.shape[1] != subset_human.shape[1]
        or subset_human.shape[0] != 512
        or np.any(~np.isfinite(all_human))
        or np.any(~np.isfinite(subset_human))
        or np.any(~np.isfinite(subset_policy))
    ):
        raise M3ObjectiveError("witness residual shapes or values are invalid")
    center = all_human.mean(axis=0)
    scale = all_human.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    all_standard = (all_human - center) / scale
    human_standard = (subset_human - center) / scale
    policy_standard = (subset_policy - center) / scale
    gap = human_standard.mean(axis=0) - policy_standard.mean(axis=0)
    raw_scores = all_standard @ gap
    score_scale = raw_scores.std()
    z = (raw_scores - raw_scores.mean()) / (score_scale if score_scale >= 1e-8 else 1.0)
    weights = np.clip(np.exp(0.5 * z), 0.5, 2.0)
    weights = weights / weights.mean()
    if np.any(~np.isfinite(weights)) or not np.isclose(weights.mean(), 1.0, atol=1e-10):
        raise M3ObjectiveError("witness weights are invalid")
    return weights, {"center": center, "scale": scale, "gap": gap, "z": z}


__all__ = [
    "M3ObjectiveError",
    "MOMENT_COEFFICIENTS",
    "SURFACE_FEATURE_NAMES",
    "calibrate_moment_coefficient",
    "select_frequent_tokens",
    "surface_features",
    "token_moment_loss",
    "witness_weights",
]

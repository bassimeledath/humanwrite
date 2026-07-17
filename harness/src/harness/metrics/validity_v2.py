"""Prospective same-cardinality collapse and repetition measurements."""
from __future__ import annotations

import math
from typing import Any, Sequence

from .distribution_v2 import MeasurementV2Error
from .validity import repeated_sentence_start_rate, self_bleu


ONE_SIDED_Z_95 = 1.6448536269514722


def _wilson(successes: int, trials: int, z: float = ONE_SIDED_Z_95) -> tuple[float, float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise MeasurementV2Error("Wilson interval requires 0 <= successes <= positive trials")
    proportion = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (proportion + z2 / (2.0 * trials)) / denominator
    half = (z / denominator) * math.sqrt(
        proportion * (1.0 - proportion) / trials + z2 / (4.0 * trials * trials)
    )
    return max(0.0, center - half), min(1.0, center + half)


def newcombe_rate_difference_interval(
    candidate_successes: int,
    human_successes: int,
    trials: int,
) -> tuple[float, float]:
    """One-sided-score Newcombe interval for candidate minus human rate."""
    candidate = candidate_successes / trials
    human = human_successes / trials
    candidate_low, candidate_high = _wilson(candidate_successes, trials)
    human_low, human_high = _wilson(human_successes, trials)
    difference = candidate - human
    low = difference - math.sqrt((candidate - candidate_low) ** 2 + (human_high - human) ** 2)
    high = difference + math.sqrt((candidate_high - candidate) ** 2 + (human - human_low) ** 2)
    return max(-1.0, low), min(1.0, high)


def repetition_noninferiority(
    candidate_texts: Sequence[str],
    human_texts: Sequence[str],
    *,
    margin: float,
    power_plan_passed: bool,
    minimum_n: int = 64,
) -> dict[str, Any]:
    candidate, humans = list(candidate_texts), list(human_texts)
    if not candidate or len(candidate) != len(humans):
        raise MeasurementV2Error("repetition requires equal non-zero candidate/human n")
    if not 0 <= margin <= 1:
        raise MeasurementV2Error("repetition non-inferiority margin must be in [0,1]")
    n = len(candidate)
    candidate_successes = int(round(repeated_sentence_start_rate(candidate) * n))
    human_successes = int(round(repeated_sentence_start_rate(humans) * n))
    low, high = newcombe_rate_difference_interval(candidate_successes, human_successes, n)
    powered = n >= int(minimum_n) and bool(power_plan_passed)
    passes = powered and high <= margin
    return {
        "status": "ready" if powered else "underpowered",
        "decision": "pass" if passes else ("fail" if powered else "not_promoting"),
        "documents_per_panel": n,
        "candidate_successes": candidate_successes,
        "human_successes": human_successes,
        "candidate_rate": candidate_successes / n,
        "human_rate": human_successes / n,
        "rate_difference": (candidate_successes - human_successes) / n,
        "difference_interval": {
            "low": low,
            "high": high,
            "method": "one-sided Newcombe score interval",
        },
        "noninferiority_margin": float(margin),
        "power_plan_passed": bool(power_plan_passed),
        "zero_candidate_events_never_fail_as_too_low": True,
    }

def same_n_self_bleu(candidate_texts: Sequence[str], human_texts: Sequence[str]) -> dict[str, Any]:
    candidate, humans = list(candidate_texts), list(human_texts)
    if len(candidate) < 2 or len(candidate) != len(humans):
        raise MeasurementV2Error("self-BLEU requires equal candidate/human n >= 2")
    return {
        "documents_per_panel": len(candidate),
        "references_per_document": len(candidate) - 1,
        "candidate_corpus_self_bleu": float(self_bleu(candidate)),
        "human_corpus_self_bleu": float(self_bleu(humans)),
    }

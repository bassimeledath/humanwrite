"""Deterministic hard validity metrics and calibrated range checks."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from math import exp, log

import numpy as np


_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _tokens(text):
    return [token.casefold() for token in _WORD_RE.findall(str(text))]


def _sentences(text):
    return [part.strip() for part in _SENTENCE_RE.split(str(text)) if part.strip()]


def _facts(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        result = []
        for key in ("supported_facts", "facts", "quotations"):
            result.extend(_facts(value.get(key)))
        return result
    result = []
    for item in value:
        result.extend(_facts(item))
    return result


def _overlap(source, target):
    source_tokens, target_tokens = set(_tokens(source)), set(_tokens(target))
    if not source_tokens:
        return 1.0
    return len(source_tokens & target_tokens) / len(source_tokens)


def outline_fact_recall(gen_texts, outlines):
    """Micro-average fraction of required facts expressed by token coverage."""
    generated, outlines = list(gen_texts), list(outlines)
    if len(generated) != len(outlines):
        raise ValueError("generated texts and outlines must have equal length")
    required = [(text, fact) for text, outline in zip(generated, outlines) for fact in _facts(outline)]
    if not required:
        return 1.0
    return sum(_overlap(fact, text) >= 0.8 for text, fact in required) / len(required)


def unsupported_claim_rate(gen_texts, fact_tables):
    """Fraction of non-trivial sentences unsupported by any supplied fact.

    This conservative deterministic proxy treats each sentence containing at
    least four lexical tokens as a claim. A claim is supported when at least
    60% of its content tokens occur in one fact, or vice versa. Fact tables
    accept the canonical outline structure.
    """
    generated, tables = list(gen_texts), list(fact_tables)
    if len(generated) != len(tables):
        raise ValueError("generated texts and fact tables must have equal length")
    unsupported = total = 0
    for text, table in zip(generated, tables):
        facts = _facts(table)
        for claim in _sentences(text):
            if len(_tokens(claim)) < 4:
                continue
            total += 1
            supported = any(max(_overlap(claim, fact), _overlap(fact, claim)) >= 0.6 for fact in facts)
            unsupported += not supported
    return unsupported / total if total else 0.0


def non_target_script_char_rate(text):
    letters = [char for char in str(text) if char.isalpha()]
    if not letters:
        return 0.0
    target = 0
    for char in letters:
        try:
            if "LATIN" in unicodedata.name(char):
                target += 1
        except ValueError:
            pass
    return 1.0 - target / len(letters)


def _in_range(value, bounds):
    bounds = bounds or {}
    low, high = bounds.get("low"), bounds.get("high")
    # Nulls in the checked-in file are deliberately uncalibrated placeholders,
    # not infinite bounds. Hard gates fail closed until calibration is run.
    if low is None or high is None:
        return False
    return (low is None or value >= float(low)) and (high is None or value <= float(high))


def language_integrity(gen_texts, calibration):
    """Whether corpus non-Latin letter rate lies inside calibration bounds."""
    texts = list(gen_texts)
    if not texts:
        raise ValueError("at least one generated text is required")
    value = float(np.mean([non_target_script_char_rate(text) for text in texts]))
    return _in_range(value, calibration.get("non_target_script_char_rate", {}))


def self_bleu(texts):
    """Mean sentence BLEU of every document against all other documents."""
    texts = list(texts)
    if len(texts) < 2:
        return 0.0
    scores = []
    for index, text in enumerate(texts):
        references = [other for other_index, other in enumerate(texts) if other_index != index]
        scores.append(sentence_bleu(str(text), references))
    return float(np.mean(scores))


def sentence_bleu(hypothesis, references, max_order=4):
    """Small deterministic BLEU with effective order and add-one smoothing."""
    hypothesis_tokens = _tokens(hypothesis)
    reference_tokens = [_tokens(reference) for reference in references]
    if not hypothesis_tokens or not reference_tokens:
        return 0.0
    precisions = []
    effective_order = min(max_order, len(hypothesis_tokens))
    for order in range(1, effective_order + 1):
        hypothesis_ngrams = Counter(
            tuple(hypothesis_tokens[start : start + order])
            for start in range(len(hypothesis_tokens) - order + 1)
        )
        maximum_reference_counts = Counter()
        for tokens in reference_tokens:
            counts = Counter(
                tuple(tokens[start : start + order]) for start in range(len(tokens) - order + 1)
            )
            for ngram, count in counts.items():
                maximum_reference_counts[ngram] = max(maximum_reference_counts[ngram], count)
        clipped = sum(min(count, maximum_reference_counts[ngram]) for ngram, count in hypothesis_ngrams.items())
        total = sum(hypothesis_ngrams.values())
        precisions.append((clipped + 1.0) / (total + 1.0))
    reference_length = min(
        (len(tokens) for tokens in reference_tokens),
        key=lambda length: (abs(length - len(hypothesis_tokens)), length),
    )
    brevity = 1.0 if len(hypothesis_tokens) > reference_length else exp(
        1.0 - reference_length / len(hypothesis_tokens)
    )
    return brevity * exp(sum(log(value) for value in precisions) / effective_order)


def repeated_sentence_start_rate(texts, run_length=3):
    """Rosmine rate: fraction of docs with a repeated first-word run.

    A document is positive when at least ``run_length`` consecutive sentences
    begin with the identical first lexical word (case-insensitive). The frozen
    disclosed metric uses a run length of three.
    """
    texts = list(texts)
    if not texts:
        return 0.0
    positives = 0
    for text in texts:
        first_words = []
        for sentence in _sentences(text):
            tokens = _tokens(sentence)
            if tokens:
                first_words.append(tokens[0])
        positive = any(
            len(set(first_words[start : start + run_length])) == 1
            for start in range(len(first_words) - run_length + 1)
        )
        positives += positive
    return positives / len(texts)


def collapse_flags(gen_texts, calibration):
    """Return measured collapse indicators, individual checks, and aggregate pass."""
    texts = list(gen_texts)
    if not texts:
        raise ValueError("at least one generated text is required")
    bleu = self_bleu(texts)
    repetition = repeated_sentence_start_rate(texts)
    bleu_ok = _in_range(bleu, calibration.get("self_bleu", {}))
    repetition_ok = _in_range(repetition, calibration.get("repeated_sentence_start_rate", {}))
    return {
        "self_bleu": bleu,
        "repetition_rate": repetition,
        "self_bleu_in_range": bleu_ok,
        "repetition_in_range": repetition_ok,
        "pass": bleu_ok and repetition_ok,
    }

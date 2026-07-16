"""Quality preference and fresh-probe authorship metrics."""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline


_JUDGE_SYSTEM = (
    "Compare two responses to the user prompt for correctness, relevance, "
    "clarity, and faithfulness. Reply with exactly A, B, or TIE."
)


def _judge_call(judge, prompt, candidate_a, candidate_b):
    if hasattr(judge, "compare"):
        return judge.compare(prompt=prompt, candidate_a=candidate_a, candidate_b=candidate_b)
    if callable(judge):
        try:
            return judge(prompt=prompt, candidate_a=candidate_a, candidate_b=candidate_b)
        except TypeError:
            return judge(prompt, candidate_a, candidate_b)
    # OpenAI/OpenRouter-compatible client abstraction. The caller constructs
    # and owns the client; this module never reads API keys.
    completions = getattr(getattr(judge, "chat", None), "completions", None)
    if completions is None or not hasattr(completions, "create"):
        raise TypeError("judge must be callable, expose compare(), or be an OpenAI-compatible client")
    model = getattr(judge, "model", None) or getattr(judge, "model_id", None)
    if not model:
        raise ValueError("OpenAI-compatible judge must expose model or model_id")
    response = completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {
                "role": "user",
                "content": f"Prompt:\n{prompt}\n\nA:\n{candidate_a}\n\nB:\n{candidate_b}",
            },
        ],
    )
    return response.choices[0].message.content


def _winner(value):
    if isinstance(value, dict):
        value = value.get("winner", value.get("choice"))
    if isinstance(value, (int, float)):
        if float(value) == 0.5:
            return "TIE"
        return "A" if float(value) > 0.5 else "B"
    raw_text = str(value).strip()
    try:
        decoded = json.loads(raw_text)
        if isinstance(decoded, dict):
            return _winner(decoded)
    except (ValueError, TypeError):
        pass
    text = raw_text.upper()
    match = re.search(r"\b(TIE|A|B)\b", text)
    if not match:
        raise ValueError(f"judge returned an unparseable winner: {value!r}")
    return match.group(1)


def quality_preference(gen_texts, human_texts, prompts, judge_model):
    """Pairwise win rate with stable hash-based order randomization."""
    generated, humans, prompts = list(gen_texts), list(human_texts), list(prompts)
    if not generated or len(generated) != len(humans) or len(generated) != len(prompts):
        raise ValueError("generated texts, human texts, and prompts must have equal non-zero length")
    wins = 0.0
    for generated_text, human_text, prompt in zip(generated, humans, prompts):
        digest = hashlib.sha256(
            (str(prompt) + "\0" + str(generated_text) + "\0" + str(human_text)).encode()
        ).digest()
        generated_first = bool(digest[0] & 1)
        a, b = (generated_text, human_text) if generated_first else (human_text, generated_text)
        result = _winner(_judge_call(judge_model, prompt, a, b))
        if result == "TIE":
            wins += 0.5
        elif (result == "A") == generated_first:
            wins += 1.0
    return wins / len(generated)


def jmq(gen_texts, human_texts, prompts, judge_model):
    """Rosmine-exact JMQ: twice the generated response win rate."""
    return 2.0 * quality_preference(gen_texts, human_texts, prompts, judge_model)


def _probe_scores(probe, texts):
    if hasattr(probe, "predict_proba"):
        values = np.asarray(probe.predict_proba(texts), dtype=float)
        values = values[:, 1] if values.ndim == 2 else values
    elif hasattr(probe, "decision_function"):
        values = np.asarray(probe.decision_function(texts), dtype=float)
    elif callable(probe):
        values = np.asarray(probe(texts), dtype=float)
    else:
        raise TypeError("probe must be callable or expose predict_proba/decision_function")
    values = values.reshape(-1)
    if len(values) != len(texts) or not np.isfinite(values).all():
        raise ValueError("probe returned invalid scores")
    return values


def authorship_auc(gen_texts, human_texts, freshly_trained_probe):
    """Return machine-positive AUC and deterministic stratified bootstrap CI."""
    generated, humans = list(gen_texts), list(human_texts)
    if not generated or not humans:
        raise ValueError("both generated and human texts are required")
    texts = humans + generated
    labels = np.array([0] * len(humans) + [1] * len(generated))
    scores = _probe_scores(freshly_trained_probe, texts)
    return _auc_with_ci(labels, scores)


def _auc_with_ci(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    auc = float(roc_auc_score(labels, scores))
    class_counts = np.bincount(labels.astype(int), minlength=2)
    if class_counts.min() < 2:
        # The point estimate exists with one item per class, but a meaningful
        # resampling interval does not.
        return auc, 0.0, 1.0
    rng = np.random.default_rng(0)
    human_indices = np.flatnonzero(labels == 0)
    generated_indices = np.flatnonzero(labels == 1)
    boot = []
    for _ in range(1000):
        indices = np.concatenate(
            [
                rng.choice(human_indices, len(human_indices), replace=True),
                rng.choice(generated_indices, len(generated_indices), replace=True),
            ]
        )
        boot.append(roc_auc_score(labels[indices], scores[indices]))
    return auc, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def fresh_authorship_auc(gen_texts, human_texts):
    """Train a deterministic fresh character probe and score held-out folds.

    Every reported score is out-of-fold: vectorizer vocabulary/IDF and the
    regularized logistic model are fitted independently inside each fold.
    """
    generated, humans = list(gen_texts), list(human_texts)
    if len(generated) < 2 or len(humans) < 2:
        raise ValueError(
            "fresh authorship probe requires at least two generated and two human documents"
        )
    texts = np.asarray(humans + generated, dtype=object)
    labels = np.array([0] * len(humans) + [1] * len(generated))
    folds = min(5, len(generated), len(humans))
    classifier = make_pipeline(
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            lowercase=True,
            sublinear_tf=True,
            max_features=50_000,
        ),
        LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="liblinear",
            max_iter=1_000,
            random_state=0,
        ),
    )
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    try:
        probabilities = cross_val_predict(
            classifier,
            texts,
            labels,
            cv=splitter,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
    except ValueError as error:
        raise ValueError(f"fresh authorship probe could not be trained: {error}") from error
    return _auc_with_ci(labels, probabilities)

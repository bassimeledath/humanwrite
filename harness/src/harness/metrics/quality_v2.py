"""Prompt-linked quality and grouped authorship measurement for v2."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

from .distribution_v2 import MeasurementV2Error


PROMOTION_ENDPOINT_TOKENS = {
    "auc",
    "authorship",
    "mmd",
    "jmq",
    "quality_pref",
    "lexical",
    "structural",
    "semantic",
    "repetition",
    "self_bleu",
    "bleu",
    "score",
    "s",
}


def _required(record: dict[str, Any], key: str) -> str:
    value = str(record.get(key) or "").strip()
    if not value:
        raise MeasurementV2Error(f"prompt-linked quality record is missing {key}")
    return value


def align_prompt_linked_references(
    generated_records: Sequence[dict[str, Any]],
    human_records: Sequence[dict[str, Any]],
    *,
    expected_human_split: str = "quality_visible_human",
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Validate and align one prompt-matched human per generated row."""
    generated, humans = list(generated_records), list(human_records)
    if not generated or len(generated) != len(humans):
        raise MeasurementV2Error("quality requires equal non-zero generated/human counts")
    human_index: dict[str, dict[str, Any]] = {}
    fingerprints = set()
    for human in humans:
        prompt_id = _required(human, "prompt_id")
        fingerprint = _required(human, "reference_fingerprint")
        if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise MeasurementV2Error("quality reference fingerprint must be a lowercase SHA-256")
        if prompt_id in human_index or fingerprint in fingerprints:
            raise MeasurementV2Error("quality references must be one-to-one and unique")
        if _required(human, "split") != expected_human_split:
            raise MeasurementV2Error("quality reference has the wrong split provenance")
        _required(human, "brief_sha256")
        _required(human, "text")
        human_index[prompt_id] = human
        fingerprints.add(fingerprint)
    aligned = []
    seen_prompts = set()
    for generated_row in generated:
        prompt_id = _required(generated_row, "prompt_id")
        if prompt_id in seen_prompts or prompt_id not in human_index:
            raise MeasurementV2Error("generated prompt IDs must be unique and exactly matched")
        reference = human_index[prompt_id]
        if _required(generated_row, "brief_sha256") != _required(reference, "brief_sha256"):
            raise MeasurementV2Error("generated/reference brief hash mismatch")
        declared_fingerprint = _required(generated_row, "reference_fingerprint")
        if declared_fingerprint != reference["reference_fingerprint"]:
            raise MeasurementV2Error("generated/reference fingerprint mismatch")
        _required(generated_row, "text")
        aligned.append((generated_row, reference))
        seen_prompts.add(prompt_id)
    if seen_prompts != set(human_index):
        raise MeasurementV2Error("quality prompt sets do not match")
    return aligned


def _winner(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("winner", value.get("choice"))
    text = str(value).strip().upper()
    match = re.search(r"\b(TIE|A|B)\b", text)
    if not match:
        raise MeasurementV2Error(f"quality judge returned an invalid winner: {value!r}")
    return match.group(1)


def _clustered_score_interval(
    scores: Sequence[float], cluster_ids: Sequence[str], *, draws: int, seed: int
) -> tuple[float, float]:
    values = np.asarray(scores, dtype=np.float64)
    clusters = np.asarray([str(value) for value in cluster_ids], dtype=object)
    unique = np.asarray(sorted(set(clusters.tolist())), dtype=object)
    if len(unique) < 2:
        return 0.0, 1.0
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(draws):
        chosen = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(clusters == item) for item in chosen])
        estimates.append(float(np.mean(values[indices])))
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def prompt_linked_quality(
    generated_records: Sequence[dict[str, Any]],
    human_records: Sequence[dict[str, Any]],
    judge: Callable[..., Any] | None,
    *,
    bootstrap_draws: int = 2_000,
    seed: int = 0,
) -> dict[str, Any]:
    if judge is None:
        return {"status": "not_measured", "reason": "prompt-linked judge unavailable"}
    aligned = align_prompt_linked_references(generated_records, human_records)
    scores, cluster_ids = [], []
    counts = {"wins": 0, "losses": 0, "ties": 0}
    for generated, human in aligned:
        prompt_id = generated["prompt_id"]
        digest = hashlib.sha256(
            (prompt_id + "\0" + generated["text"] + "\0" + human["text"]).encode("utf-8")
        ).digest()
        generated_first = bool(digest[0] & 1)
        candidate_a, candidate_b = (
            (generated["text"], human["text"])
            if generated_first
            else (human["text"], generated["text"])
        )
        result = _winner(
            judge(
                prompt=generated.get("prompt", ""),
                candidate_a=candidate_a,
                candidate_b=candidate_b,
            )
        )
        if result == "TIE":
            score, bucket = 0.5, "ties"
        elif (result == "A") == generated_first:
            score, bucket = 1.0, "wins"
        else:
            score, bucket = 0.0, "losses"
        scores.append(score)
        counts[bucket] += 1
        cluster_ids.append(str(generated.get("cluster_id") or prompt_id))
    low, high = _clustered_score_interval(
        scores, cluster_ids, draws=bootstrap_draws, seed=seed
    )
    return {
        "status": "measured",
        "win_rate": float(np.mean(scores)),
        "jmq": float(2.0 * np.mean(scores)),
        **counts,
        "interval": {"low": low, "high": high, "method": "prompt-cluster bootstrap"},
        "effective_prompt_clusters": len(set(cluster_ids)),
    }


def _probe_pipeline():
    return make_pipeline(
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            lowercase=True,
            sublinear_tf=True,
            max_features=50_000,
        ),
        LogisticRegression(
            C=1.0,
            solver="liblinear",
            max_iter=1_000,
            random_state=0,
        ),
    )


def _grouped_oof_auc(texts, labels, groups, *, folds: int, fold_seed: int) -> tuple[float, int]:
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=fold_seed)
    scores = np.full(len(texts), np.nan, dtype=np.float64)
    for train, test in splitter.split(texts, labels, groups):
        model = _probe_pipeline()
        model.fit(texts[train].tolist(), labels[train])
        scores[test] = model.predict_proba(texts[test].tolist())[:, 1]
    if not np.isfinite(scores).all():
        raise MeasurementV2Error("grouped authorship cross-fit left unscored rows")
    return float(roc_auc_score(labels, scores)), folds


def grouped_authorship_auc(
    generated_records: Sequence[dict[str, Any]],
    human_records: Sequence[dict[str, Any]],
    *,
    fold_seeds: Sequence[int] = (701, 702, 703, 704, 705),
    uncertainty_refits: int = 100,
    min_effective_clusters: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """Repeated grouped cross-fit with full-pipeline cluster-bootstrap refits."""
    generated, humans = list(generated_records), list(human_records)
    if not generated or len(generated) != len(humans):
        raise MeasurementV2Error("authorship requires equal non-zero human/generated counts")
    rows = humans + generated
    texts = np.asarray([_required(row, "text") for row in rows], dtype=object)
    labels = np.asarray([0] * len(humans) + [1] * len(generated), dtype=int)
    groups = np.asarray([_required(row, "cluster_id") for row in rows], dtype=object)
    unique_groups = sorted(set(groups.tolist()))
    class_group_counts = [len(set(groups[labels == label].tolist())) for label in (0, 1)]
    folds = min(5, *class_group_counts)
    if folds < 2:
        raise MeasurementV2Error("authorship requires at least two groups per class")
    fold_seeds = tuple(int(value) for value in fold_seeds)
    if not fold_seeds:
        raise MeasurementV2Error("authorship fold seeds must be frozen")
    point_results = [
        _grouped_oof_auc(texts, labels, groups, folds=folds, fold_seed=value)
        for value in fold_seeds
    ]
    point_estimates = [value for value, _ in point_results]
    fit_count = sum(count for _, count in point_results)
    rng = np.random.default_rng(seed)
    refit_estimates = []
    successful_refit_seeds = []
    for replicate in range(int(uncertainty_refits)):
        sampled_indices, sampled_groups = [], []
        selected = rng.choice(
            np.asarray(unique_groups, dtype=object), len(unique_groups), replace=True
        )
        for occurrence, group in enumerate(selected):
            group_indices = np.flatnonzero(groups == group)
            sampled_indices.extend(group_indices.tolist())
            sampled_groups.extend([f"{occurrence}:{group}"] * len(group_indices))
        indices = np.asarray(sampled_indices, dtype=int)
        boot_labels = labels[indices]
        boot_groups = np.asarray(sampled_groups, dtype=object)
        boot_class_groups = [
            len(set(boot_groups[boot_labels == label].tolist())) for label in (0, 1)
        ]
        boot_folds = min(5, *boot_class_groups)
        if boot_folds < 2:
            continue
        try:
            refit_seed = fold_seeds[replicate % len(fold_seeds)] + replicate
            estimate, replicate_fit_count = _grouped_oof_auc(
                texts[indices],
                boot_labels,
                boot_groups,
                folds=boot_folds,
                fold_seed=refit_seed,
            )
            refit_estimates.append(estimate)
            fit_count += replicate_fit_count
            successful_refit_seeds.append(refit_seed)
        except (ValueError, MeasurementV2Error):
            continue
    if len(refit_estimates) < max(10, uncertainty_refits // 2):
        raise MeasurementV2Error("too few successful grouped authorship uncertainty refits")
    auc = float(np.mean(point_estimates))
    return {
        "status": "ready" if len(unique_groups) >= min_effective_clusters else "underpowered",
        "auc": auc,
        "separability": abs(auc - 0.5),
        "interval": {
            "low": float(np.quantile(refit_estimates, 0.025)),
            "high": float(np.quantile(refit_estimates, 0.975)),
            "method": "full-pipeline grouped cluster-bootstrap refits",
        },
        "effective_clusters": len(unique_groups),
        "fold_seeds": list(fold_seeds),
        "fit_count": fit_count,
        "uncertainty_refit_fold_seeds": successful_refit_seeds,
    }


def validate_selection_firewall(selection_manifest: dict[str, Any]) -> dict[str, Any]:
    """Reject endpoint-selected checkpoints before v2 scoring."""
    selection = selection_manifest.get("selection")
    if not isinstance(selection, dict):
        raise MeasurementV2Error("v2 checkpoint manifest requires a selection block")
    rule_type = str(selection.get("rule_type") or "")
    if rule_type not in {"fixed_seed", "hash_selected_seed", "all_preregistered_seeds", "training_only"}:
        raise MeasurementV2Error("checkpoint selection is not endpoint-independent")
    serialized = json.dumps(selection, sort_keys=True).casefold()
    words = set(re.findall(r"[a-z]+", serialized))
    words.update(re.findall(r"[a-z_]+", serialized))
    forbidden = sorted(words & PROMOTION_ENDPOINT_TOKENS)
    if forbidden:
        raise MeasurementV2Error(
            "checkpoint selection references promotion endpoints: " + ", ".join(forbidden)
        )
    if rule_type == "fixed_seed":
        seed = selection.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise MeasurementV2Error("fixed_seed selection requires a nonnegative seed")
    elif rule_type == "all_preregistered_seeds":
        seeds = selection.get("seeds")
        if (
            not isinstance(seeds, list)
            or not seeds
            or len(seeds) != len(set(seeds))
            or any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds)
        ):
            raise MeasurementV2Error("all_preregistered_seeds requires unique nonnegative seeds")
    elif rule_type == "hash_selected_seed":
        seed = selection.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise MeasurementV2Error("hash_selected_seed requires a nonnegative seed")
        digest = str(selection.get("selector_input_sha256") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise MeasurementV2Error("hash_selected_seed requires selector_input_sha256")
    elif rule_type == "training_only":
        digest = str(selection.get("training_objective_sha256") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise MeasurementV2Error("training_only requires training_objective_sha256")
    return {"status": "pass", "rule_type": rule_type}

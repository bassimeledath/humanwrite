"""Prospective measurement-v2 distribution metrics.

This module is intentionally independent of the historical v1 helpers.  V2
uses one human-only bandwidth array for candidate, control, and human floor;
requires equal-cardinality, unique, disjoint panels; and exposes deterministic
permutation/cluster uncertainty interfaces.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np


class MeasurementV2Error(ValueError):
    """Raised when a measurement-v2 invariant is violated."""


@dataclass(frozen=True)
class EmbeddingPanel:
    panel_id: str
    document_ids: tuple[str, ...]
    embeddings: np.ndarray

    @classmethod
    def build(cls, panel_id: str, document_ids: Sequence[str], embeddings) -> "EmbeddingPanel":
        ids = tuple(str(value) for value in document_ids)
        array = np.asarray(embeddings, dtype=np.float64)
        if not panel_id or not ids:
            raise MeasurementV2Error("panel_id and document_ids are required")
        if len(ids) != len(set(ids)):
            raise MeasurementV2Error(f"panel {panel_id} contains duplicate document IDs")
        if array.ndim != 2 or array.shape[0] != len(ids) or not np.isfinite(array).all():
            raise MeasurementV2Error(f"panel {panel_id} has invalid embeddings")
        return cls(panel_id=str(panel_id), document_ids=ids, embeddings=array)


def _sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.maximum(
        np.sum(x * x, axis=1)[:, None]
        + np.sum(y * y, axis=1)[None, :]
        - 2.0 * x @ y.T,
        0.0,
    )


def _require_same_dimension(*panels: EmbeddingPanel) -> None:
    dimensions = {panel.embeddings.shape[1] for panel in panels}
    if len(dimensions) != 1:
        raise MeasurementV2Error("embedding panels have different dimensions")


def require_disjoint_equal_human_panels(
    human_eval: EmbeddingPanel,
    floor_a: EmbeddingPanel,
    floor_b: EmbeddingPanel,
) -> int:
    _require_same_dimension(human_eval, floor_a, floor_b)
    counts = {len(human_eval.document_ids), len(floor_a.document_ids), len(floor_b.document_ids)}
    if len(counts) != 1:
        raise MeasurementV2Error("human panels must have equal cardinality")
    all_ids = human_eval.document_ids + floor_a.document_ids + floor_b.document_ids
    if len(all_ids) != len(set(all_ids)):
        raise MeasurementV2Error("human eval/floor panels must be mutually disjoint")
    return counts.pop()


def human_only_bandwidths(
    floor_a: EmbeddingPanel,
    floor_b: EmbeddingPanel,
    scales: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> tuple[float, ...]:
    """Derive bandwidths once from two disjoint human-only panels."""
    if len(floor_a.document_ids) != len(floor_b.document_ids):
        raise MeasurementV2Error("floor panels must have equal cardinality")
    if set(floor_a.document_ids) & set(floor_b.document_ids):
        raise MeasurementV2Error("floor panels must be disjoint; replacement is prohibited")
    _require_same_dimension(floor_a, floor_b)
    scales = tuple(float(value) for value in scales)
    if not scales or any(value <= 0 for value in scales):
        raise MeasurementV2Error("bandwidth scales must be positive")
    pooled = np.concatenate([floor_a.embeddings, floor_b.embeddings], axis=0)
    distances = _sq_dists(pooled, pooled)
    positive = distances[distances > 0]
    median_squared_distance = float(np.median(positive)) if positive.size else 1.0
    return tuple(
        max(median_squared_distance * scale * scale, np.finfo(float).eps)
        for scale in scales
    )


def bandwidth_hash(bandwidths: Sequence[float]) -> str:
    values = [format(float(value), ".17g") for value in bandwidths]
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def fixed_kernel(x, y, bandwidths: Sequence[float]) -> np.ndarray:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    values = tuple(float(value) for value in bandwidths)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise MeasurementV2Error("kernel inputs must be compatible matrices")
    if not values or any(value <= 0 for value in values):
        raise MeasurementV2Error("frozen bandwidths must be positive")
    distances = _sq_dists(x, y)
    return sum(np.exp(-distances / (2.0 * value)) for value in values) / len(values)


def mmd_unbiased_fixed(x, y, bandwidths: Sequence[float]) -> float:
    """Unbiased MMD-squared under an externally frozen kernel."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 2 or len(y) < 2:
        raise MeasurementV2Error("unbiased MMD requires at least two rows per panel")
    k_xx = fixed_kernel(x, x, bandwidths)
    k_yy = fixed_kernel(y, y, bandwidths)
    k_xy = fixed_kernel(x, y, bandwidths)
    xx = (k_xx.sum() - np.trace(k_xx)) / (len(x) * (len(x) - 1))
    yy = (k_yy.sum() - np.trace(k_yy)) / (len(y) * (len(y) - 1))
    return float(xx + yy - 2.0 * k_xy.mean())


def label_permutation_pvalue(
    x,
    y,
    bandwidths: Sequence[float],
    *,
    draws: int = 10_000,
    seed: int = 0,
) -> float:
    """Two-sided deterministic label-permutation p-value for absolute MMD."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) != len(y):
        raise MeasurementV2Error("permutation inference requires equal cardinality")
    if draws < 1:
        raise MeasurementV2Error("draws must be positive")
    observed = abs(mmd_unbiased_fixed(x, y, bandwidths))
    pooled = np.concatenate([x, y], axis=0)
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(draws):
        order = rng.permutation(len(pooled))
        value = abs(
            mmd_unbiased_fixed(pooled[order[: len(x)]], pooled[order[len(x) :]], bandwidths)
        )
        exceedances += value >= observed
    return float((exceedances + 1) / (draws + 1))


def paired_prompt_swap_pvalue(
    candidate,
    control,
    human_eval,
    bandwidths: Sequence[float],
    *,
    prompt_ids: Sequence[str],
    draws: int = 10_000,
    seed: int = 0,
) -> float:
    """Prompt-clustered paired-swap test for candidate-minus-control MMD."""
    candidate = np.asarray(candidate, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    human_eval = np.asarray(human_eval, dtype=np.float64)
    prompts = tuple(str(value) for value in prompt_ids)
    if len(candidate) != len(control) or len(candidate) != len(human_eval):
        raise MeasurementV2Error("paired MMD comparison requires equal cardinality")
    if len(prompts) != len(candidate) or len(prompts) != len(set(prompts)):
        raise MeasurementV2Error("paired MMD comparison requires unique aligned prompt IDs")
    observed = mmd_unbiased_fixed(candidate, human_eval, bandwidths) - mmd_unbiased_fixed(
        control, human_eval, bandwidths
    )
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(draws):
        swap = rng.integers(0, 2, len(prompts), dtype=np.int8).astype(bool)
        left, right = candidate.copy(), control.copy()
        left[swap], right[swap] = control[swap], candidate[swap]
        effect = mmd_unbiased_fixed(left, human_eval, bandwidths) - mmd_unbiased_fixed(
            right, human_eval, bandwidths
        )
        exceedances += abs(effect) >= abs(observed)
    return float((exceedances + 1) / (draws + 1))


def clustered_paired_interval(
    candidate_values: Sequence[float],
    control_values: Sequence[float],
    cluster_ids: Sequence[str],
    *,
    draws: int = 2_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Cluster bootstrap interface for additive paired effects."""
    candidate = np.asarray(candidate_values, dtype=np.float64)
    control = np.asarray(control_values, dtype=np.float64)
    clusters = np.asarray([str(value) for value in cluster_ids], dtype=object)
    if len(candidate) != len(control) or len(candidate) != len(clusters) or not len(candidate):
        raise MeasurementV2Error("paired values and cluster IDs must align")
    unique = np.asarray(sorted(set(clusters.tolist())), dtype=object)
    if len(unique) < 2 or draws < 1 or not 0 < confidence < 1:
        raise MeasurementV2Error("cluster interval requires >=2 clusters and valid settings")
    effects = candidate - control
    cluster_effects = np.asarray(
        [float(np.mean(effects[clusters == cluster])) for cluster in unique],
        dtype=np.float64,
    )
    observed = float(np.mean(cluster_effects))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        selected = rng.integers(0, len(unique), len(unique))
        samples.append(float(np.mean(cluster_effects[selected])))
    alpha = (1.0 - confidence) / 2.0
    return {
        "effect": observed,
        "low": float(np.quantile(samples, alpha)),
        "high": float(np.quantile(samples, 1.0 - alpha)),
        "effective_clusters": int(len(unique)),
    }


def common_kernel_report(
    candidate: EmbeddingPanel,
    control: EmbeddingPanel,
    human_eval: EmbeddingPanel,
    floor_a: EmbeddingPanel,
    floor_b: EmbeddingPanel,
    bandwidths: Sequence[float],
    *,
    permutation_draws: int = 10_000,
    seed: int = 0,
) -> dict[str, object]:
    """Report candidate/control/floor under one equal-n fixed kernel."""
    n = require_disjoint_equal_human_panels(human_eval, floor_a, floor_b)
    _require_same_dimension(candidate, control, human_eval)
    if len(candidate.document_ids) != n or len(control.document_ids) != n:
        raise MeasurementV2Error("candidate, control, and every human panel must use equal n")
    if set(candidate.document_ids) != set(control.document_ids):
        raise MeasurementV2Error("candidate and control prompt ID sets must match exactly")
    control_index = {document_id: index for index, document_id in enumerate(control.document_ids)}
    aligned_control = control.embeddings[
        [control_index[document_id] for document_id in candidate.document_ids]
    ]
    values = tuple(float(value) for value in bandwidths)
    candidate_mmd = mmd_unbiased_fixed(candidate.embeddings, human_eval.embeddings, values)
    control_mmd = mmd_unbiased_fixed(aligned_control, human_eval.embeddings, values)
    floor_mmd = mmd_unbiased_fixed(floor_a.embeddings, floor_b.embeddings, values)
    return {
        "documents_per_cell": n,
        "human_documents_per_panel": n,
        "bandwidths": list(values),
        "bandwidth_sha256": bandwidth_hash(values),
        "candidate_mmd2_unbiased": candidate_mmd,
        "control_mmd2_unbiased": control_mmd,
        "human_floor_mmd2_unbiased": floor_mmd,
        "candidate_delta_vs_floor": candidate_mmd - floor_mmd,
        "control_delta_vs_floor": control_mmd - floor_mmd,
        "candidate_minus_control": candidate_mmd - control_mmd,
        "candidate_absolute_null_p": label_permutation_pvalue(
            candidate.embeddings,
            human_eval.embeddings,
            values,
            draws=permutation_draws,
            seed=seed,
        ),
        "control_absolute_null_p": label_permutation_pvalue(
            aligned_control,
            human_eval.embeddings,
            values,
            draws=permutation_draws,
            seed=seed + 1,
        ),
        "paired_candidate_control_p": paired_prompt_swap_pvalue(
            candidate.embeddings,
            aligned_control,
            human_eval.embeddings,
            values,
            prompt_ids=candidate.document_ids,
            draws=permutation_draws,
            seed=seed + 2,
        ),
        "negative_estimate_interpretation": "sampling variation around zero",
    }

"""Fresh, unpaired measurement-v3 statistical core.

This module does not import or modify measurement v2.  V3 treats prompt
sources as quality/adherence material only; semantic distribution references
and human floors must come from disjoint documents.  It also keeps absolute
generated-vs-human inference separate from the prompt-paired treatment effect.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from typing import Any, Callable, Hashable, Sequence

import numpy as np


class MeasurementV3Error(ValueError):
    """Raised when a measurement-v3 statistical invariant is violated."""


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _ids(values: Sequence[str], label: str, *, minimum: int = 1) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if len(result) < minimum or any(not value for value in result):
        raise MeasurementV3Error(f"{label} requires at least {minimum} nonempty IDs")
    if len(result) != len(set(result)):
        raise MeasurementV3Error(f"{label} contains duplicate IDs")
    return result


@dataclass(frozen=True)
class UnpairedPanelDesign:
    """Document identities for a prompt-paired treatment and unpaired humans."""

    prompt_ids: tuple[str, ...]
    prompt_source_document_ids: tuple[str, ...]
    distribution_reference_ids: tuple[str, ...]
    human_floor_a_ids: tuple[str, ...]
    human_floor_b_ids: tuple[str, ...]

    @classmethod
    def build(
        cls,
        *,
        prompt_ids: Sequence[str],
        prompt_source_document_ids: Sequence[str],
        distribution_reference_ids: Sequence[str],
        human_floor_a_ids: Sequence[str],
        human_floor_b_ids: Sequence[str],
    ) -> "UnpairedPanelDesign":
        prompts = _ids(prompt_ids, "prompt panel", minimum=2)
        sources = _ids(prompt_source_document_ids, "prompt-source panel", minimum=2)
        references = _ids(
            distribution_reference_ids, "distribution-reference panel", minimum=2
        )
        floor_a = _ids(human_floor_a_ids, "human-floor-a panel", minimum=2)
        floor_b = _ids(human_floor_b_ids, "human-floor-b panel", minimum=2)
        if len(prompts) != len(sources):
            raise MeasurementV3Error(
                "each evaluation prompt must bind one distinct prompt-source document"
            )
        human_panels = (set(references), set(floor_a), set(floor_b))
        if any(
            human_panels[i] & human_panels[j] for i in range(3) for j in range(i + 1, 3)
        ):
            raise MeasurementV3Error(
                "distribution references and human floors must be mutually disjoint"
            )
        human_ids = set().union(*human_panels)
        if set(sources) & human_ids:
            raise MeasurementV3Error(
                "prompt-source documents cannot be semantic references or human floors"
            )
        return cls(prompts, sources, references, floor_a, floor_b)

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(
            {
                "prompt_ids": self.prompt_ids,
                "prompt_source_document_ids": self.prompt_source_document_ids,
                "distribution_reference_ids": self.distribution_reference_ids,
                "human_floor_a_ids": self.human_floor_a_ids,
                "human_floor_b_ids": self.human_floor_b_ids,
            }
        )


def _matrix(values: Any, rows: int, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 2
        or array.shape[0] != rows
        or array.shape[1] < 1
        or not np.isfinite(array).all()
    ):
        raise MeasurementV3Error(f"{label} is not a finite {rows}-row matrix")
    return array


@dataclass(frozen=True)
class EmbeddingFamily:
    """All five panels embedded by one immutable, evaluation-only family."""

    family_id: str
    model_id: str
    model_revision: str
    treatment: np.ndarray
    control: np.ndarray
    distribution_reference: np.ndarray
    human_floor_a: np.ndarray
    human_floor_b: np.ndarray

    @classmethod
    def build(
        cls,
        design: UnpairedPanelDesign,
        *,
        family_id: str,
        model_id: str,
        model_revision: str,
        treatment: Any,
        control: Any,
        distribution_reference: Any,
        human_floor_a: Any,
        human_floor_b: Any,
    ) -> "EmbeddingFamily":
        if not family_id or not model_id or not model_revision:
            raise MeasurementV3Error(
                "embedding family, model ID, and immutable revision are required"
            )
        arrays = (
            _matrix(treatment, len(design.prompt_ids), "treatment embeddings"),
            _matrix(control, len(design.prompt_ids), "control embeddings"),
            _matrix(
                distribution_reference,
                len(design.distribution_reference_ids),
                "distribution-reference embeddings",
            ),
            _matrix(human_floor_a, len(design.human_floor_a_ids), "floor-a embeddings"),
            _matrix(human_floor_b, len(design.human_floor_b_ids), "floor-b embeddings"),
        )
        if len({array.shape[1] for array in arrays}) != 1:
            raise MeasurementV3Error("one embedding family has inconsistent dimensions")
        return cls(str(family_id), str(model_id), str(model_revision), *arrays)

    @property
    def identity(self) -> str:
        return f"{self.family_id}:{self.model_id}@{self.model_revision}"


def _squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.maximum(
        np.sum(x * x, axis=1)[:, None] + np.sum(y * y, axis=1)[None, :] - 2.0 * x @ y.T,
        0.0,
    )


def human_floor_bandwidths(
    floor_a: Any,
    floor_b: Any,
    scales: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> tuple[float, ...]:
    """Freeze a family-specific kernel using human floors only."""
    left, right = (
        np.asarray(floor_a, dtype=np.float64),
        np.asarray(floor_b, dtype=np.float64),
    )
    if (
        left.ndim != 2
        or right.ndim != 2
        or left.shape[1:] != right.shape[1:]
        or len(left) < 2
        or len(right) < 2
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise MeasurementV3Error("human floors must be compatible finite matrices")
    scale_values = tuple(float(value) for value in scales)
    if not scale_values or any(
        not math.isfinite(value) or value <= 0 for value in scale_values
    ):
        raise MeasurementV3Error("bandwidth scales must be finite and positive")
    pooled = np.concatenate((left, right), axis=0)
    distances = _squared_distances(pooled, pooled)
    positive = distances[distances > 0]
    if not positive.size:
        raise MeasurementV3Error("human-floor geometry is degenerate")
    median = float(np.median(positive))
    return tuple(
        max(median * scale * scale, np.finfo(float).eps) for scale in scale_values
    )


def fixed_rbf_kernel(x: Any, y: Any, bandwidths: Sequence[float]) -> np.ndarray:
    left, right = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    values = tuple(float(value) for value in bandwidths)
    if (
        left.ndim != 2
        or right.ndim != 2
        or left.shape[1] != right.shape[1]
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise MeasurementV3Error("kernel inputs must be compatible finite matrices")
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise MeasurementV3Error("frozen bandwidths must be finite and positive")
    distances = _squared_distances(left, right)
    return sum(np.exp(-distances / (2.0 * value)) for value in values) / len(values)


def mmd2_unbiased(x: Any, y: Any, bandwidths: Sequence[float]) -> float:
    """Unbiased MMD-squared; negative estimates remain negative sampling noise."""
    left, right = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(left) < 2 or len(right) < 2:
        raise MeasurementV3Error("unbiased MMD requires at least two rows per sample")
    k_xx = fixed_rbf_kernel(left, left, bandwidths)
    k_yy = fixed_rbf_kernel(right, right, bandwidths)
    k_xy = fixed_rbf_kernel(left, right, bandwidths)
    xx = (k_xx.sum() - np.trace(k_xx)) / (len(left) * (len(left) - 1))
    yy = (k_yy.sum() - np.trace(k_yy)) / (len(right) * (len(right) - 1))
    return float(xx + yy - 2.0 * k_xy.mean())


def one_sided_absolute_mmd_permutation_test(
    sample: Any,
    human_reference: Any,
    bandwidths: Sequence[float],
    *,
    draws: int = 10_000,
    seed: int = 0,
    exact: bool = False,
    max_exact_partitions: int = 200_000,
) -> dict[str, Any]:
    """Right-tail generated-vs-human MMD test without ``abs(MMD2)``.

    "Absolute" distinguishes this one-sample-status diagnostic from the
    treatment-minus-control contrast.  The statistic itself is the signed
    unbiased estimate and only large positive values are evidence against the
    exchangeability null.  Negative estimates therefore cannot become
    significant merely by taking their absolute value.
    """
    left, right = (
        np.asarray(sample, dtype=np.float64),
        np.asarray(human_reference, dtype=np.float64),
    )
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise MeasurementV3Error("absolute MMD samples are incompatible")
    if draws < 1:
        raise MeasurementV3Error("permutation draws must be positive")
    observed = mmd2_unbiased(left, right, bandwidths)
    pooled = np.concatenate((left, right), axis=0)
    n_left = len(left)
    total_partitions = math.comb(len(pooled), n_left)
    exceedances = 0
    if exact:
        if total_partitions > max_exact_partitions:
            raise MeasurementV3Error(
                "exact MMD permutation space exceeds its frozen cap"
            )
        trial_count = total_partitions
        all_indices = np.arange(len(pooled))
        for selected_tuple in itertools.combinations(range(len(pooled)), n_left):
            selected = np.asarray(selected_tuple, dtype=np.int64)
            complement = np.setdiff1d(all_indices, selected, assume_unique=True)
            statistic = mmd2_unbiased(pooled[selected], pooled[complement], bandwidths)
            exceedances += statistic >= observed
        pvalue = exceedances / trial_count
        mode = "exact_label_enumeration"
    else:
        trial_count = int(draws)
        rng = np.random.default_rng(seed)
        for _ in range(trial_count):
            order = rng.permutation(len(pooled))
            statistic = mmd2_unbiased(
                pooled[order[:n_left]], pooled[order[n_left:]], bandwidths
            )
            exceedances += statistic >= observed
        pvalue = (exceedances + 1) / (trial_count + 1)
        mode = "monte_carlo_plus_one"
    return {
        "statistic": observed,
        "alternative": "mmd2_greater_than_exchangeability_null",
        "tail": "right",
        "absolute_value_transform": False,
        "pvalue": float(pvalue),
        "exceedances": int(exceedances),
        "permutations": int(trial_count),
        "mode": mode,
        "seed": None if exact else int(seed),
    }


def paired_treatment_control_swap_test(
    treatment: Any,
    control: Any,
    human_reference: Any,
    bandwidths: Sequence[float],
    *,
    prompt_ids: Sequence[str],
    alternative: str = "treatment_lower",
    draws: int = 10_000,
    seed: int = 0,
    exact: bool = False,
    max_exact_assignments: int = 1_048_576,
) -> dict[str, Any]:
    """One-sided prompt-paired randomization test of treatment minus control."""
    treated, baseline = (
        np.asarray(treatment, dtype=np.float64),
        np.asarray(control, dtype=np.float64),
    )
    reference = np.asarray(human_reference, dtype=np.float64)
    prompts = _ids(prompt_ids, "paired treatment prompts", minimum=2)
    if (
        treated.ndim != 2
        or baseline.ndim != 2
        or reference.ndim != 2
        or treated.shape != baseline.shape
        or treated.shape[0] != len(prompts)
        or treated.shape[1] != reference.shape[1]
    ):
        raise MeasurementV3Error(
            "paired treatment/control rows must align and share reference dimension"
        )
    if alternative not in {"treatment_lower", "treatment_higher"}:
        raise MeasurementV3Error("paired MMD alternative is invalid")
    if draws < 1:
        raise MeasurementV3Error("paired-swap draws must be positive")
    observed = mmd2_unbiased(treated, reference, bandwidths) - mmd2_unbiased(
        baseline, reference, bandwidths
    )

    def is_extreme(value: float) -> bool:
        return (
            value <= observed if alternative == "treatment_lower" else value >= observed
        )

    assignment_count = 1 << len(prompts)
    exceedances = 0
    if exact:
        if assignment_count > max_exact_assignments:
            raise MeasurementV3Error("exact paired-swap space exceeds its frozen cap")
        trial_count = assignment_count
        masks = (
            np.fromiter(((bits >> index) & 1 for index in range(len(prompts))), bool)
            for bits in range(assignment_count)
        )
        mode = "exact_paired_swap_enumeration"
    else:
        trial_count = int(draws)
        rng = np.random.default_rng(seed)
        masks = (
            rng.integers(0, 2, len(prompts), dtype=np.int8).astype(bool)
            for _ in range(trial_count)
        )
        mode = "monte_carlo_plus_one"
    for swap in masks:
        left, right = treated.copy(), baseline.copy()
        left[swap], right[swap] = baseline[swap], treated[swap]
        statistic = mmd2_unbiased(left, reference, bandwidths) - mmd2_unbiased(
            right, reference, bandwidths
        )
        exceedances += is_extreme(statistic)
    pvalue = (
        exceedances / trial_count if exact else (exceedances + 1) / (trial_count + 1)
    )
    return {
        "effect": observed,
        "effect_definition": "treatment_mmd2_minus_control_mmd2",
        "alternative": alternative,
        "tail": "left" if alternative == "treatment_lower" else "right",
        "pvalue": float(pvalue),
        "extreme_assignments": int(exceedances),
        "assignments": int(trial_count),
        "effective_prompt_clusters": len(prompts),
        "mode": mode,
        "seed": None if exact else int(seed),
    }


def two_family_distribution_report(
    design: UnpairedPanelDesign,
    families: Sequence[EmbeddingFamily],
    *,
    training_reward_model_ids: Sequence[str] = (),
    permutation_draws: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Evaluate the same unpaired design under exactly two independent families."""
    values = tuple(families)
    if len(values) != 2:
        raise MeasurementV3Error("v3 requires exactly two embedding families")
    if (
        len({family.family_id for family in values}) != 2
        or len({family.model_id for family in values}) != 2
    ):
        raise MeasurementV3Error(
            "v3 embedding families must have distinct families and models"
        )
    reward_models = {str(value).casefold() for value in training_reward_model_ids}
    if any(family.model_id.casefold() in reward_models for family in values):
        raise MeasurementV3Error(
            "evaluation embedding families cannot reuse a training reward model"
        )
    reports: dict[str, Any] = {}
    for offset, family in enumerate(values):
        bandwidths = human_floor_bandwidths(family.human_floor_a, family.human_floor_b)
        treatment_mmd = mmd2_unbiased(
            family.treatment, family.distribution_reference, bandwidths
        )
        control_mmd = mmd2_unbiased(
            family.control, family.distribution_reference, bandwidths
        )
        reports[family.family_id] = {
            "model_id": family.model_id,
            "model_revision": family.model_revision,
            "bandwidths": list(bandwidths),
            "bandwidth_sha256": _canonical_sha256(
                [format(value, ".17g") for value in bandwidths]
            ),
            "treatment_mmd2": treatment_mmd,
            "control_mmd2": control_mmd,
            "treatment_minus_control": treatment_mmd - control_mmd,
            "treatment_absolute_test": one_sided_absolute_mmd_permutation_test(
                family.treatment,
                family.distribution_reference,
                bandwidths,
                draws=permutation_draws,
                seed=seed + 10 * offset,
            ),
            "control_absolute_test": one_sided_absolute_mmd_permutation_test(
                family.control,
                family.distribution_reference,
                bandwidths,
                draws=permutation_draws,
                seed=seed + 10 * offset + 1,
            ),
            "paired_treatment_test": paired_treatment_control_swap_test(
                family.treatment,
                family.control,
                family.distribution_reference,
                bandwidths,
                prompt_ids=design.prompt_ids,
                draws=permutation_draws,
                seed=seed + 10 * offset + 2,
            ),
        }
    effects = [
        reports[family.family_id]["treatment_minus_control"] for family in values
    ]
    return {
        "artifact_schema": "dftr.measurement.distribution_report.v3",
        "panel_design_sha256": design.identity_sha256,
        "prompt_source_role": "quality_and_adherence_only",
        "distribution_reference_role": "unpaired_semantic_reference",
        "family_count": 2,
        "families": reports,
        "primary_direction": "lower_mmd_is_better",
        "primary_direction_agreement": all(effect < 0 for effect in effects),
    }


def token_unigram_l2(
    sample_documents: Sequence[Sequence[Hashable]],
    human_documents: Sequence[Sequence[Hashable]],
) -> dict[str, Any]:
    """Corpus-frequency L2 on caller-supplied token sequences.

    Tokenization is deliberately outside the metric and must be frozen by the
    caller.  This avoids silently changing model tokenizer, Unicode, casing,
    or punctuation policy inside the endpoint.
    """

    def counts(documents: Sequence[Sequence[Hashable]], label: str) -> Counter:
        if not documents:
            raise MeasurementV3Error(f"{label} token documents are empty")
        result: Counter = Counter()
        for document in documents:
            if isinstance(document, (str, bytes)):
                raise MeasurementV3Error(
                    "token unigram L2 requires token sequences, not raw strings"
                )
            result.update(document)
        if not result:
            raise MeasurementV3Error(f"{label} token corpus has no tokens")
        return result

    sample_counts = counts(sample_documents, "sample")
    human_counts = counts(human_documents, "human")
    sample_total, human_total = sum(sample_counts.values()), sum(human_counts.values())
    vocabulary = set(sample_counts) | set(human_counts)
    squared = sum(
        (sample_counts[token] / sample_total - human_counts[token] / human_total) ** 2
        for token in vocabulary
    )
    return {
        "l2": float(math.sqrt(squared)),
        "sample_tokens": int(sample_total),
        "human_tokens": int(human_total),
        "vocabulary_size": len(vocabulary),
        "normalization": "corpus_relative_frequency",
        "tokenizer_policy": "caller_frozen",
    }


def _numeric(values: Sequence[float], label: str, *, minimum: int = 2) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < minimum or not np.isfinite(array).all():
        raise MeasurementV3Error(f"{label} requires {minimum}+ finite values")
    return array


def human_floor_margin(
    floor_a_values: Sequence[float],
    floor_b_values: Sequence[float],
    *,
    draws: int = 5_000,
    seed: int = 0,
    coverage: float = 0.95,
    multiplier: float = 1.0,
) -> dict[str, Any]:
    """Calibrate a symmetric margin from unpaired human-vs-human variation."""
    left = _numeric(floor_a_values, "human floor A")
    right = _numeric(floor_b_values, "human floor B")
    if (
        draws < 1
        or not 0 < coverage < 1
        or not math.isfinite(multiplier)
        or multiplier <= 0
    ):
        raise MeasurementV3Error("human-floor margin settings are invalid")
    rng = np.random.default_rng(seed)
    differences = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        a = left[rng.integers(0, len(left), len(left))]
        b = right[rng.integers(0, len(right), len(right))]
        differences[index] = abs(float(np.mean(a) - np.mean(b)))
    margin = float(np.quantile(differences, coverage) * multiplier)
    return {
        "margin": margin,
        "coverage": float(coverage),
        "multiplier": float(multiplier),
        "draws": int(draws),
        "seed": int(seed),
        "source": "unpaired_human_floor_bootstrap",
    }


def equivalence_decision(
    *, interval_low: float, interval_high: float, margin: float
) -> dict[str, Any]:
    """Two-sided equivalence: the complete interval must lie in +/- margin."""
    low, high, bound = float(interval_low), float(interval_high), float(margin)
    if (
        not all(math.isfinite(value) for value in (low, high, bound))
        or low > high
        or bound < 0
    ):
        raise MeasurementV3Error("equivalence interval or margin is invalid")
    passed = low >= -bound and high <= bound
    return {
        "decision": "pass" if passed else "fail",
        "interval": {"low": low, "high": high},
        "equivalence_bounds": {"low": -bound, "high": bound},
    }


def noninferiority_decision(
    *,
    interval_low: float,
    interval_high: float,
    margin: float,
    lower_is_better: bool,
) -> dict[str, Any]:
    """One-sided noninferiority for candidate-minus-human effects."""
    low, high, bound = float(interval_low), float(interval_high), float(margin)
    if (
        not all(math.isfinite(value) for value in (low, high, bound))
        or low > high
        or bound < 0
    ):
        raise MeasurementV3Error("noninferiority interval or margin is invalid")
    passed = high <= bound if lower_is_better else low >= -bound
    return {
        "decision": "pass" if passed else "fail",
        "interval": {"low": low, "high": high},
        "noninferiority_boundary": bound if lower_is_better else -bound,
        "orientation": "lower_is_better" if lower_is_better else "higher_is_better",
    }


def human_calibrated_equivalence(
    candidate_values: Sequence[float],
    human_values: Sequence[float],
    *,
    margin: float,
    draws: int = 5_000,
    seed: int = 0,
    confidence: float = 0.90,
) -> dict[str, Any]:
    """Unpaired bootstrap equivalence of candidate-minus-human mean."""
    candidate = _numeric(candidate_values, "candidate equivalence values")
    human = _numeric(human_values, "human equivalence values")
    if draws < 1 or not 0.5 < confidence < 1:
        raise MeasurementV3Error("equivalence bootstrap settings are invalid")
    rng = np.random.default_rng(seed)
    differences = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        candidate_sample = candidate[rng.integers(0, len(candidate), len(candidate))]
        human_sample = human[rng.integers(0, len(human), len(human))]
        differences[index] = np.mean(candidate_sample) - np.mean(human_sample)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(differences, (alpha, 1.0 - alpha))
    return {
        **equivalence_decision(
            interval_low=float(low), interval_high=float(high), margin=margin
        ),
        "effect": float(np.mean(candidate) - np.mean(human)),
        "confidence": float(confidence),
        "draws": int(draws),
        "seed": int(seed),
        "calibration": "unpaired_candidate_vs_human_bootstrap",
    }


def human_calibrated_noninferiority(
    candidate_values: Sequence[float],
    human_values: Sequence[float],
    *,
    margin: float,
    lower_is_better: bool,
    draws: int = 5_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """One-sided unpaired bootstrap noninferiority against humans."""
    candidate = _numeric(candidate_values, "candidate noninferiority values")
    human = _numeric(human_values, "human noninferiority values")
    if draws < 1 or not 0.5 < confidence < 1:
        raise MeasurementV3Error("noninferiority bootstrap settings are invalid")
    rng = np.random.default_rng(seed)
    differences = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        candidate_sample = candidate[rng.integers(0, len(candidate), len(candidate))]
        human_sample = human[rng.integers(0, len(human), len(human))]
        differences[index] = np.mean(candidate_sample) - np.mean(human_sample)
    low = float(np.quantile(differences, 1.0 - confidence))
    high = float(np.quantile(differences, confidence))
    return {
        **noninferiority_decision(
            interval_low=low,
            interval_high=high,
            margin=margin,
            lower_is_better=lower_is_better,
        ),
        "effect": float(np.mean(candidate) - np.mean(human)),
        "confidence": float(confidence),
        "draws": int(draws),
        "seed": int(seed),
        "calibration": "unpaired_candidate_vs_human_bootstrap",
    }


TrialGenerator = Callable[[np.random.Generator, int], Any]
DecisionRule = Callable[[Any], bool]


def resample_exact_decision_rule(
    generator: TrialGenerator,
    decision_rule: DecisionRule,
    *,
    trials: int,
    seed: int,
    scenario: str,
    rule_id: str,
) -> dict[str, Any]:
    """Run the actual frozen decision callable once per simulated trial."""
    if trials < 1 or not scenario or not rule_id:
        raise MeasurementV3Error("resampling requires trials, scenario, and rule ID")
    rng = np.random.default_rng(seed)
    decisions: list[bool] = []
    for index in range(trials):
        decision = decision_rule(generator(rng, index))
        if type(decision) is not bool:
            raise MeasurementV3Error("the exact decision rule must return bool")
        decisions.append(decision)
    successes = sum(decisions)
    packed = bytes(
        sum(
            (1 << bit)
            for bit, value in enumerate(decisions[start : start + 8])
            if value
        )
        for start in range(0, len(decisions), 8)
    )
    return {
        "scenario": scenario,
        "rule_id": rule_id,
        "trials": int(trials),
        "seed": int(seed),
        "successes": int(successes),
        "rate": float(successes / trials),
        "decision_vector_sha256": hashlib.sha256(packed).hexdigest(),
    }


def prospective_exact_decision_power(
    *,
    null_generator: TrialGenerator,
    alternative_generator: TrialGenerator,
    decision_rule: DecisionRule,
    rule_id: str,
    trials: int,
    seed: int,
    decision_boundary: float,
    alternative_effect: float,
    effect_direction: str,
    type_i_max: float = 0.05,
    power_min: float = 0.80,
) -> dict[str, Any]:
    """Type-I and power simulation through the exact intended decision rule.

    The declared alternative must lie strictly beyond, not exactly on, the
    decision boundary.  Composite/intersection decisions belong inside
    ``decision_rule`` and are therefore exercised on every trial.
    """
    if trials < 1_000:
        raise MeasurementV3Error(
            "prospective decision power requires at least 1000 trials"
        )
    boundary, effect = float(decision_boundary), float(alternative_effect)
    if not math.isfinite(boundary) or not math.isfinite(effect):
        raise MeasurementV3Error("power boundary and alternative must be finite")
    if effect_direction == "greater":
        beyond = effect > boundary
    elif effect_direction == "less":
        beyond = effect < boundary
    else:
        raise MeasurementV3Error("power effect direction must be greater or less")
    if not beyond:
        raise MeasurementV3Error(
            "power alternative must be strictly beyond the decision boundary"
        )
    if not 0 <= type_i_max <= 1 or not 0 <= power_min <= 1:
        raise MeasurementV3Error("power targets must be probabilities")
    seed_sequence = np.random.SeedSequence(seed)
    null_seed, alternative_seed = (
        int(child.generate_state(1, dtype=np.uint64)[0])
        for child in seed_sequence.spawn(2)
    )
    null = resample_exact_decision_rule(
        null_generator,
        decision_rule,
        trials=trials,
        seed=null_seed,
        scenario="null",
        rule_id=rule_id,
    )
    alternative = resample_exact_decision_rule(
        alternative_generator,
        decision_rule,
        trials=trials,
        seed=alternative_seed,
        scenario="alternative",
        rule_id=rule_id,
    )
    type_i_pass = null["rate"] <= type_i_max
    power_pass = alternative["rate"] >= power_min
    return {
        "artifact_schema": "dftr.measurement.exact_decision_power.v3",
        "rule_id": rule_id,
        "trials_per_scenario": int(trials),
        "master_seed": int(seed),
        "decision_boundary": boundary,
        "alternative_effect": effect,
        "effect_direction": effect_direction,
        "alternative_strictly_beyond_boundary": True,
        "type_i_max": float(type_i_max),
        "power_min": float(power_min),
        "null": null,
        "alternative": alternative,
        "type_i_pass": bool(type_i_pass),
        "power_pass": bool(power_pass),
        "all_targets_pass": bool(type_i_pass and power_pass),
    }

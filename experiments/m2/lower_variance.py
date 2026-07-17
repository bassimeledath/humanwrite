"""Pure objective primitives for the prospective lower-variance M2 screen.

The functions in this module deliberately do not load models, read artifacts,
or own training state.  Callers must align next-token logits and targets before
calling the teacher-forced objective, and must persist MMD witness weights if
they want a genuinely one-round (rather than adaptive) reweighting scheme.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


class LowerVarianceError(ValueError):
    """Raised when a lower-variance objective contract is malformed."""


@dataclass(frozen=True)
class TokenDistributionMoments:
    """Corpus moments of per-document token-bucket distributions."""

    predicted_first: Tensor
    target_first: Tensor
    predicted_second: Tensor
    target_second: Tensor


@dataclass(frozen=True)
class MMDWitnessResult:
    """Frozen human witness scores and their mean-one exponential weights."""

    witness: Tensor
    weights: Tensor
    human_leave_one_out_density: Tensor
    generated_density: Tensor


def _finite_nonnegative(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LowerVarianceError(f"{field} must be a finite nonnegative number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise LowerVarianceError(f"{field} must be a finite nonnegative number")
    return number


def _finite_positive(value: object, field: str) -> float:
    number = _finite_nonnegative(value, field)
    if number == 0.0:
        raise LowerVarianceError(f"{field} must be positive")
    return number


def validate_frequent_token_ids(
    frequent_token_ids: Sequence[int], vocab_size: int
) -> tuple[int, ...]:
    """Validate the ordered, frozen frequent-token vocabulary.

    The IDs must be strictly increasing so their order is canonical.  At least
    one vocabulary item must remain outside the list for the explicit ``other``
    bucket.
    """
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 2:
        raise LowerVarianceError("vocab_size must be an integer of at least two")
    if isinstance(frequent_token_ids, (str, bytes)) or not isinstance(
        frequent_token_ids, Sequence
    ):
        raise LowerVarianceError("frequent_token_ids must be a frozen integer sequence")
    token_ids = tuple(frequent_token_ids)
    if not token_ids or len(token_ids) >= vocab_size:
        raise LowerVarianceError(
            "frequent_token_ids must be nonempty and leave at least one other token"
        )
    if any(type(token_id) is not int for token_id in token_ids):
        raise LowerVarianceError("frequent_token_ids must contain only integer IDs")
    if any(token_id < 0 or token_id >= vocab_size for token_id in token_ids):
        raise LowerVarianceError("frequent_token_ids contains an out-of-range ID")
    if any(left >= right for left, right in zip(token_ids, token_ids[1:])):
        raise LowerVarianceError("frequent_token_ids must be unique and strictly increasing")
    return token_ids


def _validate_teacher_forced_inputs(
    logits: Tensor,
    target_token_ids: Tensor,
    token_mask: Tensor,
    frequent_token_ids: Sequence[int],
) -> tuple[int, ...]:
    if not isinstance(logits, Tensor) or logits.ndim != 3 or not logits.is_floating_point():
        raise LowerVarianceError("logits must be a floating [documents, tokens, vocab] tensor")
    if logits.shape[0] < 1 or logits.shape[1] < 1 or logits.shape[2] < 2:
        raise LowerVarianceError("logits dimensions must be nonempty and vocab must be at least two")
    expected_shape = logits.shape[:2]
    if (
        not isinstance(target_token_ids, Tensor)
        or target_token_ids.dtype != torch.long
        or target_token_ids.shape != expected_shape
    ):
        raise LowerVarianceError("target_token_ids must be a long tensor aligned with logits")
    if (
        not isinstance(token_mask, Tensor)
        or token_mask.dtype != torch.bool
        or token_mask.shape != expected_shape
    ):
        raise LowerVarianceError("token_mask must be a boolean tensor aligned with logits")
    if target_token_ids.device != logits.device or token_mask.device != logits.device:
        raise LowerVarianceError("logits, targets, and mask must be on the same device")
    if not bool(torch.isfinite(logits).all().item()):
        raise LowerVarianceError("logits must be finite")
    vocab_size = int(logits.shape[-1])
    if bool(((target_token_ids < 0) | (target_token_ids >= vocab_size)).any().item()):
        raise LowerVarianceError("target_token_ids contains an out-of-range ID")
    if bool((token_mask.sum(dim=1) == 0).any().item()):
        raise LowerVarianceError("every document must contain at least one selected token")
    return validate_frequent_token_ids(frequent_token_ids, vocab_size)


def teacher_forced_token_distribution_moments(
    logits: Tensor,
    target_token_ids: Tensor,
    token_mask: Tensor,
    frequent_token_ids: Sequence[int],
) -> TokenDistributionMoments:
    """Compute differentiable first/second document-distribution moments.

    ``logits[d, t]`` must predict ``target_token_ids[d, t]``; the function does
    not guess or apply a causal shift.  Frequent token probabilities are kept
    individually and all remaining vocabulary probability is summed into one
    final ``other`` bucket.  Moments are taken across per-document bucket
    frequencies, rather than across individual positions, so the second moment
    captures between-document heterogeneity.
    """
    token_ids = _validate_teacher_forced_inputs(
        logits, target_token_ids, token_mask, frequent_token_ids
    )
    compute_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    probabilities = torch.softmax(logits.to(dtype=compute_dtype), dim=-1)
    frequent_index = torch.tensor(token_ids, dtype=torch.long, device=logits.device)
    frequent_probabilities = probabilities.index_select(-1, frequent_index)
    other_mask = torch.ones(int(logits.shape[-1]), dtype=torch.bool, device=logits.device)
    other_mask[frequent_index] = False
    other_index = torch.arange(int(logits.shape[-1]), device=logits.device)[other_mask]
    other_probability = probabilities.index_select(-1, other_index).sum(
        dim=-1, keepdim=True
    )
    predicted_buckets = torch.cat((frequent_probabilities, other_probability), dim=-1)

    other_bucket = len(token_ids)
    bucket_lookup = torch.full(
        (int(logits.shape[-1]),), other_bucket, dtype=torch.long, device=logits.device
    )
    bucket_lookup[frequent_index] = torch.arange(
        other_bucket, dtype=torch.long, device=logits.device
    )
    target_buckets = F.one_hot(
        bucket_lookup[target_token_ids], num_classes=other_bucket + 1
    ).to(dtype=compute_dtype)

    mask = token_mask.to(dtype=compute_dtype).unsqueeze(-1)
    counts = mask.sum(dim=1)
    predicted_documents = (predicted_buckets * mask).sum(dim=1) / counts
    target_documents = (target_buckets * mask).sum(dim=1) / counts
    return TokenDistributionMoments(
        predicted_first=predicted_documents.mean(dim=0),
        target_first=target_documents.mean(dim=0),
        predicted_second=predicted_documents.square().mean(dim=0),
        target_second=target_documents.square().mean(dim=0),
    )


def teacher_forced_token_moment_loss(
    logits: Tensor,
    target_token_ids: Tensor,
    token_mask: Tensor,
    frequent_token_ids: Sequence[int],
    *,
    first_moment_weight: float = 1.0,
    second_moment_weight: float = 1.0,
) -> Tensor:
    """Return weighted mean-squared error for frozen token-bucket moments."""
    first_weight = _finite_nonnegative(first_moment_weight, "first_moment_weight")
    second_weight = _finite_nonnegative(second_moment_weight, "second_moment_weight")
    if first_weight == 0.0 and second_weight == 0.0:
        raise LowerVarianceError("at least one moment weight must be positive")
    moments = teacher_forced_token_distribution_moments(
        logits, target_token_ids, token_mask, frequent_token_ids
    )
    first_loss = (moments.predicted_first - moments.target_first).square().mean()
    second_loss = (moments.predicted_second - moments.target_second).square().mean()
    return first_weight * first_loss + second_weight * second_loss


def _validate_bandwidths(bandwidths: Sequence[float]) -> tuple[float, ...]:
    if isinstance(bandwidths, (str, bytes)) or not isinstance(bandwidths, Sequence):
        raise LowerVarianceError("bandwidths must be a nonempty numeric sequence")
    values = tuple(_finite_positive(value, "bandwidth") for value in bandwidths)
    if not values:
        raise LowerVarianceError("bandwidths must be nonempty")
    if len(set(values)) != len(values):
        raise LowerVarianceError("bandwidths must not contain duplicates")
    return values


def _validate_frozen_embeddings(generated: Tensor, humans: Tensor) -> None:
    for value, field in ((generated, "generated"), (humans, "human")):
        if not isinstance(value, Tensor) or value.ndim != 2 or not value.is_floating_point():
            raise LowerVarianceError(f"{field} embeddings must be a floating matrix")
        if value.requires_grad:
            raise LowerVarianceError(f"{field} embeddings must be detached and frozen")
        if not bool(torch.isfinite(value).all().item()):
            raise LowerVarianceError(f"{field} embeddings must be finite")
    if generated.shape[0] < 1 or humans.shape[0] < 2 or generated.shape[1] < 1:
        raise LowerVarianceError(
            "witness estimation requires one generated and two human embeddings"
        )
    if generated.shape[1] != humans.shape[1]:
        raise LowerVarianceError("generated and human embedding dimensions must match")
    if generated.device != humans.device or generated.dtype != humans.dtype:
        raise LowerVarianceError("generated and human embeddings must share device and dtype")


def _multiscale_rbf(left: Tensor, right: Tensor, bandwidths: tuple[float, ...]) -> Tensor:
    squared_distance = torch.cdist(left, right).square()
    kernels = [
        torch.exp(-squared_distance / (2.0 * bandwidth)) for bandwidth in bandwidths
    ]
    return torch.stack(kernels, dim=0).mean(dim=0)


def one_round_mmd_human_witness_weights(
    generated_embeddings: Tensor,
    human_embeddings: Tensor,
    bandwidths: Sequence[float],
    *,
    temperature: float = 1.0,
) -> MMDWitnessResult:
    """Compute fixed mean-one human weights from the empirical MMD witness.

    The witness orientation is ``human density - generated density``.  Human
    self-similarity is excluded, then a temperature-scaled exponential tilt
    maps witness scores to strictly positive weights whose arithmetic mean is
    one.  Inputs must already be detached; callers should compute this once
    from a frozen shared rollout and persist the result before weighted SFT.

    Each bandwidth is a squared RBF scale, matching
    ``exp(-||x-y||^2 / (2 * bandwidth))``.
    """
    _validate_frozen_embeddings(generated_embeddings, human_embeddings)
    scales = _validate_bandwidths(bandwidths)
    tilt_temperature = _finite_positive(temperature, "temperature")
    compute_dtype = (
        torch.float64 if generated_embeddings.dtype == torch.float64 else torch.float32
    )
    generated = generated_embeddings.detach().to(dtype=compute_dtype)
    humans = human_embeddings.detach().to(dtype=compute_dtype)
    with torch.no_grad():
        human_kernel = _multiscale_rbf(humans, humans, scales)
        generated_kernel = _multiscale_rbf(humans, generated, scales)
        human_density = (
            human_kernel.sum(dim=1) - torch.diagonal(human_kernel)
        ) / (int(humans.shape[0]) - 1)
        generated_density = generated_kernel.mean(dim=1)
        witness = human_density - generated_density
        weights = torch.softmax(witness / tilt_temperature, dim=0) * int(humans.shape[0])
    if not bool(torch.isfinite(weights).all().item()):
        raise LowerVarianceError("MMD witness weights are non-finite")
    return MMDWitnessResult(
        witness=witness,
        weights=weights,
        human_leave_one_out_density=human_density,
        generated_density=generated_density,
    )


__all__ = [
    "LowerVarianceError",
    "MMDWitnessResult",
    "TokenDistributionMoments",
    "one_round_mmd_human_witness_weights",
    "teacher_forced_token_distribution_moments",
    "teacher_forced_token_moment_loss",
    "validate_frequent_token_ids",
]

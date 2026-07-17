from __future__ import annotations

import math

import pytest
import torch

from experiments.m2.lower_variance import (
    LowerVarianceError,
    one_round_mmd_human_witness_weights,
    teacher_forced_token_distribution_moments,
    teacher_forced_token_moment_loss,
    validate_frequent_token_ids,
)


def test_teacher_forced_moments_match_document_level_manual_oracle():
    logits = torch.zeros((2, 2, 3), dtype=torch.float64)
    targets = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
    mask = torch.ones((2, 2), dtype=torch.bool)

    moments = teacher_forced_token_distribution_moments(
        logits, targets, mask, (0, 1)
    )
    predicted_document = torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=torch.float64)
    target_documents = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]], dtype=torch.float64
    )
    assert torch.allclose(moments.predicted_first, predicted_document)
    assert torch.allclose(moments.target_first, target_documents.mean(dim=0))
    assert torch.allclose(moments.predicted_second, predicted_document.square())
    assert torch.allclose(
        moments.target_second, target_documents.square().mean(dim=0)
    )

    observed = teacher_forced_token_moment_loss(
        logits,
        targets,
        mask,
        (0, 1),
        first_moment_weight=0.75,
        second_moment_weight=0.25,
    )
    expected = 0.75 * (
        predicted_document - target_documents.mean(dim=0)
    ).square().mean() + 0.25 * (
        predicted_document.square() - target_documents.square().mean(dim=0)
    ).square().mean()
    assert observed == pytest.approx(expected.item(), abs=1e-12)


def test_teacher_forced_loss_has_finite_nonzero_gradients_including_other_bucket():
    logits = torch.tensor(
        [
            [[2.0, -1.0, 0.5, 0.0, -0.5], [0.2, 0.1, -0.3, 1.1, -0.7]],
            [[-0.4, 0.8, 0.3, -0.2, 1.2], [1.0, -0.8, 0.1, 0.4, -0.1]],
        ],
        requires_grad=True,
    )
    targets = torch.tensor([[0, 3], [2, 4]], dtype=torch.long)
    mask = torch.tensor([[True, True], [True, False]])
    loss = teacher_forced_token_moment_loss(logits, targets, mask, (0, 2))
    loss.backward()

    assert math.isfinite(loss.item())
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0
    assert logits.grad[..., [1, 3, 4]].abs().sum() > 0
    assert torch.count_nonzero(logits.grad[1, 1]) == 0


@pytest.mark.parametrize(
    "token_ids",
    ((1, 0), (0, 0), (0, True), (0, 3), (0, 1, 2)),
)
def test_frequent_token_vocabulary_validation_is_exact(token_ids):
    with pytest.raises(LowerVarianceError):
        validate_frequent_token_ids(token_ids, 3)


def test_teacher_forced_input_and_weight_validation_fails_closed():
    logits = torch.zeros((2, 2, 4))
    targets = torch.zeros((2, 2), dtype=torch.long)
    mask = torch.ones((2, 2), dtype=torch.bool)
    with pytest.raises(LowerVarianceError, match="boolean"):
        teacher_forced_token_moment_loss(logits, targets, mask.to(torch.int64), (0, 1))
    empty_document = mask.clone()
    empty_document[1] = False
    with pytest.raises(LowerVarianceError, match="every document"):
        teacher_forced_token_moment_loss(logits, targets, empty_document, (0, 1))
    nonfinite = logits.clone()
    nonfinite[0, 0, 0] = float("nan")
    with pytest.raises(LowerVarianceError, match="finite"):
        teacher_forced_token_moment_loss(nonfinite, targets, mask, (0, 1))
    with pytest.raises(LowerVarianceError, match="at least one"):
        teacher_forced_token_moment_loss(
            logits,
            targets,
            mask,
            (0, 1),
            first_moment_weight=0.0,
            second_moment_weight=0.0,
        )


def _rbf(left: torch.Tensor, right: torch.Tensor, bandwidth: float) -> torch.Tensor:
    return torch.exp(-torch.sum((left - right).square()) / (2.0 * bandwidth))


def test_one_round_human_witness_weights_match_manual_leave_one_out_oracle():
    generated = torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
    humans = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]], dtype=torch.float64)
    temperature = 0.7
    result = one_round_mmd_human_witness_weights(
        generated, humans, [1.0], temperature=temperature
    )

    manual_witness = []
    for index, human in enumerate(humans):
        human_density = sum(
            _rbf(human, other, 1.0)
            for other_index, other in enumerate(humans)
            if other_index != index
        ) / (len(humans) - 1)
        generated_density = sum(_rbf(human, row, 1.0) for row in generated) / len(
            generated
        )
        manual_witness.append(human_density - generated_density)
    expected_witness = torch.stack(manual_witness)
    expected_weights = torch.softmax(expected_witness / temperature, dim=0) * len(humans)

    assert torch.allclose(result.witness, expected_witness, atol=1e-12)
    assert torch.allclose(result.weights, expected_weights, atol=1e-12)
    assert result.weights.mean() == pytest.approx(1.0, abs=1e-12)
    assert torch.all(result.weights > 0)
    assert not result.weights.requires_grad


@pytest.mark.parametrize("bandwidths", ([], [0.0], [-1.0], [1.0, 1.0], [True]))
def test_one_round_witness_rejects_invalid_bandwidths(bandwidths):
    generated = torch.zeros((2, 2))
    humans = torch.ones((2, 2))
    with pytest.raises(LowerVarianceError):
        one_round_mmd_human_witness_weights(generated, humans, bandwidths)


def test_one_round_witness_requires_finite_compatible_frozen_panels():
    generated = torch.zeros((2, 2), requires_grad=True)
    humans = torch.ones((2, 2))
    with pytest.raises(LowerVarianceError, match="detached and frozen"):
        one_round_mmd_human_witness_weights(generated, humans, [1.0])
    with pytest.raises(LowerVarianceError, match="two human"):
        one_round_mmd_human_witness_weights(generated.detach(), humans[:1], [1.0])
    with pytest.raises(LowerVarianceError, match="dimensions"):
        one_round_mmd_human_witness_weights(
            generated.detach(), torch.ones((2, 3)), [1.0]
        )
    nonfinite = humans.clone()
    nonfinite[0, 0] = float("inf")
    with pytest.raises(LowerVarianceError, match="finite"):
        one_round_mmd_human_witness_weights(generated.detach(), nonfinite, [1.0])
    with pytest.raises(LowerVarianceError, match="temperature"):
        one_round_mmd_human_witness_weights(
            generated.detach(), humans, [1.0], temperature=True
        )

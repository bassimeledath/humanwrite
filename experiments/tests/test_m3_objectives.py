from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.m3.objectives import (
    M3ObjectiveError,
    calibrate_moment_coefficient,
    select_frequent_tokens,
    surface_features,
    token_moment_loss,
    witness_weights,
)


def test_frequent_token_selection_is_deterministic_and_excludes_specials() -> None:
    sequences = [list(range(300)) + [7] * 20, list(range(300)) + [8] * 10]
    selected, frequency = select_frequent_tokens(sequences, special_ids={0, 1}, count=256)
    assert selected[:2] == [7, 8]
    assert 0 not in selected and 1 not in selected
    assert frequency.shape == (256,)


def test_token_moment_loss_is_finite_and_differentiable() -> None:
    logits = torch.randn(2, 5, 300, requires_grad=True)
    labels = torch.tensor([[-100, -100, 2, 3, 4], [-100, 8, 9, 10, 11]])
    selected = list(range(256))
    target = np.full(256, 1 / 256, dtype=np.float64)
    loss = token_moment_loss(logits, labels, selected, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_moment_coefficient_calibration_uses_frozen_grid_and_tie_break() -> None:
    ce = [10.0] * 32
    raw = [20.0] * 32
    assert calibrate_moment_coefficient(ce, raw) == 0.1
    with pytest.raises(M3ObjectiveError, match="32"):
        calibrate_moment_coefficient(ce[:-1], raw[:-1])


def test_surface_features_are_exactly_twelve_and_sensitive_to_style() -> None:
    plain = surface_features("One sentence. Another sentence.")
    marked = surface_features("One clause, another clause; a third: (note)—done.\n\nNext paragraph.")
    assert plain.shape == marked.shape == (12,)
    assert marked[6] > plain[6]
    assert marked[10] > plain[10]
    assert marked[2] > plain[2]


def test_witness_weights_upweight_undercovered_direction_and_normalize() -> None:
    rng = np.random.default_rng(17)
    all_human = rng.normal(size=(2048, 20))
    subset_human = all_human[:512]
    subset_policy = subset_human.copy()
    subset_policy[:, 0] -= 1.0
    weights, state = witness_weights(all_human, subset_human, subset_policy)
    assert weights.shape == (2048,)
    assert np.isclose(weights.mean(), 1.0)
    assert weights.max() <= 2.0 / weights.mean() + 1e-12
    assert state["gap"][0] > 0
    assert np.corrcoef(weights, all_human[:, 0])[0, 1] > 0.5

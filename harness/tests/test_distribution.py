import numpy as np
import pytest

from harness.metrics import distribution


def embed(texts):
    return np.array([[len(text), sum(map(ord, text)) % 101] for text in texts], dtype=float)


def test_semantic_mmd_is_deterministic_and_injected():
    human = ["one alpha", "two beta", "three gamma", "four delta"]
    generated = ["far away x" * 4, "far away y" * 5, "far away z" * 6]
    first = distribution.semantic_mmd(generated, human, embed)
    second = distribution.semantic_mmd(generated, human, embed)
    assert first == second
    assert np.isfinite(first)


def test_semantic_mmd_validates_inputs():
    with pytest.raises(ValueError, match="at least two"):
        distribution.semantic_mmd(["one"], ["a", "b"], embed)
    with pytest.raises(ValueError, match="positive"):
        distribution.semantic_mmd(["a", "b"], ["c", "d"], embed, (0,))


def test_human_floor_is_stable():
    texts = [f"document number {number}" for number in range(8)]
    assert distribution.human_floor_mmd(texts, embed, n_boot=12) == distribution.human_floor_mmd(
        texts, embed, n_boot=12
    )


def test_lexical_l2_explicit_and_hashed_features():
    assert distribution.lexical_l2(["red blue"], ["red blue"], {"hash_dim": 32}) == 0.0
    distance = distribution.lexical_l2(
        ["red red"], ["blue blue"], {"features": ["red", "blue"], "ngram_range": (1, 1)}
    )
    assert distance == pytest.approx(np.sqrt(2))
    assert distance == distribution.lexical_l2(
        ["red red"], ["blue blue"], {"features": ["red", "blue"], "ngram_range": (1, 1)}
    )


def test_structural_js_distance():
    texts = ["The short sentence. We continue.\n\nAnother paragraph."]
    assert distribution.structural_distance(texts, texts) == pytest.approx(0.0)
    changed = ["And " + "word " * 250 + "."]
    assert 0 < distribution.structural_distance(changed, texts) <= 1

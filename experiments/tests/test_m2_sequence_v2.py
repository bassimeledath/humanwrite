from types import SimpleNamespace

import pytest
import torch

from experiments.m2.sequence_v2 import (
    FULL_BRIEF_SCHEMA_V2,
    FULL_BRIEF_SERIALIZER_V2_SHA256,
    SequenceV2Error,
    materialize_sampled_batch,
    normalize_legacy_brief_as_tokens,
    render_full_brief_v2,
    sample_raw_policy_eos_aware,
    sequence_log_probs_eos_aware,
)


def _brief():
    return {
        "user_prompt": "Explain the orchard harvest.",
        "use_case": "article",
        "style_kind": "reported",
        "style": "plain",
        "detail_mode": "strict",
        "target_length": 64,
        "em_dashes_allowed": False,
        "outline": [],
    }


def test_v2_serializer_migrates_known_legacy_token_estimate_explicitly():
    migrated = normalize_legacy_brief_as_tokens(_brief())
    rendered = render_full_brief_v2(migrated)
    assert FULL_BRIEF_SCHEMA_V2 == "dft.full-brief.v2"
    assert len(FULL_BRIEF_SERIALIZER_V2_SHA256) == 64
    assert "Target length: about 64 tokens" in rendered
    assert "words" not in rendered


def test_v2_serializer_rejects_ambiguous_or_wrong_unit():
    with pytest.raises(SequenceV2Error, match="missing fields"):
        render_full_brief_v2(_brief())
    row = {**_brief(), "target_length_unit": "words"}
    with pytest.raises(SequenceV2Error, match="must be tokens"):
        render_full_brief_v2(row)


def test_eos_aware_sampler_masks_post_eos_actions_and_pads_shape():
    class Model:
        def __call__(
            self, *, input_ids, attention_mask, past_key_values, use_cache, return_dict
        ):
            logits = torch.full((*input_ids.shape, 6), -1000.0)
            logits[:, -1, 2] = 1000.0
            return SimpleNamespace(logits=logits, past_key_values=object())

    prompt = torch.tensor([[0, 4], [0, 5]])
    mask = torch.tensor([[0, 1], [0, 1]])
    batch = sample_raw_policy_eos_aware(
        Model(),
        prompt,
        mask,
        max_new_tokens=4,
        eos_token_id=2,
        pad_token_id=0,
    )
    assert batch.sequences.tolist() == [[0, 4, 2, 0, 0, 0], [0, 5, 2, 0, 0, 0]]
    assert batch.action_mask.tolist() == [[1, 0, 0, 0], [1, 0, 0, 0]]
    assert batch.sequences.is_inference()
    materialized = materialize_sampled_batch(batch)
    assert not materialized.sequences.is_inference()
    assert not materialized.action_mask.is_inference()


def test_sequence_log_probs_scores_eos_but_not_padded_post_eos_positions():
    class Model:
        def __call__(self, *, input_ids, attention_mask, return_dict):
            self.attention_mask = attention_mask
            return SimpleNamespace(
                logits=torch.zeros((*input_ids.shape, 8), dtype=torch.float32)
            )

    model = Model()
    sequences = torch.tensor([[0, 4, 5, 2, 0, 0]])
    prompt_mask = torch.tensor([[0, 1]])
    action_mask = torch.tensor([[1, 1, 0, 0]])
    result = sequence_log_probs_eos_aware(
        model, sequences, prompt_mask, prompt_width=2, action_mask=action_mask
    )
    assert torch.equal(model.attention_mask, torch.tensor([[0, 1, 1, 1, 0, 0]]))
    assert result.item() == pytest.approx(-2 * torch.log(torch.tensor(8.0)).item())

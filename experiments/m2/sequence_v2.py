"""Versioned prompt and rollout semantics for the post-audit M2 cycle.

The historical v1 experiment intentionally remains unchanged.  This module
repairs two defects prospectively: target length has an explicit token unit,
and EOS ends both visible generation and the score-function action mask.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any


FULL_BRIEF_SCHEMA_V2 = "dft.full-brief.v2"
TARGET_LENGTH_UNIT = "tokens"


class SequenceV2Error(ValueError):
    pass


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


FULL_BRIEF_SERIALIZER_V2_SHA256 = canonical_hash(
    {
        "schema": FULL_BRIEF_SCHEMA_V2,
        "required_fields": [
            "user_prompt",
            "use_case",
            "style_kind",
            "style",
            "detail_mode",
            "target_length",
            "target_length_unit",
            "em_dashes_allowed",
            "outline",
        ],
        "target_length_unit": TARGET_LENGTH_UNIT,
        "brief_lines": [
            "Writing request: {user_prompt}",
            "Use case: {use_case}",
            "Style category: {style_kind}",
            "Style: {style}",
            "Detail mode: {detail_mode}",
            "Target length: about {target_length} tokens",
            "Em dashes allowed: {yes_or_no}",
            "Grounding outline (use only these supported facts when non-empty): {outline_json}",
        ],
        "outline_json": {
            "ensure_ascii": False,
            "sort_keys": True,
            "separators": [",", ":"],
        },
        "prompt_format": "USER:\n{brief}\nASSISTANT:",
    }
)


def normalize_legacy_brief_as_tokens(record: dict[str, Any]) -> dict[str, Any]:
    """Make the article-disclosed unit explicit without changing its value.

    Existing synthesis asked the provider for a token estimate, but the stored
    rows omitted the unit.  This migration is valid only for those known rows;
    callers must opt into it rather than silently guessing a unit.
    """
    result = dict(record)
    if "target_length_unit" in result:
        if result["target_length_unit"] != TARGET_LENGTH_UNIT:
            raise SequenceV2Error("target_length_unit must be tokens")
        return result
    result["target_length_unit"] = TARGET_LENGTH_UNIT
    return result


def render_full_brief_v2(record: dict[str, Any]) -> str:
    required = (
        "user_prompt",
        "use_case",
        "style_kind",
        "style",
        "detail_mode",
        "target_length",
        "target_length_unit",
        "em_dashes_allowed",
        "outline",
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise SequenceV2Error(
            f"canonical v2 full brief is missing fields: {', '.join(missing)}"
        )
    if record["target_length_unit"] != TARGET_LENGTH_UNIT:
        raise SequenceV2Error("target_length_unit must be tokens")
    target_length = record["target_length"]
    if (
        isinstance(target_length, bool)
        or not isinstance(target_length, int)
        or target_length <= 0
    ):
        raise SequenceV2Error("target_length must be a positive integer")
    outline = record["outline"]
    if not isinstance(outline, list):
        raise SequenceV2Error("outline must be a list")
    user_prompt = str(record["user_prompt"]).strip()
    if not user_prompt:
        raise SequenceV2Error("user_prompt must be non-empty")
    return "\n".join(
        (
            f"Writing request: {user_prompt}",
            f"Use case: {str(record['use_case']).strip()}",
            f"Style category: {str(record['style_kind']).strip()}",
            f"Style: {str(record['style']).strip()}",
            f"Detail mode: {str(record['detail_mode']).strip()}",
            f"Target length: about {target_length} tokens",
            f"Em dashes allowed: {'yes' if bool(record['em_dashes_allowed']) else 'no'}",
            "Grounding outline (use only these supported facts when non-empty): "
            + json.dumps(
                outline,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    )


@dataclass(frozen=True)
class SampledSequenceBatch:
    sequences: Any
    action_mask: Any


def sample_raw_policy_eos_aware(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    *,
    max_new_tokens: int,
    eos_token_id: int,
    pad_token_id: int,
) -> SampledSequenceBatch:
    """Sample raw categorical logits and mask every action after first EOS."""
    import torch

    if max_new_tokens <= 0:
        raise SequenceV2Error("max_new_tokens must be positive")
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise SequenceV2Error("input_ids and attention_mask must be aligned rank-2 tensors")
    sequences = input_ids
    running_mask = attention_mask
    next_input = input_ids
    past_key_values = None
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    action_columns = []
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            output = model(
                input_ids=next_input,
                attention_mask=running_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            logits = output.logits[:, -1].float()
            if not torch.all(torch.isfinite(logits)):
                raise SequenceV2Error("raw policy sampler produced non-finite logits")
            active = ~finished
            sampled = torch.full(
                (input_ids.shape[0], 1),
                int(pad_token_id),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            if bool(active.any()):
                sampled[active] = torch.multinomial(
                    torch.softmax(logits[active], dim=-1), num_samples=1
                )
            action_column = active.to(dtype=attention_mask.dtype).unsqueeze(1)
            action_columns.append(action_column)
            sequences = torch.cat((sequences, sampled), dim=1)
            running_mask = torch.cat((running_mask, action_column), dim=1)
            finished = finished | (active & sampled.squeeze(1).eq(int(eos_token_id)))
            next_input = sampled
            past_key_values = output.past_key_values
            if past_key_values is None:
                raise SequenceV2Error(
                    "raw policy sampler requires an autoregressive cache"
                )
            if bool(finished.all()):
                remaining = max_new_tokens - len(action_columns)
                if remaining:
                    padding = torch.full(
                        (input_ids.shape[0], remaining),
                        int(pad_token_id),
                        dtype=input_ids.dtype,
                        device=input_ids.device,
                    )
                    sequences = torch.cat((sequences, padding), dim=1)
                    action_columns.extend(
                        torch.zeros(
                            (input_ids.shape[0], 1),
                            dtype=attention_mask.dtype,
                            device=input_ids.device,
                        )
                        for _ in range(remaining)
                    )
                break
    return SampledSequenceBatch(
        sequences=sequences,
        action_mask=torch.cat(action_columns, dim=1),
    )


def materialize_sampled_batch(batch: SampledSequenceBatch) -> SampledSequenceBatch:
    """Convert inference tensors to ordinary tensors for teacher-forced autograd."""
    return SampledSequenceBatch(
        sequences=batch.sequences.clone(),
        action_mask=batch.action_mask.clone(),
    )


def sequence_log_probs_eos_aware(
    model: Any,
    sequences: Any,
    prompt_attention_mask: Any,
    prompt_width: int,
    action_mask: Any,
) -> Any:
    import torch.nn.functional as F

    expected = (sequences.shape[0], sequences.shape[1] - prompt_width)
    if tuple(action_mask.shape) != expected:
        raise SequenceV2Error("action_mask shape does not match continuation")
    attention_mask = __import__("torch").cat(
        (prompt_attention_mask, action_mask.to(prompt_attention_mask.dtype)), dim=1
    )
    output = model(input_ids=sequences, attention_mask=attention_mask, return_dict=True)
    log_probs = F.log_softmax(output.logits[:, :-1].float(), dim=-1)
    labels = sequences[:, 1:]
    selected = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    continuation = selected[:, prompt_width - 1 :]
    return (continuation * action_mask.to(continuation.dtype)).sum(dim=1)

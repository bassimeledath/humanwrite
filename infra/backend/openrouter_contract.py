"""Pure request-shape helpers for sanctioned OpenRouter calls."""
from __future__ import annotations

from typing import Any


def structured_chat_request(
    *,
    model: str,
    prompt: str,
    response_format: dict[str, Any],
    max_completion_tokens: int,
    reasoning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a chat-completions payload that requires parameter-compatible routing."""

    request: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": response_format,
        "max_completion_tokens": max_completion_tokens,
        "provider": {"require_parameters": True},
    }
    if reasoning is not None:
        request["reasoning"] = reasoning
    return request


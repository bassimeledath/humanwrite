"""Pure request-shape helpers for sanctioned OpenRouter calls."""
from __future__ import annotations

from typing import Any


def chat_request(
    *,
    model: str,
    prompt: str,
    max_completion_tokens: int,
    response_format: dict[str, Any] | None = None,
    reasoning: dict[str, Any] | None = None,
    plugins: list[dict[str, Any]] | None = None,
    require_parameters: bool = True,
) -> dict[str, Any]:
    """Build a sanctioned chat-completions request payload."""

    request: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        # OpenRouter's Qwen and Gemini routes advertise the portable
        # chat-completions field `max_tokens`, not the OpenAI-specific
        # `max_completion_tokens`. With require_parameters enabled, using the
        # latter incorrectly eliminates otherwise compatible endpoints.
        "max_tokens": max_completion_tokens,
    }
    if response_format is not None:
        request["response_format"] = response_format
    if require_parameters:
        request["provider"] = {"require_parameters": True}
    if reasoning is not None:
        request["reasoning"] = reasoning
    if plugins:
        request["plugins"] = plugins
    return request


def structured_chat_request(
    *,
    model: str,
    prompt: str,
    response_format: dict[str, Any],
    max_completion_tokens: int,
    reasoning: dict[str, Any] | None = None,
    plugins: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a structured request that requires parameter-compatible routing."""

    return chat_request(
        model=model,
        prompt=prompt,
        response_format=response_format,
        max_completion_tokens=max_completion_tokens,
        reasoning=reasoning,
        plugins=plugins,
    )

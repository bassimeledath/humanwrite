from __future__ import annotations

from backend.openrouter_contract import (
    chat_request,
    schema_transport_fallback_allowed,
    structured_chat_request,
)


def test_structured_chat_request_requires_parameter_compatible_routing() -> None:
    request = structured_chat_request(
        model="qwen/qwen3-32b",
        prompt="Return JSON.",
        response_format={"type": "json_object"},
        max_completion_tokens=512,
    )

    assert request["provider"] == {"require_parameters": True}
    assert request["messages"] == [{"role": "user", "content": "Return JSON."}]
    assert request["max_tokens"] == 512
    assert "max_completion_tokens" not in request


def test_structured_chat_request_preserves_optional_reasoning() -> None:
    request = structured_chat_request(
        model="openai/gpt-5-mini",
        prompt="Return JSON.",
        response_format={"type": "json_object"},
        max_completion_tokens=256,
        reasoning={"effort": "minimal", "exclude": True},
    )

    assert request["reasoning"] == {"effort": "minimal", "exclude": True}


def test_structured_chat_request_preserves_optional_plugins() -> None:
    request = structured_chat_request(
        model="qwen/qwen3-32b",
        prompt="Return JSON.",
        response_format={"type": "json_object"},
        max_completion_tokens=256,
        plugins=[{"id": "response-healing"}],
    )

    assert request["plugins"] == [{"id": "response-healing"}]


def test_chat_request_allows_unstructured_qwen_fallback() -> None:
    request = chat_request(
        model="qwen/qwen3-32b",
        prompt="Return only JSON.",
        max_completion_tokens=192,
    )

    assert "response_format" not in request
    assert request["provider"] == {"require_parameters": True}
    assert request["max_tokens"] == 192
    assert "max_completion_tokens" not in request


def test_schema_transport_fallback_is_limited_to_compatibility_failures() -> None:
    assert schema_transport_fallback_allowed(
        "anthropic/claude-haiku-4.5",
        "json_schema provider HTTP 400: output_config.format.schema rejects maxItems",
    )
    assert schema_transport_fallback_allowed(
        "qwen/qwen3-32b",
        "provider HTTP 404: No endpoints found that can handle the requested parameters",
    )
    assert not schema_transport_fallback_allowed(
        "anthropic/claude-haiku-4.5", "provider HTTP 401: invalid API key"
    )

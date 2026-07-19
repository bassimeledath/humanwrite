from __future__ import annotations

from backend.openrouter_contract import chat_request, structured_chat_request


def test_structured_chat_request_requires_parameter_compatible_routing() -> None:
    request = structured_chat_request(
        model="qwen/qwen3-32b",
        prompt="Return JSON.",
        response_format={"type": "json_object"},
        max_completion_tokens=512,
    )

    assert request["provider"] == {"require_parameters": True}
    assert request["messages"] == [{"role": "user", "content": "Return JSON."}]
    assert request["max_completion_tokens"] == 512


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

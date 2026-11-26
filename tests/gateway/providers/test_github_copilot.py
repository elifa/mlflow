from unittest import mock

import pytest
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import EndpointConfig
from mlflow.gateway.providers.base import PassthroughAction
from mlflow.gateway.providers.github_copilot import (
    _DEFAULT_COPILOT_HEADERS,
    GitHubCopilotProvider,
)
from mlflow.gateway.schemas import chat
from mlflow.tracing.client import TracingClient
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.uri import append_to_uri_path

from tests.gateway.tools import MockAsyncResponse, mock_http_client


def _make_provider(config: dict | None = None) -> GitHubCopilotProvider:
    endpoint_config = EndpointConfig(
        name="github-copilot-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": "github-copilot",
            "name": "gpt-4o",
            "config": config or {"api_key": "server_key"},
        },
    )
    return GitHubCopilotProvider(endpoint_config)


def _chat_response():
    return {
        "id": "chatcmpl-copilot-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o",
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 22,
            "total_tokens": 33,
        },
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello from Copilot!"},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
    }


def test_default_api_base():
    provider = _make_provider()
    assert provider._api_base == "https://api.githubcopilot.com"


def test_configured_api_base_overrides_default():
    provider = _make_provider({"api_key": "server_key", "api_base": "https://copilot.internal/api"})
    assert provider._api_base == "https://copilot.internal/api"


def test_name():
    provider = _make_provider()
    assert provider.DISPLAY_NAME == "GitHub Copilot"


def test_headers():
    provider = _make_provider()
    assert provider.headers == {
        "Authorization": "Bearer server_key",
        "Copilot-Integration-Id": "copilot-developer-cli",
        "X-GitHub-Api-Version": "2026-01-09",
    }


@pytest.mark.asyncio
async def test_chat():
    provider = _make_provider()
    mock_client = mock_http_client(MockAsyncResponse(_chat_response()))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client):
        payload = chat.RequestPayload(
            messages=[{"role": "user", "content": "Hello"}],
        )
        response = await provider.chat(payload)

    result = jsonable_encoder(response)
    assert result["id"] == "chatcmpl-copilot-123"
    assert result["choices"][0]["message"]["content"] == "Hello from Copilot!"
    assert result["usage"]["total_tokens"] == 33


@pytest.mark.asyncio
async def test_chat_sends_copilot_headers_upstream():
    provider = _make_provider()
    mock_client = mock_http_client(MockAsyncResponse(_chat_response()))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client) as mock_session:
        await provider.chat(chat.RequestPayload(messages=[{"role": "user", "content": "Hello"}]))

    sent_headers = mock_session.call_args.kwargs["headers"]
    assert sent_headers["Copilot-Integration-Id"] == "copilot-developer-cli"
    assert sent_headers["X-GitHub-Api-Version"] == "2026-01-09"
    assert sent_headers["Authorization"] == "Bearer server_key"


def test_get_headers_injects_copilot_defaults():
    provider = _make_provider()
    headers = provider._get_headers(None)
    assert headers["Copilot-Integration-Id"] == "copilot-developer-cli"
    assert headers["X-GitHub-Api-Version"] == "2026-01-09"
    assert headers["Authorization"] == "Bearer server_key"


@pytest.mark.parametrize(
    ("header_name", "header_value"),
    [
        ("Copilot-Integration-Id", "my-integration"),
        ("copilot-integration-id", "my-integration"),
        ("X-GitHub-Api-Version", "2025-05-01"),
        ("x-github-api-version", "2025-05-01"),
    ],
)
def test_get_headers_client_values_win_over_defaults(header_name, header_value):
    provider = _make_provider()
    headers = provider._get_headers({header_name: header_value})
    assert headers[header_name] == header_value
    assert sum(k.lower() == header_name.lower() for k in headers) == 1


def test_get_headers_preserves_client_credentials_for_copilot_cli():
    provider = _make_provider()
    headers = provider._get_headers({
        "user-agent": "copilot-linux-x64/1.0.75 (linux v24.18.0) term/vscode",
        "authorization": "Bearer user-token",
    })
    assert headers["authorization"] == "Bearer user-token"
    assert "Authorization" not in headers


def test_get_headers_uses_server_key_for_unknown_agent():
    provider = _make_provider()
    headers = provider._get_headers({
        "user-agent": "python-httpx/0.27.0",
        "authorization": "Bearer user-token",
    })
    assert headers["Authorization"] == "Bearer server_key"
    assert "authorization" not in headers


def test_passthrough_provider_paths_resolve_to_copilot_urls():
    provider = _make_provider()
    resolved = {
        action: append_to_uri_path(provider._api_base, path)
        for action, path in GitHubCopilotProvider.PASSTHROUGH_PROVIDER_PATHS.items()
    }
    assert resolved == {
        PassthroughAction.OPENAI_CHAT: "https://api.githubcopilot.com/chat/completions",
        PassthroughAction.OPENAI_EMBEDDINGS: "https://api.githubcopilot.com/embeddings",
        PassthroughAction.OPENAI_RESPONSES: "https://api.githubcopilot.com/responses",
        PassthroughAction.ANTHROPIC_MESSAGES: "https://api.githubcopilot.com/v1/messages",
    }


@pytest.mark.asyncio
async def test_passthrough_forwards_client_token_for_copilot_cli():
    provider = _make_provider()
    mock_client = mock_http_client(MockAsyncResponse(_chat_response()))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client) as mock_session:
        await provider._passthrough(
            PassthroughAction.OPENAI_CHAT,
            {"messages": [{"role": "user", "content": "Hello"}]},
            headers={
                "user-agent": "copilot-linux-x64/1.0.75 (linux v24.18.0) term/vscode",
                "authorization": "Bearer user-token",
            },
        )

    sent_headers = mock_session.call_args.kwargs["headers"]
    assert sent_headers["authorization"] == "Bearer user-token"
    assert "Bearer server_key" not in sent_headers.values()
    assert sent_headers["Copilot-Integration-Id"] == "copilot-developer-cli"


def test_extract_passthrough_token_usage_anthropic_messages():
    provider = _make_provider()
    usage = provider._extract_passthrough_token_usage(
        PassthroughAction.ANTHROPIC_MESSAGES,
        {
            "usage": {
                "input_tokens": 5,
                "output_tokens": 7,
                "cache_read_input_tokens": 2,
                "cache_creation_input_tokens": 3,
                "cache_creation": {"ephemeral_5m_input_tokens": 2, "ephemeral_1h_input_tokens": 1},
            }
        },
    )
    # Anthropic reports input_tokens excluding cache tokens; they are folded back in.
    assert usage == {
        "input_tokens": 10,
        "output_tokens": 7,
        "total_tokens": 17,
        "cache_read_input_tokens": 2,
        "cache_creation_input_tokens": 3,
        "cache_creation_input_tokens_above_1hr": 1,
    }


def test_extract_passthrough_token_usage_openai_responses():
    provider = _make_provider()
    usage = provider._extract_passthrough_token_usage(
        PassthroughAction.OPENAI_RESPONSES,
        {
            "usage": {
                "input_tokens": 12,
                "output_tokens": 8,
                "total_tokens": 20,
                "input_tokens_details": {"cached_tokens": 4},
            }
        },
    )
    assert usage["input_tokens"] == 12
    assert usage["output_tokens"] == 8
    assert usage["total_tokens"] == 20
    assert usage["cache_read_input_tokens"] == 4


def test_extract_streaming_token_usage_openai_chat():
    provider = _make_provider()
    chunk = (
        b'data: {"choices": [], "usage": {"prompt_tokens": 10, '
        b'"completion_tokens": 20, "total_tokens": 30}}\n\n'
    )
    assert provider._extract_streaming_token_usage(chunk) == {
        "input_tokens": 10,
        "output_tokens": 20,
        "total_tokens": 30,
    }


def test_extract_streaming_token_usage_openai_responses():
    provider = _make_provider()
    chunk = (
        b'data: {"type": "response.completed", "response": {"usage": '
        b'{"input_tokens": 12, "output_tokens": 8, "total_tokens": 20, '
        b'"input_tokens_details": {"cached_tokens": 4}}}}\n\n'
    )
    assert provider._extract_streaming_token_usage(chunk) == {
        "input_tokens": 12,
        "output_tokens": 8,
        "total_tokens": 20,
        "cache_read_input_tokens": 4,
    }


def test_extract_streaming_token_usage_returns_empty_without_usage():
    provider = _make_provider()
    assert provider._extract_streaming_token_usage(b'data: {"choices": []}\n\n') == {}


def test_extract_passthrough_token_usage_openai_chat():
    provider = _make_provider()
    usage = provider._extract_passthrough_token_usage(
        PassthroughAction.OPENAI_CHAT,
        {"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}},
    )
    assert usage["input_tokens"] == 10
    assert usage["output_tokens"] == 20
    assert usage["total_tokens"] == 30


def test_default_copilot_headers_constant():
    assert set(_DEFAULT_COPILOT_HEADERS) == {"Copilot-Integration-Id", "X-GitHub-Api-Version"}


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("chat/completions", PassthroughAction.OPENAI_CHAT),
        ("v1/chat/completions", PassthroughAction.OPENAI_CHAT),
        ("/chat/completions?stream=true", PassthroughAction.OPENAI_CHAT),
        ("responses", PassthroughAction.OPENAI_RESPONSES),
        ("v1/responses", PassthroughAction.OPENAI_RESPONSES),
        ("v1/messages", PassthroughAction.ANTHROPIC_MESSAGES),
        ("messages", PassthroughAction.ANTHROPIC_MESSAGES),
        ("embeddings", PassthroughAction.OPENAI_EMBEDDINGS),
        ("models", None),
    ],
)
def test_passthrough_action_for_proxy_path(path, expected):
    assert _make_provider()._passthrough_action_for_path(path) == expected


def _mock_proxy_request(*items):
    def factory(headers, base_url, method, path, payload):
        async def gen():
            for item in items:
                yield item

        return gen()

    return mock.patch(
        "mlflow.gateway.providers.openai_compatible.send_proxy_request", side_effect=factory
    )


def _traces():
    return TracingClient().search_traces(locations=[_get_experiment_id()])


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["chat/completions", "v1/chat/completions"])
async def test_proxy_records_token_usage_non_streaming(path):
    provider = _make_provider()
    provider._enable_tracing = True

    with _mock_proxy_request({"is_streaming": False}, _chat_response()):
        result = await provider.proxy(path, {"messages": [{"role": "user", "content": "Hi"}]})

    assert result["id"] == "chatcmpl-copilot-123"

    traces = _traces()
    assert len(traces) == 1
    usage = traces[0].data.spans[0].attributes.get(SpanAttributeKey.CHAT_USAGE)
    assert usage[TokenUsageKey.INPUT_TOKENS] == 11
    assert usage[TokenUsageKey.OUTPUT_TOKENS] == 22
    assert usage[TokenUsageKey.TOTAL_TOKENS] == 33


_MESSAGE_START = (
    b'data: {"type": "message_start", "message": {"usage": {"input_tokens": 9, '
    b'"cache_read_input_tokens": 2, "cache_creation_input_tokens": 3, '
    b'"cache_creation": {"ephemeral_1h_input_tokens": 1}}}}\n\n'
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message_delta", "expected"),
    [
        (
            b'data: {"type": "message_delta", "usage": {"output_tokens": 4}}\n\n',
            {
                TokenUsageKey.INPUT_TOKENS: 14,
                TokenUsageKey.OUTPUT_TOKENS: 4,
                TokenUsageKey.TOTAL_TOKENS: 18,
                TokenUsageKey.CACHE_READ_INPUT_TOKENS: 2,
                TokenUsageKey.CACHE_CREATION_INPUT_TOKENS: 3,
                TokenUsageKey.CACHE_CREATION_INPUT_TOKENS_ABOVE_1HR: 1,
            },
        ),
        # With server tools, message_delta carries the final cumulative input and cache counts.
        (
            b'data: {"type": "message_delta", "usage": {"input_tokens": 20, '
            b'"cache_read_input_tokens": 5, "cache_creation_input_tokens": 3, '
            b'"output_tokens": 4}}\n\n',
            {
                TokenUsageKey.INPUT_TOKENS: 28,
                TokenUsageKey.OUTPUT_TOKENS: 4,
                TokenUsageKey.TOTAL_TOKENS: 32,
                TokenUsageKey.CACHE_READ_INPUT_TOKENS: 5,
                TokenUsageKey.CACHE_CREATION_INPUT_TOKENS: 3,
                TokenUsageKey.CACHE_CREATION_INPUT_TOKENS_ABOVE_1HR: 1,
            },
        ),
    ],
)
async def test_proxy_records_token_usage_streaming_anthropic_messages(message_delta, expected):
    # The Anthropic shape reports usage across two events without any opt-in field, and they
    # arrive in separate chunks, so the counts can only be normalized once the stream ends.
    provider = _make_provider()
    provider._enable_tracing = True

    chunks = [_MESSAGE_START, message_delta]
    with _mock_proxy_request({"is_streaming": True}, *chunks):
        stream = await provider.proxy("v1/messages", {"stream": True})
        assert [chunk async for chunk in stream] == chunks

    traces = _traces()
    assert len(traces) == 1
    assert traces[0].data.spans[0].attributes.get(SpanAttributeKey.CHAT_USAGE) == expected


@pytest.mark.asyncio
async def test_proxy_records_token_usage_streaming_openai_responses():
    # Cached input is already part of the OpenAI input_tokens, so it must not be folded in again.
    provider = _make_provider()
    provider._enable_tracing = True

    chunks = [
        b'data: {"type": "response.created", "response": {"usage": null}}\n\n',
        b'data: {"type": "response.completed", "response": {"usage": {"input_tokens": 12, '
        b'"output_tokens": 8, "total_tokens": 20, "input_tokens_details": '
        b'{"cached_tokens": 4}}}}\n\n',
    ]
    with _mock_proxy_request({"is_streaming": True}, *chunks):
        stream = await provider.proxy("responses", {"stream": True})
        assert [chunk async for chunk in stream] == chunks

    traces = _traces()
    assert len(traces) == 1
    assert traces[0].data.spans[0].attributes.get(SpanAttributeKey.CHAT_USAGE) == {
        TokenUsageKey.INPUT_TOKENS: 12,
        TokenUsageKey.OUTPUT_TOKENS: 8,
        TokenUsageKey.TOTAL_TOKENS: 20,
        TokenUsageKey.CACHE_READ_INPUT_TOKENS: 4,
    }


@pytest.mark.asyncio
async def test_proxy_records_no_token_usage_for_model_discovery():
    provider = _make_provider()
    provider._enable_tracing = True

    with _mock_proxy_request({"is_streaming": False}, {"object": "list", "data": []}):
        result = await provider.proxy("models", {}, method="GET")

    assert result == {"object": "list", "data": []}

    traces = _traces()
    assert len(traces) == 1
    assert traces[0].data.spans[0].attributes.get(SpanAttributeKey.CHAT_USAGE) is None

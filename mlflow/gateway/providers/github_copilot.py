from typing import Any

from mlflow.gateway.config import _OpenAICompatibleConfig
from mlflow.gateway.providers.anthropic import _normalize_anthropic_input_tokens
from mlflow.gateway.providers.base import PassthroughAction
from mlflow.gateway.providers.openai_compatible import OpenAICompatibleProvider
from mlflow.gateway.utils import parse_sse_lines
from mlflow.tracing.constant import TokenUsageKey

# Values GitHub's own clients send: the integration id from the Copilot CLI, the API version
# from Copilot Chat's CAPI chat requests.
_DEFAULT_COPILOT_HEADERS = {
    "Copilot-Integration-Id": "copilot-developer-cli",
    "X-GitHub-Api-Version": "2026-01-09",
}


class GitHubCopilotProvider(OpenAICompatibleProvider):
    DISPLAY_NAME = "GitHub Copilot"
    CONFIG_TYPE = _OpenAICompatibleConfig
    DEFAULT_API_BASE = "https://api.githubcopilot.com"

    PASSTHROUGH_PROVIDER_PATHS = {
        **OpenAICompatibleProvider.PASSTHROUGH_PROVIDER_PATHS,
        PassthroughAction.OPENAI_RESPONSES: "responses",
        PassthroughAction.ANTHROPIC_MESSAGES: "v1/messages",
    }

    def get_provider_name(self) -> str:
        return "github-copilot"

    @property
    def headers(self) -> dict[str, str]:
        return {**super().headers, **_DEFAULT_COPILOT_HEADERS}

    def _get_headers(self, headers: dict[str, str] | None = None) -> dict[str, str]:
        result_headers = super()._get_headers(headers)
        client_headers = {k.lower(): (k, v) for k, v in (headers or {}).items()}
        # The base merge lets the server defaults win; hand the client's own value and casing
        # back so only one variant of each Copilot header goes on the wire.
        for name in _DEFAULT_COPILOT_HEADERS:
            if (client_header := client_headers.get(name.lower())) is not None:
                result_headers.pop(name, None)
                client_name, client_value = client_header
                result_headers[client_name] = client_value
        return result_headers

    def _extract_passthrough_token_usage(
        self, action: PassthroughAction, result: dict[str, Any]
    ) -> dict[str, int] | None:
        """Extract token usage from a non-streaming passthrough response.

        Copilot serves three different response shapes behind one endpoint, so the usage keys
        depend on the action: Anthropic messages report ``input_tokens`` alongside separate
        cache counters, the OpenAI Responses API reports ``input_tokens``/``output_tokens``,
        and OpenAI chat (handled by ``super()``) reports ``prompt_tokens``/``completion_tokens``.
        """
        if action is PassthroughAction.ANTHROPIC_MESSAGES:
            token_usage = self._extract_token_usage_from_dict(
                result.get("usage"),
                input_tokens_key="input_tokens",
                output_tokens_key="output_tokens",
                cache_read_key="cache_read_input_tokens",
                cache_creation_key="cache_creation_input_tokens",
            )
            return _normalize_anthropic_input_tokens(token_usage)
        if action is PassthroughAction.OPENAI_RESPONSES:
            return self._extract_token_usage_from_dict(
                result.get("usage"),
                "input_tokens",
                "output_tokens",
                "total_tokens",
                cache_read_key="input_tokens_details.cached_tokens",
            )
        return super()._extract_passthrough_token_usage(action, result)

    def _extract_streaming_token_usage(self, chunk: bytes) -> dict[str, int]:
        """Extract token usage from a streaming passthrough chunk.

        Case order matters. Anthropic splits its usage across two events, so ``message_start``
        and ``message_delta`` accumulate into ``anthropic_usage`` and fall through rather than
        returning; the ``message_delta`` case must therefore precede the generic ``{"usage":
        ...}`` case, which would otherwise misread Anthropic's cumulative delta as OpenAI chat
        usage. The OpenAI Responses (``{"response": {"usage": ...}}``) and OpenAI chat shapes
        each carry complete usage in a single event and return as soon as one is populated.
        """
        anthropic_usage: dict[str, int] = {}
        for data in parse_sse_lines(chunk):
            match data:
                case {"type": "message_start", "message": {"usage": dict(msg_usage)}}:
                    if (input_tokens := msg_usage.get("input_tokens")) is not None:
                        anthropic_usage[TokenUsageKey.INPUT_TOKENS] = input_tokens
                    if (cached := msg_usage.get("cache_read_input_tokens")) is not None:
                        anthropic_usage[TokenUsageKey.CACHE_READ_INPUT_TOKENS] = cached
                    if (created := msg_usage.get("cache_creation_input_tokens")) is not None:
                        anthropic_usage[TokenUsageKey.CACHE_CREATION_INPUT_TOKENS] = created
                case {"type": "message_delta", "usage": {"output_tokens": int(output_tokens)}}:
                    anthropic_usage[TokenUsageKey.OUTPUT_TOKENS] = output_tokens
                case {"response": {"usage": dict(resp_usage)}}:
                    if token_usage := self._extract_token_usage_from_dict(
                        resp_usage,
                        "input_tokens",
                        "output_tokens",
                        "total_tokens",
                        cache_read_key="input_tokens_details.cached_tokens",
                    ):
                        return token_usage
                case {"usage": dict(chat_usage)}:
                    if token_usage := self._extract_token_usage_from_dict(
                        chat_usage,
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                        cache_read_key="prompt_tokens_details.cached_tokens",
                    ):
                        return token_usage
        return _normalize_anthropic_input_tokens(anthropic_usage) or {}

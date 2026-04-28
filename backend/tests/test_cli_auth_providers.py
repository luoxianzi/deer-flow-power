from __future__ import annotations

import json

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from deerflow.models import openai_codex_provider as codex_provider_module
from deerflow.models.claude_provider import ClaudeAuthenticationUnavailableError, ClaudeChatModel
from deerflow.models.credential_loader import CodexCliCredential
from deerflow.models.openai_codex_provider import CodexChatModel


def test_codex_provider_rejects_non_positive_retry_attempts():
    with pytest.raises(ValueError, match="retry_max_attempts must be >= 1"):
        CodexChatModel(retry_max_attempts=0)


def test_codex_provider_defers_missing_credentials_until_first_request(monkeypatch):
    monkeypatch.setattr(CodexChatModel, "_load_codex_auth", lambda self: None)

    model = CodexChatModel()
    assert model._credential_available is False

    with pytest.raises(ValueError, match="Codex CLI credential not found"):
        model.invoke([HumanMessage(content="hello")])


def test_codex_provider_concatenates_multiple_system_messages(monkeypatch):
    monkeypatch.setattr(
        CodexChatModel,
        "_load_codex_auth",
        lambda self: CodexCliCredential(access_token="token", account_id="acct"),
    )

    model = CodexChatModel()
    instructions, input_items = model._convert_messages(
        [
            SystemMessage(content="First system prompt."),
            SystemMessage(content="Second system prompt."),
            HumanMessage(content="Hello"),
        ]
    )

    assert instructions == "First system prompt.\n\nSecond system prompt."
    assert input_items == [{"role": "user", "content": "Hello"}]


def test_codex_provider_flattens_structured_text_blocks(monkeypatch):
    monkeypatch.setattr(
        CodexChatModel,
        "_load_codex_auth",
        lambda self: CodexCliCredential(access_token="token", account_id="acct"),
    )

    model = CodexChatModel()
    instructions, input_items = model._convert_messages(
        [
            HumanMessage(content=[{"type": "text", "text": "Hello from blocks"}]),
        ]
    )

    assert instructions == "You are a helpful assistant."
    assert input_items == [{"role": "user", "content": "Hello from blocks"}]


def test_claude_provider_rejects_non_positive_retry_attempts():
    with pytest.raises(ValueError, match="retry_max_attempts must be >= 1"):
        ClaudeChatModel(model="claude-sonnet-4-6", retry_max_attempts=0)


def test_claude_provider_requires_valid_auth(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_CREDENTIALS_PATH", raising=False)

    monkeypatch.setattr(
        "deerflow.models.credential_loader.load_claude_code_credential",
        lambda: None,
    )

    with pytest.raises(ValidationError, match="Claude authentication unavailable"):
        ClaudeChatModel(model="claude-sonnet-4-6", retry_max_attempts=1)


def test_codex_provider_skips_terminal_sse_markers(monkeypatch):
    monkeypatch.setattr(
        CodexChatModel,
        "_load_codex_auth",
        lambda self: CodexCliCredential(access_token="token", account_id="acct"),
    )

    model = CodexChatModel()

    assert model._parse_sse_data_line("data: [DONE]") is None
    assert model._parse_sse_data_line("event: response.completed") is None


def test_codex_provider_skips_non_json_sse_frames(monkeypatch):
    monkeypatch.setattr(
        CodexChatModel,
        "_load_codex_auth",
        lambda self: CodexCliCredential(access_token="token", account_id="acct"),
    )

    model = CodexChatModel()

    assert model._parse_sse_data_line("data: not-json") is None


def test_codex_provider_marks_invalid_tool_call_arguments(monkeypatch):
    monkeypatch.setattr(
        CodexChatModel,
        "_load_codex_auth",
        lambda self: CodexCliCredential(access_token="token", account_id="acct"),
    )

    model = CodexChatModel()
    result = model._parse_response(
        {
            "model": "gpt-5.4",
            "output": [
                {
                    "type": "function_call",
                    "name": "bash",
                    "arguments": "{invalid",
                    "call_id": "tc-1",
                }
            ],
            "usage": {},
        }
    )

    message = result.generations[0].message
    assert message.tool_calls == []
    assert len(message.invalid_tool_calls) == 1
    assert message.invalid_tool_calls[0]["type"] == "invalid_tool_call"
    assert message.invalid_tool_calls[0]["name"] == "bash"
    assert message.invalid_tool_calls[0]["args"] == "{invalid"
    assert message.invalid_tool_calls[0]["id"] == "tc-1"
    assert "Failed to parse tool arguments" in message.invalid_tool_calls[0]["error"]


def test_codex_provider_parses_valid_tool_arguments(monkeypatch):
    monkeypatch.setattr(
        CodexChatModel,
        "_load_codex_auth",
        lambda self: CodexCliCredential(access_token="token", account_id="acct"),
    )

    model = CodexChatModel()
    result = model._parse_response(
        {
            "model": "gpt-5.4",
            "output": [
                {
                    "type": "function_call",
                    "name": "bash",
                    "arguments": json.dumps({"cmd": "pwd"}),
                    "call_id": "tc-1",
                }
            ],
            "usage": {},
        }
    )

    assert result.generations[0].message.tool_calls == [{"name": "bash", "args": {"cmd": "pwd"}, "id": "tc-1", "type": "tool_call"}]


class _FakeResponseStream:
    def __init__(self, lines: list[str]):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        yield from self._lines


class _FakeHttpxClient:
    def __init__(self, lines: list[str], *_args, **_kwargs):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def stream(self, *_args, **_kwargs):
        return _FakeResponseStream(self._lines)


def test_codex_provider_merges_streamed_output_items_when_completed_output_is_empty(monkeypatch):
    monkeypatch.setattr(
        CodexChatModel,
        "_load_codex_auth",
        lambda self: CodexCliCredential(access_token="token", account_id="acct"),
    )

    lines = [
        'data: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","content":[{"type":"output_text","text":"Hello from stream"}]}}',
        'data: {"type":"response.completed","response":{"model":"gpt-5.4","output":[],"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}}',
    ]

    monkeypatch.setattr(
        codex_provider_module.httpx,
        "Client",
        lambda *args, **kwargs: _FakeHttpxClient(lines, *args, **kwargs),
    )

    model = CodexChatModel()
    response = model._stream_response(headers={}, payload={})
    parsed = model._parse_response(response)

    assert response["output"] == [
        {
            "type": "message",
            "content": [{"type": "output_text", "text": "Hello from stream"}],
        }
    ]
    assert parsed.generations[0].message.content == "Hello from stream"


def test_codex_provider_orders_streamed_output_items_by_output_index(monkeypatch):
    monkeypatch.setattr(
        CodexChatModel,
        "_load_codex_auth",
        lambda self: CodexCliCredential(access_token="token", account_id="acct"),
    )

    lines = [
        'data: {"type":"response.output_item.done","output_index":1,"item":{"type":"message","content":[{"type":"output_text","text":"Second"}]}}',
        'data: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","content":[{"type":"output_text","text":"First"}]}}',
        'data: {"type":"response.completed","response":{"model":"gpt-5.4","output":[],"usage":{}}}',
    ]

    monkeypatch.setattr(
        codex_provider_module.httpx,
        "Client",
        lambda *args, **kwargs: _FakeHttpxClient(lines, *args, **kwargs),
    )

    model = CodexChatModel()
    response = model._stream_response(headers={}, payload={})

    assert [item["content"][0]["text"] for item in response["output"]] == [
        "First",
        "Second",
    ]


def test_codex_provider_preserves_completed_output_when_stream_only_has_placeholder(monkeypatch):
    monkeypatch.setattr(
        CodexChatModel,
        "_load_codex_auth",
        lambda self: CodexCliCredential(access_token="token", account_id="acct"),
    )

    lines = [
        'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","status":"in_progress","content":[]}}',
        'data: {"type":"response.completed","response":{"model":"gpt-5.4","output":[{"type":"message","content":[{"type":"output_text","text":"Final from completed"}]}],"usage":{}}}',
    ]

    monkeypatch.setattr(
        codex_provider_module.httpx,
        "Client",
        lambda *args, **kwargs: _FakeHttpxClient(lines, *args, **kwargs),
    )

    model = CodexChatModel()
    response = model._stream_response(headers={}, payload={})
    parsed = model._parse_response(response)

    assert response["output"] == [
        {
            "type": "message",
            "content": [{"type": "output_text", "text": "Final from completed"}],
        }
    ]
    assert parsed.generations[0].message.content == "Final from completed"


def test_codex_provider_falls_back_to_mini_after_429(monkeypatch):
    monkeypatch.setattr(
        CodexChatModel,
        "_load_codex_auth",
        lambda self: CodexCliCredential(access_token="token", account_id="acct"),
    )

    model = CodexChatModel(model="gpt-5.4", retry_max_attempts=1)
    seen_models: list[str] = []

    def fake_stream_response(_headers, payload):
        seen_models.append(payload["model"])
        if payload["model"] == "gpt-5.4":
            request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses")
            response = httpx.Response(429, request=request)
            raise httpx.HTTPStatusError("rate limited", request=request, response=response)
        return {
            "model": payload["model"],
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "OK"}],
                }
            ],
            "usage": {},
        }

    monkeypatch.setattr(model, "_stream_response", fake_stream_response)

    result = model.invoke([HumanMessage(content="hello")])

    assert seen_models == ["gpt-5.4", "gpt-5.4-mini"]
    assert result.content == "OK"


def test_codex_provider_does_not_fallback_when_model_has_no_mapping(monkeypatch):
    monkeypatch.setattr(
        CodexChatModel,
        "_load_codex_auth",
        lambda self: CodexCliCredential(access_token="token", account_id="acct"),
    )

    model = CodexChatModel(model="gpt-5.4-mini", retry_max_attempts=1)
    seen_models: list[str] = []

    def fake_stream_response(_headers, payload):
        seen_models.append(payload["model"])
        request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses")
        response = httpx.Response(429, request=request)
        raise httpx.HTTPStatusError("rate limited", request=request, response=response)

    monkeypatch.setattr(model, "_stream_response", fake_stream_response)

    with pytest.raises(httpx.HTTPStatusError):
        model.invoke([HumanMessage(content="hello")])

    assert seen_models == ["gpt-5.4-mini"]


def test_codex_provider_refreshes_credentials_between_requests(monkeypatch):
    creds = iter(
        [
            CodexCliCredential(access_token="token-1", account_id="acct-1"),
            CodexCliCredential(access_token="token-1", account_id="acct-1"),
            CodexCliCredential(access_token="token-2", account_id="acct-2"),
        ]
    )

    monkeypatch.setattr(CodexChatModel, "_load_codex_auth", lambda self: next(creds))

    model = CodexChatModel(model="gpt-5.4", retry_max_attempts=1)
    seen_headers: list[tuple[str, str]] = []

    def fake_stream_response(headers, payload):
        seen_headers.append((headers["Authorization"], headers["ChatGPT-Account-ID"]))
        return {
            "model": payload["model"],
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "OK"}],
                }
            ],
            "usage": {},
        }

    monkeypatch.setattr(model, "_stream_response", fake_stream_response)

    model.invoke([HumanMessage(content="hello once")])
    model.invoke([HumanMessage(content="hello twice")])

    assert seen_headers == [
        ("Bearer token-1", "acct-1"),
        ("Bearer token-2", "acct-2"),
    ]


def test_codex_provider_keeps_last_known_credential_when_reload_is_temporarily_empty(monkeypatch):
    creds = iter(
        [
            CodexCliCredential(access_token="token-1", account_id="acct-1"),
            None,
        ]
    )

    monkeypatch.setattr(CodexChatModel, "_load_codex_auth", lambda self: next(creds))

    model = CodexChatModel(model="gpt-5.4", retry_max_attempts=1)
    seen_headers: list[tuple[str, str]] = []

    def fake_stream_response(headers, payload):
        seen_headers.append((headers["Authorization"], headers["ChatGPT-Account-ID"]))
        return {
            "model": payload["model"],
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "OK"}],
                }
            ],
            "usage": {},
        }

    monkeypatch.setattr(model, "_stream_response", fake_stream_response)

    result = model.invoke([HumanMessage(content="hello")])

    assert result.content == "OK"
    assert seen_headers == [("Bearer token-1", "acct-1")]


def test_codex_provider_refreshes_auth_and_retries_once_after_401(monkeypatch):
    creds = iter(
        [
            CodexCliCredential(access_token="token-old", account_id="acct-old"),
            CodexCliCredential(access_token="token-old", account_id="acct-old"),
            CodexCliCredential(access_token="token-new", account_id="acct-new"),
        ]
    )

    monkeypatch.setattr(CodexChatModel, "_load_codex_auth", lambda self: next(creds))

    model = CodexChatModel(model="gpt-5.4", retry_max_attempts=1)
    seen_headers: list[tuple[str, str]] = []

    def fake_stream_response(headers, payload):
        seen_headers.append((headers["Authorization"], headers["ChatGPT-Account-ID"]))
        if len(seen_headers) == 1:
            request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses")
            response = httpx.Response(401, request=request)
            raise httpx.HTTPStatusError("unauthorized", request=request, response=response)
        return {
            "model": payload["model"],
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "OK"}],
                }
            ],
            "usage": {},
        }

    monkeypatch.setattr(model, "_stream_response", fake_stream_response)

    result = model.invoke([HumanMessage(content="hello")])

    assert result.content == "OK"
    assert seen_headers == [
        ("Bearer token-old", "acct-old"),
        ("Bearer token-new", "acct-new"),
    ]

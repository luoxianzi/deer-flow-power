from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from deerflow.models.openai_codex_provider import CodexChatModel


@pytest.fixture
def codex_model(monkeypatch):
    monkeypatch.setenv("API_POOL_BASE_URL", "")
    monkeypatch.delenv("API_POOL_KEYS", raising=False)
    monkeypatch.setattr(
        "deerflow.models.openai_codex_provider.load_codex_cli_credential",
        lambda: None,
    )
    m = CodexChatModel(model="gpt-5.4", reasoning_effort="medium")
    m._access_token = "test-token"
    m._account_id = "00000000-0000-0000-0000-000000000001"
    m._credential_available = True
    return m


class _FakeStreamResp:
    def __init__(self, lines: list[str]):
        self._lines = lines

    def raise_for_status(self) -> None:
        return None

    def __enter__(self) -> _FakeStreamResp:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def iter_lines(self):
        yield from self._lines


class _FakeHttpxClient:
    def __init__(self, lines: list[str], *args: object, **kwargs: object):
        self._lines = lines

    def __enter__(self) -> _FakeHttpxClient:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def stream(self, method: str, url: str, headers=None, json=None):
        return _FakeStreamResp(self._lines)


def test_codex_stream_completes_when_type_only_on_sse_event_line(codex_model, monkeypatch):
    lines = [
        "event: response.completed",
        'data: {"id":"resp_evt","output":[],"model":"gpt-5.4","object":"response"}',
    ]
    monkeypatch.setattr(
        "deerflow.models.openai_codex_provider.httpx.Client",
        lambda *a, **k: _FakeHttpxClient(lines),
    )
    out = codex_model._stream_response({}, {"model": "gpt-5.4", "input": []})
    assert out["id"] == "resp_evt"
    assert out.get("object") == "response"


def test_codex_stream_nested_response_key(codex_model, monkeypatch):
    lines = [
        'data: {"type":"response.completed","response":{"id":"resp_nested","output":[]}}',
    ]
    monkeypatch.setattr(
        "deerflow.models.openai_codex_provider.httpx.Client",
        lambda *a, **k: _FakeHttpxClient(lines),
    )
    out = codex_model._stream_response({}, {})
    assert out["id"] == "resp_nested"


def test_codex_stream_response_failed_raises(codex_model, monkeypatch):
    lines = [
        'data: {"type":"response.failed","response":{"error":{"code":"usage_not_included","message":"Plus required"}}}',
    ]
    monkeypatch.setattr(
        "deerflow.models.openai_codex_provider.httpx.Client",
        lambda *a, **k: _FakeHttpxClient(lines),
    )
    with pytest.raises(RuntimeError, match="response.failed.*usage_not_included"):
        codex_model._stream_response({}, {})


def test_codex_stream_response_incomplete_raises(codex_model, monkeypatch):
    lines = [
        'data: {"type":"response.incomplete","response":{"incomplete_details":{"reason":"max_output_tokens"}}}',
    ]
    monkeypatch.setattr(
        "deerflow.models.openai_codex_provider.httpx.Client",
        lambda *a, **k: _FakeHttpxClient(lines),
    )
    with pytest.raises(RuntimeError, match="response.incomplete"):
        codex_model._stream_response({}, {})


def test_parse_response_accepts_text_content_part_type(codex_model):
    resp = {
        "model": "gpt-5.4",
        "output": [
            {
                "type": "message",
                "content": [{"type": "text", "text": "你好"}],
            }
        ],
        "usage": {},
    }
    result = codex_model._parse_response(resp)
    assert result.generations[0].message.content == "你好"


def test_parse_response_reasoning_only_falls_back_to_visible_content(codex_model):
    resp = {
        "model": "gpt-5.4",
        "output": [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "reasoning-only body"}],
            }
        ],
        "usage": {},
    }
    result = codex_model._parse_response(resp)
    assert result.generations[0].message.content == "reasoning-only body"
    assert result.generations[0].message.additional_kwargs.get("reasoning_content") == "reasoning-only body"

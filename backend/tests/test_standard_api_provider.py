from __future__ import annotations
async def _no_async_sleep(*_a, **_k):
    return None


import asyncio
import json
import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deerflow.models.standard_api_provider import StandardAPIChatModel


@pytest.fixture(autouse=True)
def _reset_api_pool_wire_api(monkeypatch):
    monkeypatch.setenv("API_POOL_WIRE_API", "chat_completions")
    monkeypatch.delenv("API_POOL_BASE_URL", raising=False)
    monkeypatch.delenv("API_POOL_BASE_URLS", raising=False)
    monkeypatch.delenv("API_POOL_KEYS", raising=False)


def test_standard_api_provider_requires_env(monkeypatch):
    monkeypatch.delenv("API_POOL_BASE_URL", raising=False)
    monkeypatch.delenv("API_POOL_BASE_URLS", raising=False)
    monkeypatch.delenv("API_POOL_KEYS", raising=False)

    with pytest.raises(ValueError, match="API_POOL_BASE_URL or API_POOL_BASE_URLS, plus API_POOL_KEYS"):
        StandardAPIChatModel()


def test_standard_api_provider_parses_tool_calls(monkeypatch):
    monkeypatch.setenv("API_POOL_BASE_URL", "https://pool.example.com/v1")
    monkeypatch.setenv("API_POOL_KEYS", "pool-key-a")

    model = StandardAPIChatModel(model="pool-model")

    async def fake_request_chat_completions(messages, *, tools=None, stop=None):
        return {
            "model": "pool-model",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": "{\"cmd\":\"pwd\"}",
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    monkeypatch.setattr(model, "_request_chat_completions", fake_request_chat_completions)

    result = model.invoke([HumanMessage(content="hello")])

    assert result.tool_calls == [
        {
            "name": "bash",
            "args": {"cmd": "pwd"},
            "id": "call-1",
            "type": "tool_call",
        }
    ]


def test_standard_api_provider_uses_explicit_proxy_over_env_proxies(monkeypatch):
    monkeypatch.setenv("API_POOL_BASE_URL", "https://pool.example.com/v1")
    monkeypatch.setenv("API_POOL_KEYS", "pool-key-a")
    monkeypatch.setenv("API_POOL_PROXY_URL", "socks5://host.docker.internal:17890")
    monkeypatch.setenv("ALL_PROXY", "http://stale-proxy.invalid:9999")
    monkeypatch.setenv("HTTPS_PROXY", "http://stale-proxy.invalid:9999")

    captured: dict = {}

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, *, headers, json):
            request = httpx.Request("POST", url)
            return httpx.Response(
                200,
                request=request,
                json={
                    "model": "pool-model",
                    "choices": [{"finish_reason": "stop", "message": {"content": "OK"}}],
                    "usage": {},
                },
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    model = StandardAPIChatModel(model="pool-model")
    result = model.invoke([HumanMessage(content="hello")])

    assert result.content == "OK"
    assert captured["proxy"] == "socks5://host.docker.internal:17890"
    assert captured["trust_env"] is False


def test_standard_api_provider_maps_reasoning_content_to_additional_kwargs(monkeypatch):
    monkeypatch.setenv("API_POOL_BASE_URL", "https://pool.example.com/v1")
    monkeypatch.setenv("API_POOL_KEYS", "pool-key-a")

    model = StandardAPIChatModel(model="gpt-5.4")

    async def fake_request_chat_completions(messages, *, tools=None, stop=None):
        return {
            "model": "gpt-5.4",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "最终答案",
                        "reasoning_content": "先想清楚，再回答。",
                        "tool_calls": [],
                    },
                }
            ],
            "usage": {},
        }

    monkeypatch.setattr(model, "_request_chat_completions", fake_request_chat_completions)

    result = model.invoke([HumanMessage(content="hello")])

    assert result.content == "最终答案"
    assert result.additional_kwargs["reasoning_content"] == "先想清楚，再回答。"


def test_standard_api_provider_parses_responses_api_text(monkeypatch):
    monkeypatch.setenv("API_POOL_BASE_URL", "https://pool.example.com/v1")
    monkeypatch.setenv("API_POOL_KEYS", "pool-key-a")
    monkeypatch.setenv("API_POOL_WIRE_API", "responses")

    model = StandardAPIChatModel(model="gpt-5.4")

    async def fake_request_responses(messages, *, tools=None, stop=None):
        return {
            "model": "gpt-5.4",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "RESP_OK"}],
                }
            ],
            "reasoning": {"effort": "high", "summary": [{"type": "summary_text", "text": "先思考。"}]},
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }

    monkeypatch.setattr(model, "_request_responses", fake_request_responses)

    result = model.invoke([HumanMessage(content="hello")])

    assert result.content == "RESP_OK"
    assert result.additional_kwargs["reasoning_content"] == "先思考。"


def test_standard_api_provider_parses_responses_api_function_calls(monkeypatch):
    monkeypatch.setenv("API_POOL_BASE_URL", "https://pool.example.com/v1")
    monkeypatch.setenv("API_POOL_KEYS", "pool-key-a")
    monkeypatch.setenv("API_POOL_WIRE_API", "responses")

    model = StandardAPIChatModel(model="gpt-5.4")

    async def fake_request_responses(messages, *, tools=None, stop=None):
        return {
            "model": "gpt-5.4",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_123",
                    "name": "bash",
                    "arguments": "{\"cmd\":\"pwd\"}",
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }

    monkeypatch.setattr(model, "_request_responses", fake_request_responses)

    result = model.invoke([HumanMessage(content="hello")])

    assert result.tool_calls == [
        {
            "name": "bash",
            "args": {"cmd": "pwd"},
            "id": "call_123",
            "type": "tool_call",
        }
    ]


def test_standard_api_provider_compacts_responses_context_after_413(monkeypatch):
    monkeypatch.setenv("API_POOL_BASE_URL", "https://pool.example.com/v1")
    monkeypatch.setenv("API_POOL_KEYS", "pool-key-a")
    monkeypatch.setenv("API_POOL_WIRE_API", "responses")

    model = StandardAPIChatModel(model="gpt-5.4")
    input_lengths: list[int] = []

    async def fake_post(self, url, *, headers, json):
        input_length = len(json["input"])
        input_lengths.append(input_length)
        request = httpx.Request("POST", url)
        if input_length > 40:
            response = httpx.Response(
                413,
                request=request,
                headers={"content-type": "text/html"},
                text="<!DOCTYPE html><html><title>413</title></html>",
            )
            raise httpx.HTTPStatusError("payload too large", request=request, response=response)
        return httpx.Response(
            200,
            request=request,
            json={
                "model": "gpt-5.4",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "RESP_COMPACT_OK"}],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    messages = [HumanMessage(content=f"msg-{idx}") for idx in range(100)]
    result = model.invoke(messages)

    assert result.content == "RESP_COMPACT_OK"
    assert input_lengths[0] == 100
    assert input_lengths[1] == 80


def test_standard_api_provider_precompacts_oversized_responses_payload(monkeypatch):
    monkeypatch.setenv("API_POOL_BASE_URL", "https://pool.example.com/v1")
    monkeypatch.setenv("API_POOL_KEYS", "pool-key-a")
    monkeypatch.setenv("API_POOL_WIRE_API", "responses")
    # The provider enforces a 10_000-char floor on API_POOL_MAX_REQUEST_BODY_CHARS to
    # avoid configurations that would reject every normal request. Use the smallest
    # limit the provider will honor so the precompaction branch is exercised.
    monkeypatch.setenv("API_POOL_MAX_REQUEST_BODY_CHARS", "10000")

    model = StandardAPIChatModel(model="gpt-5.4")
    # Enough large-ish messages that even after per-field truncation the payload
    # still exceeds the floor and the compactor has to drop context with the
    # canonical "gateway request size limit" placeholder.
    messages = [HumanMessage(content=f"msg-{i} " + ("X" * 800)) for i in range(40)]

    payload = model._build_responses_payload(messages)

    assert len(json.dumps(payload, ensure_ascii=False)) <= 10000
    assert any("gateway request size limit" in str(item.get("content", "")) for item in payload["input"])


def test_standard_api_provider_strips_inline_think_tags(monkeypatch):
    monkeypatch.setenv("API_POOL_BASE_URL", "https://pool.example.com/v1")
    monkeypatch.setenv("API_POOL_KEYS", "pool-key-a")

    model = StandardAPIChatModel(model="gpt-5.4")

    async def fake_request_chat_completions(messages, *, tools=None, stop=None):
        return {
            "model": "gpt-5.4",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "<think>这是思考过程。</think>\n\n真正答案。",
                        "tool_calls": [],
                    },
                }
            ],
            "usage": {},
        }

    monkeypatch.setattr(model, "_request_chat_completions", fake_request_chat_completions)

    result = model.invoke([HumanMessage(content="hello")])

    assert result.content == "真正答案。"
    assert result.additional_kwargs["reasoning_content"] == "这是思考过程。"


def test_standard_api_provider_rotates_key_after_401(monkeypatch):
    monkeypatch.setenv("API_POOL_BASE_URL", "https://pool.example.com/v1")
    monkeypatch.setenv("API_POOL_KEYS", "sk-first, sk-second")

    model = StandardAPIChatModel(model="pool-model")
    seen_auth_headers: list[str] = []

    async def fake_post(self, url, *, headers, json):
        seen_auth_headers.append(headers["Authorization"])
        request = httpx.Request("POST", url)
        if len(seen_auth_headers) == 1:
            response = httpx.Response(401, request=request, json={"error": {"message": "expired"}})
            raise httpx.HTTPStatusError("unauthorized", request=request, response=response)
        return httpx.Response(
            200,
            request=request,
            json={
                "model": "pool-model",
                "choices": [{"finish_reason": "stop", "message": {"content": "OK"}}],
                "usage": {},
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = model.invoke([HumanMessage(content="hello")])

    assert result.content == "OK"
    assert seen_auth_headers == ["Bearer sk-first", "Bearer sk-second"]


def test_standard_api_provider_exhausts_all_keys_on_retryable_errors(monkeypatch):
    monkeypatch.setenv("API_POOL_BASE_URL", "https://pool.example.com/v1")
    monkeypatch.setenv("API_POOL_KEYS", "sk-first, sk-second, sk-third")

    model = StandardAPIChatModel(model="pool-model")
    seen_auth_headers: list[str] = []

    async def fake_post(self, url, *, headers, json):
        seen_auth_headers.append(headers["Authorization"])
        request = httpx.Request("POST", url)
        response = httpx.Response(429, request=request, json={"error": {"message": "rate limit"}})
        raise httpx.HTTPStatusError("rate limited", request=request, response=response)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    with pytest.raises(RuntimeError, match="API pool key rotation exhausted after 3 keys. Last error HTTP 429: rate limit"):
        model.invoke([HumanMessage(content="hello")])

    assert seen_auth_headers == [
        "Bearer sk-first",
        "Bearer sk-second",
        "Bearer sk-third",
    ]


def test_standard_api_provider_does_not_retry_timeout_on_single_gateway(monkeypatch):
    monkeypatch.setenv("API_POOL_BASE_URL", "https://pool.example.com/v1")
    monkeypatch.setenv("API_POOL_KEYS", "sk-first")
    monkeypatch.setenv("API_POOL_MAX_TRANSIENT_RETRIES", "1")

    model = StandardAPIChatModel(model="pool-model")
    call_count = 0

    async def fake_post(self, url, *, headers, json):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise httpx.TimeoutException("upstream timeout")
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            request=request,
            json={
                "model": "pool-model",
                "choices": [{"finish_reason": "stop", "message": {"content": "OK"}}],
                "usage": {},
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    with pytest.raises(TimeoutError) as exc_info:
        model.invoke([HumanMessage(content="hello")])

    assert call_count == 1
    assert f"request timed out after {model.timeout_seconds}s" in str(exc_info.value)


def test_standard_api_provider_summarizes_html_gateway_errors(monkeypatch):
    monkeypatch.setenv("API_POOL_BASE_URL", "https://newapi.zikl.dev/v1")
    monkeypatch.setenv("API_POOL_KEYS", "sk-first")
    monkeypatch.setenv("API_POOL_MAX_TRANSIENT_RETRIES", "0")

    model = StandardAPIChatModel(model="pool-model")

    async def fake_post(self, url, *, headers, json):
        request = httpx.Request("POST", url)
        response = httpx.Response(
            504,
            request=request,
            headers={"content-type": "text/html"},
            text="<!DOCTYPE html><html><title>504</title></html>",
        )
        raise httpx.HTTPStatusError("gateway timeout", request=request, response=response)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    with pytest.raises(
        RuntimeError,
        match=r"API pool transient retries exhausted after 1 attempts\. Last error HTTP 504: API pool upstream gateway timed out \(newapi\.zikl\.dev\)",
    ):
        model.invoke([HumanMessage(content="hello")])


def test_standard_api_provider_retries_same_gateway_after_504(monkeypatch):
    monkeypatch.setenv("API_POOL_BASE_URL", "https://newapi.zikl.dev/v1")
    monkeypatch.setenv("API_POOL_KEYS", "sk-first")
    monkeypatch.setenv("API_POOL_MAX_TRANSIENT_RETRIES", "1")

    model = StandardAPIChatModel(model="gpt-5.4")
    call_count = 0

    async def fake_post(self, url, *, headers, json):
        nonlocal call_count
        call_count += 1
        request = httpx.Request("POST", url)
        if call_count == 1:
            response = httpx.Response(
                504,
                request=request,
                headers={"content-type": "text/html"},
                text="<!DOCTYPE html><html><title>504</title></html>",
            )
            raise httpx.HTTPStatusError("gateway timeout", request=request, response=response)
        return httpx.Response(
            200,
            request=request,
            json={
                "model": "gpt-5.4",
                "choices": [{"finish_reason": "stop", "message": {"content": "OK"}}],
                "usage": {},
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = model.invoke([HumanMessage(content="hello")])

    assert result.content == "OK"
    assert call_count == 2


def test_standard_api_provider_retries_transport_disconnect_same_gateway(monkeypatch):
    monkeypatch.setenv("API_POOL_BASE_URL", "https://codesurf.ccwu.cc/v1")
    monkeypatch.setenv("API_POOL_KEYS", "sk-first")
    monkeypatch.setenv("API_POOL_MAX_TRANSIENT_RETRIES", "1")

    model = StandardAPIChatModel(model="gpt-5.4")
    call_count = 0

    async def fake_post(self, url, *, headers, json):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            request=request,
            json={
                "model": "gpt-5.4",
                "choices": [{"finish_reason": "stop", "message": {"content": "OK"}}],
                "usage": {},
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = model.invoke([HumanMessage(content="hello")])

    assert result.content == "OK"
    assert call_count == 2


def test_standard_api_provider_409_allows_more_retries_than_configured_cap_for_other_transients(monkeypatch):
    """HTTP 409 (per-account concurrency) uses a higher retry ceiling than 502/504 etc."""
    monkeypatch.setenv("API_POOL_BASE_URL", "https://pool.example.com/v1")
    monkeypatch.setenv("API_POOL_KEYS", "sk-first")
    monkeypatch.setenv("API_POOL_MAX_TRANSIENT_RETRIES", "2")
    monkeypatch.setattr(asyncio, "sleep", _no_async_sleep)

    model = StandardAPIChatModel(model="glm-4.7")
    call_count = 0

    async def fake_post(self, url, *, headers, json):
        nonlocal call_count
        call_count += 1
        request = httpx.Request("POST", url)
        if call_count <= 4:
            response = httpx.Response(
                409,
                request=request,
                json={"error": {"message": "Local concurrency limit exceeded"}},
            )
            raise httpx.HTTPStatusError("conflict", request=request, response=response)
        return httpx.Response(
            200,
            request=request,
            json={
                "model": "glm-4.7",
                "choices": [{"finish_reason": "stop", "message": {"content": "OK"}}],
                "usage": {},
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = model.invoke([HumanMessage(content="hello")])

    assert result.content == "OK"
    assert call_count == 5


def test_standard_api_provider_retries_same_key_before_rotating_on_null_content(monkeypatch):
    """HTTP 200 + null content is often a flaky gateway; retry same key before burning the pool."""
    monkeypatch.setenv("API_POOL_BASE_URL", "https://pool.example.com/v1")
    monkeypatch.setenv("API_POOL_KEYS", "sk-only")
    monkeypatch.setenv("API_POOL_MAX_TRANSIENT_RETRIES", "2")
    monkeypatch.setattr(asyncio, "sleep", _no_async_sleep)

    model = StandardAPIChatModel(model="gpt-5.4")
    call_count = 0

    def null_json():
        return {
            "model": "gpt-5.4",
            "choices": [{"finish_reason": "stop", "message": {"content": None}}],
            "usage": {},
        }

    async def fake_post(self, url, *, headers, json):
        nonlocal call_count
        call_count += 1
        request = httpx.Request("POST", url)
        if call_count <= 2:
            return httpx.Response(200, request=request, json=null_json())
        return httpx.Response(
            200,
            request=request,
            json={
                "model": "gpt-5.4",
                "choices": [{"finish_reason": "stop", "message": {"content": "OK"}}],
                "usage": {},
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = model.invoke([HumanMessage(content="hello")])

    assert result.content == "OK"
    assert call_count == 3


def test_standard_api_provider_compacts_context_on_transient_409_before_retry(monkeypatch):
    """Long threads: same behavior as responses API — shrink context on 409 before backoff/rotation."""
    monkeypatch.setenv("API_POOL_BASE_URL", "https://pool.example.com/v1")
    monkeypatch.setenv("API_POOL_KEYS", "sk-first")
    monkeypatch.setenv("API_POOL_MAX_TRANSIENT_RETRIES", "0")

    model = StandardAPIChatModel(model="m")
    message_counts: list[int] = []
    n = 0

    async def fake_post(self, url, *, headers, json):
        nonlocal n
        n += 1
        message_counts.append(len(json.get("messages", [])))
        request = httpx.Request("POST", url)
        if n == 1:
            r = httpx.Response(409, request=request, json={"error": {"message": "concurrency"}})
            raise httpx.HTTPStatusError("c", request=request, response=r)
        return httpx.Response(
            200,
            request=request,
            json={"model": "m", "choices": [{"finish_reason": "stop", "message": {"content": "hi"}}], "usage": {}},
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    msgs = [HumanMessage(content=f"x{i}") for i in range(5)]
    result = model.invoke(msgs)

    assert result.content == "hi"
    assert message_counts == [5, 4]


def test_standard_api_provider_compacts_on_null_content_before_key_rotation(monkeypatch):
    monkeypatch.setenv("API_POOL_BASE_URL", "https://pool.example.com/v1")
    monkeypatch.setenv("API_POOL_KEYS", "sk-first")
    monkeypatch.setenv("API_POOL_MAX_TRANSIENT_RETRIES", "0")

    model = StandardAPIChatModel(model="m")
    message_counts: list[int] = []
    n = 0

    def null_json():
        return {"model": "m", "choices": [{"finish_reason": "stop", "message": {"content": None}}], "usage": {}}

    async def fake_post(self, url, *, headers, json):
        nonlocal n
        n += 1
        message_counts.append(len(json.get("messages", [])))
        request = httpx.Request("POST", url)
        if n == 1:
            return httpx.Response(200, request=request, json=null_json())
        return httpx.Response(
            200,
            request=request,
            json={"model": "m", "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}], "usage": {}},
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    msgs = [HumanMessage(content=f"x{i}") for i in range(5)]
    result = model.invoke(msgs)

    assert result.content == "ok"
    assert message_counts == [5, 4]


def test_standard_api_provider_sanitize_drops_orphan_tool_messages(monkeypatch):
    """Compaction tail-slices can leave tool output without assistant tool_calls (HTTP 400 from gateway)."""
    monkeypatch.setenv("API_POOL_BASE_URL", "https://pool.example.com/v1")
    monkeypatch.setenv("API_POOL_KEYS", "sk-first")
    model = StandardAPIChatModel(model="m")
    cid = "fcuamo18DEUNcx1EPmvjsswboZ"

    only_tool = [
        HumanMessage(content="user says"),
        ToolMessage(content="output", tool_call_id=cid),
    ]
    cleaned = model._sanitize_compacted_messages_for_tool_protocol(only_tool)
    assert len(cleaned) == 1
    assert isinstance(cleaned[0], HumanMessage)

    sliced_off_assistant = [
        HumanMessage(content="u"),
        AIMessage(content="", tool_calls=[{"name": "x", "id": cid, "args": {}}]),
        ToolMessage(content="out", tool_call_id=cid),
    ]
    bad = [sliced_off_assistant[0], sliced_off_assistant[2]]
    cleaned2 = model._sanitize_compacted_messages_for_tool_protocol(bad)
    assert len(cleaned2) == 1
    assert isinstance(cleaned2[0], HumanMessage)

    intact = model._sanitize_compacted_messages_for_tool_protocol(sliced_off_assistant)
    assert len(intact) == 3


def test_standard_api_provider_rotates_base_url_after_transient_gateway_error(monkeypatch):
    monkeypatch.delenv("API_POOL_BASE_URL", raising=False)
    monkeypatch.setenv("API_POOL_BASE_URLS", "https://gw-a.example.com/v1,https://gw-b.example.com/v1")
    monkeypatch.setenv("API_POOL_KEYS", "sk-first")

    model = StandardAPIChatModel(model="pool-model")
    seen_urls: list[str] = []

    async def fake_post(self, url, *, headers, json):
        seen_urls.append(str(url))
        request = httpx.Request("POST", url)
        if "gw-a.example.com" in str(url):
            response = httpx.Response(
                504,
                request=request,
                headers={"content-type": "text/html"},
                text="<!DOCTYPE html><html><title>504</title></html>",
            )
            raise httpx.HTTPStatusError("gateway timeout", request=request, response=response)

        return httpx.Response(
            200,
            request=request,
            json={
                "model": "pool-model",
                "choices": [{"finish_reason": "stop", "message": {"content": "OK"}}],
                "usage": {},
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = model.invoke([HumanMessage(content="hello")])

    assert result.content == "OK"
    assert seen_urls == [
        "https://gw-a.example.com/v1/chat/completions",
        "https://gw-b.example.com/v1/chat/completions",
    ]


def test_standard_api_provider_surfaces_cloudflare_1010_as_gateway_block(monkeypatch):
    monkeypatch.setenv("API_POOL_BASE_URL", "https://newapi.zikl.dev/v1")
    monkeypatch.setenv("API_POOL_KEYS", "sk-first,sk-second,sk-third")

    model = StandardAPIChatModel(model="gpt-5.4")
    seen_auth_headers: list[str] = []

    async def fake_post(self, url, *, headers, json):
        seen_auth_headers.append(headers["Authorization"])
        request = httpx.Request("POST", url)
        response = httpx.Response(
            403,
            request=request,
            headers={"content-type": "text/plain"},
            text="error code: 1010",
        )
        raise httpx.HTTPStatusError("forbidden", request=request, response=response)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    with pytest.raises(
        RuntimeError,
        match=r"API pool upstream denied the current egress path or proxy \(HTTP 403 on newapi\.zikl\.dev: error code: 1010\)",
    ):
        model.invoke([HumanMessage(content="hello")])

    assert seen_auth_headers == ["Bearer sk-first"]


def test_standard_api_provider_surfaces_non_retryable_http_errors(monkeypatch):
    monkeypatch.setenv("API_POOL_BASE_URL", "https://pool.example.com/v1")
    monkeypatch.setenv("API_POOL_KEYS", "pool-key-a,pool-key-b")

    model = StandardAPIChatModel(model="pool-model")

    async def fake_post(self, url, *, headers, json):
        request = httpx.Request("POST", url)
        response = httpx.Response(
            500,
            request=request,
            json={"error": {"message": "upstream exploded"}},
        )
        raise httpx.HTTPStatusError("server error", request=request, response=response)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    with pytest.raises(RuntimeError, match="HTTP 500: upstream exploded"):
        model.invoke([HumanMessage(content="hello")])


def test_standard_api_provider_includes_reasoning_effort_and_extra_body(monkeypatch):
    monkeypatch.setenv("API_POOL_BASE_URL", "https://pool.example.com/v1")
    monkeypatch.setenv("API_POOL_KEYS", "pool-key-a")

    model = StandardAPIChatModel(
        model="gpt-5.4",
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
    )
    captured_payloads: list[dict] = []

    async def fake_post(self, url, *, headers, json):
        captured_payloads.append(json)
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            request=request,
            json={
                "model": "gpt-5.4",
                "choices": [{"finish_reason": "stop", "message": {"content": "OK"}}],
                "usage": {},
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = model.invoke([HumanMessage(content="think hard")])

    assert result.content == "OK"
    assert captured_payloads[0]["reasoning_effort"] == "high"
    assert captured_payloads[0]["thinking"] == {"type": "enabled"}


def test_standard_api_provider_normalizes_minimal_reasoning_for_gpt5(monkeypatch):
    monkeypatch.setenv("API_POOL_BASE_URL", "https://pool.example.com/v1")
    monkeypatch.setenv("API_POOL_KEYS", "pool-key-a")

    model = StandardAPIChatModel(model="gpt-5.4", reasoning_effort="minimal")
    captured_payloads: list[dict] = []

    async def fake_post(self, url, *, headers, json):
        captured_payloads.append(json)
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            request=request,
            json={
                "model": "gpt-5.4",
                "choices": [{"finish_reason": "stop", "message": {"content": "OK"}}],
                "usage": {},
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    model.invoke([HumanMessage(content="hello")])

    assert captured_payloads[0]["reasoning_effort"] == "none"


def test_standard_api_provider_streams_reasoning_and_content(monkeypatch):
    monkeypatch.setenv("API_POOL_BASE_URL", "https://pool.example.com/v1")
    monkeypatch.setenv("API_POOL_KEYS", "pool-key-a")

    model = StandardAPIChatModel(model="gpt-5.4", reasoning_effort="high")
    captured_payloads: list[dict] = []

    class FakeStreamResponse:
        def __init__(self, payload: dict):
            self.payload = payload

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}'
            yield 'data: {"choices":[{"delta":{"reasoning_content":"先想"},"finish_reason":null}]}'
            yield 'data: {"choices":[{"delta":{"reasoning_content":"后答"},"finish_reason":null}]}'
            yield 'data: {"choices":[{"delta":{"content":"你好"},"finish_reason":null}]}'
            yield 'data: {"choices":[{"delta":{"content":"。"},"finish_reason":"stop"}]}'
            yield "data: [DONE]"

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, *, headers, json):
            captured_payloads.append(json)
            return FakeStreamResponse(json)

    monkeypatch.setattr(model, "_create_http_client", lambda: FakeAsyncClient())

    async def collect():
        chunks = []
        async for chunk in model.astream([HumanMessage(content="hello")]):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(collect())
    assert captured_payloads[0]["stream"] is True

    merged = chunks[0]
    for chunk in chunks[1:]:
        merged = merged + chunk

    assert merged.content == "你好。"
    assert merged.additional_kwargs["reasoning_content"] == "先想后答"


def test_standard_api_provider_streams_tool_calls(monkeypatch):
    monkeypatch.setenv("API_POOL_BASE_URL", "https://pool.example.com/v1")
    monkeypatch.setenv("API_POOL_KEYS", "pool-key-a")

    model = StandardAPIChatModel(model="gpt-5.4", reasoning_effort="high")

    class FakeStreamResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}'
            yield 'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"bash","arguments":""}}]},"finish_reason":null}]}'
            yield 'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"","arguments":"{\\"cmd\\":\\"pwd"}}]},"finish_reason":null}]}'
            yield 'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"","arguments":"\\"}"}}]},"finish_reason":"tool_calls"}]}'
            yield "data: [DONE]"

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, *, headers, json):
            return FakeStreamResponse()

    monkeypatch.setattr(model, "_create_http_client", lambda: FakeAsyncClient())

    async def collect():
        chunks = []
        async for chunk in model.astream(
            [HumanMessage(content="hello")],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": "run command",
                        "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
                    },
                }
            ],
        ):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(collect())

    merged = chunks[0]
    for chunk in chunks[1:]:
        merged = merged + chunk

    assert merged.tool_calls == [
        {
            "name": "bash",
            "args": {"cmd": "pwd"},
            "id": "call_1",
            "type": "tool_call",
        }
    ]


def test_standard_api_provider_stream_retries_transport_disconnect_same_gateway(monkeypatch):
    monkeypatch.setenv("API_POOL_BASE_URL", "https://codesurf.ccwu.cc/v1")
    monkeypatch.setenv("API_POOL_KEYS", "sk-first")
    monkeypatch.setenv("API_POOL_MAX_TRANSIENT_RETRIES", "1")

    model = StandardAPIChatModel(model="gpt-5.4", reasoning_effort="high")
    stream_attempts = 0

    class FakeStreamResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"OK"},"finish_reason":"stop"}]}'
            yield "data: [DONE]"

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, *, headers, json):
            nonlocal stream_attempts
            stream_attempts += 1
            if stream_attempts == 1:
                raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
            return FakeStreamResponse()

    monkeypatch.setattr(model, "_create_http_client", lambda: FakeAsyncClient())

    async def collect():
        chunks = []
        async for chunk in model.astream([HumanMessage(content="hello")]):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(collect())

    assert stream_attempts == 2
    merged = chunks[0]
    for chunk in chunks[1:]:
        merged = merged + chunk
    assert merged.content == "OK"


def test_standard_api_provider_stream_reads_error_body_before_extracting_detail(monkeypatch):
    monkeypatch.setenv("API_POOL_BASE_URL", "https://codesurf.ccwu.cc/v1")
    monkeypatch.setenv("API_POOL_KEYS", "sk-first")
    monkeypatch.setenv("API_POOL_MAX_TRANSIENT_RETRIES", "0")

    model = StandardAPIChatModel(model="gpt-5.4", reasoning_effort="high")

    class FakeResponse:
        def __init__(self):
            self.status_code = 403
            self.headers = {"content-type": "application/json"}
            self.request = httpx.Request("POST", "https://codesurf.ccwu.cc/v1/chat/completions")
            self._read = False

        async def aread(self):
            self._read = True
            return b'{"error":{"message":"forbidden by upstream"}}'

        def raise_for_status(self):
            raise httpx.HTTPStatusError("forbidden", request=self.request, response=self)

        @property
        def text(self):
            if not self._read:
                raise httpx.ResponseNotRead()
            return '{"error":{"message":"forbidden by upstream"}}'

        def json(self):
            if not self._read:
                raise httpx.ResponseNotRead()
            return {"error": {"message": "forbidden by upstream"}}

    class FakeStreamContext:
        def __init__(self):
            self.response = FakeResponse()

        async def __aenter__(self):
            return self.response

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, *, headers, json):
            return FakeStreamContext()

    monkeypatch.setattr(model, "_create_http_client", lambda: FakeAsyncClient())

    async def collect():
        async for _ in model.astream([HumanMessage(content="hello")]):
            pass

    with pytest.raises(RuntimeError, match="HTTP 403: forbidden by upstream"):
        asyncio.run(collect())


def test_standard_api_provider_stream_falls_back_to_non_stream_after_transport_disconnect(monkeypatch):
    monkeypatch.setenv("API_POOL_BASE_URL", "https://codesurf.ccwu.cc/v1")
    monkeypatch.setenv("API_POOL_KEYS", "sk-first")
    monkeypatch.setenv("API_POOL_MAX_TRANSIENT_RETRIES", "0")

    model = StandardAPIChatModel(model="gpt-5.4", reasoning_effort="high")

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, *, headers, json):
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")

    async def fake_request_chat_completions(messages, *, tools=None, stop=None):
        return {
            "model": "gpt-5.4",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "FALLBACK_OK",
                        "reasoning_content": "先完整返回再展示。",
                    },
                }
            ],
            "usage": {},
        }

    monkeypatch.setattr(model, "_create_http_client", lambda: FakeAsyncClient())
    monkeypatch.setattr(model, "_request_chat_completions", fake_request_chat_completions)

    async def collect():
        chunks = []
        async for chunk in model.astream([HumanMessage(content="hello")]):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(collect())

    merged = chunks[0]
    for chunk in chunks[1:]:
        merged = merged + chunk

    assert merged.content == "FALLBACK_OK"
    assert merged.additional_kwargs["reasoning_content"] == "先完整返回再展示。"


def test_standard_api_provider_non_stream_falls_back_to_stream_collection(monkeypatch):
    monkeypatch.setenv("API_POOL_BASE_URL", "https://codesurf.ccwu.cc/v1")
    monkeypatch.setenv("API_POOL_KEYS", "sk-first")
    monkeypatch.setenv("API_POOL_MAX_TRANSIENT_RETRIES", "0")

    model = StandardAPIChatModel(model="gpt-5.4", reasoning_effort="high")
    post_attempts = 0

    class FakeStreamResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            yield 'data: {"model":"gpt-5.4","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}'
            yield 'data: {"choices":[{"delta":{"reasoning_content":"先恢复流式。"},"finish_reason":null}]}'
            yield 'data: {"choices":[{"delta":{"content":"STREAM_BACKFILL_OK"},"finish_reason":"stop"}]}'
            yield "data: [DONE]"

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, *, headers, json):
            nonlocal post_attempts
            post_attempts += 1
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")

        def stream(self, method, url, *, headers, json):
            return FakeStreamResponse()

    monkeypatch.setattr(model, "_create_http_client", lambda: FakeAsyncClient())

    result = model.invoke([HumanMessage(content="hello")])

    assert post_attempts == 1
    assert result.content == "STREAM_BACKFILL_OK"
    assert result.additional_kwargs["reasoning_content"] == "先恢复流式。"


# ---------------------------------------------------------------------------
# Per-gateway key binding (pipe-delimited API_POOL_KEYS)
# ---------------------------------------------------------------------------


def test_per_gateway_keys_parsed_correctly(monkeypatch):
    monkeypatch.setenv("API_POOL_BASE_URLS", "https://gw-a.example.com/v1,https://gw-b.example.com/v1")
    monkeypatch.setenv("API_POOL_KEYS", "sk-a1,sk-a2|sk-b1,sk-b2,sk-b3")

    model = StandardAPIChatModel(model="pool-model")

    assert model._api_keys_per_gateway == [["sk-a1", "sk-a2"], ["sk-b1", "sk-b2", "sk-b3"]]
    assert model._current_key_indices == [0, 0]


def test_per_gateway_keys_backward_compatible_without_pipe(monkeypatch):
    monkeypatch.setenv("API_POOL_BASE_URLS", "https://gw-a.example.com/v1,https://gw-b.example.com/v1")
    monkeypatch.setenv("API_POOL_KEYS", "sk-shared1,sk-shared2")

    model = StandardAPIChatModel(model="pool-model")

    assert model._api_keys_per_gateway == [["sk-shared1", "sk-shared2"], ["sk-shared1", "sk-shared2"]]


def test_per_gateway_keys_mismatch_raises(monkeypatch):
    monkeypatch.setenv("API_POOL_BASE_URLS", "https://gw-a.example.com/v1,https://gw-b.example.com/v1")
    monkeypatch.setenv("API_POOL_KEYS", "sk-a|sk-b|sk-c")

    with pytest.raises(ValueError, match="3 pipe-delimited groups but.*2 gateways"):
        StandardAPIChatModel(model="pool-model")


def test_per_gateway_key_rotation_stays_within_gateway(monkeypatch):
    monkeypatch.setenv("API_POOL_BASE_URLS", "https://gw-a.example.com/v1,https://gw-b.example.com/v1")
    monkeypatch.setenv("API_POOL_KEYS", "sk-a1,sk-a2|sk-b1")

    model = StandardAPIChatModel(model="pool-model")

    assert model._current_api_key() == "sk-a1"
    model._rotate_to_next_key()
    assert model._current_api_key() == "sk-a2"
    model._rotate_to_next_key()
    assert model._current_api_key() == "sk-a1"

    model._rotate_to_next_base_url()
    assert model._current_api_key() == "sk-b1"
    model._rotate_to_next_key()
    assert model._current_api_key() == "sk-b1"


def test_per_gateway_401_exhaustion_triggers_gateway_failover(monkeypatch):
    monkeypatch.setenv("API_POOL_BASE_URLS", "https://gw-a.example.com/v1,https://gw-b.example.com/v1")
    monkeypatch.setenv("API_POOL_KEYS", "sk-a1|sk-b1")

    model = StandardAPIChatModel(model="pool-model")
    seen: list[tuple[str, str]] = []

    async def fake_post(self, url, *, headers, json):
        auth = headers["Authorization"]
        seen.append((url, auth))
        request = httpx.Request("POST", url)
        if "gw-a" in url:
            response = httpx.Response(401, request=request, json={"error": {"message": "Invalid API key"}})
            raise httpx.HTTPStatusError("unauthorized", request=request, response=response)
        return httpx.Response(
            200,
            request=request,
            json={
                "model": "pool-model",
                "choices": [{"finish_reason": "stop", "message": {"content": "from-gw-b"}}],
                "usage": {},
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = model.invoke([HumanMessage(content="hello")])

    assert result.content == "from-gw-b"
    assert seen[0] == ("https://gw-a.example.com/v1/chat/completions", "Bearer sk-a1")
    assert seen[1] == ("https://gw-b.example.com/v1/chat/completions", "Bearer sk-b1")


def test_key_rotation_exhaustion_message_adds_balance_hint():
    msg = StandardAPIChatModel._key_rotation_exhaustion_message(
        status=403,
        detail="Insufficient account balance",
        max_key_rotations=3,
    )
    assert "key rotation exhausted after 3 keys" in msg
    assert "Insufficient account balance" in msg
    assert "充值" in msg
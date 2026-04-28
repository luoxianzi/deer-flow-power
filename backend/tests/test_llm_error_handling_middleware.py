from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langgraph.errors import GraphBubbleUp

from deerflow.agents.middlewares.llm_error_handling_middleware import (
    LLMErrorHandlingMiddleware,
)


class FakeError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        headers: dict[str, str] | None = None,
        body: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.body = body
        self.response = SimpleNamespace(status_code=status_code, headers=headers or {}) if status_code is not None or headers else None


def _build_middleware(**attrs: int) -> LLMErrorHandlingMiddleware:
    middleware = LLMErrorHandlingMiddleware()
    for key, value in attrs.items():
        setattr(middleware, key, value)
    return middleware


def _request(*, llm_type: str = "standard-api-chat-completions", reasoning_effort: str | None = "high", model_settings=None):
    request = MagicMock()
    request.model = SimpleNamespace(_llm_type=llm_type, reasoning_effort=reasoning_effort)
    request.model_settings = model_settings or {}
    patched_request = MagicMock()
    request.override.return_value = patched_request
    return request, patched_request


def test_async_model_call_retries_busy_provider_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    middleware = _build_middleware(retry_max_attempts=3, retry_base_delay_ms=25, retry_cap_delay_ms=25)
    attempts = 0
    waits: list[float] = []
    events: list[dict] = []

    async def fake_sleep(delay: float) -> None:
        waits.append(delay)

    def fake_writer():
        return events.append

    async def handler(_request) -> AIMessage:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise FakeError("当前服务集群负载较高，请稍后重试，感谢您的耐心等待。 (2064)")
        return AIMessage(content="ok")

    monkeypatch.setattr("asyncio.sleep", fake_sleep)
    monkeypatch.setattr(
        "langgraph.config.get_stream_writer",
        fake_writer,
    )

    result = asyncio.run(middleware.awrap_model_call(SimpleNamespace(), handler))

    assert isinstance(result, AIMessage)
    assert result.content == "ok"
    assert attempts == 3
    assert waits == [0.025, 0.025]
    assert [event["type"] for event in events] == ["llm_retry", "llm_retry"]


def test_async_model_call_returns_user_message_for_quota_errors() -> None:
    middleware = _build_middleware(retry_max_attempts=3)

    async def handler(_request) -> AIMessage:
        raise FakeError(
            "insufficient_quota: account balance is empty",
            status_code=429,
            code="insufficient_quota",
        )

    result = asyncio.run(middleware.awrap_model_call(SimpleNamespace(), handler))

    assert isinstance(result, AIMessage)
    assert "out of quota" in str(result.content)


def test_sync_model_call_uses_retry_after_header(monkeypatch: pytest.MonkeyPatch) -> None:
    middleware = _build_middleware(retry_max_attempts=2, retry_base_delay_ms=10, retry_cap_delay_ms=10)
    waits: list[float] = []
    attempts = 0

    def fake_sleep(delay: float) -> None:
        waits.append(delay)

    def handler(_request) -> AIMessage:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise FakeError(
                "server busy",
                status_code=503,
                headers={"Retry-After": "2"},
            )
        return AIMessage(content="ok")

    monkeypatch.setattr("time.sleep", fake_sleep)

    result = middleware.wrap_model_call(SimpleNamespace(), handler)

    assert isinstance(result, AIMessage)
    assert result.content == "ok"
    assert waits == [2.0]


def test_sync_model_call_propagates_graph_bubble_up() -> None:
    middleware = _build_middleware()

    def handler(_request) -> AIMessage:
        raise GraphBubbleUp()

    with pytest.raises(GraphBubbleUp):
        middleware.wrap_model_call(SimpleNamespace(), handler)


def test_async_model_call_propagates_graph_bubble_up() -> None:
    middleware = _build_middleware()

    async def handler(_request) -> AIMessage:
        raise GraphBubbleUp()

    with pytest.raises(GraphBubbleUp):
        asyncio.run(middleware.awrap_model_call(SimpleNamespace(), handler))


def test_circuit_half_open_graph_bubble_up_resets_probe() -> None:
    """Verify that GraphBubbleUp in half_open state resets probe_in_flight."""
    middleware = _build_middleware()

    # Step 1: Manually set state to half_open and check_circuit() to set probe_in_flight=True
    middleware._circuit_state = "half_open"
    middleware._circuit_probe_in_flight = False
    # Call _check_circuit() once to simulate the probe being allowed through
    assert middleware._check_circuit() is False
    assert middleware._circuit_probe_in_flight is True

    # Step 2: Now trigger handler that raises GraphBubbleUp
    def handler(_request) -> AIMessage:
        raise GraphBubbleUp()

    # Mock _check_circuit() to return False (since we already did the probe check)
    import unittest.mock

    with unittest.mock.patch.object(middleware, "_check_circuit", return_value=False):
        with pytest.raises(GraphBubbleUp):
            middleware.wrap_model_call(SimpleNamespace(), handler)

    # Verify probe_in_flight was reset, state should remain half_open
    assert middleware._circuit_probe_in_flight is False
    assert middleware._circuit_state == "half_open"


@pytest.mark.anyio
async def test_async_circuit_half_open_graph_bubble_up_resets_probe() -> None:
    """Verify that GraphBubbleUp in half_open state resets probe_in_flight (async version)."""
    middleware = _build_middleware()

    # Step 1: Manually set state to half_open and check_circuit() to set probe_in_flight=True
    middleware._circuit_state = "half_open"
    middleware._circuit_probe_in_flight = False
    # Call _check_circuit() once to simulate the probe being allowed through
    assert middleware._check_circuit() is False
    assert middleware._circuit_probe_in_flight is True

    # Step 2: Now trigger handler that raises GraphBubbleUp
    async def handler(_request) -> AIMessage:
        raise GraphBubbleUp()

    # Mock _check_circuit() to return False (since we already did the probe check)
    import unittest.mock

    with unittest.mock.patch.object(middleware, "_check_circuit", return_value=False):
        with pytest.raises(GraphBubbleUp):
            await middleware.awrap_model_call(SimpleNamespace(), handler)

    # Verify probe_in_flight was reset, state should remain half_open
    assert middleware._circuit_probe_in_flight is False
    assert middleware._circuit_state == "half_open"


# ---------- Circuit Breaker Tests ----------


def transient_failing_handler(request: Any) -> Any:
    raise FakeError("Server Error", status_code=502)  # Used for transient error


def quota_failing_handler(request: Any) -> Any:
    raise FakeError("Quota exceeded", body={"error": {"code": "insufficient_quota"}})  # Used for quota error


def success_handler(request: Any) -> Any:
    return AIMessage(content="Success")


def mock_classify_retriable(exc: BaseException) -> tuple[bool, str]:
    return True, "transient"


def mock_classify_non_retriable(exc: BaseException) -> tuple[bool, str]:
    return False, "quota"


def test_circuit_breaker_trips_and_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that circuit breaker trips, fast fails, correctly transitions to Half-Open, and recovers or re-opens."""

    # Mock time.sleep to avoid slow tests during retry loops (Speed up from ~4s to 0.1s)
    waits: list[float] = []
    monkeypatch.setattr("time.sleep", lambda d: waits.append(d))

    # Mock time.time to decouple from private implementation details and enable time travel
    current_time = 1000.0
    monkeypatch.setattr("time.time", lambda: current_time)

    middleware = LLMErrorHandlingMiddleware()
    middleware.circuit_failure_threshold = 3
    middleware.circuit_recovery_timeout_sec = 10
    monkeypatch.setattr(middleware, "_classify_error", mock_classify_retriable)

    request: Any = {"messages": []}

    # --- 0. Test initial state & Success ---
    # Success handler does not increase count. If it's already 0, it stays 0.
    middleware.wrap_model_call(request, success_handler)
    assert middleware._circuit_failure_count == 0
    assert middleware._check_circuit() is False

    # --- 1. Trip the circuit ---
    # Fails 3 overall calls. Threshold (3) is reached.
    middleware.wrap_model_call(request, transient_failing_handler)
    assert middleware._circuit_failure_count == 1
    middleware.wrap_model_call(request, transient_failing_handler)
    assert middleware._circuit_failure_count == 2
    middleware.wrap_model_call(request, transient_failing_handler)
    assert middleware._circuit_failure_count == 3
    assert middleware._check_circuit() is True  # Circuit is OPEN

    # --- 2. Fast Fail ---
    # 2nd call: fast fail immediately without calling handler.
    # Count should not increase during OPEN state.
    result = middleware.wrap_model_call(request, success_handler)
    assert result.content == middleware._build_circuit_breaker_message()
    assert middleware._circuit_failure_count == 3

    # --- 3. Half-Open -> Fail -> Re-Open ---
    # Time travel 11 seconds (timeout is 10s). Current time becomes 1011.0
    current_time += 11.0

    # Verify that the timeout was set EXACTLY relative to current_time + timeout_sec
    assert middleware._circuit_open_until == current_time - 11.0 + middleware.circuit_recovery_timeout_sec

    # Fails again! The request will go through the 3-attempt retry loop again.
    middleware.wrap_model_call(request, transient_failing_handler)
    assert middleware._circuit_failure_count == middleware.circuit_failure_threshold
    assert middleware._circuit_state == "open"  # Re-OPENed

    # --- 4. Half-Open -> Success -> Reset ---
    # Time travel another 11 seconds
    current_time += 11.0

    # Succeeds this time! Should completely reset.
    result = middleware.wrap_model_call(request, success_handler)
    assert result.content == "Success"
    assert middleware._circuit_failure_count == 0  # Fully RESET!
    assert middleware._check_circuit() is False


def test_circuit_breaker_does_not_trip_on_non_retriable_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that circuit breaker ignores business errors like Quota or Auth."""
    waits: list[float] = []
    monkeypatch.setattr("time.sleep", lambda d: waits.append(d))

    middleware = LLMErrorHandlingMiddleware()
    middleware.circuit_failure_threshold = 3
    monkeypatch.setattr(middleware, "_classify_error", mock_classify_non_retriable)

    request: Any = {"messages": []}

    for _ in range(3):
        middleware.wrap_model_call(request, quota_failing_handler)

    assert middleware._circuit_failure_count == 0
    assert middleware._check_circuit() is False


@pytest.mark.anyio
async def test_async_circuit_breaker_trips_and_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify async version of circuit breaker correctly handles state transitions."""
    waits: list[float] = []

    async def fake_sleep(d: float) -> None:
        waits.append(d)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    current_time = 1000.0
    monkeypatch.setattr("time.time", lambda: current_time)

    middleware = LLMErrorHandlingMiddleware()
    middleware.circuit_failure_threshold = 3
    middleware.circuit_recovery_timeout_sec = 10
    monkeypatch.setattr(middleware, "_classify_error", mock_classify_retriable)

    async def async_failing_handler(request: Any) -> Any:
        raise FakeError("Server Error", status_code=502)

    request: Any = {"messages": []}

    # --- 1. Trip the circuit ---
    # Fails 3 overall calls. Threshold (3) is reached.
    await middleware.awrap_model_call(request, async_failing_handler)
    assert middleware._circuit_failure_count == 1
    await middleware.awrap_model_call(request, async_failing_handler)
    assert middleware._circuit_failure_count == 2
    await middleware.awrap_model_call(request, async_failing_handler)
    assert middleware._circuit_failure_count == 3
    assert middleware._check_circuit() is True

    # --- 2. Fast Fail ---
    # 2nd call: fast fail immediately without calling handler
    async def async_success_handler(request: Any) -> Any:
        return AIMessage(content="Success")

    result = await middleware.awrap_model_call(request, async_success_handler)
    assert result.content == middleware._build_circuit_breaker_message()
    assert middleware._circuit_failure_count == 3  # Unchanged

    # --- 3. Half-Open -> Fail -> Re-Open ---
    # Time travel 11 seconds
    current_time += 11.0

    # Verify timeout formula
    assert middleware._circuit_open_until == current_time - 11.0 + middleware.circuit_recovery_timeout_sec

    # Fails again! The request goes through the 3-attempt retry loop.
    await middleware.awrap_model_call(request, async_failing_handler)
    assert middleware._circuit_failure_count == middleware.circuit_failure_threshold
    assert middleware._circuit_state == "open"  # Re-OPENed

    # --- 4. Half-Open -> Success -> Reset ---
    # Time travel another 11 seconds
    current_time += 11.0

    result = await middleware.awrap_model_call(request, async_success_handler)
    assert result.content == "Success"
    assert middleware._circuit_failure_count == 0  # RESET
    assert middleware._check_circuit() is False


# ---------- Local power-profile tests: API pool failover, network-exit-block detection ----------


def test_wrap_model_call_retries_transient_error_then_succeeds(monkeypatch):
    middleware = LLMErrorHandlingMiddleware()
    request, _patched_request = _request(llm_type="codex")

    state = {"calls": 0}

    def _handler(req):
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError("HTTP 503: provider temporarily unavailable")
        return AIMessage(content="OK")

    monkeypatch.setattr("deerflow.agents.middlewares.llm_error_handling_middleware.time.sleep", lambda *_args, **_kwargs: None)

    result = middleware.wrap_model_call(request, _handler)

    assert isinstance(result, AIMessage)
    assert result.content == "OK"
    assert state["calls"] == 2


def test_wrap_model_call_does_not_fail_over_api_pool_by_default(monkeypatch):
    # .env may set FAILOVER_ENABLED=1 for operators; neutralize for this case.
    monkeypatch.delenv("DEER_FLOW_API_POOL_FAILOVER_ENABLED", raising=False)
    middleware = LLMErrorHandlingMiddleware()
    request, patched_request = _request()

    def _handler(req):
        if req is request:
            raise RuntimeError("API pool transient retries exhausted after 1 attempts. Last error HTTP 504: API pool upstream gateway timed out (newapi.zikl.dev)")
        assert req is not patched_request
        return AIMessage(content="UNEXPECTED")

    result = middleware.wrap_model_call(request, _handler)

    request.override.assert_not_called()
    assert isinstance(result, AIMessage)
    assert "已禁用自动切换到本机后备模型" in result.content
    assert result.additional_kwargs["llm_error"] is True
    assert result.additional_kwargs["llm_error_reason"] == "transient"


def test_wrap_model_call_returns_user_message_when_no_fallback(monkeypatch):
    # Ensure failover is OFF for this case; we're asserting the "no fallback" path.
    monkeypatch.delenv("DEER_FLOW_API_POOL_FAILOVER_ENABLED", raising=False)
    middleware = LLMErrorHandlingMiddleware()
    request, _patched_request = _request()

    monkeypatch.setattr(
        "deerflow.agents.middlewares.llm_error_handling_middleware.get_app_config",
        lambda: SimpleNamespace(models=[SimpleNamespace(name="api-pool-default", use="deerflow.models.standard_api_provider:StandardAPIChatModel")]),
    )

    result = middleware.wrap_model_call(
        request,
        lambda _req: (_ for _ in ()).throw(
            RuntimeError("API pool transient retries exhausted after 1 attempts. Last error HTTP 504: API pool upstream gateway timed out (newapi.zikl.dev)")
        ),
    )

    assert isinstance(result, AIMessage)
    assert "暂时不可用" in result.content
    assert result.additional_kwargs["llm_error"] is True


def test_wrap_model_call_skips_outer_retry_for_api_pool_transient_error(monkeypatch):
    # Failover-off path: StandardAPI's internal retries already exhausted, so
    # the outer middleware should NOT retry and NOT swap models.
    monkeypatch.delenv("DEER_FLOW_API_POOL_FAILOVER_ENABLED", raising=False)
    middleware = LLMErrorHandlingMiddleware()
    request, _patched_request = _request()
    state = {"calls": 0}

    def _handler(_req):
        state["calls"] += 1
        raise RuntimeError("API pool transient retries exhausted after 1 attempts. Last error: request timed out after 90.0s")

    result = middleware.wrap_model_call(request, _handler)

    assert isinstance(result, AIMessage)
    assert "暂时不可用" in result.content
    assert state["calls"] == 1


def test_wrap_model_call_returns_network_exit_block_message(monkeypatch):
    middleware = LLMErrorHandlingMiddleware()
    request, _patched_request = _request()

    result = middleware.wrap_model_call(
        request,
        lambda _req: (_ for _ in ()).throw(
            RuntimeError(
                "API pool upstream denied the current egress path or proxy "
                "(HTTP 403 on newapi.zikl.dev: error code: 1010). "
                "This is not a DeerFlow or Docker container failure."
            )
        ),
    )

    assert isinstance(result, AIMessage)
    assert "网络出口" in result.content
    assert "1010" in result.content
    assert result.additional_kwargs["llm_error"] is True
    assert result.additional_kwargs["llm_error_reason"] == "network_exit_block"


def test_wrap_model_call_reraises_graph_bubble_up():
    middleware = LLMErrorHandlingMiddleware()
    request, _patched_request = _request(llm_type="codex")

    def _interrupt(_req):
        raise GraphBubbleUp()

    with pytest.raises(GraphBubbleUp):
        middleware.wrap_model_call(request, _interrupt)


@pytest.mark.anyio
async def test_awrap_model_call_does_not_fail_over_api_pool_by_default(monkeypatch):
    monkeypatch.delenv("DEER_FLOW_API_POOL_FAILOVER_ENABLED", raising=False)
    middleware = LLMErrorHandlingMiddleware()
    request, patched_request = _request()

    async def _handler(req):
        if req is request:
            raise RuntimeError("API pool transient retries exhausted after 1 attempts. Last error HTTP 504: API pool upstream gateway timed out (newapi.zikl.dev)")
        assert req is not patched_request
        return AIMessage(content="UNEXPECTED")

    result = await middleware.awrap_model_call(request, _handler)

    assert isinstance(result, AIMessage)
    request.override.assert_not_called()
    assert "已禁用自动切换到本机后备模型" in result.content
    assert result.additional_kwargs["llm_error"] is True
    assert result.additional_kwargs["llm_error_reason"] == "transient"


@pytest.mark.anyio
async def test_awrap_model_call_skips_outer_retry_for_api_pool_transient_error(monkeypatch):
    monkeypatch.delenv("DEER_FLOW_API_POOL_FAILOVER_ENABLED", raising=False)
    middleware = LLMErrorHandlingMiddleware()
    request, _patched_request = _request()
    state = {"calls": 0}

    async def _handler(_req):
        state["calls"] += 1
        raise RuntimeError("API pool transient retries exhausted after 1 attempts. Last error: request timed out after 90.0s")

    result = await middleware.awrap_model_call(request, _handler)

    assert isinstance(result, AIMessage)
    assert "暂时不可用" in result.content
    assert state["calls"] == 1
    assert result.additional_kwargs["llm_error"] is True


def test_wrap_model_call_fails_over_api_pool_when_explicitly_enabled(monkeypatch):
    middleware = LLMErrorHandlingMiddleware()
    request, patched_request = _request()
    # Plain object() does not allow setattr — use a minimal type that does, so
    # the middleware can stamp the failover flag on the instance.
    fallback_model = SimpleNamespace()

    monkeypatch.setenv("DEER_FLOW_API_POOL_FAILOVER_ENABLED", "true")
    monkeypatch.setattr(
        "deerflow.agents.middlewares.llm_error_handling_middleware.get_app_config",
        lambda: SimpleNamespace(
            models=[
                SimpleNamespace(name="gpt-5.4", use="deerflow.models.openai_codex_provider:CodexChatModel"),
                SimpleNamespace(name="api-pool-default", use="deerflow.models.standard_api_provider:StandardAPIChatModel"),
            ]
        ),
    )
    monkeypatch.setattr(
        "deerflow.models.create_chat_model",
        lambda **kwargs: fallback_model,
    )

    def _handler(req):
        if req is request:
            raise RuntimeError("API pool transient retries exhausted after 1 attempts. Last error HTTP 504: API pool upstream gateway timed out (newapi.zikl.dev)")
        assert req is patched_request
        return AIMessage(content="FALLBACK_OK")

    result = middleware.wrap_model_call(request, _handler)

    request.override.assert_called_once()
    kwargs = request.override.call_args.kwargs
    assert kwargs["model"] is fallback_model
    # The failover flag lives on the fallback model instance itself (not in
    # model_settings — that would leak into the provider SDK's .create() kwargs).
    assert getattr(fallback_model, "_deerflow_api_pool_failover_done", False) is True
    # And critically: model_settings must NOT be set by the override, because
    # any unknown key there crashes the provider SDK.
    assert "model_settings" not in kwargs
    assert isinstance(result, AIMessage)
    assert result.content == "FALLBACK_OK"


@pytest.mark.anyio
async def test_awrap_model_call_reraises_graph_bubble_up():
    middleware = LLMErrorHandlingMiddleware()
    request, _patched_request = _request(llm_type="codex")

    async def _interrupt(_req):
        raise GraphBubbleUp()

    with pytest.raises(GraphBubbleUp):
        await middleware.awrap_model_call(request, _interrupt)

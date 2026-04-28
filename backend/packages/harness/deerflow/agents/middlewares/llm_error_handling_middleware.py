"""LLM error handling middleware with retry/backoff, circuit breaker, and user-facing fallbacks.

Combines upstream's circuit-breaker-aware retry loop with local power-profile
extensions (network-exit-block detection, StandardAPI-aware retry skipping,
optional API pool failover to a local fallback model).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time
from collections.abc import Awaitable, Callable
from email.utils import parsedate_to_datetime
from typing import Any, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage
from langgraph.errors import GraphBubbleUp

from deerflow.config import get_app_config

logger = logging.getLogger(__name__)

_RETRIABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504, 524}
_BUSY_PATTERNS = (
    "server busy",
    "temporarily unavailable",
    "try again later",
    "please retry",
    "please try again",
    "overloaded",
    "high demand",
    "rate limit",
    "gateway timed out",
    "timed out",
    "负载较高",
    "服务繁忙",
    "稍后重试",
    "请稍后重试",
)
_QUOTA_PATTERNS = (
    "insufficient_quota",
    "quota",
    "billing",
    "credit",
    "payment",
    "余额不足",
    "超出限额",
    "额度不足",
    "欠费",
)
_AUTH_PATTERNS = (
    "authentication",
    "unauthorized",
    "invalid api key",
    "invalid_api_key",
    "invalid token",
    "permission",
    "forbidden",
    "access denied",
    "无权",
    "未授权",
    "无效的令牌",
)
_NETWORK_EXIT_BLOCK_PATTERNS = (
    "error code: 1010",
    "denied the current egress path or proxy",
    "cloudflare",
    "waf",
)
_HTTP_STATUS_RE = re.compile(r"\bHTTP\s+(\d{3})\b", re.IGNORECASE)

# Attribute we set on the fallback model instance itself. Must NOT live on
# request.model_settings — LangGraph passes model_settings straight through to
# the provider SDK's .create(**kwargs), so any unknown key (like this one) is
# rejected with "got an unexpected keyword argument".
_API_POOL_FAILOVER_MODEL_ATTR = "_deerflow_api_pool_failover_done"
_API_POOL_FAILOVER_ENV = "DEER_FLOW_API_POOL_FAILOVER_ENABLED"
_STANDARD_API_LLM_TYPE = "standard-api-chat-completions"


class LLMErrorHandlingMiddleware(AgentMiddleware[AgentState]):
    """Retry transient LLM errors and surface graceful assistant messages.

    Extends upstream middleware with:
      - optional API-pool -> local-fallback failover (opt-in via env flag)
      - StandardAPI-aware short-circuit (provider already retries internally)
      - network-exit-block (Cloudflare/WAF 1010) classification
    """

    retry_max_attempts: int = 3
    retry_base_delay_ms: int = 1000
    retry_cap_delay_ms: int = 8000

    circuit_failure_threshold: int = 5
    circuit_recovery_timeout_sec: int = 60

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Load Circuit Breaker configs from app config if available, fall back to defaults
        try:
            app_config = get_app_config()
            self.circuit_failure_threshold = app_config.circuit_breaker.failure_threshold
            self.circuit_recovery_timeout_sec = app_config.circuit_breaker.recovery_timeout_sec
        except (FileNotFoundError, RuntimeError):
            # Gracefully fall back to class defaults in test environments
            pass

        # Circuit Breaker state
        self._circuit_lock = threading.Lock()
        self._circuit_failure_count = 0
        self._circuit_open_until = 0.0
        self._circuit_state = "closed"
        self._circuit_probe_in_flight = False

    # ------------------------------------------------------------------
    # Circuit breaker (upstream)
    # ------------------------------------------------------------------

    def _check_circuit(self) -> bool:
        """Returns True if circuit is OPEN (fast fail), False otherwise."""
        with self._circuit_lock:
            now = time.time()

            if self._circuit_state == "open":
                if now < self._circuit_open_until:
                    return True
                self._circuit_state = "half_open"
                self._circuit_probe_in_flight = False

            if self._circuit_state == "half_open":
                if self._circuit_probe_in_flight:
                    return True
                self._circuit_probe_in_flight = True
                return False

            return False

    def _record_success(self) -> None:
        with self._circuit_lock:
            if self._circuit_state != "closed" or self._circuit_failure_count > 0:
                logger.info("Circuit breaker reset (Closed). LLM service recovered.")
            self._circuit_failure_count = 0
            self._circuit_open_until = 0.0
            self._circuit_state = "closed"
            self._circuit_probe_in_flight = False

    def _record_failure(self) -> None:
        with self._circuit_lock:
            if self._circuit_state == "half_open":
                self._circuit_open_until = time.time() + self.circuit_recovery_timeout_sec
                self._circuit_state = "open"
                self._circuit_probe_in_flight = False
                logger.error(
                    "Circuit breaker probe failed (Open). Will probe again after %ds.",
                    self.circuit_recovery_timeout_sec,
                )
                return

            self._circuit_failure_count += 1
            if self._circuit_failure_count >= self.circuit_failure_threshold:
                self._circuit_open_until = time.time() + self.circuit_recovery_timeout_sec
                if self._circuit_state != "open":
                    self._circuit_state = "open"
                    self._circuit_probe_in_flight = False
                    logger.error(
                        "Circuit breaker tripped (Open). Threshold reached (%d). Will probe after %ds.",
                        self.circuit_failure_threshold,
                        self.circuit_recovery_timeout_sec,
                    )

    def _release_half_open_probe_on_bubble_up(self) -> None:
        """Free the half-open probe slot when a GraphBubbleUp propagates."""
        with self._circuit_lock:
            if self._circuit_state == "half_open":
                self._circuit_probe_in_flight = False

    # ------------------------------------------------------------------
    # Classification & delays
    # ------------------------------------------------------------------

    def _classify_error(self, exc: BaseException) -> tuple[bool, str]:
        detail = _extract_error_detail(exc)
        lowered = detail.lower()
        error_code = _extract_error_code(exc)
        status_code = _extract_status_code(exc)

        if _matches_any(lowered, _NETWORK_EXIT_BLOCK_PATTERNS) and status_code in {403, 451, None}:
            return False, "network_exit_block"
        if _matches_any(lowered, _QUOTA_PATTERNS) or _matches_any(str(error_code).lower(), _QUOTA_PATTERNS):
            return False, "quota"
        if _matches_any(lowered, _AUTH_PATTERNS):
            return False, "auth"

        exc_name = exc.__class__.__name__
        if exc_name in {
            "APITimeoutError",
            "APIConnectionError",
            "InternalServerError",
            "TimeoutError",
        }:
            return True, "transient"
        if status_code in _RETRIABLE_STATUS_CODES:
            return True, "transient"
        if _matches_any(lowered, _BUSY_PATTERNS):
            return True, "busy"

        return False, "generic"

    def _build_retry_delay_ms(self, attempt: int, exc: BaseException) -> int:
        retry_after = _extract_retry_after_ms(exc)
        if retry_after is not None:
            return retry_after
        backoff = self.retry_base_delay_ms * (2 ** max(0, attempt - 1))
        return min(backoff, self.retry_cap_delay_ms)

    def _build_retry_message(self, attempt: int, wait_ms: int, reason: str) -> str:
        seconds = max(1, round(wait_ms / 1000))
        reason_text = "provider is busy" if reason == "busy" else "provider request failed temporarily"
        return f"LLM request retry {attempt}/{self.retry_max_attempts}: {reason_text}. Retrying in {seconds}s."

    def _build_circuit_breaker_message(self) -> str:
        return (
            "The configured LLM provider is currently unavailable due to continuous failures. "
            "Circuit breaker is engaged to protect the system. Please wait a moment before trying again."
        )

    def _build_user_message(self, exc: BaseException, reason: str, *, request: ModelRequest | None = None) -> str:
        detail = _extract_error_detail(exc)
        if reason == "quota":
            return (
                "The configured LLM provider rejected the request because the account is out of quota, "
                "billing is unavailable, or usage is restricted. Please fix the provider account and try again."
            )
        if reason == "auth":
            return (
                "The configured LLM provider rejected the request because authentication or access is invalid. "
                "Please check the provider credentials and try again."
            )
        if reason == "network_exit_block":
            return (
                "当前 API Pool 网关拒绝了这条网络出口（例如 Cloudflare/WAF 1010）。"
                "这不是 DeerFlow、Docker 或 key 数量本身的问题；请改用可用的代理/隧道出口后重试。"
            )
        if reason in {"busy", "transient"}:
            # Local-power-profile user-facing copy: Chinese for "temporarily unavailable",
            # plus an explicit note when automatic API-pool failover is disabled (the default).
            base = (
                "模型服务暂时不可用（已达到上限或网关超时）。The configured LLM provider is "
                "temporarily unavailable after multiple retries. Please wait a moment and "
                "continue the conversation."
            )
            if (
                request is not None
                and self._provider_handles_transient_retries(request)
                and not self._is_api_pool_failover_enabled()
            ):
                base = (
                    "已禁用自动切换到本机后备模型（DEER_FLOW_API_POOL_FAILOVER_ENABLED 未启用）。"
                    + base
                )
            return base
        return f"LLM request failed: {detail}"

    def _build_error_message(self, exc: BaseException, reason: str, *, request: ModelRequest | None = None) -> AIMessage:
        return AIMessage(
            content=self._build_user_message(exc, reason, request=request),
            additional_kwargs={
                "llm_error": True,
                "llm_error_reason": reason,
            },
        )

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _emit_retry_event(self, attempt: int, wait_ms: int, reason: str) -> None:
        try:
            from langgraph.config import get_stream_writer

            writer = get_stream_writer()
            writer(
                {
                    "type": "llm_retry",
                    "attempt": attempt,
                    "max_attempts": self.retry_max_attempts,
                    "wait_ms": wait_ms,
                    "reason": reason,
                    "message": self._build_retry_message(attempt, wait_ms, reason),
                }
            )
        except Exception:
            logger.debug("Failed to emit llm_retry event", exc_info=True)

    def _emit_failover_event(self, target_model_name: str) -> None:
        try:
            from langgraph.config import get_stream_writer

            writer = get_stream_writer()
            writer(
                {
                    "type": "llm_failover",
                    "from": "api_pool",
                    "to": target_model_name,
                    "message": f"API pool unavailable. Falling back to {target_model_name}.",
                }
            )
        except Exception:
            logger.debug("Failed to emit llm_failover event", exc_info=True)

    # ------------------------------------------------------------------
    # API pool failover (local power-profile feature, opt-in)
    # ------------------------------------------------------------------

    def _is_api_pool_failover_enabled(self) -> bool:
        return os.getenv(_API_POOL_FAILOVER_ENV, "").strip().lower() in {"1", "true", "yes", "on"}

    def _provider_handles_transient_retries(self, request: ModelRequest) -> bool:
        model = getattr(request, "model", None)
        if model is None:
            return False
        llm_type = getattr(model, "_llm_type", "")
        return llm_type == _STANDARD_API_LLM_TYPE

    def _should_fail_over_api_pool(self, request: ModelRequest, reason: str) -> bool:
        if not self._is_api_pool_failover_enabled():
            return False
        if reason not in {"busy", "transient"}:
            return False
        if not self._provider_handles_transient_retries(request):
            return False

        # Don't re-enter failover if we already swapped this request's model.
        model = getattr(request, "model", None)
        if model is not None and getattr(model, _API_POOL_FAILOVER_MODEL_ATTR, False):
            return False
        return True

    def _find_api_pool_fallback_model_name(self) -> str | None:
        """Choose the backup model when the api-pool pipeline exhausts retries.

        Priority: Claude OAuth → Codex CLI → any other non-api-pool model.

        Rationale: Claude via OAuth/Max is the most reliable backup — it goes
        direct to Anthropic with no third-party gateway in between. Codex is
        second (its own quota) and then anything that isn't api-pool.

        Override with DEER_FLOW_API_POOL_FALLBACK_MODEL=<model_name> if an
        operator wants a specific pin.
        """
        # Imported lazily so that absence of these providers does not break
        # module import in environments that don't use them.
        try:
            from deerflow.models.engines import (
                CLAUDE_PROVIDER_PATH,
                CODEX_PROVIDER_PATH,
                STANDARD_API_PROVIDER_PATH,
            )
        except Exception:
            logger.debug("engines import failed during failover lookup", exc_info=True)
            return None

        try:
            app_config = get_app_config()
        except (FileNotFoundError, RuntimeError):
            return None

        override = os.getenv("DEER_FLOW_API_POOL_FALLBACK_MODEL", "").strip()
        if override:
            # app_config may be a Pydantic AppConfig (has get_model_config) or
            # a test-time SimpleNamespace (doesn't). Accept either shape.
            lookup = getattr(app_config, "get_model_config", None)
            if callable(lookup):
                if lookup(override) is not None:
                    return override
            else:
                # Fallback: scan the models list directly.
                for candidate in getattr(app_config, "models", []):
                    if getattr(candidate, "name", None) == override:
                        return override

        for candidate in app_config.models:
            if candidate.use == CLAUDE_PROVIDER_PATH:
                return candidate.name

        for candidate in app_config.models:
            if candidate.use == CODEX_PROVIDER_PATH:
                return candidate.name

        for candidate in app_config.models:
            if candidate.use != STANDARD_API_PROVIDER_PATH:
                return candidate.name

        return None

    def _build_api_pool_fallback_request(self, request: ModelRequest) -> ModelRequest | None:
        fallback_model_name = self._find_api_pool_fallback_model_name()
        if not fallback_model_name:
            return None

        try:
            from deerflow.models import create_chat_model
        except Exception:
            logger.debug("create_chat_model import failed during failover", exc_info=True)
            return None

        reasoning_effort = getattr(request.model, "reasoning_effort", None)
        thinking_enabled = reasoning_effort not in (None, "none")

        try:
            fallback_model = create_chat_model(
                name=fallback_model_name,
                thinking_enabled=thinking_enabled,
                reasoning_effort=reasoning_effort,
            )
        except Exception:
            logger.warning(
                "Failed to construct API-pool fallback model %s",
                fallback_model_name,
                exc_info=True,
            )
            return None

        # Mark the fallback model instance (not model_settings — that would leak
        # the flag into the SDK's .create() kwargs and raise "unexpected keyword
        # argument"). object.__setattr__ bypasses Pydantic's assignment
        # validation; instance attrs never enter the request payload.
        try:
            object.__setattr__(fallback_model, _API_POOL_FAILOVER_MODEL_ATTR, True)
        except Exception:
            logger.debug("Could not mark fallback model with failover attr", exc_info=True)

        self._emit_failover_event(fallback_model_name)
        return request.override(model=fallback_model)

    # ------------------------------------------------------------------
    # Core wrap_model_call / awrap_model_call
    # ------------------------------------------------------------------

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        if self._check_circuit():
            return AIMessage(content=self._build_circuit_breaker_message())

        attempt = 1
        while True:
            try:
                response = handler(request)
                self._record_success()
                return response
            except GraphBubbleUp:
                # Preserve LangGraph control-flow signals (interrupt/pause/resume).
                self._release_half_open_probe_on_bubble_up()
                raise
            except Exception as exc:
                retriable, reason = self._classify_error(exc)

                # Optional local-only failover: swap to a non-api_pool model once
                # before giving up when the upstream StandardAPI provider fails.
                if self._should_fail_over_api_pool(request, reason):
                    fallback_request = self._build_api_pool_fallback_request(request)
                    if fallback_request is not None:
                        logger.warning(
                            "API pool failed after %d attempt(s); falling back to local model. Reason: %s",
                            attempt,
                            _extract_error_detail(exc),
                        )
                        try:
                            response = handler(fallback_request)
                            self._record_success()
                            return response
                        except GraphBubbleUp:
                            self._release_half_open_probe_on_bubble_up()
                            raise
                        except Exception as fallback_exc:
                            retriable, reason = self._classify_error(fallback_exc)
                            exc = fallback_exc
                            # fall through to normal handling below

                # StandardAPI provider already does its own retry loop — don't
                # double-retry on top of it, just surface the error.
                if retriable and self._provider_handles_transient_retries(request):
                    logger.warning(
                        "LLM call failed after provider-managed retries; skipping outer retry: %s",
                        _extract_error_detail(exc),
                        exc_info=exc,
                    )
                    self._record_failure()
                    return self._build_error_message(exc, reason, request=request)

                if retriable and attempt < self.retry_max_attempts:
                    wait_ms = self._build_retry_delay_ms(attempt, exc)
                    logger.warning(
                        "Transient LLM error on attempt %d/%d; retrying in %dms: %s",
                        attempt,
                        self.retry_max_attempts,
                        wait_ms,
                        _extract_error_detail(exc),
                    )
                    self._emit_retry_event(attempt, wait_ms, reason)
                    time.sleep(wait_ms / 1000)
                    attempt += 1
                    continue

                logger.warning(
                    "LLM call failed after %d attempt(s): %s",
                    attempt,
                    _extract_error_detail(exc),
                    exc_info=exc,
                )
                if retriable:
                    self._record_failure()
                return self._build_error_message(exc, reason, request=request)

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        if self._check_circuit():
            return AIMessage(content=self._build_circuit_breaker_message())

        attempt = 1
        while True:
            try:
                response = await handler(request)
                self._record_success()
                return response
            except GraphBubbleUp:
                # Preserve LangGraph control-flow signals (interrupt/pause/resume).
                self._release_half_open_probe_on_bubble_up()
                raise
            except Exception as exc:
                retriable, reason = self._classify_error(exc)

                if self._should_fail_over_api_pool(request, reason):
                    fallback_request = self._build_api_pool_fallback_request(request)
                    if fallback_request is not None:
                        logger.warning(
                            "API pool failed after %d attempt(s); falling back to local model. Reason: %s",
                            attempt,
                            _extract_error_detail(exc),
                        )
                        try:
                            response = await handler(fallback_request)
                            self._record_success()
                            return response
                        except GraphBubbleUp:
                            self._release_half_open_probe_on_bubble_up()
                            raise
                        except Exception as fallback_exc:
                            retriable, reason = self._classify_error(fallback_exc)
                            exc = fallback_exc

                if retriable and self._provider_handles_transient_retries(request):
                    logger.warning(
                        "LLM call failed after provider-managed retries; skipping outer retry: %s",
                        _extract_error_detail(exc),
                        exc_info=exc,
                    )
                    self._record_failure()
                    return self._build_error_message(exc, reason, request=request)

                if retriable and attempt < self.retry_max_attempts:
                    wait_ms = self._build_retry_delay_ms(attempt, exc)
                    logger.warning(
                        "Transient LLM error on attempt %d/%d; retrying in %dms: %s",
                        attempt,
                        self.retry_max_attempts,
                        wait_ms,
                        _extract_error_detail(exc),
                    )
                    self._emit_retry_event(attempt, wait_ms, reason)
                    await asyncio.sleep(wait_ms / 1000)
                    attempt += 1
                    continue

                logger.warning(
                    "LLM call failed after %d attempt(s): %s",
                    attempt,
                    _extract_error_detail(exc),
                    exc_info=exc,
                )
                if retriable:
                    self._record_failure()
                return self._build_error_message(exc, reason, request=request)


def _matches_any(detail: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in detail for pattern in patterns)


def _extract_error_code(exc: BaseException) -> Any:
    for attr in ("code", "error_code"):
        value = getattr(exc, attr, None)
        if value not in (None, ""):
            return value

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            for key in ("code", "type"):
                value = error.get(key)
                if value not in (None, ""):
                    return value
    return None


def _extract_status_code(exc: BaseException) -> int | None:
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status

    match = _HTTP_STATUS_RE.search(_extract_error_detail(exc))
    if not match:
        return None
    return int(match.group(1))


def _extract_retry_after_ms(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None

    raw = None
    header_name = ""
    for key in ("retry-after-ms", "Retry-After-Ms", "retry-after", "Retry-After"):
        header_name = key
        if hasattr(headers, "get"):
            raw = headers.get(key)
        if raw:
            break
    if not raw:
        return None

    try:
        multiplier = 1 if "ms" in header_name.lower() else 1000
        return max(0, int(float(raw) * multiplier))
    except (TypeError, ValueError):
        try:
            target = parsedate_to_datetime(str(raw))
            delta = target.timestamp() - time.time()
            return max(0, int(delta * 1000))
        except (TypeError, ValueError, OverflowError):
            return None


def _extract_error_detail(exc: BaseException) -> str:
    detail = str(exc).strip()
    if detail:
        return detail
    message = getattr(exc, "message", None)
    if isinstance(message, str) and message.strip():
        return message.strip()
    return exc.__class__.__name__

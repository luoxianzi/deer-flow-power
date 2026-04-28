"""In-process diagnostics for the AIO sandbox tool runtime (HTTP bridge)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

_lock = threading.Lock()


@dataclass
class ToolRuntimeDiagnostics:
    """Lightweight snapshot of tool-bridge health for doctor / logging."""

    sandbox_provider_class: str | None = None
    last_success_ts: float | None = None
    last_failure_ts: float | None = None
    last_failure_tool: str | None = None
    last_failure_class: str | None = None
    last_failure_message: str | None = None
    last_sandbox_id: str | None = None
    last_sandbox_url: str | None = None
    last_request_context: dict[str, Any] = field(default_factory=dict)
    degraded: bool = False
    degraded_reason: str | None = None
    last_recovery_action: str | None = None
    last_recovery_ts: float | None = None
    total_recoveries: int = 0
    total_retries: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sandbox_provider_class": self.sandbox_provider_class,
            "last_success_ts": self.last_success_ts,
            "last_failure_ts": self.last_failure_ts,
            "last_failure_tool": self.last_failure_tool,
            "last_failure_class": self.last_failure_class,
            "last_failure_message": self.last_failure_message,
            "last_sandbox_id": self.last_sandbox_id,
            "last_sandbox_url": self.last_sandbox_url,
            "last_request_context": dict(self.last_request_context),
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
            "last_recovery_action": self.last_recovery_action,
            "last_recovery_ts": self.last_recovery_ts,
            "total_recoveries": self.total_recoveries,
            "total_retries": self.total_retries,
        }


_state = ToolRuntimeDiagnostics()


def snapshot() -> dict[str, Any]:
    with _lock:
        return _state.to_dict()


def set_provider_class(name: str | None) -> None:
    with _lock:
        _state.sandbox_provider_class = name


def record_success(
    *,
    tool_name: str | None = None,
    sandbox_id: str | None = None,
    sandbox_url: str | None = None,
    context: dict[str, Any] | None = None,
) -> None:
    with _lock:
        _state.last_success_ts = time.time()
        if tool_name is not None:
            _state.last_request_context["tool_name"] = tool_name
        if sandbox_id is not None:
            _state.last_sandbox_id = sandbox_id
        if sandbox_url is not None:
            _state.last_sandbox_url = sandbox_url
        if context:
            _state.last_request_context.update(context)
        _state.degraded = False
        _state.degraded_reason = None


def record_failure(
    *,
    tool_name: str | None,
    exc: BaseException | None = None,
    message: str | None = None,
    sandbox_id: str | None = None,
    sandbox_url: str | None = None,
    context: dict[str, Any] | None = None,
) -> None:
    with _lock:
        _state.last_failure_ts = time.time()
        _state.last_failure_tool = tool_name
        if exc is not None:
            _state.last_failure_class = type(exc).__name__
            _state.last_failure_message = str(exc)
        elif message is not None:
            _state.last_failure_class = None
            _state.last_failure_message = message
        if sandbox_id is not None:
            _state.last_sandbox_id = sandbox_id
        if sandbox_url is not None:
            _state.last_sandbox_url = sandbox_url
        if context:
            _state.last_request_context.update(context)


def record_retry(tool_name: str | None = None) -> None:
    with _lock:
        _state.total_retries += 1
        if tool_name:
            _state.last_request_context["last_retry_tool"] = tool_name


def record_recovery(action: str, *, sandbox_id: str | None = None) -> None:
    with _lock:
        _state.last_recovery_ts = time.time()
        _state.last_recovery_action = action
        _state.total_recoveries += 1
        if sandbox_id:
            _state.last_request_context["recovery_sandbox_id"] = sandbox_id


def mark_degraded(reason: str) -> None:
    with _lock:
        _state.degraded = True
        _state.degraded_reason = reason

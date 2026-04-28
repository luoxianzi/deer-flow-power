"""Helpers for AIO sandbox HTTP tool-bridge resilience (connection refused, stale handles)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, TypeVar

from deerflow.sandbox.exceptions import SandboxNotFoundError

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime
    from langgraph.typing import ContextT

    from deerflow.agents.thread_state import ThreadState
    from deerflow.sandbox.sandbox import Sandbox

logger = logging.getLogger(__name__)

T = TypeVar("T")


def is_aio_sandbox_provider(provider: object) -> bool:
    return type(provider).__name__ == "AioSandboxProvider"


def tool_output_suggests_connection_failure(text: str) -> bool:
    m = text.lower()
    return (
        "connection refused" in m
        or "[errno 111]" in m
        or "errno 111" in m
        or "failed to establish a new connection" in m
    )


def exception_suggests_connection_failure(exc: BaseException) -> bool:
    if isinstance(exc, BrokenPipeError):
        return True
    if isinstance(exc, ConnectionError):
        return True
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == 111:
        return True
    name = type(exc).__name__
    if name in ("ConnectError", "ReadTimeout", "WriteTimeout", "RemoteProtocolError"):
        return True
    msg = str(exc).lower()
    return "connection refused" in msg or "errno 111" in msg or "[errno 111]" in msg


def ephemeral_local_sandbox_for_fallback():
    """Host-path sandbox with the same mount mappings as LocalSandboxProvider (read/write under mounts only)."""
    from deerflow.sandbox.local.local_sandbox import LocalSandbox
    from deerflow.sandbox.local.local_sandbox_provider import LocalSandboxProvider

    prov = LocalSandboxProvider()
    return LocalSandbox("__tool_fallback__", path_mappings=prov._path_mappings)


def run_with_bridge_retry(
    *,
    runtime: "ToolRuntime[ContextT, ThreadState]",
    tool_name: str,
    thread_id: str,
    sandbox_id: str,
    fn: Callable[["Sandbox"], T],
) -> T:
    """Run fn(sandbox); on connection-style failure, recover AIO bridge once and retry."""
    from deerflow.community.aio_sandbox import tool_runtime_diagnostics as trd
    from deerflow.community.aio_sandbox import tool_runtime_env as tre
    from deerflow.sandbox.sandbox_provider import get_sandbox_provider

    provider = get_sandbox_provider()
    sandbox = provider.get(sandbox_id)
    if sandbox is None:
        raise SandboxNotFoundError(f"Sandbox {sandbox_id} not found for {tool_name}", sandbox_id=sandbox_id)

    try:
        return fn(sandbox)
    except BaseException as exc:
        if not (
            is_aio_sandbox_provider(provider)
            and tre.tool_autorecover_enabled()
            and exception_suggests_connection_failure(exc)
        ):
            raise
        trd.record_failure(tool_name=tool_name, exc=exc, sandbox_id=sandbox_id, context={"phase": "pre_recover"})
        trd.record_retry(tool_name)
        recover = getattr(provider, "recover_sandbox_for_thread", None)
        if not callable(recover):
            raise
        logger.warning("Tool %s: connection failure, recovering sandbox for thread %s", tool_name, thread_id)
        new_id = recover(thread_id, sandbox_id)
        if runtime.state is not None:
            runtime.state["sandbox"] = {"sandbox_id": new_id}
        if runtime.context is not None:
            runtime.context["sandbox_id"] = new_id
        sandbox2 = provider.get(new_id)
        if sandbox2 is None:
            raise SandboxNotFoundError(
                f"Sandbox not found after recovery: {new_id}",
                sandbox_id=new_id,
            ) from exc
        return fn(sandbox2)


def run_aio_shell_with_recover(
    *,
    runtime: "ToolRuntime[ContextT, ThreadState]",
    tool_name: str,
    thread_id: str,
    sandbox_id: str,
    execute: Callable[["Sandbox"], str],
) -> str:
    """Run sandbox shell helper; if output looks like connection failure, recover once and retry."""
    from deerflow.community.aio_sandbox import tool_runtime_diagnostics as trd
    from deerflow.community.aio_sandbox import tool_runtime_env as tre
    from deerflow.sandbox.sandbox_provider import get_sandbox_provider

    provider = get_sandbox_provider()
    sandbox = provider.get(sandbox_id)
    if sandbox is None:
        return f"Error: sandbox {sandbox_id} not found"

    out = execute(sandbox)
    if not (
        is_aio_sandbox_provider(provider)
        and tre.tool_autorecover_enabled()
        and tool_output_suggests_connection_failure(out)
    ):
        return out

    trd.record_failure(
        tool_name=tool_name,
        message=out[:500],
        sandbox_id=sandbox_id,
        context={"phase": "shell_output_recover"},
    )
    trd.record_retry(tool_name)
    recover = getattr(provider, "recover_sandbox_for_thread", None)
    if not callable(recover):
        return out
    logger.warning("Tool %s: shell output suggests connection failure; recovering", tool_name)
    try:
        new_id = recover(thread_id, sandbox_id)
    except Exception as e:
        trd.mark_degraded(f"recover_failed:{e}")
        return f"Error: sandbox recovery failed: {e}"
    if runtime.state is not None:
        runtime.state["sandbox"] = {"sandbox_id": new_id}
    if runtime.context is not None:
        runtime.context["sandbox_id"] = new_id
    sandbox2 = provider.get(new_id)
    if sandbox2 is None:
        return f"Error: sandbox not found after recovery ({new_id})"
    return execute(sandbox2)


def run_aio_read_file_with_recover(
    *,
    runtime: "ToolRuntime[ContextT, ThreadState]",
    tool_name: str,
    thread_id: str,
    sandbox_id: str,
    read_fn: Callable[["Sandbox"], str],
) -> str:
    """read_file returns errors as strings; recover once if the message looks like a connection failure."""
    from deerflow.community.aio_sandbox import tool_runtime_diagnostics as trd
    from deerflow.community.aio_sandbox import tool_runtime_env as tre
    from deerflow.sandbox.sandbox_provider import get_sandbox_provider

    provider = get_sandbox_provider()
    sandbox = provider.get(sandbox_id)
    if sandbox is None:
        return "Error: sandbox not found"

    content = read_fn(sandbox)
    if not (
        is_aio_sandbox_provider(provider)
        and tre.tool_autorecover_enabled()
        and content.startswith("Error:")
        and tool_output_suggests_connection_failure(content)
    ):
        return content

    trd.record_failure(
        tool_name=tool_name,
        message=content[:500],
        sandbox_id=sandbox_id,
        context={"phase": "read_file_output_recover"},
    )
    trd.record_retry(tool_name)
    recover = getattr(provider, "recover_sandbox_for_thread", None)
    if not callable(recover):
        return content
    try:
        new_id = recover(thread_id, sandbox_id)
    except Exception as e:
        trd.mark_degraded(f"recover_failed:{e}")
        return f"Error: sandbox recovery failed: {e}"
    if runtime.state is not None:
        runtime.state["sandbox"] = {"sandbox_id": new_id}
    if runtime.context is not None:
        runtime.context["sandbox_id"] = new_id
    sandbox2 = provider.get(new_id)
    if sandbox2 is None:
        return "Error: sandbox not found after recovery"
    return read_fn(sandbox2)

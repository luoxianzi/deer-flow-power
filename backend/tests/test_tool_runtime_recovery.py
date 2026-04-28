"""Tests for sandbox HTTP tool-bridge recovery helpers."""

from unittest.mock import MagicMock

import pytest

from deerflow.community.aio_sandbox.backend import probe_sandbox_http
from deerflow.sandbox.tool_bridge_recovery import (
    exception_suggests_connection_failure,
    tool_output_suggests_connection_failure,
)


def test_probe_sandbox_http_false_on_invalid_url(monkeypatch):
    assert probe_sandbox_http("http://127.0.0.1:59999", timeout=0.01) is False


def test_tool_output_suggests_connection_failure():
    assert tool_output_suggests_connection_failure("Error: [Errno 111] Connection refused")
    assert tool_output_suggests_connection_failure("connection refused")
    assert not tool_output_suggests_connection_failure("hello world")


def test_exception_suggests_connection_failure():
    exc = OSError(111, "Connection refused")
    assert exception_suggests_connection_failure(exc)
    assert exception_suggests_connection_failure(ConnectionRefusedError())

    class ConnectError(Exception):
        pass

    assert exception_suggests_connection_failure(ConnectError("[Errno 111] Connection refused"))


def test_run_aio_shell_with_recover_retries(monkeypatch):
    from deerflow.sandbox import tool_bridge_recovery as tbr
    from deerflow.sandbox.sandbox_provider import reset_sandbox_provider, set_sandbox_provider

    reset_sandbox_provider()
    provider = MagicMock()
    provider.__class__.__name__ = "AioSandboxProvider"

    sb1 = MagicMock()
    sb1.execute_command.return_value = "Error: [Errno 111] Connection refused"
    sb2 = MagicMock()
    sb2.execute_command.return_value = "ok"

    provider.get.side_effect = [sb1, sb2]
    provider.recover_sandbox_for_thread = MagicMock(return_value="newid")

    set_sandbox_provider(provider)

    monkeypatch.setenv("DEERFLOW_TOOL_AUTORECOVER", "1")

    runtime = MagicMock()
    runtime.state = {"sandbox": {"sandbox_id": "old"}}
    runtime.context = {"thread_id": "t1"}

    out = tbr.run_aio_shell_with_recover(
        runtime=runtime,
        tool_name="bash",
        thread_id="t1",
        sandbox_id="old",
        execute=lambda sb: sb.execute_command("true"),
    )
    assert out == "ok"
    provider.recover_sandbox_for_thread.assert_called_once_with("t1", "old")
    assert runtime.state["sandbox"]["sandbox_id"] == "newid"

    reset_sandbox_provider()

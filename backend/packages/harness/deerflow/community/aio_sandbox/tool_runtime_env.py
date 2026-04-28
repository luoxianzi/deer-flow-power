"""Environment flags for Deer-Flow tool runtime (sandbox HTTP bridge) resilience."""

from __future__ import annotations

import os


def _truthy(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def tool_healthcheck_enabled() -> bool:
    return _truthy("DEERFLOW_TOOL_HEALTHCHECK", default=True)


def tool_autorecover_enabled() -> bool:
    return _truthy("DEERFLOW_TOOL_AUTORECOVER", default=True)


def tool_fallback_local_enabled() -> bool:
    return _truthy("DEERFLOW_TOOL_FALLBACK_LOCAL", default=False)


def tool_debug_enabled() -> bool:
    return _truthy("DEERFLOW_TOOL_DEBUG", default=False)


def tool_probe_timeout_seconds() -> float:
    try:
        return max(0.5, float(os.environ.get("DEERFLOW_TOOL_PROBE_TIMEOUT", "2.5")))
    except ValueError:
        return 2.5


def tool_recover_max_attempts() -> int:
    try:
        return max(1, min(5, int(os.environ.get("DEERFLOW_TOOL_RECOVER_MAX_ATTEMPTS", "3"))))
    except ValueError:
        return 3


def tool_recover_backoff_base_seconds() -> float:
    try:
        return max(0.1, float(os.environ.get("DEERFLOW_TOOL_RECOVER_BACKOFF_BASE", "0.4")))
    except ValueError:
        return 0.4

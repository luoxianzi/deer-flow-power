"""Lightweight usage telemetry for custom agents."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from deerflow.config.paths import get_paths

logger = logging.getLogger(__name__)

_USAGE_LOG_FILENAME = "agent_usage.jsonl"


@dataclass(slots=True)
class AgentUsageStats:
    """Aggregated usage counters for a custom agent."""

    manual_runs: int = 0
    delegated_tasks: int = 0
    last_used_at: str | None = None
    last_manual_run_at: str | None = None
    last_delegated_task_at: str | None = None


def _usage_log_path():
    return get_paths().base_dir / _USAGE_LOG_FILENAME


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_agent_name(agent_name: str) -> str:
    return agent_name.strip().lower()


def record_agent_usage(
    agent_name: str,
    usage_type: Literal["manual", "delegated"],
    *,
    source_agent_name: str | None = None,
    thread_id: str | None = None,
    task_id: str | None = None,
) -> None:
    """Append a usage event for a custom agent."""
    normalized_name = _normalize_agent_name(agent_name)
    if not normalized_name:
        return

    payload = {
        "ts": _utcnow_iso(),
        "agent_name": normalized_name,
        "usage_type": usage_type,
        "source_agent_name": source_agent_name.strip().lower() if isinstance(source_agent_name, str) and source_agent_name.strip() else None,
        "thread_id": thread_id,
        "task_id": task_id,
    }

    try:
        path = _usage_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        logger.exception("Failed to record custom-agent usage for %s", normalized_name)


def get_agent_usage_snapshot() -> dict[str, AgentUsageStats]:
    """Read and aggregate recorded custom-agent usage telemetry."""
    path = _usage_log_path()
    if not path.exists():
        return {}

    snapshot: dict[str, AgentUsageStats] = {}

    try:
        with path.open(encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed agent usage record: %s", line[:200])
                    continue

                agent_name = record.get("agent_name")
                if not isinstance(agent_name, str) or not agent_name:
                    continue

                usage_type = record.get("usage_type")
                timestamp = record.get("ts")
                stats = snapshot.setdefault(agent_name, AgentUsageStats())

                if usage_type == "manual":
                    stats.manual_runs += 1
                    stats.last_manual_run_at = timestamp if isinstance(timestamp, str) else stats.last_manual_run_at
                elif usage_type == "delegated":
                    stats.delegated_tasks += 1
                    stats.last_delegated_task_at = timestamp if isinstance(timestamp, str) else stats.last_delegated_task_at
                else:
                    continue

                if isinstance(timestamp, str):
                    stats.last_used_at = timestamp
    except Exception:
        logger.exception("Failed to load custom-agent usage telemetry from %s", path)

    return snapshot

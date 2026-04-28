"""Tests for lightweight custom-agent usage telemetry."""

from unittest.mock import patch

from deerflow.config.paths import Paths


def test_record_and_aggregate_agent_usage(tmp_path):
    with patch("deerflow.config.agent_usage.get_paths", return_value=Paths(base_dir=tmp_path)):
        from deerflow.config.agent_usage import get_agent_usage_snapshot, record_agent_usage

        record_agent_usage("Data-Architect-Agent", "manual", thread_id="thread-1")
        record_agent_usage(
            "data-architect-agent",
            "delegated",
            source_agent_name="global-chief-architect",
            thread_id="thread-1",
            task_id="task-1",
        )

        snapshot = get_agent_usage_snapshot()

    stats = snapshot["data-architect-agent"]
    assert stats.manual_runs == 1
    assert stats.delegated_tasks == 1
    assert stats.last_used_at is not None
    assert stats.last_manual_run_at is not None
    assert stats.last_delegated_task_at is not None


def test_get_agent_usage_snapshot_skips_malformed_lines(tmp_path):
    usage_file = tmp_path / "agent_usage.jsonl"
    usage_file.write_text('{"agent_name":"good-agent","usage_type":"manual","ts":"2026-04-03T00:00:00Z"}\nnot-json\n', encoding="utf-8")

    with patch("deerflow.config.agent_usage.get_paths", return_value=Paths(base_dir=tmp_path)):
        from deerflow.config.agent_usage import get_agent_usage_snapshot

        snapshot = get_agent_usage_snapshot()

    assert snapshot["good-agent"].manual_runs == 1

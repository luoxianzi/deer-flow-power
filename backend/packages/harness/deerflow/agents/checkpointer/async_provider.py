"""Async checkpointer factory.

Provides an **async context manager** for long-running async servers that need
proper resource cleanup.

Supported backends: memory, sqlite, postgres.

Usage (e.g. FastAPI lifespan)::

    from deerflow.agents.checkpointer.async_provider import make_checkpointer

    async with make_checkpointer() as checkpointer:
        app.state.checkpointer = checkpointer  # InMemorySaver if not configured

For sync usage see :mod:`deerflow.agents.checkpointer.provider`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator

from langgraph.types import Checkpointer

from deerflow.agents.checkpointer.provider import (
    POSTGRES_CONN_REQUIRED,
    POSTGRES_INSTALL,
    SQLITE_INSTALL,
)
from deerflow.config.app_config import get_app_config
from deerflow.runtime.store._sqlite_utils import ensure_sqlite_parent_dir, resolve_sqlite_conn_str

logger = logging.getLogger(__name__)


async def _adelete_for_runs(saver, run_ids) -> None:
    """Delete checkpoints belonging to the given run ids (rollback support).

    Detached from a mixin so it can be bound to arbitrary saver instances
    (including mocks) without triggering metaclass conflicts.
    """
    run_id_set = {str(run_id) for run_id in run_ids if run_id}
    if not run_id_set:
        return

    await saver.setup()
    checkpoint_keys: list[tuple[str, str, str]] = []

    async with saver.lock, saver.conn.execute(
        "SELECT thread_id, checkpoint_ns, checkpoint_id, metadata FROM checkpoints"
    ) as cur:
        async for thread_id, checkpoint_ns, checkpoint_id, metadata in cur:
            try:
                raw_metadata = (
                    metadata.decode("utf-8", "ignore")
                    if isinstance(metadata, (bytes, bytearray))
                    else str(metadata)
                )
                parsed_metadata = json.loads(raw_metadata)
            except Exception:
                continue

            if str(parsed_metadata.get("run_id")) in run_id_set:
                checkpoint_keys.append((thread_id, checkpoint_ns, checkpoint_id))

        if not checkpoint_keys:
            return

        await saver.conn.executemany(
            "DELETE FROM writes WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?",
            checkpoint_keys,
        )
        await saver.conn.executemany(
            "DELETE FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?",
            checkpoint_keys,
        )
        await saver.conn.commit()


def _attach_run_cleanup(saver) -> None:
    """Bind the run-cleanup coroutine to the saver instance unless one is already present."""
    if getattr(saver, "adelete_for_runs", None) is None:
        async def bound(run_ids):
            await _adelete_for_runs(saver, run_ids)

        saver.adelete_for_runs = bound

# ---------------------------------------------------------------------------
# Async factory
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def _async_checkpointer(config) -> AsyncIterator[Checkpointer]:
    """Async context manager that constructs and tears down a checkpointer."""
    if config.type == "memory":
        from langgraph.checkpoint.memory import InMemorySaver

        yield InMemorySaver()
        return

    if config.type == "sqlite":
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        except ImportError as exc:
            raise ImportError(SQLITE_INSTALL) from exc

        conn_str = resolve_sqlite_conn_str(config.connection_string or "store.db")
        await asyncio.to_thread(ensure_sqlite_parent_dir, conn_str)

        async with AsyncSqliteSaver.from_conn_string(conn_str) as saver:
            await saver.setup()
            _attach_run_cleanup(saver)
            yield saver
        return

    if config.type == "postgres":
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        except ImportError as exc:
            raise ImportError(POSTGRES_INSTALL) from exc

        if not config.connection_string:
            raise ValueError(POSTGRES_CONN_REQUIRED)

        async with AsyncPostgresSaver.from_conn_string(config.connection_string) as saver:
            await saver.setup()
            yield saver
        return

    raise ValueError(f"Unknown checkpointer type: {config.type!r}")


# ---------------------------------------------------------------------------
# Public async context manager
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def make_checkpointer() -> AsyncIterator[Checkpointer]:
    """Async context manager that yields a checkpointer for the caller's lifetime.
    Resources are opened on enter and closed on exit — no global state::

        async with make_checkpointer() as checkpointer:
            app.state.checkpointer = checkpointer

    Yields an ``InMemorySaver`` when no checkpointer is configured in *config.yaml*.
    """

    config = get_app_config()

    if config.checkpointer is None:
        from langgraph.checkpoint.memory import InMemorySaver

        yield InMemorySaver()
        return

    async with _async_checkpointer(config.checkpointer) as saver:
        yield saver

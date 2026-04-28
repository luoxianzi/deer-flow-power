"""Task tool for delegating work to subagents."""

import asyncio
import logging
import uuid
from dataclasses import replace
from typing import Annotated

from langchain.tools import InjectedToolCallId, ToolRuntime, tool
from langgraph.config import get_stream_writer
from langgraph.typing import ContextT

from deerflow.agents.lead_agent.prompt import get_skills_prompt_section
from deerflow.agents.thread_state import ThreadState
from deerflow.config.agent_usage import record_agent_usage
from deerflow.config.agents_config import classify_agent_config, is_agent_delegation_target, load_agent_config
from deerflow.sandbox.security import LOCAL_BASH_SUBAGENT_DISABLED_MESSAGE, is_host_bash_allowed
from deerflow.subagents import SubagentExecutor, get_available_subagent_names, get_subagent_config
from deerflow.subagents.executor import SubagentStatus, cleanup_background_task, get_background_task_result, request_cancel_background_task

logger = logging.getLogger(__name__)


def _resolve_runtime_thread_id(runtime: ToolRuntime[ContextT, ThreadState] | None) -> str | None:
    """Resolve the current thread id from runtime context or configurable metadata."""
    if runtime is None:
        return None

    if runtime.context:
        thread_id = runtime.context.get("thread_id")
        if isinstance(thread_id, str) and thread_id:
            return thread_id

    configurable = runtime.config.get("configurable", {}) if runtime.config else {}
    thread_id = configurable.get("thread_id")
    if isinstance(thread_id, str) and thread_id:
        return thread_id

    return None


def _resolve_subagent_max_turns(subagent_type: str, requested: int | None, default_value: int) -> int:
    """Resolve an effective subagent max_turns without allowing harmful downgrades.

    Built-in subagent defaults are chosen to leave enough room for exploration,
    tool use, and a final answer. Model-generated overrides like ``3`` or ``8``
    can prematurely trip LangGraph's recursion limit, so we only allow overrides
    that keep or increase the configured default.
    """
    if requested is None:
        return default_value

    if requested < default_value:
        logger.warning(
            "Ignoring undersized max_turns=%d for subagent %s; keeping configured default=%d",
            requested,
            subagent_type,
            default_value,
        )
        return default_value

    return requested


def _resolve_parent_reasoning(runtime: ToolRuntime[ContextT, ThreadState] | None) -> tuple[bool | None, str | None]:
    """Resolve parent thinking/reasoning settings from runtime config."""
    if runtime is None or runtime.config is None:
        return None, None

    configurable = runtime.config.get("configurable", {})
    metadata = runtime.config.get("metadata", {})

    thinking_enabled = configurable.get("thinking_enabled")
    if thinking_enabled is None:
        thinking_enabled = metadata.get("thinking_enabled")

    reasoning_effort = configurable.get("reasoning_effort")
    if reasoning_effort is None:
        reasoning_effort = metadata.get("reasoning_effort")

    return thinking_enabled, reasoning_effort


def _resolve_tool_model_name(config, parent_model: str | None) -> str | None:
    """Resolve the model name that should determine tool availability."""
    if config.model == "inherit":
        return parent_model
    return config.model


def _summarize_task_error(error: str | None) -> str:
    """Collapse noisy upstream failures into short, readable task errors."""
    if not error:
        return "Unknown subagent failure"

    normalized = " ".join(error.split())
    lower = normalized.lower()

    if "http 504" in lower and ("gateway timed out" in lower or "gateway time-out" in lower):
        return "API pool upstream gateway timed out (HTTP 504). Ultra subtask is active; retry the task or add more API pool capacity."

    if len(normalized) > 400:
        return normalized[:397] + "..."

    return normalized


def _completed_result_should_fail(result_text: str | None) -> bool:
    """Catch synthetic LLM failure messages that reached the task layer as plain text."""
    if not result_text:
        return False

    normalized = " ".join(result_text.split())
    return normalized.startswith("当前模型供应商暂时不可用") or normalized.startswith(
        "当前配置的模型供应商"
    ) or normalized.startswith("当前 API Pool 网关拒绝了这条网络出口") or normalized.startswith("LLM request failed:")


@tool("task", parse_docstring=True)
async def task_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    description: str,
    prompt: str,
    subagent_type: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    max_turns: int | None = None,
) -> str:
    """Delegate a task to a specialized subagent that runs in its own context.

    Subagents help you:
    - Preserve context by keeping exploration and implementation separate
    - Handle complex multi-step tasks autonomously
    - Execute commands or operations in isolated contexts

    Available subagent types depend on the active sandbox configuration:
    - **general-purpose**: A capable agent for complex, multi-step tasks that require
      both exploration and action. Use when the task requires complex reasoning,
      multiple dependent steps, or would benefit from isolated context.
    - **bash**: Command execution specialist for running bash commands. This is only
      available when host bash is explicitly allowed or when using an isolated shell
      sandbox such as `AioSandboxProvider`. Use for git operations, build processes,
      or when command output would be verbose.
    - **custom worker agent name**: Any delegatable custom agent, such as
      `data-architect-agent`, `security-audit-agent`, or other worker agents
      exposed by the current DeerFlow runtime.

    When to use this tool:
    - Complex tasks requiring multiple steps or tools
    - Tasks that produce verbose output
    - When you want to isolate context from the main conversation
    - Parallel research or exploration tasks

    When NOT to use this tool:
    - Simple, single-step operations (use tools directly)
    - Tasks requiring user interaction or clarification

    Args:
        description: A short (3-5 word) description of the task for logging/display. ALWAYS PROVIDE THIS PARAMETER FIRST.
        prompt: The task description for the subagent. Be specific and clear about what needs to be done. ALWAYS PROVIDE THIS PARAMETER SECOND.
        subagent_type: The type of subagent to use. ALWAYS PROVIDE THIS PARAMETER THIRD.
        max_turns: Optional maximum number of agent turns. Defaults to subagent's configured max.
    """
    available_subagent_names = get_available_subagent_names()

    # Get subagent configuration
    config = get_subagent_config(subagent_type)
    if config is None:
        try:
            agent_cfg = load_agent_config(subagent_type)
        except FileNotFoundError:
            agent_cfg = None
        except Exception as exc:
            logger.warning("Failed to inspect subagent target '%s': %s", subagent_type, exc)
            agent_cfg = None

        if agent_cfg is not None and not is_agent_delegation_target(agent_cfg):
            classification = classify_agent_config(agent_cfg)
            available = ", ".join(available_subagent_names)
            return (
                f"Error: Agent '{subagent_type}' exists but is classified as '{classification}' "
                f"and cannot be delegated via task(). Available: {available}"
            )

        available = ", ".join(available_subagent_names)
        return f"Error: Unknown subagent type '{subagent_type}'. Available: {available}"
    if subagent_type == "bash" and not is_host_bash_allowed():
        return f"Error: {LOCAL_BASH_SUBAGENT_DISABLED_MESSAGE}"

    # Build config overrides
    overrides: dict = {}

    skills_section = get_skills_prompt_section()
    if config.append_skills_prompt and skills_section:
        overrides["system_prompt"] = config.system_prompt + "\n\n" + skills_section

    effective_max_turns = _resolve_subagent_max_turns(subagent_type, max_turns, config.max_turns)
    if effective_max_turns != config.max_turns:
        overrides["max_turns"] = effective_max_turns

    if overrides:
        config = replace(config, **overrides)

    # Extract parent context from runtime
    sandbox_state = None
    thread_data = None
    thread_id = None
    parent_model = None
    parent_thinking_enabled = None
    parent_reasoning_effort = None
    trace_id = None

    if runtime is not None:
        sandbox_state = runtime.state.get("sandbox")
        thread_data = runtime.state.get("thread_data")
        thread_id = _resolve_runtime_thread_id(runtime)
        parent_thinking_enabled, parent_reasoning_effort = _resolve_parent_reasoning(runtime)

        # Try to get parent model from configurable
        metadata = runtime.config.get("metadata", {})
        parent_model = metadata.get("model_name")

        # Get or generate trace_id for distributed tracing
        trace_id = metadata.get("trace_id") or str(uuid.uuid4())[:8]

    # Get available tools (excluding task tool to prevent nesting)
    # Lazy import to avoid circular dependency
    from deerflow.tools import get_available_tools

    # Subagents should not have subagent tools enabled (prevent recursive nesting)
    tool_kwargs = {
        "model_name": _resolve_tool_model_name(config, parent_model),
        "subagent_enabled": False,
    }
    if config.tool_groups is not None:
        tool_kwargs["groups"] = config.tool_groups
    tools = get_available_tools(**tool_kwargs)

    # Create executor
    executor = SubagentExecutor(
        config=config,
        tools=tools,
        parent_model=parent_model,
        parent_thinking_enabled=parent_thinking_enabled,
        parent_reasoning_effort=parent_reasoning_effort,
        sandbox_state=sandbox_state,
        thread_data=thread_data,
        thread_id=thread_id,
        trace_id=trace_id,
    )

    # Start background execution (always async to prevent blocking)
    # Use tool_call_id as task_id for better traceability
    task_id = executor.execute_async(prompt, task_id=tool_call_id)

    if config.agent_name:
        parent_agent_name = None
        if runtime is not None and runtime.config is not None:
            metadata = runtime.config.get("metadata", {})
            parent_agent_name = metadata.get("agent_name")
        record_agent_usage(
            config.agent_name,
            "delegated",
            source_agent_name=parent_agent_name,
            thread_id=thread_id,
            task_id=task_id,
        )

    # Poll for task completion in backend (removes need for LLM to poll)
    poll_count = 0
    last_status = None
    last_message_count = 0  # Track how many AI messages we've already sent
    # Polling timeout: execution timeout + 60s buffer, checked every 5s
    max_poll_count = (config.timeout_seconds + 60) // 5

    logger.info(
        "[trace=%s] Started background task %s (subagent=%s source=%s classification=%s timeout=%ss polling_limit=%s polls)",
        trace_id,
        task_id,
        subagent_type,
        config.source,
        config.classification,
        config.timeout_seconds,
        max_poll_count,
    )

    writer = get_stream_writer()
    # Send Task Started message'
    writer(
        {
            "type": "task_started",
            "task_id": task_id,
            "description": description,
            "subagent_type": subagent_type,
            "subagent_source": config.source,
            "agent_name": config.agent_name,
        }
    )

    try:
        while True:
            result = get_background_task_result(task_id)

            if result is None:
                logger.error(f"[trace={trace_id}] Task {task_id} not found in background tasks")
                writer({"type": "task_failed", "task_id": task_id, "error": "Task disappeared from background tasks"})
                cleanup_background_task(task_id)
                return f"Error: Task {task_id} disappeared from background tasks"

            # Log status changes for debugging
            if result.status != last_status:
                logger.info(f"[trace={trace_id}] Task {task_id} status: {result.status.value}")
                last_status = result.status

            # Check for new AI messages and send task_running events
            current_message_count = len(result.ai_messages)
            if current_message_count > last_message_count:
                # Send task_running event for each new message
                for i in range(last_message_count, current_message_count):
                    message = result.ai_messages[i]
                    writer(
                        {
                            "type": "task_running",
                            "task_id": task_id,
                            "message": message,
                            "message_index": i + 1,  # 1-based index for display
                            "total_messages": current_message_count,
                        }
                    )
                    logger.info(f"[trace={trace_id}] Task {task_id} sent message #{i + 1}/{current_message_count}")
                last_message_count = current_message_count

            # Check if task completed, failed, cancelled, or timed out
            if result.status == SubagentStatus.COMPLETED:
                if _completed_result_should_fail(result.result):
                    summarized_error = _summarize_task_error(result.result)
                    writer({"type": "task_failed", "task_id": task_id, "error": summarized_error})
                    logger.error(
                        f"[trace={trace_id}] Task {task_id} completed with synthetic error payload; treating as failed: {summarized_error}"
                    )
                    cleanup_background_task(task_id)
                    return f"Task failed. Error: {summarized_error}"
                writer({"type": "task_completed", "task_id": task_id, "result": result.result})
                logger.info(f"[trace={trace_id}] Task {task_id} completed after {poll_count} polls")
                cleanup_background_task(task_id)
                return f"Task Succeeded. Result: {result.result}"
            elif result.status == SubagentStatus.FAILED:
                summarized_error = _summarize_task_error(result.error)
                writer({"type": "task_failed", "task_id": task_id, "error": summarized_error})
                logger.error(f"[trace={trace_id}] Task {task_id} failed: {summarized_error}")
                cleanup_background_task(task_id)
                return f"Task failed. Error: {summarized_error}"
            elif result.status == SubagentStatus.CANCELLED:
                writer({"type": "task_cancelled", "task_id": task_id, "error": result.error})
                logger.info(f"[trace={trace_id}] Task {task_id} cancelled: {result.error}")
                cleanup_background_task(task_id)
                return "Task cancelled by user."
            elif result.status == SubagentStatus.TIMED_OUT:
                summarized_error = _summarize_task_error(result.error)
                writer({"type": "task_timed_out", "task_id": task_id, "error": summarized_error})
                logger.warning(f"[trace={trace_id}] Task {task_id} timed out: {summarized_error}")
                cleanup_background_task(task_id)
                return f"Task timed out. Error: {summarized_error}"

            # Still running, wait before next poll
            await asyncio.sleep(5)
            poll_count += 1

            # Polling timeout as a safety net (in case thread pool timeout doesn't work)
            # Set to execution timeout + 60s buffer, in 5s poll intervals
            # This catches edge cases where the background task gets stuck
            # Note: We don't call cleanup_background_task here because the task may
            # still be running in the background. The cleanup will happen when the
            # executor completes and sets a terminal status.
            if poll_count > max_poll_count:
                timeout_minutes = config.timeout_seconds // 60
                logger.error(f"[trace={trace_id}] Task {task_id} polling timed out after {poll_count} polls (should have been caught by thread pool timeout)")
                writer({"type": "task_timed_out", "task_id": task_id})
                return f"Task polling timed out after {timeout_minutes} minutes. This may indicate the background task is stuck. Status: {result.status.value}"
    except asyncio.CancelledError:
        # Signal the background subagent thread to stop cooperatively.
        # Without this, the thread (running in ThreadPoolExecutor with its
        # own event loop via asyncio.run) would continue executing even
        # after the parent task is cancelled.
        request_cancel_background_task(task_id)

        async def cleanup_when_done() -> None:
            max_cleanup_polls = max_poll_count
            cleanup_poll_count = 0

            while True:
                result = get_background_task_result(task_id)
                if result is None:
                    return

                if result.status in {SubagentStatus.COMPLETED, SubagentStatus.FAILED, SubagentStatus.CANCELLED, SubagentStatus.TIMED_OUT} or getattr(result, "completed_at", None) is not None:
                    cleanup_background_task(task_id)
                    return

                if cleanup_poll_count > max_cleanup_polls:
                    logger.warning(f"[trace={trace_id}] Deferred cleanup for task {task_id} timed out after {cleanup_poll_count} polls")
                    return

                await asyncio.sleep(5)
                cleanup_poll_count += 1

        def log_cleanup_failure(cleanup_task: asyncio.Task[None]) -> None:
            if cleanup_task.cancelled():
                return

            exc = cleanup_task.exception()
            if exc is not None:
                logger.error(f"[trace={trace_id}] Deferred cleanup failed for task {task_id}: {exc}")

        logger.debug(f"[trace={trace_id}] Scheduling deferred cleanup for cancelled task {task_id}")
        asyncio.create_task(cleanup_when_done()).add_done_callback(log_cleanup_failure)
        raise

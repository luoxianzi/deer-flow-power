"""Subagent execution engine."""

import asyncio
import logging
import os
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from langchain.agents import create_agent
from langchain.tools import BaseTool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError

from deerflow.agents.thread_state import SandboxState, ThreadDataState, ThreadState
from deerflow.models import create_chat_model
from deerflow.subagents.config import SubagentConfig

logger = logging.getLogger(__name__)

_DEFAULT_RECURSION_CEILING = 200


def _get_recursion_ceiling() -> int:
    """Global maximum recursion budget, configurable via env var."""
    raw = os.getenv("DEER_FLOW_SUBAGENT_RECURSION_CEILING", "").strip()
    if raw:
        try:
            val = int(raw)
            return max(50, val)
        except ValueError:
            logger.warning("Invalid DEER_FLOW_SUBAGENT_RECURSION_CEILING=%r; using default %s", raw, _DEFAULT_RECURSION_CEILING)
    return _DEFAULT_RECURSION_CEILING


def _build_turn_budgets(max_turns: int, source: str) -> list[int]:
    """Build progressive retry budgets for a subagent.

    Builtin agents get no retry (single budget).
    Custom agents get up to two retries with escalating budgets,
    capped by the global recursion ceiling.
    """
    ceiling = _get_recursion_ceiling()
    budgets = [min(max_turns, ceiling)]

    if source != "custom":
        return budgets

    tier2 = min(int(max_turns * 1.5), ceiling)
    if tier2 > budgets[-1]:
        budgets.append(tier2)

    tier3 = min(max_turns * 2, ceiling)
    if tier3 > budgets[-1]:
        budgets.append(tier3)

    return budgets


def _looks_like_llm_error_message(message: AIMessage | None) -> tuple[bool, str | None]:
    """Detect middleware-generated fallback messages that should fail the subagent."""
    if message is None:
        return False, None

    if message.additional_kwargs.get("llm_error") is True:
        content = message.content
        if isinstance(content, str):
            return True, content
        return True, str(content)

    return False, None


class SubagentStatus(Enum):
    """Status of a subagent execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


@dataclass
class SubagentResult:
    """Result of a subagent execution.

    Attributes:
        task_id: Unique identifier for this execution.
        trace_id: Trace ID for distributed tracing (links parent and subagent logs).
        status: Current status of the execution.
        result: The final result message (if completed).
        error: Error message (if failed).
        started_at: When execution started.
        completed_at: When execution completed.
        ai_messages: List of complete AI messages (as dicts) generated during execution.
    """

    task_id: str
    trace_id: str
    status: SubagentStatus
    result: str | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    ai_messages: list[dict[str, Any]] | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.ai_messages is None:
            self.ai_messages = []


# Global storage for background task results
_background_tasks: dict[str, SubagentResult] = {}
_background_tasks_lock = threading.Lock()

# Thread pool for background task scheduling and orchestration.
# Sized to the hard concurrency ceiling (MAX_SUBAGENT_LIMIT=8) so even a full
# batch of parallel subagent submissions does not queue behind a saturated
# executor.
_scheduler_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="subagent-scheduler-")

# Thread pool for actual subagent execution (with timeout support).
# Must be >= SubagentLimitMiddleware's ceiling so a full parallel batch can run
# without pool contention. LLM calls are I/O bound → the memory cost of extra
# workers is minimal compared to the wall-clock speedup from true parallelism.
_execution_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="subagent-exec-")

# Dedicated pool for sync execute() calls made from an already-running event loop.
_isolated_loop_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="subagent-isolated-")


def _filter_tools(
    all_tools: list[BaseTool],
    allowed: list[str] | None,
    disallowed: list[str] | None,
) -> list[BaseTool]:
    """Filter tools based on subagent configuration.

    Args:
        all_tools: List of all available tools.
        allowed: Optional allowlist of tool names. If provided, only these tools are included.
        disallowed: Optional denylist of tool names. These tools are always excluded.

    Returns:
        Filtered list of tools.
    """
    filtered = all_tools

    # Apply allowlist if specified
    if allowed is not None:
        allowed_set = set(allowed)
        filtered = [t for t in filtered if t.name in allowed_set]

    # Apply denylist
    if disallowed is not None:
        disallowed_set = set(disallowed)
        filtered = [t for t in filtered if t.name not in disallowed_set]

    return filtered


def _get_model_name(config: SubagentConfig, parent_model: str | None) -> str | None:
    """Resolve the model name for a subagent.

    Args:
        config: Subagent configuration.
        parent_model: The parent agent's model name.

    Returns:
        Model name to use, or None to use default.
    """
    if config.model == "inherit":
        return parent_model
    return config.model


def _resolve_model_runtime(
    config: SubagentConfig,
    parent_model: str | None,
    parent_thinking_enabled: bool | None,
    parent_reasoning_effort: str | None,
) -> tuple[str | None, bool, str | None]:
    """Resolve effective model/runtime settings for a subagent."""
    model_name = _get_model_name(config, parent_model)

    if config.thinking_enabled is None:
        thinking_enabled = bool(parent_thinking_enabled)
    else:
        thinking_enabled = config.thinking_enabled

    reasoning_effort = config.reasoning_effort
    if reasoning_effort is None and thinking_enabled:
        reasoning_effort = parent_reasoning_effort

    return model_name, thinking_enabled, reasoning_effort


class SubagentExecutor:
    """Executor for running subagents."""

    def __init__(
        self,
        config: SubagentConfig,
        tools: list[BaseTool],
        parent_model: str | None = None,
        parent_thinking_enabled: bool | None = None,
        parent_reasoning_effort: str | None = None,
        sandbox_state: SandboxState | None = None,
        thread_data: ThreadDataState | None = None,
        thread_id: str | None = None,
        trace_id: str | None = None,
    ):
        """Initialize the executor.

        Args:
            config: Subagent configuration.
            tools: List of all available tools (will be filtered).
            parent_model: The parent agent's model name for inheritance.
            parent_thinking_enabled: Parent thinking flag for inheritable custom agents.
            parent_reasoning_effort: Parent reasoning effort for inheritable custom agents.
            sandbox_state: Sandbox state from parent agent.
            thread_data: Thread data from parent agent.
            thread_id: Thread ID for sandbox operations.
            trace_id: Trace ID from parent for distributed tracing.
        """
        self.config = config
        self.parent_model = parent_model
        self.parent_thinking_enabled = parent_thinking_enabled
        self.parent_reasoning_effort = parent_reasoning_effort
        self.sandbox_state = sandbox_state
        self.thread_data = thread_data
        self.thread_id = thread_id
        # Generate trace_id if not provided (for top-level calls)
        self.trace_id = trace_id or str(uuid.uuid4())[:8]

        # Filter tools based on config
        self.tools = _filter_tools(
            tools,
            config.tools,
            config.disallowed_tools,
        )

        logger.info(
            "[trace=%s] SubagentExecutor initialized: name=%s source=%s classification=%s agent_name=%s tools=%s",
            self.trace_id,
            config.name,
            config.source,
            config.classification,
            config.agent_name,
            len(self.tools),
        )

    def _create_agent(self):
        """Create the agent instance."""
        model_name, thinking_enabled, reasoning_effort = _resolve_model_runtime(
            self.config,
            self.parent_model,
            self.parent_thinking_enabled,
            self.parent_reasoning_effort,
        )
        # Cap subagent reasoning to 'high' — xhigh causes excessive thinking
        # with minimal actionable output on third-party API gateways.
        if reasoning_effort == "xhigh":
            reasoning_effort = "high"
        model = create_chat_model(
            name=model_name,
            thinking_enabled=thinking_enabled,
            reasoning_effort=reasoning_effort,
        )

        from deerflow.agents.middlewares.tool_error_handling_middleware import build_subagent_runtime_middlewares

        # Reuse shared middleware composition with lead agent.
        middlewares = build_subagent_runtime_middlewares(lazy_init=True)

        system_prompt = self._enrich_system_prompt(self.config.system_prompt)

        return create_agent(
            model=model,
            tools=self.tools,
            middleware=middlewares,
            system_prompt=system_prompt,
            state_schema=ThreadState,
        )

    @staticmethod
    def _enrich_system_prompt(prompt: str) -> str:
        """Append configured mount paths and tool-use mandate to the system prompt."""
        extra = ""
        try:
            from deerflow.config import get_app_config

            mounts = get_app_config().sandbox.mounts
            if mounts:
                lines = []
                for mount in mounts:
                    access = "read-only" if mount.read_only else "read/write"
                    lines.append(f"- Project mount: `{mount.container_path}` ({access})")
                extra += (
                    "\n<mounted_projects>\n"
                    + "\n".join(lines)
                    + "\n</mounted_projects>\n"
                )
        except Exception:
            pass

        extra += (
            "\n<tool_use_mandate>"
            "\nYou MUST use tools (bash, ls, read_file, write_file, str_replace) to investigate and act on the codebase."
            "\nNEVER respond with only text — always call at least one tool before providing your final answer."
            "\nStart by using ls or read_file to explore the relevant directories, then proceed with concrete actions."
            "\n</tool_use_mandate>\n"
        )
        return prompt + extra

    def _build_initial_state(self, task: str) -> dict[str, Any]:
        """Build the initial state for agent execution.

        Args:
            task: The task description.

        Returns:
            Initial state dictionary.
        """
        state: dict[str, Any] = {
            "messages": [HumanMessage(content=task)],
        }

        # Pass through sandbox and thread data from parent
        if self.sandbox_state is not None:
            state["sandbox"] = self.sandbox_state
        if self.thread_data is not None:
            state["thread_data"] = self.thread_data

        return state

    def _salvage_partial_result(self, result: "SubagentResult", final_state: dict | None) -> str | None:
        """Extract the best available text from collected AI messages or final_state
        when all recursion budgets are exhausted.  Returns None if nothing useful."""
        parts: list[str] = []
        for msg_dict in reversed(result.ai_messages):
            content = msg_dict.get("content", "")
            if isinstance(content, list):
                text_parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                text = "\n".join(t for t in text_parts if t)
            else:
                text = str(content) if content else ""
            if text.strip():
                parts.append(text.strip())
                break

        if not parts and final_state is not None:
            messages = final_state.get("messages", [])
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    c = msg.content
                    if isinstance(c, str) and c.strip():
                        parts.append(c.strip())
                        break

        if not parts:
            return None

        salvaged = parts[0]
        return (
            f"[注意：此子任務已耗盡所有遞迴預算，以下為截至中斷時的最佳部分結果]\n\n{salvaged}"
        )

    async def _aexecute(self, task: str, result_holder: SubagentResult | None = None) -> SubagentResult:
        """Execute a task asynchronously.

        Args:
            task: The task description for the subagent.
            result_holder: Optional pre-created result object to update during execution.

        Returns:
            SubagentResult with the execution result.
        """
        if result_holder is not None:
            # Use the provided result holder (for async execution with real-time updates)
            result = result_holder
        else:
            # Create a new result for synchronous execution
            task_id = str(uuid.uuid4())[:8]
            result = SubagentResult(
                task_id=task_id,
                trace_id=self.trace_id,
                status=SubagentStatus.RUNNING,
                started_at=datetime.now(),
            )

        try:
            agent = self._create_agent()
            state = self._build_initial_state(task)
            context = {}
            configurable: dict[str, Any] = {}
            if self.thread_id:
                configurable["thread_id"] = self.thread_id
                context["thread_id"] = self.thread_id
            if self.config.agent_name:
                configurable["agent_name"] = self.config.agent_name
                context["agent_name"] = self.config.agent_name

            turn_budgets = _build_turn_budgets(self.config.max_turns, self.config.source)

            logger.info(
                f"[trace={self.trace_id}] Subagent {self.config.name} starting async execution with turn_budgets={turn_budgets}"
            )

            final_state = None

            # Pre-check: bail out immediately if already cancelled before streaming starts
            if result.cancel_event.is_set():
                logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} cancelled before streaming")
                with _background_tasks_lock:
                    if result.status == SubagentStatus.RUNNING:
                        result.status = SubagentStatus.CANCELLED
                        result.error = "Cancelled by user"
                        result.completed_at = datetime.now()
                return result

            hit_recursion_limit = False
            for attempt_index, turn_budget in enumerate(turn_budgets, start=1):
                run_config: RunnableConfig = {"recursion_limit": turn_budget}
                if configurable:
                    run_config["configurable"] = configurable

                hit_recursion_limit = False
                try:
                    async for chunk in agent.astream(state, config=run_config, context=context, stream_mode="values"):  # type: ignore[arg-type]
                        # Cooperative cancellation: check if parent requested stop.
                        # Note: cancellation is only detected at astream iteration boundaries,
                        # so long-running tool calls within a single iteration will not be
                        # interrupted until the next chunk is yielded.
                        if result.cancel_event.is_set():
                            logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} cancelled by parent")
                            with _background_tasks_lock:
                                if result.status == SubagentStatus.RUNNING:
                                    result.status = SubagentStatus.CANCELLED
                                    result.error = "Cancelled by user"
                                    result.completed_at = datetime.now()
                            return result

                        final_state = chunk

                        messages = chunk.get("messages", [])
                        if messages:
                            last_message = messages[-1]
                            if isinstance(last_message, AIMessage):
                                message_dict = last_message.model_dump()
                                message_id = message_dict.get("id")
                                is_duplicate = False
                                if message_id:
                                    is_duplicate = any(msg.get("id") == message_id for msg in result.ai_messages)
                                else:
                                    is_duplicate = message_dict in result.ai_messages

                                if not is_duplicate:
                                    result.ai_messages.append(message_dict)
                                    logger.info(
                                        f"[trace={self.trace_id}] Subagent {self.config.name} captured AI message #{len(result.ai_messages)}"
                                    )
                    logger.info(
                        f"[trace={self.trace_id}] Subagent {self.config.name} completed on attempt={attempt_index} recursion_limit={turn_budget}"
                    )
                    break
                except GraphRecursionError:
                    hit_recursion_limit = True
                    if attempt_index < len(turn_budgets):
                        logger.warning(
                            "[trace=%s] Subagent %s hit recursion_limit=%s; escalating to recursion_limit=%s (attempt %s/%s)",
                            self.trace_id,
                            self.config.name,
                            turn_budget,
                            turn_budgets[attempt_index],
                            attempt_index,
                            len(turn_budgets),
                        )
                        continue
                    logger.warning(
                        "[trace=%s] Subagent %s exhausted all turn budgets %s; salvaging partial results.",
                        self.trace_id,
                        self.config.name,
                        turn_budgets,
                    )

            if hit_recursion_limit:
                partial = self._salvage_partial_result(result, final_state)
                if partial:
                    result.status = SubagentStatus.COMPLETED
                    result.result = partial
                    result.completed_at = datetime.now()
                    logger.info(
                        "[trace=%s] Subagent %s returned salvaged partial result (%d chars).",
                        self.trace_id,
                        self.config.name,
                        len(partial),
                    )
                    return result
                result.status = SubagentStatus.FAILED
                result.error = (
                    f"Subagent '{self.config.name}' exhausted all recursion budgets {turn_budgets} "
                    f"without completing. Consider adding max_turns in the agent's config.yaml."
                )
                result.completed_at = datetime.now()
                return result

            if final_state is None:
                logger.warning(f"[trace={self.trace_id}] Subagent {self.config.name} no final state")
                result.result = "No response generated"
            else:
                # Extract the final message - find the last AIMessage
                messages = final_state.get("messages", [])
                logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} final messages count: {len(messages)}")

                # Find the last AIMessage in the conversation
                last_ai_message = None
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage):
                        last_ai_message = msg
                        break

                if last_ai_message is not None:
                    content = last_ai_message.content
                    # Handle both str and list content types for the final result
                    if isinstance(content, str):
                        result.result = content
                    elif isinstance(content, list):
                        # Extract text from list of content blocks for final result only.
                        # Concatenate raw string chunks directly, but preserve separation
                        # between full text blocks for readability.
                        text_parts = []
                        pending_str_parts = []
                        for block in content:
                            if isinstance(block, str):
                                pending_str_parts.append(block)
                            elif isinstance(block, dict):
                                if pending_str_parts:
                                    text_parts.append("".join(pending_str_parts))
                                    pending_str_parts.clear()
                                text_val = block.get("text")
                                if isinstance(text_val, str):
                                    text_parts.append(text_val)
                        if pending_str_parts:
                            text_parts.append("".join(pending_str_parts))
                        result.result = "\n".join(text_parts) if text_parts else "No text content in response"
                    else:
                        result.result = str(content)

                    is_llm_error, llm_error_text = _looks_like_llm_error_message(last_ai_message)
                    if is_llm_error:
                        result.status = SubagentStatus.FAILED
                        result.error = llm_error_text or "Subagent model call failed"
                        result.completed_at = datetime.now()
                        return result
                elif messages:
                    # Fallback: use the last message if no AIMessage found
                    last_message = messages[-1]
                    logger.warning(f"[trace={self.trace_id}] Subagent {self.config.name} no AIMessage found, using last message: {type(last_message)}")
                    raw_content = last_message.content if hasattr(last_message, "content") else str(last_message)
                    if isinstance(raw_content, str):
                        result.result = raw_content
                    elif isinstance(raw_content, list):
                        parts = []
                        pending_str_parts = []
                        for block in raw_content:
                            if isinstance(block, str):
                                pending_str_parts.append(block)
                            elif isinstance(block, dict):
                                if pending_str_parts:
                                    parts.append("".join(pending_str_parts))
                                    pending_str_parts.clear()
                                text_val = block.get("text")
                                if isinstance(text_val, str):
                                    parts.append(text_val)
                        if pending_str_parts:
                            parts.append("".join(pending_str_parts))
                        result.result = "\n".join(parts) if parts else "No text content in response"
                    else:
                        result.result = str(raw_content)
                else:
                    logger.warning(f"[trace={self.trace_id}] Subagent {self.config.name} no messages in final state")
                    result.result = "No response generated"

            result.status = SubagentStatus.COMPLETED
            result.completed_at = datetime.now()

        except Exception as e:
            logger.exception(f"[trace={self.trace_id}] Subagent {self.config.name} async execution failed")
            result.status = SubagentStatus.FAILED
            result.error = str(e)
            result.completed_at = datetime.now()

        return result

    def _execute_in_isolated_loop(self, task: str, result_holder: SubagentResult | None = None) -> SubagentResult:
        """Execute the subagent in a completely fresh event loop.

        This method is designed to run in a separate thread to ensure complete
        isolation from any parent event loop, preventing conflicts with asyncio
        primitives that may be bound to the parent loop (e.g., httpx clients).
        """
        try:
            previous_loop = asyncio.get_event_loop()
        except RuntimeError:
            previous_loop = None

        # Create and set a new event loop for this thread
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._aexecute(task, result_holder))
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                if pending:
                    for task_obj in pending:
                        task_obj.cancel()
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(loop.shutdown_default_executor())
            except Exception:
                logger.debug(
                    f"[trace={self.trace_id}] Failed while cleaning up isolated event loop for subagent {self.config.name}",
                    exc_info=True,
                )
            finally:
                try:
                    loop.close()
                finally:
                    asyncio.set_event_loop(previous_loop)

    def execute(self, task: str, result_holder: SubagentResult | None = None) -> SubagentResult:
        """Execute a task synchronously (wrapper around async execution).

        This method runs the async execution in a new event loop, allowing
        asynchronous tools (like MCP tools) to be used within the thread pool.

        When called from within an already-running event loop (e.g., when the
        parent agent is async), this method isolates the subagent execution in
        a separate thread to avoid event loop conflicts with shared async
        primitives like httpx clients.

        Args:
            task: The task description for the subagent.
            result_holder: Optional pre-created result object to update during execution.

        Returns:
            SubagentResult with the execution result.
        """
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                logger.debug(f"[trace={self.trace_id}] Subagent {self.config.name} detected running event loop, using isolated thread")
                future = _isolated_loop_pool.submit(self._execute_in_isolated_loop, task, result_holder)
                return future.result()

            # Standard path: no running event loop, use asyncio.run
            return asyncio.run(self._aexecute(task, result_holder))
        except Exception as e:
            logger.exception(f"[trace={self.trace_id}] Subagent {self.config.name} execution failed")
            # Create a result with error if we don't have one
            if result_holder is not None:
                result = result_holder
            else:
                result = SubagentResult(
                    task_id=str(uuid.uuid4())[:8],
                    trace_id=self.trace_id,
                    status=SubagentStatus.FAILED,
                )
            result.status = SubagentStatus.FAILED
            result.error = str(e)
            result.completed_at = datetime.now()
            return result

    def execute_async(self, task: str, task_id: str | None = None) -> str:
        """Start a task execution in the background.

        Args:
            task: The task description for the subagent.
            task_id: Optional task ID to use. If not provided, a random UUID will be generated.

        Returns:
            Task ID that can be used to check status later.
        """
        # Use provided task_id or generate a new one
        if task_id is None:
            task_id = str(uuid.uuid4())[:8]

        # Create initial pending result
        result = SubagentResult(
            task_id=task_id,
            trace_id=self.trace_id,
            status=SubagentStatus.PENDING,
        )

        logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} starting async execution, task_id={task_id}, timeout={self.config.timeout_seconds}s")

        with _background_tasks_lock:
            _background_tasks[task_id] = result

        # Submit to scheduler pool
        def run_task():
            with _background_tasks_lock:
                _background_tasks[task_id].status = SubagentStatus.RUNNING
                _background_tasks[task_id].started_at = datetime.now()
                result_holder = _background_tasks[task_id]

            try:
                # Submit execution to execution pool with timeout
                # Pass result_holder so execute() can update it in real-time
                execution_future: Future = _execution_pool.submit(self.execute, task, result_holder)
                try:
                    # Wait for execution with timeout
                    exec_result = execution_future.result(timeout=self.config.timeout_seconds)
                    with _background_tasks_lock:
                        _background_tasks[task_id].status = exec_result.status
                        _background_tasks[task_id].result = exec_result.result
                        _background_tasks[task_id].error = exec_result.error
                        _background_tasks[task_id].completed_at = datetime.now()
                        _background_tasks[task_id].ai_messages = exec_result.ai_messages
                except FuturesTimeoutError:
                    logger.error(f"[trace={self.trace_id}] Subagent {self.config.name} execution timed out after {self.config.timeout_seconds}s")
                    with _background_tasks_lock:
                        if _background_tasks[task_id].status == SubagentStatus.RUNNING:
                            _background_tasks[task_id].status = SubagentStatus.TIMED_OUT
                            _background_tasks[task_id].error = f"Execution timed out after {self.config.timeout_seconds} seconds"
                            _background_tasks[task_id].completed_at = datetime.now()
                    # Signal cooperative cancellation and cancel the future
                    result_holder.cancel_event.set()
                    execution_future.cancel()
            except Exception as e:
                logger.exception(f"[trace={self.trace_id}] Subagent {self.config.name} async execution failed")
                with _background_tasks_lock:
                    _background_tasks[task_id].status = SubagentStatus.FAILED
                    _background_tasks[task_id].error = str(e)
                    _background_tasks[task_id].completed_at = datetime.now()

        _scheduler_pool.submit(run_task)
        return task_id


MAX_CONCURRENT_SUBAGENTS = 6


def request_cancel_background_task(task_id: str) -> None:
    """Signal a running background task to stop.

    Sets the cancel_event on the task, which is checked cooperatively
    by ``_aexecute`` during ``agent.astream()`` iteration.  This allows
    subagent threads — which cannot be force-killed via ``Future.cancel()``
    — to stop at the next iteration boundary.

    Args:
        task_id: The task ID to cancel.
    """
    with _background_tasks_lock:
        result = _background_tasks.get(task_id)
        if result is not None:
            result.cancel_event.set()
            logger.info("Requested cancellation for background task %s", task_id)


def get_background_task_result(task_id: str) -> SubagentResult | None:
    """Get the result of a background task.

    Args:
        task_id: The task ID returned by execute_async.

    Returns:
        SubagentResult if found, None otherwise.
    """
    with _background_tasks_lock:
        return _background_tasks.get(task_id)


def list_background_tasks() -> list[SubagentResult]:
    """List all background tasks.

    Returns:
        List of all SubagentResult instances.
    """
    with _background_tasks_lock:
        return list(_background_tasks.values())


def cleanup_background_task(task_id: str) -> None:
    """Remove a completed task from background tasks.

    Should be called by task_tool after it finishes polling and returns the result.
    This prevents memory leaks from accumulated completed tasks.

    Only removes tasks that are in a terminal state (COMPLETED/FAILED/TIMED_OUT)
    to avoid race conditions with the background executor still updating the task entry.

    Args:
        task_id: The task ID to remove.
    """
    with _background_tasks_lock:
        result = _background_tasks.get(task_id)
        if result is None:
            # Nothing to clean up; may have been removed already.
            logger.debug("Requested cleanup for unknown background task %s", task_id)
            return

        # Only clean up tasks that are in a terminal state to avoid races with
        # the background executor still updating the task entry.
        is_terminal_status = result.status in {
            SubagentStatus.COMPLETED,
            SubagentStatus.FAILED,
            SubagentStatus.CANCELLED,
            SubagentStatus.TIMED_OUT,
        }
        if is_terminal_status or result.completed_at is not None:
            del _background_tasks[task_id]
            logger.debug("Cleaned up background task: %s", task_id)
        else:
            logger.debug(
                "Skipping cleanup for non-terminal background task %s (status=%s)",
                task_id,
                result.status.value if hasattr(result.status, "value") else result.status,
            )

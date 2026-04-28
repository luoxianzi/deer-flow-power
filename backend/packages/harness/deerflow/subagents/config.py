"""Subagent configuration definitions."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SubagentConfig:
    """Configuration for a subagent.

    Attributes:
        name: Unique identifier for the subagent.
        description: When Claude should delegate to this subagent.
        system_prompt: The system prompt that guides the subagent's behavior.
        tools: Optional list of tool names to allow. If None, inherits all tools.
        tool_groups: Optional list of tool group names to load before allow/deny filtering.
        disallowed_tools: Optional list of tool names to deny.
        model: Model to use - 'inherit' uses parent's model.
        max_turns: Maximum number of agent turns before stopping.
        timeout_seconds: Maximum execution time in seconds (default: 900 = 15 minutes).
        agent_name: Optional custom agent name when this subagent maps to a named custom agent.
        source: Where this subagent definition came from.
        classification: Derived runtime class (worker/orchestrator/legacy/manual_only).
        append_skills_prompt: Whether task_tool should append the shared skills section.
        thinking_enabled: Whether to force thinking mode. ``None`` means inherit from parent.
        reasoning_effort: Optional reasoning effort override for supported models.
    """

    name: str
    description: str
    system_prompt: str
    tools: list[str] | None = None
    tool_groups: list[str] | None = None
    disallowed_tools: list[str] | None = field(default_factory=lambda: ["task"])
    model: str = "inherit"
    max_turns: int = 80
    timeout_seconds: int = 900
    agent_name: str | None = None
    source: Literal["builtin", "custom"] = "builtin"
    classification: Literal["worker", "orchestrator", "legacy", "manual_only"] = "worker"
    append_skills_prompt: bool = True
    thinking_enabled: bool | None = False
    reasoning_effort: str | None = None

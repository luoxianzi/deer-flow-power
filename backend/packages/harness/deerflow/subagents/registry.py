"""Subagent registry for managing available subagents."""

import logging
from dataclasses import replace

from deerflow.config.agents_config import (
    AgentConfig,
    classify_agent_config,
    is_agent_delegation_target,
    list_delegatable_custom_agents,
    load_agent_config,
)
from deerflow.sandbox.security import is_host_bash_allowed
from deerflow.subagents.builtins import BUILTIN_SUBAGENTS
from deerflow.subagents.config import SubagentConfig

logger = logging.getLogger(__name__)


DELEGATED_TASK_BOUNDARY_PROMPT = """
You are executing a bounded delegated task for the lead agent.

- Prioritize the explicit task in the user prompt over broad role exploration.
- Read directly referenced files and answer once you have enough evidence.
- If the task is an audit/review/diagnosis request, do not drift into unrelated redesign work.
- Prefer a concise partial answer with blockers over continuing to explore indefinitely.
""".strip()


def _apply_timeout_override(config: SubagentConfig) -> SubagentConfig:
    """Apply config.yaml runtime overrides (timeout, max_turns, model) to a subagent config."""
    from deerflow.config.subagents_config import get_subagents_app_config

    app_config = get_subagents_app_config()
    overrides: dict = {}

    effective_timeout = app_config.get_timeout_for(config.name)
    if effective_timeout != config.timeout_seconds:
        logger.debug(
            "Subagent '%s': timeout overridden by config.yaml (%ss -> %ss)",
            config.name,
            config.timeout_seconds,
            effective_timeout,
        )
        overrides["timeout_seconds"] = effective_timeout

    effective_max_turns = app_config.get_max_turns_for(config.name, config.max_turns)
    if effective_max_turns != config.max_turns:
        logger.debug(
            "Subagent '%s': max_turns overridden by config.yaml (%s -> %s)",
            config.name,
            config.max_turns,
            effective_max_turns,
        )
        overrides["max_turns"] = effective_max_turns

    effective_model = app_config.get_model_for(config.name)
    if effective_model is not None and effective_model != config.model:
        logger.debug(
            "Subagent '%s': model overridden by config.yaml (%s -> %s)",
            config.name,
            config.model,
            effective_model,
        )
        overrides["model"] = effective_model

    if overrides:
        config = replace(config, **overrides)
    return config


_DEFAULT_CUSTOM_AGENT_MAX_TURNS = 80


def _build_custom_subagent_config(agent_cfg: AgentConfig) -> SubagentConfig:
    """Convert a custom worker agent into a task() runnable config."""
    from deerflow.agents.lead_agent.prompt import apply_prompt_template

    effective_max_turns = agent_cfg.max_turns if agent_cfg.max_turns is not None else _DEFAULT_CUSTOM_AGENT_MAX_TURNS
    if effective_max_turns < 20:
        logger.warning(
            "Agent '%s' config.yaml max_turns=%s is too low; clamping to 20.",
            agent_cfg.name,
            effective_max_turns,
        )
        effective_max_turns = 20

    description = agent_cfg.role or agent_cfg.description or agent_cfg.mission or f"Custom worker agent '{agent_cfg.name}'"
    return SubagentConfig(
        name=agent_cfg.name,
        description=description,
        system_prompt=apply_prompt_template(subagent_enabled=False, agent_name=agent_cfg.name)
        + "\n\n"
        + DELEGATED_TASK_BOUNDARY_PROMPT,
        tool_groups=agent_cfg.tool_groups,
        disallowed_tools=["task", "ask_clarification", "present_files"],
        model=agent_cfg.model or "inherit",
        max_turns=effective_max_turns,
        agent_name=agent_cfg.name,
        source="custom",
        classification=classify_agent_config(agent_cfg),
        append_skills_prompt=False,
        thinking_enabled=None,
    )


def get_subagent_config(name: str) -> SubagentConfig | None:
    """Get a subagent configuration by name, with config.yaml overrides applied.

    Args:
        name: The name of the subagent.

    Returns:
        SubagentConfig if found (with any config.yaml overrides applied), None otherwise.
    """
    config = BUILTIN_SUBAGENTS.get(name)
    if config is not None:
        return _apply_timeout_override(config)

    try:
        agent_cfg = load_agent_config(name)
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.warning("Failed to load custom agent '%s' as subagent target: %s", name, exc)
        return None

    if not is_agent_delegation_target(agent_cfg):
        return None

    config = _build_custom_subagent_config(agent_cfg)
    config = _apply_timeout_override(config)

    return config


def list_subagents() -> list[SubagentConfig]:
    """List all available subagent configurations (with config.yaml overrides applied).

    Returns:
        List of all registered SubagentConfig instances.
    """
    configs = [get_subagent_config(name) for name in BUILTIN_SUBAGENTS]
    configs.extend(_apply_timeout_override(_build_custom_subagent_config(agent_cfg)) for agent_cfg in list_delegatable_custom_agents())
    return [config for config in configs if config is not None]


def get_subagent_names() -> list[str]:
    """Get all available subagent names.

    Returns:
        List of subagent names.
    """
    return list(BUILTIN_SUBAGENTS.keys()) + [agent_cfg.name for agent_cfg in list_delegatable_custom_agents()]


def get_available_subagent_names() -> list[str]:
    """Get subagent names that should be exposed to the active runtime.

    Filters builtin ``bash`` when host bash is disallowed, and includes custom
    delegatable agents discovered from config.

    Returns:
        List of subagent names visible to the current sandbox configuration.
    """
    names = list(BUILTIN_SUBAGENTS.keys())
    try:
        host_bash_allowed = is_host_bash_allowed()
    except Exception:
        logger.debug("Could not determine host bash availability; exposing all built-in subagents")
        host_bash_allowed = True

    if not host_bash_allowed:
        names = [name for name in names if name != "bash"]

    try:
        names.extend(agent_cfg.name for agent_cfg in list_delegatable_custom_agents())
    except Exception:
        logger.debug("Failed to enumerate delegatable custom agents; skipping", exc_info=True)

    return names

"""CRUD API for custom agents."""

import logging
import re
import shutil

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from deerflow.config.agent_usage import AgentUsageStats, get_agent_usage_snapshot
from deerflow.config.agents_api_config import get_agents_api_config
from deerflow.config.agents_config import (
    AgentConfig,
    classify_agent_config,
    is_agent_delegation_target,
    list_custom_agents,
    load_agent_config,
    load_agent_soul,
)
from deerflow.config.paths import get_paths

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["agents"])

AGENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9-]+$")


class AgentUsageStatsResponse(BaseModel):
    """Observed usage counters for a custom agent."""

    manual_runs: int = Field(default=0, description="Number of direct custom-agent chat activations observed")
    delegated_tasks: int = Field(default=0, description="Number of task() delegations observed")
    last_used_at: str | None = Field(default=None, description="Last observed usage timestamp (UTC ISO8601)")
    last_manual_run_at: str | None = Field(default=None, description="Last direct custom-agent chat timestamp")
    last_delegated_task_at: str | None = Field(default=None, description="Last task() delegation timestamp")


class AgentResponse(BaseModel):
    """Response model for a custom agent."""

    name: str = Field(..., description="Agent name (hyphen-case)")
    description: str = Field(default="", description="Agent description")
    model: str | None = Field(default=None, description="Optional model override")
    tool_groups: list[str] | None = Field(default=None, description="Optional tool group whitelist")
    tags: list[str] | None = Field(default=None, description="Optional agent tags for grouping and visibility")
    role: str | None = Field(default=None, description="Agent role positioning")
    mission: str | None = Field(default=None, description="Agent mission statement")
    in_scope: list[str] | None = Field(default=None, description="Explicit in-scope responsibilities")
    out_of_scope: list[str] | None = Field(default=None, description="Explicit out-of-scope responsibilities")
    tool_permissions: list[str] | None = Field(default=None, description="Declared tool permission boundaries")
    constraints: list[str] | None = Field(default=None, description="Agent-specific hard constraints")
    escalation_rules: list[str] | None = Field(default=None, description="Conditions that require escalation")
    input_schema: dict | None = Field(default=None, description="Structured task input card schema")
    output_schema: dict | None = Field(default=None, description="Structured result package schema")
    completion_definition: list[str] | None = Field(default=None, description="Definition of done checklist")
    classification: str = Field(default="worker", description="Derived runtime classification: worker|orchestrator|legacy|manual_only")
    delegation_enabled: bool = Field(default=True, description="Whether this agent can be used as a task() delegation target")
    usage_stats: AgentUsageStatsResponse = Field(default_factory=AgentUsageStatsResponse, description="Observed usage counters from lightweight telemetry")
    soul: str | None = Field(default=None, description="SOUL.md content (included on GET /{name})")


class AgentsListResponse(BaseModel):
    """Response model for listing all custom agents."""

    agents: list[AgentResponse]


class AgentCreateRequest(BaseModel):
    """Request body for creating a custom agent."""

    name: str = Field(..., description="Agent name (must match ^[A-Za-z0-9-]+$, stored as lowercase)")
    description: str = Field(default="", description="Agent description")
    model: str | None = Field(default=None, description="Optional model override")
    tool_groups: list[str] | None = Field(default=None, description="Optional tool group whitelist")
    tags: list[str] | None = Field(default=None, description="Optional agent tags")
    role: str | None = Field(default=None, description="Agent role positioning")
    mission: str | None = Field(default=None, description="Agent mission statement")
    in_scope: list[str] | None = Field(default=None, description="Explicit in-scope responsibilities")
    out_of_scope: list[str] | None = Field(default=None, description="Explicit out-of-scope responsibilities")
    tool_permissions: list[str] | None = Field(default=None, description="Declared tool permission boundaries")
    constraints: list[str] | None = Field(default=None, description="Agent-specific hard constraints")
    escalation_rules: list[str] | None = Field(default=None, description="Conditions that require escalation")
    input_schema: dict | None = Field(default=None, description="Structured task input card schema")
    output_schema: dict | None = Field(default=None, description="Structured result package schema")
    completion_definition: list[str] | None = Field(default=None, description="Definition of done checklist")
    soul: str = Field(default="", description="SOUL.md content — agent personality and behavioral guardrails")


class AgentUpdateRequest(BaseModel):
    """Request body for updating a custom agent."""

    description: str | None = Field(default=None, description="Updated description")
    model: str | None = Field(default=None, description="Updated model override")
    tool_groups: list[str] | None = Field(default=None, description="Updated tool group whitelist")
    tags: list[str] | None = Field(default=None, description="Updated agent tags")
    role: str | None = Field(default=None, description="Updated role positioning")
    mission: str | None = Field(default=None, description="Updated mission statement")
    in_scope: list[str] | None = Field(default=None, description="Updated in-scope responsibilities")
    out_of_scope: list[str] | None = Field(default=None, description="Updated out-of-scope responsibilities")
    tool_permissions: list[str] | None = Field(default=None, description="Updated tool permission boundaries")
    constraints: list[str] | None = Field(default=None, description="Updated hard constraints")
    escalation_rules: list[str] | None = Field(default=None, description="Updated escalation rules")
    input_schema: dict | None = Field(default=None, description="Updated structured task input card schema")
    output_schema: dict | None = Field(default=None, description="Updated structured result package schema")
    completion_definition: list[str] | None = Field(default=None, description="Updated definition of done checklist")
    soul: str | None = Field(default=None, description="Updated SOUL.md content")


def _validate_agent_name(name: str) -> None:
    """Validate agent name against allowed pattern.

    Args:
        name: The agent name to validate.

    Raises:
        HTTPException: 422 if the name is invalid.
    """
    if not AGENT_NAME_PATTERN.match(name):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid agent name '{name}'. Must match ^[A-Za-z0-9-]+$ (letters, digits, and hyphens only).",
        )


def _normalize_agent_name(name: str) -> str:
    """Normalize agent name to lowercase for filesystem storage."""
    return name.lower()


def _usage_stats_to_response(usage_stats: AgentUsageStats | None) -> AgentUsageStatsResponse:
    if usage_stats is None:
        return AgentUsageStatsResponse()
    return AgentUsageStatsResponse(
        manual_runs=usage_stats.manual_runs,
        delegated_tasks=usage_stats.delegated_tasks,
        last_used_at=usage_stats.last_used_at,
        last_manual_run_at=usage_stats.last_manual_run_at,
        last_delegated_task_at=usage_stats.last_delegated_task_at,
    )


def _require_agents_api_enabled() -> None:
    """Reject access unless the custom-agent management API is explicitly enabled."""
    if not get_agents_api_config().enabled:
        raise HTTPException(
            status_code=403,
            detail=("Custom-agent management API is disabled. Set agents_api.enabled=true to expose agent and user-profile routes over HTTP."),
        )


def _agent_config_to_response(
    agent_cfg: AgentConfig,
    include_soul: bool = False,
    usage_stats: AgentUsageStats | None = None,
) -> AgentResponse:
    """Convert AgentConfig to AgentResponse."""
    soul: str | None = None
    if include_soul:
        soul = load_agent_soul(agent_cfg.name) or ""

    classification = classify_agent_config(agent_cfg)
    delegation_enabled = is_agent_delegation_target(agent_cfg)

    return AgentResponse(
        name=agent_cfg.name,
        description=agent_cfg.description,
        model=agent_cfg.model,
        tool_groups=agent_cfg.tool_groups,
        tags=agent_cfg.tags,
        role=agent_cfg.role,
        mission=agent_cfg.mission,
        in_scope=agent_cfg.in_scope,
        out_of_scope=agent_cfg.out_of_scope,
        tool_permissions=agent_cfg.tool_permissions,
        constraints=agent_cfg.constraints,
        escalation_rules=agent_cfg.escalation_rules,
        input_schema=agent_cfg.input_schema,
        output_schema=agent_cfg.output_schema,
        completion_definition=agent_cfg.completion_definition,
        classification=classification,
        delegation_enabled=delegation_enabled,
        usage_stats=_usage_stats_to_response(usage_stats),
        soul=soul,
    )


@router.get(
    "/agents",
    response_model=AgentsListResponse,
    summary="List Custom Agents",
    description="List all custom agents available in the agents directory, including their soul content.",
)
async def list_agents() -> AgentsListResponse:
    """List all custom agents.

    Returns:
        List of all custom agents with their metadata and soul content.
    """
    _require_agents_api_enabled()

    try:
        agents = list_custom_agents()
        usage_snapshot = get_agent_usage_snapshot()
        return AgentsListResponse(agents=[_agent_config_to_response(a, include_soul=True, usage_stats=usage_snapshot.get(a.name)) for a in agents])
    except Exception as e:
        logger.error(f"Failed to list agents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list agents: {str(e)}")


@router.get(
    "/agents/check",
    summary="Check Agent Name",
    description="Validate an agent name and check if it is available (case-insensitive).",
)
async def check_agent_name(name: str) -> dict:
    """Check whether an agent name is valid and not yet taken.

    Args:
        name: The agent name to check.

    Returns:
        ``{"available": true/false, "name": "<normalized>"}``

    Raises:
        HTTPException: 422 if the name is invalid.
    """
    _require_agents_api_enabled()
    _validate_agent_name(name)
    normalized = _normalize_agent_name(name)
    available = not get_paths().agent_dir(normalized).exists()
    return {"available": available, "name": normalized}


@router.get(
    "/agents/{name}",
    response_model=AgentResponse,
    summary="Get Custom Agent",
    description="Retrieve details and SOUL.md content for a specific custom agent.",
)
async def get_agent(name: str) -> AgentResponse:
    """Get a specific custom agent by name.

    Args:
        name: The agent name.

    Returns:
        Agent details including SOUL.md content.

    Raises:
        HTTPException: 404 if agent not found.
    """
    _require_agents_api_enabled()
    _validate_agent_name(name)
    name = _normalize_agent_name(name)

    try:
        agent_cfg = load_agent_config(name)
        usage_snapshot = get_agent_usage_snapshot()
        return _agent_config_to_response(agent_cfg, include_soul=True, usage_stats=usage_snapshot.get(agent_cfg.name))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")
    except Exception as e:
        logger.error(f"Failed to get agent '{name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get agent: {str(e)}")


@router.post(
    "/agents",
    response_model=AgentResponse,
    status_code=201,
    summary="Create Custom Agent",
    description="Create a new custom agent with its config and SOUL.md.",
)
async def create_agent_endpoint(request: AgentCreateRequest) -> AgentResponse:
    """Create a new custom agent.

    Args:
        request: The agent creation request.

    Returns:
        The created agent details.

    Raises:
        HTTPException: 409 if agent already exists, 422 if name is invalid.
    """
    _require_agents_api_enabled()
    _validate_agent_name(request.name)
    normalized_name = _normalize_agent_name(request.name)

    agent_dir = get_paths().agent_dir(normalized_name)

    if agent_dir.exists():
        raise HTTPException(status_code=409, detail=f"Agent '{normalized_name}' already exists")

    try:
        agent_dir.mkdir(parents=True, exist_ok=True)

        # Write config.yaml
        config_data: dict = {"name": normalized_name}
        if request.description:
            config_data["description"] = request.description
        if request.model is not None:
            config_data["model"] = request.model
        if request.tool_groups is not None:
            config_data["tool_groups"] = request.tool_groups
        if request.tags is not None:
            config_data["tags"] = request.tags
        if request.role is not None:
            config_data["role"] = request.role
        if request.mission is not None:
            config_data["mission"] = request.mission
        if request.in_scope is not None:
            config_data["in_scope"] = request.in_scope
        if request.out_of_scope is not None:
            config_data["out_of_scope"] = request.out_of_scope
        if request.tool_permissions is not None:
            config_data["tool_permissions"] = request.tool_permissions
        if request.constraints is not None:
            config_data["constraints"] = request.constraints
        if request.escalation_rules is not None:
            config_data["escalation_rules"] = request.escalation_rules
        if request.input_schema is not None:
            config_data["input_schema"] = request.input_schema
        if request.output_schema is not None:
            config_data["output_schema"] = request.output_schema
        if request.completion_definition is not None:
            config_data["completion_definition"] = request.completion_definition

        config_file = agent_dir / "config.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

        # Write SOUL.md
        soul_file = agent_dir / "SOUL.md"
        soul_file.write_text(request.soul, encoding="utf-8")

        logger.info(f"Created agent '{normalized_name}' at {agent_dir}")

        agent_cfg = load_agent_config(normalized_name)
        return _agent_config_to_response(agent_cfg, include_soul=True)

    except HTTPException:
        raise
    except Exception as e:
        # Clean up on failure
        if agent_dir.exists():
            shutil.rmtree(agent_dir)
        logger.error(f"Failed to create agent '{request.name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")


@router.put(
    "/agents/{name}",
    response_model=AgentResponse,
    summary="Update Custom Agent",
    description="Update an existing custom agent's config and/or SOUL.md.",
)
async def update_agent(name: str, request: AgentUpdateRequest) -> AgentResponse:
    """Update an existing custom agent.

    Args:
        name: The agent name.
        request: The update request (all fields optional).

    Returns:
        The updated agent details.

    Raises:
        HTTPException: 404 if agent not found.
    """
    _require_agents_api_enabled()
    _validate_agent_name(name)
    name = _normalize_agent_name(name)

    try:
        agent_cfg = load_agent_config(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")

    agent_dir = get_paths().agent_dir(name)

    try:
        # Update config if any config fields changed
        config_changed = any(
            v is not None
            for v in [
                request.description,
                request.model,
                request.tool_groups,
                request.tags,
                request.role,
                request.mission,
                request.in_scope,
                request.out_of_scope,
                request.tool_permissions,
                request.constraints,
                request.escalation_rules,
                request.input_schema,
                request.output_schema,
                request.completion_definition,
            ]
        )

        if config_changed:
            updated: dict = {
                "name": agent_cfg.name,
                "description": request.description if request.description is not None else agent_cfg.description,
            }
            new_model = request.model if request.model is not None else agent_cfg.model
            if new_model is not None:
                updated["model"] = new_model

            new_tool_groups = request.tool_groups if request.tool_groups is not None else agent_cfg.tool_groups
            if new_tool_groups is not None:
                updated["tool_groups"] = new_tool_groups

            new_tags = request.tags if request.tags is not None else agent_cfg.tags
            if new_tags is not None:
                updated["tags"] = new_tags

            new_role = request.role if request.role is not None else agent_cfg.role
            if new_role is not None:
                updated["role"] = new_role

            new_mission = request.mission if request.mission is not None else agent_cfg.mission
            if new_mission is not None:
                updated["mission"] = new_mission

            new_in_scope = request.in_scope if request.in_scope is not None else agent_cfg.in_scope
            if new_in_scope is not None:
                updated["in_scope"] = new_in_scope

            new_out_of_scope = request.out_of_scope if request.out_of_scope is not None else agent_cfg.out_of_scope
            if new_out_of_scope is not None:
                updated["out_of_scope"] = new_out_of_scope

            new_tool_permissions = request.tool_permissions if request.tool_permissions is not None else agent_cfg.tool_permissions
            if new_tool_permissions is not None:
                updated["tool_permissions"] = new_tool_permissions

            new_constraints = request.constraints if request.constraints is not None else agent_cfg.constraints
            if new_constraints is not None:
                updated["constraints"] = new_constraints

            new_escalation_rules = request.escalation_rules if request.escalation_rules is not None else agent_cfg.escalation_rules
            if new_escalation_rules is not None:
                updated["escalation_rules"] = new_escalation_rules

            new_input_schema = request.input_schema if request.input_schema is not None else agent_cfg.input_schema
            if new_input_schema is not None:
                updated["input_schema"] = new_input_schema

            new_output_schema = request.output_schema if request.output_schema is not None else agent_cfg.output_schema
            if new_output_schema is not None:
                updated["output_schema"] = new_output_schema

            new_completion_definition = request.completion_definition if request.completion_definition is not None else agent_cfg.completion_definition
            if new_completion_definition is not None:
                updated["completion_definition"] = new_completion_definition

            config_file = agent_dir / "config.yaml"
            with open(config_file, "w", encoding="utf-8") as f:
                yaml.dump(updated, f, default_flow_style=False, allow_unicode=True)

        # Update SOUL.md if provided
        if request.soul is not None:
            soul_path = agent_dir / "SOUL.md"
            soul_path.write_text(request.soul, encoding="utf-8")

        logger.info(f"Updated agent '{name}'")

        refreshed_cfg = load_agent_config(name)
        return _agent_config_to_response(refreshed_cfg, include_soul=True)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update agent '{name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update agent: {str(e)}")


class UserProfileResponse(BaseModel):
    """Response model for the global user profile (USER.md)."""

    content: str | None = Field(default=None, description="USER.md content, or null if not yet created")


class UserProfileUpdateRequest(BaseModel):
    """Request body for setting the global user profile."""

    content: str = Field(default="", description="USER.md content — describes the user's background and preferences")


@router.get(
    "/user-profile",
    response_model=UserProfileResponse,
    summary="Get User Profile",
    description="Read the global USER.md file that is injected into all custom agents.",
)
async def get_user_profile() -> UserProfileResponse:
    """Return the current USER.md content.

    Returns:
        UserProfileResponse with content=None if USER.md does not exist yet.
    """
    _require_agents_api_enabled()

    try:
        user_md_path = get_paths().user_md_file
        if not user_md_path.exists():
            return UserProfileResponse(content=None)
        raw = user_md_path.read_text(encoding="utf-8").strip()
        return UserProfileResponse(content=raw or None)
    except Exception as e:
        logger.error(f"Failed to read user profile: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to read user profile: {str(e)}")


@router.put(
    "/user-profile",
    response_model=UserProfileResponse,
    summary="Update User Profile",
    description="Write the global USER.md file that is injected into all custom agents.",
)
async def update_user_profile(request: UserProfileUpdateRequest) -> UserProfileResponse:
    """Create or overwrite the global USER.md.

    Args:
        request: The update request with the new USER.md content.

    Returns:
        UserProfileResponse with the saved content.
    """
    _require_agents_api_enabled()

    try:
        paths = get_paths()
        paths.base_dir.mkdir(parents=True, exist_ok=True)
        paths.user_md_file.write_text(request.content, encoding="utf-8")
        logger.info(f"Updated USER.md at {paths.user_md_file}")
        return UserProfileResponse(content=request.content or None)
    except Exception as e:
        logger.error(f"Failed to update user profile: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update user profile: {str(e)}")


@router.delete(
    "/agents/{name}",
    status_code=204,
    summary="Delete Custom Agent",
    description="Delete a custom agent and all its files (config, SOUL.md, memory).",
)
async def delete_agent(name: str) -> None:
    """Delete a custom agent.

    Args:
        name: The agent name.

    Raises:
        HTTPException: 404 if agent not found.
    """
    _require_agents_api_enabled()
    _validate_agent_name(name)
    name = _normalize_agent_name(name)

    agent_dir = get_paths().agent_dir(name)

    if not agent_dir.exists():
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")

    try:
        shutil.rmtree(agent_dir)
        logger.info(f"Deleted agent '{name}' from {agent_dir}")
    except Exception as e:
        logger.error(f"Failed to delete agent '{name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete agent: {str(e)}")

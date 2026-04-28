"""Runtime profile configuration for assistant execution modes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ModelEngine = Literal["codex", "claude", "api_pool"]
ReasoningEffort = Literal["minimal", "low", "medium", "high", "xhigh"]


class RuntimeProfileConfig(BaseModel):
    """A named runtime profile that can shape assistant execution defaults."""

    engine: ModelEngine | None = Field(
        default=None,
        description="Default engine family to use for this profile when the request does not override it.",
    )
    model_name: str | None = Field(
        default=None,
        description="Default model name to use for this profile when the request does not override it.",
    )
    thinking_enabled: bool | None = Field(
        default=None,
        description="Default thinking toggle for this profile.",
    )
    is_plan_mode: bool | None = Field(
        default=None,
        description="Default plan mode toggle for this profile.",
    )
    subagent_enabled: bool | None = Field(
        default=None,
        description="Default subagent toggle for this profile.",
    )
    reasoning_effort: ReasoningEffort | None = Field(
        default=None,
        description="Default reasoning effort for this profile.",
    )


def default_runtime_profiles() -> dict[str, RuntimeProfileConfig]:
    """Return the built-in execution profile defaults."""
    return {
        "flash": RuntimeProfileConfig(
            thinking_enabled=False,
            is_plan_mode=False,
            subagent_enabled=False,
            reasoning_effort="minimal",
        ),
        "thinking": RuntimeProfileConfig(
            thinking_enabled=True,
            is_plan_mode=False,
            subagent_enabled=False,
            reasoning_effort="low",
        ),
        "pro": RuntimeProfileConfig(
            thinking_enabled=True,
            is_plan_mode=True,
            subagent_enabled=False,
            reasoning_effort="medium",
        ),
        "ultra": RuntimeProfileConfig(
            thinking_enabled=True,
            is_plan_mode=False,
            subagent_enabled=True,
            reasoning_effort="high",
        ),
    }

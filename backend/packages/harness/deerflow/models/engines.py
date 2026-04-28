"""Helpers for mapping runtime engine selections to configured model providers."""

from __future__ import annotations

import os
from typing import Literal

CODEX_PROVIDER_PATH = "deerflow.models.openai_codex_provider:CodexChatModel"
CLAUDE_PROVIDER_PATH = "deerflow.models.claude_provider:ClaudeChatModel"
STANDARD_API_PROVIDER_PATH = "deerflow.models.standard_api_provider:StandardAPIChatModel"

ModelEngine = Literal["codex", "claude", "api_pool"]

ENGINE_PROVIDER_PATHS: dict[ModelEngine, str] = {
    "codex": CODEX_PROVIDER_PATH,
    "claude": CLAUDE_PROVIDER_PATH,
    "api_pool": STANDARD_API_PROVIDER_PATH,
}


def infer_engine_from_provider_path(provider_path: str | None) -> ModelEngine | None:
    """Infer the frontend engine label from a configured provider path."""
    if provider_path is None:
        return None

    for engine, expected_path in ENGINE_PROVIDER_PATHS.items():
        if provider_path == expected_path:
            return engine
    return None


def find_first_model_name_for_engine(app_config, engine: str) -> str | None:
    """Return the first configured model name for the requested engine."""
    provider_path = ENGINE_PROVIDER_PATHS.get(engine)  # type: ignore[arg-type]
    if provider_path is None:
        return None

    for model in app_config.models:
        if model.use == provider_path:
            return model.name
    return None


def resolve_runtime_model_name(model_config) -> str:
    """Resolve the effective runtime model name for dynamically routed providers."""
    engine = infer_engine_from_provider_path(model_config.use)
    runtime_model = model_config.model

    if engine == "api_pool":
        api_pool_model = os.getenv("API_POOL_MODEL", "").strip()
        if api_pool_model:
            runtime_model = api_pool_model

    return runtime_model


def infer_runtime_capabilities(
    *,
    engine: str | None,
    runtime_model: str,
    configured_supports_thinking: bool,
    configured_supports_reasoning_effort: bool,
) -> tuple[bool, bool]:
    """Infer thinking and reasoning-effort support for the active runtime model."""
    if engine != "api_pool":
        return configured_supports_thinking, configured_supports_reasoning_effort

    model_lower = runtime_model.lower()
    if model_lower.startswith("gpt-5") or model_lower.startswith(("o1", "o3", "o4")):
        return True, True

    return configured_supports_thinking, configured_supports_reasoning_effort


def normalize_runtime_reasoning_effort(
    *,
    engine: str | None,
    runtime_model: str,
    reasoning_effort: str | None,
) -> str | None:
    """Normalize reasoning effort for runtime model families with different enums."""
    if reasoning_effort is None:
        return None

    if engine != "api_pool":
        return reasoning_effort

    model_lower = runtime_model.lower()
    if model_lower.startswith("gpt-5") or model_lower.startswith(("o1", "o3", "o4")):
        if reasoning_effort == "minimal":
            return "none"

    return reasoning_effort

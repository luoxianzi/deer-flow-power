import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from deerflow.config import get_app_config
from deerflow.models.credential_loader import load_codex_cli_credential
from deerflow.models.engines import (
    infer_engine_from_provider_path,
    infer_runtime_capabilities,
    resolve_runtime_model_name,
)

router = APIRouter(prefix="/api", tags=["models"])


def _mask_account_id(account_id: str) -> str:
    if not account_id:
        return ""
    if len(account_id) <= 8:
        return account_id
    return f"{account_id[:8]}..."


def _append_description(base: str | None, detail: str) -> str:
    if not base:
        return detail
    if detail in base:
        return base
    return f"{base} ({detail})"


def _resolve_codex_credential_label() -> tuple[str | None, str | None]:
    credential = load_codex_cli_credential(emit_log=False)
    if credential is None:
        detail = "当前未检测到 Codex CLI 登录"
        return detail, detail

    if credential.account_id:
        masked_account = _mask_account_id(credential.account_id)
        detail = f"当前 Codex 账号: {masked_account}"
        label = f"账号 {masked_account}"
        return label, detail

    detail = "已检测到 Codex CLI 登录"
    return detail, detail


def _resolve_runtime_model_metadata(model) -> tuple[str, str | None, str | None, bool, bool, str | None]:
    """Resolve runtime display metadata and capabilities."""
    engine = infer_engine_from_provider_path(model.use)
    runtime_model = resolve_runtime_model_name(model)
    runtime_display_name = model.display_name
    runtime_description = model.description
    credential_label = None

    if engine == "api_pool":
        api_pool_model = os.getenv("API_POOL_MODEL", "").strip()
        if api_pool_model:
            runtime_display_name = f"{model.display_name or 'API Pool'} · {api_pool_model}"
            runtime_description = (
                f"{model.description or 'Standard OpenAI-compatible API pool'} "
                f"(runtime model: {api_pool_model})"
            )
    elif engine == "codex":
        credential_label, credential_detail = _resolve_codex_credential_label()
        if credential_detail:
            runtime_description = _append_description(runtime_description, credential_detail)

    supports_thinking, supports_reasoning_effort = infer_runtime_capabilities(
        engine=engine,
        runtime_model=runtime_model,
        configured_supports_thinking=model.supports_thinking,
        configured_supports_reasoning_effort=model.supports_reasoning_effort,
    )

    return (
        runtime_model,
        runtime_display_name,
        runtime_description,
        supports_thinking,
        supports_reasoning_effort,
        credential_label,
    )


class ModelResponse(BaseModel):
    """Response model for model information."""

    name: str = Field(..., description="Unique identifier for the model")
    model: str = Field(..., description="Actual provider model identifier")
    display_name: str | None = Field(None, description="Human-readable name")
    description: str | None = Field(None, description="Model description")
    engine: str | None = Field(None, description="Provider family used for engine routing")
    credential_label: str | None = Field(None, description="Visible label for the currently detected credential")
    supports_thinking: bool = Field(default=False, description="Whether model supports thinking mode")
    supports_reasoning_effort: bool = Field(default=False, description="Whether model supports reasoning effort")


class TokenUsageResponse(BaseModel):
    """Token usage display configuration."""

    enabled: bool = Field(default=False, description="Whether token usage display is enabled")


class ModelsListResponse(BaseModel):
    """Response model for listing all models."""

    models: list[ModelResponse]
    token_usage: TokenUsageResponse


@router.get(
    "/models",
    response_model=ModelsListResponse,
    summary="List All Models",
    description="Retrieve a list of all available AI models configured in the system.",
)
async def list_models() -> ModelsListResponse:
    """List all available models from configuration.

    Returns model information suitable for frontend display,
    excluding sensitive fields like API keys and internal configuration.

    Returns:
        A list of all configured models with their metadata and token usage display settings.

    Example Response:
        ```json
        {
            "models": [
                {
                    "name": "gpt-4",
                    "model": "gpt-4",
                    "display_name": "GPT-4",
                    "description": "OpenAI GPT-4 model",
                    "supports_thinking": false,
                    "supports_reasoning_effort": false
                },
                {
                    "name": "claude-3-opus",
                    "model": "claude-3-opus",
                    "display_name": "Claude 3 Opus",
                    "description": "Anthropic Claude 3 Opus model",
                    "supports_thinking": true,
                    "supports_reasoning_effort": false
                }
            ],
            "token_usage": {
                "enabled": true
            }
        }
        ```
    """
    config = get_app_config()
    models: list[ModelResponse] = []
    for model in config.models:
        (
            runtime_model,
            runtime_display_name,
            runtime_description,
            supports_thinking,
            supports_reasoning_effort,
            credential_label,
        ) = _resolve_runtime_model_metadata(model)
        models.append(
            ModelResponse(
                name=model.name,
                model=runtime_model,
                display_name=runtime_display_name,
                description=runtime_description,
                engine=infer_engine_from_provider_path(model.use),
                credential_label=credential_label,
                supports_thinking=supports_thinking,
                supports_reasoning_effort=supports_reasoning_effort,
            )
        )
    return ModelsListResponse(
        models=models,
        token_usage=TokenUsageResponse(enabled=config.token_usage.enabled),
    )


@router.get(
    "/models/{model_name}",
    response_model=ModelResponse,
    summary="Get Model Details",
    description="Retrieve detailed information about a specific AI model by its name.",
)
async def get_model(model_name: str) -> ModelResponse:
    """Get a specific model by name.

    Args:
        model_name: The unique name of the model to retrieve.

    Returns:
        Model information if found.

    Raises:
        HTTPException: 404 if model not found.

    Example Response:
        ```json
        {
            "name": "gpt-4",
            "display_name": "GPT-4",
            "description": "OpenAI GPT-4 model",
            "supports_thinking": false
        }
        ```
    """
    config = get_app_config()
    model = config.get_model_config(model_name)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    (
        runtime_model,
        runtime_display_name,
        runtime_description,
        supports_thinking,
        supports_reasoning_effort,
        credential_label,
    ) = _resolve_runtime_model_metadata(model)

    return ModelResponse(
        name=model.name,
        model=runtime_model,
        display_name=runtime_display_name,
        description=runtime_description,
        engine=infer_engine_from_provider_path(model.use),
        credential_label=credential_label,
        supports_thinking=supports_thinking,
        supports_reasoning_effort=supports_reasoning_effort,
    )

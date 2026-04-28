import logging

from langchain.chat_models import BaseChatModel

from deerflow.config import get_app_config, get_tracing_config, is_tracing_enabled
from deerflow.models.claude_provider import ClaudeAuthenticationUnavailableError
from deerflow.models.engines import (
    CLAUDE_PROVIDER_PATH,
    infer_engine_from_provider_path,
    infer_runtime_capabilities,
    normalize_runtime_reasoning_effort,
    resolve_runtime_model_name,
)
from deerflow.reflection import resolve_class
from deerflow.tracing import build_tracing_callbacks

logger = logging.getLogger(__name__)


def _find_auth_fallback_model_name(config, failed_model_name: str) -> str | None:
    """Find a non-Claude fallback model for top-level chat when Claude auth is unavailable."""
    for candidate in config.models:
        if candidate.name == failed_model_name:
            continue
        if candidate.use == CLAUDE_PROVIDER_PATH:
            continue
        return candidate.name
    return None


def _is_claude_auth_unavailable_error(exc: Exception) -> bool:
    if isinstance(exc, ClaudeAuthenticationUnavailableError):
        return True
    return "Claude authentication unavailable" in str(exc)


def _deep_merge_dicts(base: dict | None, override: dict) -> dict:
    """Recursively merge two dictionaries without mutating the inputs."""
    merged = dict(base or {})
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _vllm_disable_chat_template_kwargs(chat_template_kwargs: dict) -> dict:
    """Build the disable payload for vLLM/Qwen chat template kwargs."""
    disable_kwargs: dict[str, bool] = {}
    if "thinking" in chat_template_kwargs:
        disable_kwargs["thinking"] = False
    if "enable_thinking" in chat_template_kwargs:
        disable_kwargs["enable_thinking"] = False
    return disable_kwargs


def create_chat_model(
    name: str | None = None,
    thinking_enabled: bool = False,
    *,
    _allow_auth_fallback: bool = False,
    **kwargs,
) -> BaseChatModel:
    """Create a chat model instance from the config.

    Args:
        name: The name of the model to create. If None, the first model in the config will be used.

    Returns:
        A chat model instance.
    """
    original_kwargs = dict(kwargs)
    config = get_app_config()
    if name is None:
        name = config.models[0].name
    model_config = config.get_model_config(name)
    if model_config is None:
        raise ValueError(f"Model {name} not found in config") from None
    engine = infer_engine_from_provider_path(model_config.use)
    runtime_model_name = resolve_runtime_model_name(model_config)
    supports_thinking, supports_reasoning_effort = infer_runtime_capabilities(
        engine=engine,
        runtime_model=runtime_model_name,
        configured_supports_thinking=model_config.supports_thinking,
        configured_supports_reasoning_effort=model_config.supports_reasoning_effort,
    )
    model_class = resolve_class(model_config.use, BaseChatModel)
    model_settings_from_config = model_config.model_dump(
        exclude_none=True,
        exclude={
            "use",
            "name",
            "display_name",
            "description",
            "supports_thinking",
            "supports_reasoning_effort",
            "when_thinking_enabled",
            "when_thinking_disabled",
            "thinking",
            "supports_vision",
        },
    )
    # Compute effective when_thinking_enabled by merging in the `thinking` shortcut field.
    # The `thinking` shortcut is equivalent to setting when_thinking_enabled["thinking"].
    has_thinking_settings = (model_config.when_thinking_enabled is not None) or (model_config.thinking is not None)
    effective_wte: dict = dict(model_config.when_thinking_enabled) if model_config.when_thinking_enabled else {}
    if model_config.thinking is not None:
        merged_thinking = {**(effective_wte.get("thinking") or {}), **model_config.thinking}
        effective_wte = {**effective_wte, "thinking": merged_thinking}
    if thinking_enabled and has_thinking_settings:
        if not supports_thinking:
            raise ValueError(f"Model {name} does not support thinking. Set `supports_thinking` to true in the `config.yaml` to enable thinking.") from None
        if effective_wte:
            model_settings_from_config.update(effective_wte)
    if not thinking_enabled:
        if model_config.when_thinking_disabled is not None:
            # User-provided disable settings take full precedence
            model_settings_from_config.update(model_config.when_thinking_disabled)
        elif has_thinking_settings and effective_wte.get("extra_body", {}).get("thinking", {}).get("type"):
            # OpenAI-compatible gateway: thinking is nested under extra_body
            model_settings_from_config["extra_body"] = _deep_merge_dicts(
                model_settings_from_config.get("extra_body"),
                {"thinking": {"type": "disabled"}},
            )
            model_settings_from_config["reasoning_effort"] = normalize_runtime_reasoning_effort(
                engine=engine,
                runtime_model=runtime_model_name,
                reasoning_effort="minimal",
            )
        elif has_thinking_settings and (disable_chat_template_kwargs := _vllm_disable_chat_template_kwargs(effective_wte.get("extra_body", {}).get("chat_template_kwargs") or {})):
            # vLLM uses chat template kwargs to switch thinking on/off.
            model_settings_from_config["extra_body"] = _deep_merge_dicts(
                model_settings_from_config.get("extra_body"),
                {"chat_template_kwargs": disable_chat_template_kwargs},
            )
        elif has_thinking_settings and effective_wte.get("thinking", {}).get("type"):
            # Native langchain_anthropic: thinking is a direct constructor parameter
            model_settings_from_config["thinking"] = {"type": "disabled"}
    # Power-profile providers (StandardAPI / Codex / Claude) normalize
    # reasoning_effort themselves downstream, so the upstream "strip when
    # unsupported" rule would accidentally drop the operator's explicit
    # selection. Defer the strip decision until after the provider branches.
    from deerflow.models.claude_provider import ClaudeChatModel
    from deerflow.models.openai_codex_provider import CodexChatModel
    from deerflow.models.standard_api_provider import StandardAPIChatModel

    _provider_manages_reasoning = issubclass(model_class, (CodexChatModel, StandardAPIChatModel, ClaudeChatModel))

    if not model_config.supports_reasoning_effort and not _provider_manages_reasoning:
        kwargs.pop("reasoning_effort", None)
        model_settings_from_config.pop("reasoning_effort", None)

    if issubclass(model_class, ClaudeChatModel):
        # Claude (Opus 4.7+) consumes `reasoning_effort` via the provider's own
        # _apply_output_config (beta task-budgets header). For Sonnet 4.6 we
        # still forward the value so a future release can pick it up, but the
        # provider will only emit output_config when _enable_output_config is True
        # (auto-set for Opus 4.7). ChatAnthropic does not accept reasoning_effort
        # as a constructor kwarg, so we move it off kwargs onto model_settings
        # (which are fed to the subclass where the field is declared).
        explicit_effort = kwargs.pop("reasoning_effort", None)
        if explicit_effort is not None:
            model_settings_from_config["reasoning_effort"] = explicit_effort
        elif not model_config.supports_reasoning_effort:
            model_settings_from_config.pop("reasoning_effort", None)
    elif issubclass(model_class, CodexChatModel):
        # The ChatGPT Codex endpoint currently rejects max_tokens/max_output_tokens.
        model_settings_from_config.pop("max_tokens", None)

        # Use explicit reasoning_effort from frontend if provided (low/medium/high)
        explicit_effort = kwargs.pop("reasoning_effort", None)
        if not thinking_enabled:
            model_settings_from_config["reasoning_effort"] = "none"
        elif explicit_effort and explicit_effort in ("low", "medium", "high", "xhigh"):
            model_settings_from_config["reasoning_effort"] = explicit_effort
        elif "reasoning_effort" not in model_settings_from_config:
            model_settings_from_config["reasoning_effort"] = "medium"
    elif issubclass(model_class, StandardAPIChatModel):
        # StandardAPIChatModel (the "API pool" / 弹药库 gateway) always normalizes
        # reasoning_effort itself — surface the operator's explicit choice regardless
        # of the config's supports_reasoning_effort flag.
        explicit_effort = normalize_runtime_reasoning_effort(
            engine=engine,
            runtime_model=runtime_model_name,
            reasoning_effort=kwargs.pop("reasoning_effort", None),
        )
        if not thinking_enabled:
            model_settings_from_config["reasoning_effort"] = normalize_runtime_reasoning_effort(
                engine=engine,
                runtime_model=runtime_model_name,
                reasoning_effort="minimal",
            )
        elif explicit_effort and explicit_effort in ("none", "minimal", "low", "medium", "high", "xhigh"):
            model_settings_from_config["reasoning_effort"] = explicit_effort
        elif "reasoning_effort" not in model_settings_from_config:
            model_settings_from_config["reasoning_effort"] = "medium"

    try:
        model_instance = model_class(**{**model_settings_from_config, **kwargs})
    except ValueError as exc:
        if not (model_config.use == CLAUDE_PROVIDER_PATH and _is_claude_auth_unavailable_error(exc)):
            raise
        fallback_model_name = None
        if _allow_auth_fallback:
            fallback_model_name = _find_auth_fallback_model_name(config, name)
        if not fallback_model_name:
            raise

        logger.warning(
            "Claude auth unavailable for model '%s'; falling back to '%s'. Reason: %s",
            name,
            fallback_model_name,
            exc,
        )
        return create_chat_model(
            name=fallback_model_name,
            thinking_enabled=thinking_enabled,
            _allow_auth_fallback=False,
            **original_kwargs,
        )

    callbacks = build_tracing_callbacks()
    if callbacks:
        existing_callbacks = model_instance.callbacks or []
        model_instance.callbacks = [*existing_callbacks, *callbacks]
        logger.debug(f"Tracing attached to model '{name}' with providers={len(callbacks)}")
    return model_instance

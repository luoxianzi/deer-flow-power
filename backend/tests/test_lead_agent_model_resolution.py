"""Tests for lead agent runtime model resolution behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from deerflow.agents.lead_agent import agent as lead_agent_module
from deerflow.config.app_config import AppConfig
from deerflow.config.memory_config import MemoryConfig
from deerflow.config.model_config import ModelConfig
from deerflow.config.sandbox_config import SandboxConfig
from deerflow.config.summarization_config import SummarizationConfig
from deerflow.config.runtime_profile_config import RuntimeProfileConfig
from deerflow.models.engines import CLAUDE_PROVIDER_PATH, CODEX_PROVIDER_PATH, STANDARD_API_PROVIDER_PATH


def _make_app_config(models: list[ModelConfig]) -> AppConfig:
    return AppConfig(
        models=models,
        sandbox=SandboxConfig(use="deerflow.sandbox.local:LocalSandboxProvider"),
    )


def _make_model(
    name: str,
    *,
    supports_thinking: bool,
    use: str = "langchain_openai:ChatOpenAI",
) -> ModelConfig:
    return ModelConfig(
        name=name,
        display_name=name,
        description=None,
        use=use,
        model=name,
        supports_thinking=supports_thinking,
        supports_vision=False,
    )


def _patch_lead_agent_runtime(monkeypatch, app_config: AppConfig) -> dict[str, object]:
    import deerflow.tools as tools_module

    monkeypatch.setattr(lead_agent_module, "get_app_config", lambda: app_config)
    monkeypatch.setattr(lead_agent_module, "load_agent_config", lambda name: None)
    monkeypatch.setattr(lead_agent_module, "record_agent_usage", lambda *args, **kwargs: None)
    monkeypatch.setattr(tools_module, "get_available_tools", lambda **kwargs: [])
    monkeypatch.setattr(lead_agent_module, "_build_middlewares", lambda config, model_name, agent_name=None: [])

    captured: dict[str, object] = {}

    def _fake_create_chat_model(*, name, thinking_enabled, reasoning_effort=None):
        captured["name"] = name
        captured["thinking_enabled"] = thinking_enabled
        captured["reasoning_effort"] = reasoning_effort
        return object()

    monkeypatch.setattr(lead_agent_module, "create_chat_model", _fake_create_chat_model)
    monkeypatch.setattr(lead_agent_module, "create_agent", lambda **kwargs: kwargs)
    return captured


def test_resolve_model_name_falls_back_to_default(monkeypatch, caplog):
    app_config = _make_app_config(
        [
            _make_model("default-model", supports_thinking=False),
            _make_model("other-model", supports_thinking=True),
        ]
    )

    monkeypatch.setattr(lead_agent_module, "get_app_config", lambda: app_config)

    with caplog.at_level("WARNING"):
        resolved = lead_agent_module._resolve_model_name("missing-model")

    assert resolved == "default-model"
    assert "fallback to default model 'default-model'" in caplog.text


def test_resolve_model_name_uses_default_when_none(monkeypatch):
    app_config = _make_app_config(
        [
            _make_model("default-model", supports_thinking=False),
            _make_model("other-model", supports_thinking=True),
        ]
    )

    monkeypatch.setattr(lead_agent_module, "get_app_config", lambda: app_config)

    resolved = lead_agent_module._resolve_model_name(None)

    assert resolved == "default-model"


def test_resolve_model_name_for_engine_uses_matching_requested_model(monkeypatch):
    app_config = _make_app_config(
        [
            _make_model("gpt-5.4", supports_thinking=True, use=CODEX_PROVIDER_PATH),
            _make_model("claude-sonnet-4-6", supports_thinking=True, use=CLAUDE_PROVIDER_PATH),
        ]
    )

    monkeypatch.setattr(lead_agent_module, "get_app_config", lambda: app_config)

    resolved = lead_agent_module._resolve_model_name_for_engine("claude-sonnet-4-6", "claude")

    assert resolved == "claude-sonnet-4-6"


def test_resolve_model_name_for_engine_falls_back_to_engine_default(monkeypatch):
    app_config = _make_app_config(
        [
            _make_model("gpt-5.4", supports_thinking=True, use=CODEX_PROVIDER_PATH),
            _make_model("claude-sonnet-4-6", supports_thinking=True, use=CLAUDE_PROVIDER_PATH),
            _make_model("api-pool-default", supports_thinking=False, use=STANDARD_API_PROVIDER_PATH),
        ]
    )

    monkeypatch.setattr(lead_agent_module, "get_app_config", lambda: app_config)

    resolved = lead_agent_module._resolve_model_name_for_engine("claude-sonnet-4-6", "api_pool")

    assert resolved == "api-pool-default"


def test_resolve_model_name_raises_when_no_models_configured(monkeypatch):
    app_config = _make_app_config([])

    monkeypatch.setattr(lead_agent_module, "get_app_config", lambda: app_config)

    with pytest.raises(
        ValueError,
        match="No chat models are configured",
    ):
        lead_agent_module._resolve_model_name("missing-model")


def test_app_config_merges_runtime_profiles_with_defaults():
    app_config = AppConfig(
        models=[_make_model("default-model", supports_thinking=True)],
        sandbox=SandboxConfig(use="deerflow.sandbox.local:LocalSandboxProvider"),
        runtime_profiles={
            "Ultra": {
                "engine": "api_pool",
                "subagent_enabled": False,
            }
        },
    )

    assert {"flash", "thinking", "pro", "ultra"}.issubset(app_config.runtime_profiles)
    assert app_config.runtime_profiles["ultra"] == RuntimeProfileConfig(
        engine="api_pool",
        model_name=None,
        thinking_enabled=True,
        is_plan_mode=False,
        subagent_enabled=False,
        reasoning_effort="high",
    )


def test_make_lead_agent_applies_ultra_runtime_profile_defaults(monkeypatch):
    app_config = _make_app_config([_make_model("gpt-5.4", supports_thinking=True)])
    captured = _patch_lead_agent_runtime(monkeypatch, app_config)

    config = {"configurable": {"runtime_profile": "ultra"}}

    lead_agent_module.make_lead_agent(config)

    assert captured["name"] == "gpt-5.4"
    assert captured["thinking_enabled"] is True
    assert captured["reasoning_effort"] == "high"
    assert config["configurable"]["runtime_profile"] == "ultra"
    assert config["configurable"]["subagent_enabled"] is True
    assert config["configurable"]["is_plan_mode"] is False


def test_make_lead_agent_runtime_profile_explicit_overrides_take_precedence(monkeypatch):
    app_config = _make_app_config(
        [
            _make_model("gpt-5.4", supports_thinking=True, use=CODEX_PROVIDER_PATH),
            _make_model("claude-sonnet-4-6", supports_thinking=True, use=CLAUDE_PROVIDER_PATH),
            _make_model("api-pool-default", supports_thinking=True, use=STANDARD_API_PROVIDER_PATH),
        ]
    )
    app_config.runtime_profiles["ultra"] = RuntimeProfileConfig(
        engine="api_pool",
        thinking_enabled=True,
        is_plan_mode=False,
        subagent_enabled=True,
        reasoning_effort="high",
    )
    captured = _patch_lead_agent_runtime(monkeypatch, app_config)

    config = {
        "configurable": {
            "runtime_profile": "ultra",
            "engine": "claude",
            "thinking_enabled": False,
            "is_plan_mode": True,
            "subagent_enabled": False,
            "reasoning_effort": "low",
        }
    }

    lead_agent_module.make_lead_agent(config)

    assert captured["name"] == "claude-sonnet-4-6"
    assert captured["thinking_enabled"] is False
    assert captured["reasoning_effort"] == "low"
    assert config["configurable"]["engine"] == "claude"
    assert config["configurable"]["subagent_enabled"] is False
    assert config["configurable"]["is_plan_mode"] is True


def test_make_lead_agent_runtime_profile_writes_back_effective_model_resolution(monkeypatch):
    app_config = _make_app_config(
        [
            _make_model("gpt-5.4", supports_thinking=True, use=CODEX_PROVIDER_PATH),
            _make_model("claude-sonnet-4-6", supports_thinking=True, use=CLAUDE_PROVIDER_PATH),
            _make_model("api-pool-default", supports_thinking=True, use=STANDARD_API_PROVIDER_PATH),
        ]
    )
    app_config.runtime_profiles["ultra"] = RuntimeProfileConfig(
        engine="api_pool",
        thinking_enabled=True,
        is_plan_mode=False,
        subagent_enabled=True,
        reasoning_effort="high",
    )
    captured = _patch_lead_agent_runtime(monkeypatch, app_config)

    config = {
        "configurable": {
            "runtime_profile": "ultra",
            "model_name": "claude-sonnet-4-6",
        }
    }

    lead_agent_module.make_lead_agent(config)

    assert captured["name"] == "api-pool-default"
    assert config["configurable"]["engine"] == "api_pool"
    assert config["configurable"]["model_name"] == "api-pool-default"
    assert config["configurable"]["model"] == "api-pool-default"


def test_make_lead_agent_unknown_runtime_profile_falls_back_to_request_flags(monkeypatch, caplog):
    app_config = _make_app_config([_make_model("gpt-5.4", supports_thinking=True)])
    captured = _patch_lead_agent_runtime(monkeypatch, app_config)

    config = {
        "configurable": {
            "runtime_profile": "mystery",
            "thinking_enabled": False,
            "subagent_enabled": True,
            "reasoning_effort": "low",
        }
    }

    with caplog.at_level("WARNING"):
        lead_agent_module.make_lead_agent(config)

    assert "Unknown runtime profile 'mystery'" in caplog.text
    assert captured["thinking_enabled"] is False
    assert captured["reasoning_effort"] == "low"
    assert config["configurable"]["runtime_profile"] == "mystery"
    assert config["configurable"]["subagent_enabled"] is True


def test_make_lead_agent_disables_thinking_when_model_does_not_support_it(monkeypatch):
    app_config = _make_app_config([_make_model("safe-model", supports_thinking=False)])

    import deerflow.tools as tools_module

    monkeypatch.setattr(lead_agent_module, "get_app_config", lambda: app_config)
    monkeypatch.setattr(tools_module, "get_available_tools", lambda **kwargs: [])
    monkeypatch.setattr(lead_agent_module, "_build_middlewares", lambda config, model_name, agent_name=None: [])

    captured: dict[str, object] = {}

    def _fake_create_chat_model(*, name, thinking_enabled, reasoning_effort=None):
        captured["name"] = name
        captured["thinking_enabled"] = thinking_enabled
        captured["reasoning_effort"] = reasoning_effort
        return object()

    monkeypatch.setattr(lead_agent_module, "create_chat_model", _fake_create_chat_model)
    monkeypatch.setattr(lead_agent_module, "create_agent", lambda **kwargs: kwargs)

    result = lead_agent_module.make_lead_agent(
        {
            "configurable": {
                "model_name": "safe-model",
                "thinking_enabled": True,
                "is_plan_mode": False,
                "subagent_enabled": False,
            }
        }
    )

    assert captured["name"] == "safe-model"
    assert captured["thinking_enabled"] is False
    assert result["model"] is not None


def test_make_lead_agent_prefers_pinned_custom_agent_model_over_request_override(monkeypatch):
    app_config = _make_app_config(
        [
            _make_model("gpt-5.4", supports_thinking=True),
            _make_model("claude-sonnet-4-6", supports_thinking=True),
        ]
    )

    import deerflow.tools as tools_module

    monkeypatch.setattr(lead_agent_module, "get_app_config", lambda: app_config)
    monkeypatch.setattr(
        lead_agent_module,
        "load_agent_config",
        lambda name: type(
            "AgentConfigStub",
            (),
            {"model": "gpt-5.4", "tool_groups": ["bash"], "name": name, "skills": None},
        )(),
    )
    monkeypatch.setattr(lead_agent_module, "record_agent_usage", lambda *args, **kwargs: None)
    monkeypatch.setattr(tools_module, "get_available_tools", lambda **kwargs: [])
    monkeypatch.setattr(lead_agent_module, "_build_middlewares", lambda config, model_name, agent_name=None: [])

    captured: dict[str, object] = {}

    def _fake_create_chat_model(*, name, thinking_enabled, reasoning_effort=None):
        captured["name"] = name
        captured["thinking_enabled"] = thinking_enabled
        captured["reasoning_effort"] = reasoning_effort
        return object()

    monkeypatch.setattr(lead_agent_module, "create_chat_model", _fake_create_chat_model)
    monkeypatch.setattr(lead_agent_module, "create_agent", lambda **kwargs: kwargs)

    result = lead_agent_module.make_lead_agent(
        {
            "configurable": {
                "agent_name": "omniai-chief-architect",
                "model_name": "claude-sonnet-4-6",
                "thinking_enabled": True,
                "is_plan_mode": False,
                "subagent_enabled": False,
            }
        }
    )

    # UI-selected model now wins over agent pinned model so the user's picker
    # is always authoritative. Pinned model is only the fallback when the UI
    # does not send model_name.
    assert captured["name"] == "claude-sonnet-4-6"
    assert result["model"] is not None


def test_make_lead_agent_falls_back_to_pinned_when_ui_omits_model_name(monkeypatch):
    """When UI sends no explicit model_name, the pinned model of the custom agent is used."""
    app_config = _make_app_config(
        [
            _make_model("gpt-5.4", supports_thinking=True),
            _make_model("claude-sonnet-4-6", supports_thinking=True),
        ]
    )

    import deerflow.tools as tools_module

    monkeypatch.setattr(lead_agent_module, "get_app_config", lambda: app_config)
    monkeypatch.setattr(
        lead_agent_module,
        "load_agent_config",
        lambda name: type(
            "AgentConfigStub",
            (),
            {"model": "gpt-5.4", "tool_groups": ["bash"], "name": name, "skills": None},
        )(),
    )
    monkeypatch.setattr(lead_agent_module, "record_agent_usage", lambda *args, **kwargs: None)
    monkeypatch.setattr(tools_module, "get_available_tools", lambda **kwargs: [])
    monkeypatch.setattr(lead_agent_module, "_build_middlewares", lambda config, model_name, agent_name=None: [])

    captured: dict[str, object] = {}

    def _fake_create_chat_model(*, name, thinking_enabled, reasoning_effort=None):
        captured["name"] = name
        return object()

    monkeypatch.setattr(lead_agent_module, "create_chat_model", _fake_create_chat_model)
    monkeypatch.setattr(lead_agent_module, "create_agent", lambda **kwargs: kwargs)

    result = lead_agent_module.make_lead_agent(
        {
            "configurable": {
                "agent_name": "omniai-chief-architect",
                # No model_name sent — UI did not override.
                "thinking_enabled": True,
                "is_plan_mode": False,
                "subagent_enabled": False,
            }
        }
    )

    # Without a UI override, the agent's pinned model is still honored.
    assert captured["name"] == "gpt-5.4"
    assert result["model"] is not None


def test_make_lead_agent_uses_engine_specific_default_for_unpinned_agents(monkeypatch):
    app_config = _make_app_config(
        [
            _make_model("gpt-5.4", supports_thinking=True, use=CODEX_PROVIDER_PATH),
            _make_model("claude-sonnet-4-6", supports_thinking=True, use=CLAUDE_PROVIDER_PATH),
            _make_model("api-pool-default", supports_thinking=False, use=STANDARD_API_PROVIDER_PATH),
        ]
    )

    import deerflow.tools as tools_module

    monkeypatch.setattr(lead_agent_module, "get_app_config", lambda: app_config)
    monkeypatch.setattr(lead_agent_module, "load_agent_config", lambda name: None)
    monkeypatch.setattr(tools_module, "get_available_tools", lambda **kwargs: [])
    monkeypatch.setattr(lead_agent_module, "_build_middlewares", lambda config, model_name, agent_name=None: [])

    captured: dict[str, object] = {}

    def _fake_create_chat_model(*, name, thinking_enabled, reasoning_effort=None):
        captured["name"] = name
        captured["thinking_enabled"] = thinking_enabled
        captured["reasoning_effort"] = reasoning_effort
        return object()

    monkeypatch.setattr(lead_agent_module, "create_chat_model", _fake_create_chat_model)
    monkeypatch.setattr(lead_agent_module, "create_agent", lambda **kwargs: kwargs)

    result = lead_agent_module.make_lead_agent(
        {
            "configurable": {
                "engine": "api_pool",
                "model_name": "claude-sonnet-4-6",
                "thinking_enabled": True,
                "is_plan_mode": False,
                "subagent_enabled": False,
            }
        }
    )

    assert captured["name"] == "api-pool-default"
    assert result["model"] is not None


def test_make_lead_agent_keeps_thinking_enabled_for_runtime_gpt5_api_pool(monkeypatch):
    app_config = _make_app_config(
        [
            _make_model("api-pool-default", supports_thinking=False, use=STANDARD_API_PROVIDER_PATH),
        ]
    )

    import deerflow.tools as tools_module

    monkeypatch.setattr(lead_agent_module, "get_app_config", lambda: app_config)
    monkeypatch.setattr(lead_agent_module, "load_agent_config", lambda name: None)
    monkeypatch.setattr(tools_module, "get_available_tools", lambda **kwargs: [])
    monkeypatch.setattr(lead_agent_module, "_build_middlewares", lambda config, model_name, agent_name=None: [])

    captured: dict[str, object] = {}

    def _fake_create_chat_model(*, name, thinking_enabled, reasoning_effort=None):
        captured["name"] = name
        captured["thinking_enabled"] = thinking_enabled
        captured["reasoning_effort"] = reasoning_effort
        return object()

    monkeypatch.setenv("API_POOL_MODEL", "gpt-5.4")
    monkeypatch.setattr(lead_agent_module, "create_chat_model", _fake_create_chat_model)
    monkeypatch.setattr(lead_agent_module, "create_agent", lambda **kwargs: kwargs)

    result = lead_agent_module.make_lead_agent(
        {
            "configurable": {
                "engine": "api_pool",
                "model_name": "api-pool-default",
                "thinking_enabled": True,
                "is_plan_mode": False,
                "subagent_enabled": False,
            }
        }
    )

    assert captured["name"] == "api-pool-default"
    assert captured["thinking_enabled"] is True
    assert result["model"] is not None


def _make_api_pool_ultra_config():
    return {
        "configurable": {
            "engine": "api_pool",
            "model_name": "api-pool-default",
            "thinking_enabled": True,
            "is_plan_mode": False,
            "subagent_enabled": True,
            "max_concurrent_subagents": 3,
        }
    }


def _patch_ultra_monkeys(monkeypatch, app_config):
    import deerflow.tools as tools_module

    monkeypatch.setattr(lead_agent_module, "get_app_config", lambda: app_config)
    monkeypatch.setattr(lead_agent_module, "load_agent_config", lambda name: None)
    monkeypatch.setattr(tools_module, "get_available_tools", lambda **kwargs: [])
    monkeypatch.setattr(lead_agent_module, "_build_middlewares", lambda config, model_name, agent_name=None: [])
    monkeypatch.setattr(lead_agent_module, "apply_prompt_template", lambda **kwargs: kwargs)

    def _fake_create_chat_model(*, name, thinking_enabled, reasoning_effort=None):
        return object()

    monkeypatch.setattr(lead_agent_module, "create_chat_model", _fake_create_chat_model)
    monkeypatch.setattr(lead_agent_module, "create_agent", lambda **kwargs: kwargs)


def test_make_lead_agent_forces_single_subagent_for_api_pool_ultra_when_failover_off(monkeypatch):
    """Without failover, api-pool engine still forces single-concurrency."""
    monkeypatch.delenv("DEER_FLOW_API_POOL_FAILOVER_ENABLED", raising=False)
    monkeypatch.delenv("DEER_FLOW_API_POOL_ALLOW_PARALLEL_SUBAGENTS", raising=False)

    app_config = _make_app_config(
        [
            _make_model("gpt-5.4", supports_thinking=True, use=CODEX_PROVIDER_PATH),
            _make_model("api-pool-default", supports_thinking=True, use=STANDARD_API_PROVIDER_PATH),
        ]
    )
    _patch_ultra_monkeys(monkeypatch, app_config)
    config = _make_api_pool_ultra_config()

    result = lead_agent_module.make_lead_agent(config)

    assert config["configurable"]["max_concurrent_subagents"] == 1
    assert result["system_prompt"]["max_concurrent_subagents"] == 1


def test_make_lead_agent_allows_parallel_subagents_when_failover_enabled(monkeypatch):
    """With failover on, api-pool agents honor the requested concurrency."""
    monkeypatch.setenv("DEER_FLOW_API_POOL_FAILOVER_ENABLED", "1")
    monkeypatch.delenv("DEER_FLOW_API_POOL_ALLOW_PARALLEL_SUBAGENTS", raising=False)

    app_config = _make_app_config(
        [
            _make_model("gpt-5.4", supports_thinking=True, use=CODEX_PROVIDER_PATH),
            _make_model("api-pool-default", supports_thinking=True, use=STANDARD_API_PROVIDER_PATH),
        ]
    )
    _patch_ultra_monkeys(monkeypatch, app_config)
    config = _make_api_pool_ultra_config()

    result = lead_agent_module.make_lead_agent(config)

    assert config["configurable"]["max_concurrent_subagents"] == 3
    assert result["system_prompt"]["max_concurrent_subagents"] == 3


def test_make_lead_agent_explicit_parallel_opt_in_overrides_failover_setting(monkeypatch):
    """Explicit opt-in/opt-out env wins over the failover default."""
    monkeypatch.delenv("DEER_FLOW_API_POOL_FAILOVER_ENABLED", raising=False)
    monkeypatch.setenv("DEER_FLOW_API_POOL_ALLOW_PARALLEL_SUBAGENTS", "1")

    app_config = _make_app_config(
        [
            _make_model("gpt-5.4", supports_thinking=True, use=CODEX_PROVIDER_PATH),
            _make_model("api-pool-default", supports_thinking=True, use=STANDARD_API_PROVIDER_PATH),
        ]
    )
    _patch_ultra_monkeys(monkeypatch, app_config)
    config = _make_api_pool_ultra_config()

    result = lead_agent_module.make_lead_agent(config)

    assert config["configurable"]["max_concurrent_subagents"] == 3
    assert result["system_prompt"]["max_concurrent_subagents"] == 3


def test_make_lead_agent_rejects_invalid_bootstrap_agent_name(monkeypatch):
    app_config = _make_app_config([_make_model("safe-model", supports_thinking=False)])

    monkeypatch.setattr(lead_agent_module, "get_app_config", lambda: app_config)

    with pytest.raises(ValueError, match="Invalid agent name"):
        lead_agent_module.make_lead_agent(
            {
                "configurable": {
                    "model_name": "safe-model",
                    "thinking_enabled": False,
                    "is_plan_mode": False,
                    "subagent_enabled": False,
                    "is_bootstrap": True,
                    "agent_name": "../../../tmp/evil",
                }
            }
        )


def test_build_middlewares_uses_resolved_model_name_for_vision(monkeypatch):
    app_config = _make_app_config(
        [
            _make_model("stale-model", supports_thinking=False),
            ModelConfig(
                name="vision-model",
                display_name="vision-model",
                description=None,
                use="langchain_openai:ChatOpenAI",
                model="vision-model",
                supports_thinking=False,
                supports_vision=True,
            ),
        ]
    )

    monkeypatch.setattr(lead_agent_module, "get_app_config", lambda: app_config)
    monkeypatch.setattr(lead_agent_module, "_create_summarization_middleware", lambda: None)
    monkeypatch.setattr(lead_agent_module, "_create_todo_list_middleware", lambda is_plan_mode: None)

    middlewares = lead_agent_module._build_middlewares({"configurable": {"model_name": "stale-model", "is_plan_mode": False, "subagent_enabled": False}}, model_name="vision-model", custom_middlewares=[MagicMock()])

    assert any(isinstance(m, lead_agent_module.ViewImageMiddleware) for m in middlewares)
    # verify the custom middleware is injected correctly
    assert len(middlewares) > 0 and isinstance(middlewares[-2], MagicMock)


def test_create_summarization_middleware_uses_configured_model_alias(monkeypatch):
    monkeypatch.setattr(
        lead_agent_module,
        "get_summarization_config",
        lambda: SummarizationConfig(enabled=True, model_name="model-masswork"),
    )
    monkeypatch.setattr(lead_agent_module, "get_memory_config", lambda: MemoryConfig(enabled=False))

    captured: dict[str, object] = {}
    fake_model = object()

    def _fake_create_chat_model(*, name=None, thinking_enabled, reasoning_effort=None):
        captured["name"] = name
        captured["thinking_enabled"] = thinking_enabled
        captured["reasoning_effort"] = reasoning_effort
        return fake_model

    monkeypatch.setattr(lead_agent_module, "create_chat_model", _fake_create_chat_model)
    monkeypatch.setattr(lead_agent_module, "DeerFlowSummarizationMiddleware", lambda **kwargs: kwargs)

    middleware = lead_agent_module._create_summarization_middleware()

    assert captured["name"] == "model-masswork"
    assert captured["thinking_enabled"] is False
    assert middleware["model"] is fake_model


def test_create_summarization_middleware_registers_memory_flush_hook_when_memory_enabled(monkeypatch):
    monkeypatch.setattr(
        lead_agent_module,
        "get_summarization_config",
        lambda: SummarizationConfig(enabled=True),
    )
    monkeypatch.setattr(lead_agent_module, "get_memory_config", lambda: MemoryConfig(enabled=True))
    monkeypatch.setattr(lead_agent_module, "create_chat_model", lambda **kwargs: object())

    captured: dict[str, object] = {}

    def _fake_middleware(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(lead_agent_module, "DeerFlowSummarizationMiddleware", _fake_middleware)

    lead_agent_module._create_summarization_middleware()

    assert captured["before_summarization"] == [lead_agent_module.memory_flush_hook]

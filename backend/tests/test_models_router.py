import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.gateway.routers import models as models_router
from deerflow.models.credential_loader import CodexCliCredential
from deerflow.models.engines import STANDARD_API_PROVIDER_PATH


def test_list_models_enables_thinking_for_api_pool_gpt5(monkeypatch):
    model = SimpleNamespace(
        name="api-pool-default",
        model="api-pool-default",
        display_name="API Pool",
        description="OpenAI-compatible pool",
        use=STANDARD_API_PROVIDER_PATH,
        supports_thinking=False,
        supports_reasoning_effort=False,
    )
    app_config = MagicMock()
    app_config.models = [model]

    monkeypatch.setenv("API_POOL_MODEL", "gpt-5.4")
    monkeypatch.setattr(models_router, "get_app_config", lambda: app_config)

    result = asyncio.run(models_router.list_models())

    assert len(result.models) == 1
    parsed = result.models[0]
    assert parsed.model == "gpt-5.4"
    assert parsed.display_name == "API Pool · gpt-5.4"
    assert parsed.supports_thinking is True
    assert parsed.supports_reasoning_effort is True


def test_get_model_preserves_non_reasoning_api_pool_capabilities(monkeypatch):
    model = SimpleNamespace(
        name="api-pool-default",
        model="api-pool-default",
        display_name="API Pool",
        description="OpenAI-compatible pool",
        use=STANDARD_API_PROVIDER_PATH,
        supports_thinking=False,
        supports_reasoning_effort=False,
    )
    app_config = MagicMock()
    app_config.get_model_config.return_value = model

    monkeypatch.setenv("API_POOL_MODEL", "deepseek-chat")
    monkeypatch.setattr(models_router, "get_app_config", lambda: app_config)

    parsed = asyncio.run(models_router.get_model("api-pool-default"))

    assert parsed.model == "deepseek-chat"
    assert parsed.supports_thinking is False
    assert parsed.supports_reasoning_effort is False


def test_list_models_surfaces_current_codex_account(monkeypatch):
    model = SimpleNamespace(
        name="gpt-5.4",
        model="gpt-5.4",
        display_name="GPT-5.4 (Codex CLI)",
        description="Codex CLI provider",
        use="deerflow.models.openai_codex_provider:CodexChatModel",
        supports_thinking=True,
        supports_reasoning_effort=True,
    )
    app_config = MagicMock()
    app_config.models = [model]

    monkeypatch.setattr(models_router, "get_app_config", lambda: app_config)
    monkeypatch.setattr(
        models_router,
        "load_codex_cli_credential",
        lambda **_: CodexCliCredential(access_token="token", account_id="90cf490f1234"),
    )

    result = asyncio.run(models_router.list_models())

    assert len(result.models) == 1
    parsed = result.models[0]
    assert parsed.credential_label == "账号 90cf490f..."
    assert parsed.description is not None
    assert "当前 Codex 账号: 90cf490f..." in parsed.description

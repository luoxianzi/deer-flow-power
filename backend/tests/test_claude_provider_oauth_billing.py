"""Tests for ClaudeChatModel._apply_oauth_billing."""

import asyncio
import json
from unittest import mock

import pytest

from deerflow.models.claude_provider import OAUTH_BILLING_HEADER, ClaudeChatModel


def _make_model() -> ClaudeChatModel:
    """Return a minimal ClaudeChatModel instance in OAuth mode without network calls."""
    import unittest.mock as mock

    with mock.patch.object(ClaudeChatModel, "model_post_init"):
        m = ClaudeChatModel(model="claude-sonnet-4-6", anthropic_api_key="sk-ant-oat-fake-token")  # type: ignore[call-arg]
    m._is_oauth = True
    m._oauth_access_token = "sk-ant-oat-fake-token"
    return m


@pytest.fixture()
def model() -> ClaudeChatModel:
    return _make_model()


def _billing_block() -> dict:
    return {"type": "text", "text": OAUTH_BILLING_HEADER}


# ---------------------------------------------------------------------------
# Billing block injection
# ---------------------------------------------------------------------------


def test_billing_injected_first_when_no_system(model):
    payload: dict = {}
    model._apply_oauth_billing(payload)
    assert payload["system"][0] == _billing_block()


def test_billing_injected_first_into_list(model):
    payload = {"system": [{"type": "text", "text": "You are a helpful assistant."}]}
    model._apply_oauth_billing(payload)
    assert payload["system"][0] == _billing_block()
    assert payload["system"][1]["text"] == "You are a helpful assistant."


def test_billing_injected_first_into_string_system(model):
    payload = {"system": "You are helpful."}
    model._apply_oauth_billing(payload)
    assert payload["system"][0] == _billing_block()
    assert payload["system"][1]["text"] == "You are helpful."


def test_billing_not_duplicated_on_second_call(model):
    payload = {"system": [{"type": "text", "text": "prompt"}]}
    model._apply_oauth_billing(payload)
    model._apply_oauth_billing(payload)
    billing_count = sum(1 for b in payload["system"] if isinstance(b, dict) and OAUTH_BILLING_HEADER in b.get("text", ""))
    assert billing_count == 1


def test_billing_moved_to_first_if_not_already_first(model):
    """Billing block already present but not first — must be normalized to index 0."""
    payload = {
        "system": [
            {"type": "text", "text": "other block"},
            _billing_block(),
        ]
    }
    model._apply_oauth_billing(payload)
    assert payload["system"][0] == _billing_block()
    assert len([b for b in payload["system"] if OAUTH_BILLING_HEADER in b.get("text", "")]) == 1


def test_billing_string_with_header_collapsed_to_single_block(model):
    """If system is a string that already contains the billing header, collapse to one block."""
    payload = {"system": OAUTH_BILLING_HEADER}
    model._apply_oauth_billing(payload)
    assert payload["system"] == [_billing_block()]


# ---------------------------------------------------------------------------
# metadata.user_id
# ---------------------------------------------------------------------------


def test_metadata_user_id_added_when_missing(model):
    payload: dict = {}
    model._apply_oauth_billing(payload)
    assert "metadata" in payload
    user_id = json.loads(payload["metadata"]["user_id"])
    assert "device_id" in user_id
    assert "session_id" in user_id
    assert user_id["account_uuid"] == "deerflow"


def test_metadata_user_id_not_overwritten_if_present(model):
    payload = {"metadata": {"user_id": "existing-value"}}
    model._apply_oauth_billing(payload)
    assert payload["metadata"]["user_id"] == "existing-value"


def test_metadata_non_dict_replaced_with_dict(model):
    """Non-dict metadata (e.g. None or a string) should be replaced, not crash."""
    for bad_value in (None, "string-metadata", 42):
        payload = {"metadata": bad_value}
        model._apply_oauth_billing(payload)
        assert isinstance(payload["metadata"], dict)
        assert "user_id" in payload["metadata"]


def test_sync_create_strips_cache_control_from_oauth_payload(model):
    payload = {
        "system": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}],
            }
        ],
        "tools": [{"name": "demo", "input_schema": {"type": "object"}, "cache_control": {"type": "ephemeral"}}],
    }

    with mock.patch.object(model._client.messages, "create", return_value=object()) as create:
        model._create(payload)

    sent_payload = create.call_args.kwargs
    assert "cache_control" not in sent_payload["system"][0]
    assert "cache_control" not in sent_payload["messages"][0]["content"][0]
    assert "cache_control" not in sent_payload["tools"][0]


def test_async_create_strips_cache_control_from_oauth_payload(model):
    payload = {
        "system": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}],
            }
        ],
        "tools": [{"name": "demo", "input_schema": {"type": "object"}, "cache_control": {"type": "ephemeral"}}],
    }

    with mock.patch.object(model._async_client.messages, "create", new=mock.AsyncMock(return_value=object())) as create:
        asyncio.run(model._acreate(payload))

    sent_payload = create.call_args.kwargs
    assert "cache_control" not in sent_payload["system"][0]
    assert "cache_control" not in sent_payload["messages"][0]["content"][0]
    assert "cache_control" not in sent_payload["tools"][0]


# ---------------------------------------------------------------------------
# Opus 4.7 contract: strip sampling params + convert thinking shape
# ---------------------------------------------------------------------------


def _make_opus_47_model() -> ClaudeChatModel:
    """Opus 4.7 stub that simulates what model_post_init would infer from the ID."""
    with mock.patch.object(ClaudeChatModel, "model_post_init"):
        m = ClaudeChatModel(model="claude-opus-4-7", anthropic_api_key="sk-ant-oat-fake-token")  # type: ignore[call-arg]
    m.use_adaptive_thinking = True
    m.strip_sampling_params = True
    m._is_oauth = False
    return m


def test_opus_47_strips_forbidden_sampling_params():
    m = _make_opus_47_model()
    payload = {"model": "claude-opus-4-7", "temperature": 0.5, "top_p": 0.9, "top_k": 42, "max_tokens": 1024}
    m._strip_forbidden_sampling_params(payload)
    for key in ("temperature", "top_p", "top_k"):
        assert key not in payload
    assert payload["max_tokens"] == 1024


def test_opus_47_converts_extended_thinking_to_adaptive():
    m = _make_opus_47_model()
    payload = {"thinking": {"type": "enabled", "budget_tokens": 8000}}
    m._convert_thinking_to_adaptive(payload)
    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}


def test_opus_47_leaves_disabled_thinking_untouched():
    m = _make_opus_47_model()
    payload = {"thinking": {"type": "disabled"}}
    m._convert_thinking_to_adaptive(payload)
    assert payload["thinking"] == {"type": "disabled"}


def test_opus_47_preserves_explicit_thinking_display():
    m = _make_opus_47_model()
    payload = {"thinking": {"type": "enabled", "budget_tokens": 4000, "display": "omitted"}}
    m._convert_thinking_to_adaptive(payload)
    assert payload["thinking"] == {"type": "adaptive", "display": "omitted"}


def test_opus_47_auto_thinking_budget_skips_adaptive():
    """`_apply_thinking_budget` must not inject budget_tokens into adaptive payloads."""
    m = _make_opus_47_model()
    payload = {"thinking": {"type": "adaptive", "display": "summarized"}, "max_tokens": 10000}
    m._apply_thinking_budget(payload)
    assert "budget_tokens" not in payload["thinking"]


def test_sonnet_46_does_not_enable_adaptive_thinking_by_default():
    """Sonnet 4.6 still supports extended thinking — the auto-infer flag stays off."""
    m = _make_model()  # claude-sonnet-4-6 under OAuth
    assert m.use_adaptive_thinking is False
    assert m.strip_sampling_params is False


# ---------------------------------------------------------------------------
# Opus 4.7 output_config.effort wiring (task-budgets beta)
# ---------------------------------------------------------------------------


def test_opus_47_emits_output_config_effort_for_known_level():
    m = _make_opus_47_model()
    m._enable_output_config = True
    m.reasoning_effort = "xhigh"
    payload: dict = {}
    m._apply_output_config(payload)
    assert payload["output_config"] == {"effort": "xhigh"}


def test_opus_47_coerces_frontend_minimal_to_low():
    m = _make_opus_47_model()
    m._enable_output_config = True
    m.reasoning_effort = "minimal"
    payload: dict = {}
    m._apply_output_config(payload)
    assert payload["output_config"] == {"effort": "low"}


def test_opus_47_none_reasoning_effort_omits_output_config():
    m = _make_opus_47_model()
    m._enable_output_config = True
    m.reasoning_effort = "none"
    payload: dict = {}
    m._apply_output_config(payload)
    assert "output_config" not in payload


def test_opus_47_unknown_reasoning_effort_is_dropped():
    m = _make_opus_47_model()
    m._enable_output_config = True
    m.reasoning_effort = "garbage-level"
    payload: dict = {}
    m._apply_output_config(payload)
    assert "output_config" not in payload


def test_opus_47_output_config_preserves_existing_keys():
    """Future callers may set task_budget before effort — don't clobber siblings."""
    m = _make_opus_47_model()
    m._enable_output_config = True
    m.reasoning_effort = "high"
    payload: dict = {"output_config": {"task_budget": {"type": "tokens", "total": 50000}}}
    m._apply_output_config(payload)
    assert payload["output_config"] == {
        "task_budget": {"type": "tokens", "total": 50000},
        "effort": "high",
    }

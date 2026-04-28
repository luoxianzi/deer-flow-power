"""OpenAI-compatible API pool provider backed by a rotating API key pool."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import PrivateAttr

from deerflow.models.engines import normalize_runtime_reasoning_effort

logger = logging.getLogger(__name__)

KEY_ROTATION_STATUS_CODES = {401, 403, 429}
# 409: some gateways enforce per-account concurrency; long threads + Ultra/subtasks can overlap calls.
TRANSIENT_STATUS_CODES = {408, 409, 502, 503, 504, 524}
DEFAULT_API_POOL_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_TRANSIENT_RETRIES = 1
DEFAULT_TRANSIENT_RETRY_DELAY_SECONDS = 2.0
DEFAULT_MAX_REQUEST_BODY_CHARS = 120_000
_THINK_TAG_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
_CONTEXT_COMPACTION_TARGETS = (80, 40, 20, 10, 6, 4, 2, 1)
_TRANSIENT_RECOVERY_TARGETS = (4, 2, 1)
_SYNTHETIC_PROVIDER_ERROR_PREFIXES = (
    "当前模型供应商暂时不可用",
    "当前配置的模型供应商",
    "当前 API Pool 网关拒绝了这条网络出口",
    "LLM request failed:",
)


class StandardAPIChatModel(BaseChatModel):
    """LangChain chat model for a standard OpenAI-compatible chat completions endpoint.

    Required environment variables:
      - API_POOL_BASE_URL
      - API_POOL_KEYS

    Optional environment variables:
      - API_POOL_MODEL
    """

    model: str = "api-pool-default"
    temperature: float | None = None
    timeout_seconds: float = DEFAULT_API_POOL_TIMEOUT_SECONDS
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    extra_body: dict[str, Any] | None = None
    # Per-model credentials: when set, override the global API_POOL_* env vars.
    # Useful for registering multiple named models with different providers in config.yaml.
    api_key: str | None = None
    base_url: str | None = None

    _base_urls: list[str] = PrivateAttr(default_factory=list)
    _current_base_url_index: int = PrivateAttr(default=0)
    _api_keys_per_gateway: list[list[str]] = PrivateAttr(default_factory=list)
    _current_key_indices: list[int] = PrivateAttr(default_factory=list)
    _max_transient_retries: int = PrivateAttr(default=DEFAULT_MAX_TRANSIENT_RETRIES)
    _transient_retry_delay_seconds: float = PrivateAttr(default=DEFAULT_TRANSIENT_RETRY_DELAY_SECONDS)
    # 409: gateways often mean "account concurrency full" — needs more patience, not faster retries.
    _transient_409_retry_delay_multiplier: float = PrivateAttr(default=3.0)
    _max_request_body_chars: int = PrivateAttr(default=DEFAULT_MAX_REQUEST_BODY_CHARS)
    _proxy_url: str | None = PrivateAttr(default=None)
    _wire_api: str = PrivateAttr(default="chat_completions")

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "standard-api-chat-completions"

    def model_post_init(self, __context: Any) -> None:
        # Per-model credentials from config.yaml take priority over global env vars.
        # api_key supports comma-separated multiple keys for round-robin rotation.
        if self.base_url and self.api_key:
            base_urls = [self.base_url.rstrip("/")]
            keys = [k.strip() for k in self.api_key.split(",") if k.strip()]
            key_groups = [keys]
        else:
            raw_base_urls = os.getenv("API_POOL_BASE_URLS", "").strip()
            base_url = os.getenv("API_POOL_BASE_URL", "").strip().rstrip("/")
            raw_keys = os.getenv("API_POOL_KEYS", "").strip()

            if raw_base_urls:
                base_urls = [url.strip().rstrip("/") for url in raw_base_urls.split(",") if url.strip()]
            elif base_url:
                base_urls = [base_url]
            else:
                base_urls = []

            if "|" in raw_keys:
                key_groups = [
                    [k.strip() for k in group.split(",") if k.strip()]
                    for group in raw_keys.split("|")
                ]
                key_groups = [g for g in key_groups if g]
            else:
                flat_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
                key_groups = [flat_keys] if flat_keys else []

        model_override = os.getenv("API_POOL_MODEL", "").strip()
        timeout_override = os.getenv("API_POOL_TIMEOUT_SECONDS", "").strip()
        transient_retry_override = os.getenv("API_POOL_MAX_TRANSIENT_RETRIES", "").strip()
        transient_retry_delay_override = os.getenv("API_POOL_TRANSIENT_RETRY_DELAY_SECONDS", "").strip()
        delay_409_mult_override = os.getenv("API_POOL_409_RETRY_DELAY_MULTIPLIER", "").strip()
        request_body_chars_override = os.getenv("API_POOL_MAX_REQUEST_BODY_CHARS", "").strip()
        proxy_override = os.getenv("API_POOL_PROXY_URL", "").strip()
        wire_api_override = os.getenv("API_POOL_WIRE_API", "").strip().lower()

        if not base_urls or not key_groups:
            raise ValueError(
                "API pool credentials unavailable: set API_POOL_BASE_URL or API_POOL_BASE_URLS, plus API_POOL_KEYS, before using the api_pool engine."
            )

        if len(key_groups) == 1:
            api_keys_per_gateway = [key_groups[0] for _ in base_urls]
        elif len(key_groups) == len(base_urls):
            api_keys_per_gateway = key_groups
        else:
            raise ValueError(
                f"API_POOL_KEYS has {len(key_groups)} pipe-delimited groups but "
                f"API_POOL_BASE_URLS has {len(base_urls)} gateways. "
                "They must match, or provide a single key group (no pipe) to share across all gateways."
            )

        if timeout_override:
            self.timeout_seconds = float(timeout_override)
        if transient_retry_override:
            self._max_transient_retries = max(0, int(transient_retry_override))
        if transient_retry_delay_override:
            self._transient_retry_delay_seconds = max(0.0, float(transient_retry_delay_override))
        if delay_409_mult_override:
            self._transient_409_retry_delay_multiplier = max(1.0, float(delay_409_mult_override))
        if request_body_chars_override:
            self._max_request_body_chars = max(10_000, int(request_body_chars_override))

        self._base_urls = base_urls
        self._current_base_url_index = 0
        self._api_keys_per_gateway = api_keys_per_gateway
        self._current_key_indices = [0] * len(base_urls)
        self._proxy_url = proxy_override or None
        self._wire_api = wire_api_override or "chat_completions"
        if self._wire_api not in {"chat_completions", "responses"}:
            raise ValueError("API_POOL_WIRE_API must be either 'chat_completions' or 'responses'.")
        # Only apply global API_POOL_MODEL override when NOT using per-model credentials,
        # to prevent the global pool model from overwriting a specific model's name.
        if model_override and not (self.base_url and self.api_key):
            self.model = model_override

        logger.info(
            "Using Standard API pool provider (base_urls=%s, model=%s, wire_api=%s, keys_per_gateway=%s, timeout_seconds=%s, max_transient_retries=%s, transient_retry_delay_seconds=%s, max_request_body_chars=%s, proxy=%s)",
            self._base_urls,
            self.model,
            self._wire_api,
            [len(g) for g in self._api_keys_per_gateway],
            self.timeout_seconds,
            self._max_transient_retries,
            self._transient_retry_delay_seconds,
            self._max_request_body_chars,
            self._proxy_url or "<env>",
        )
        super().model_post_init(__context)

    def _create_http_client(self) -> httpx.AsyncClient:
        client_kwargs: dict[str, Any] = {"timeout": self.timeout_seconds}
        if self._proxy_url:
            client_kwargs["proxy"] = self._proxy_url
            client_kwargs["trust_env"] = False
        return httpx.AsyncClient(**client_kwargs)

    @classmethod
    def _normalize_content(cls, content: Any) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = [cls._normalize_content(item) for item in content]
            return "\n".join(part for part in parts if part)

        if isinstance(content, dict):
            for key in ("text", "output"):
                value = content.get(key)
                if isinstance(value, str):
                    return value
            nested_content = content.get("content")
            if nested_content is not None:
                return cls._normalize_content(nested_content)
            try:
                return json.dumps(content, ensure_ascii=False)
            except TypeError:
                return str(content)

        try:
            return json.dumps(content, ensure_ascii=False)
        except TypeError:
            return str(content)

    @classmethod
    def _extract_reasoning_text(
        cls,
        reasoning: Any,
        *,
        strip_parts: bool = True,
    ) -> str | None:
        if reasoning is None:
            return None

        if isinstance(reasoning, str):
            normalized = reasoning.strip() if strip_parts else reasoning
            return normalized if normalized.strip() else None

        if isinstance(reasoning, list):
            parts: list[str] = []
            for item in reasoning:
                extracted = cls._extract_reasoning_text(item, strip_parts=strip_parts)
                if extracted:
                    parts.append(extracted)
            return "\n\n".join(parts) if parts else None

        if isinstance(reasoning, Mapping):
            for key in ("text", "reasoning_content", "content", "summary"):
                if key in reasoning:
                    extracted = cls._extract_reasoning_text(reasoning.get(key), strip_parts=strip_parts)
                    if extracted:
                        return extracted
            return None

        return None

    @staticmethod
    def _strip_inline_think_tags(content: str) -> tuple[str, str | None]:
        reasoning_parts: list[str] = []

        def _replace(match: re.Match[str]) -> str:
            reasoning = match.group(1).strip()
            if reasoning:
                reasoning_parts.append(reasoning)
            return ""

        cleaned = _THINK_TAG_RE.sub(_replace, content).strip()
        reasoning = "\n\n".join(reasoning_parts) if reasoning_parts else None
        return cleaned, reasoning

    @staticmethod
    def _merge_reasoning(*values: str | None) -> str | None:
        merged: list[str] = []
        for value in values:
            if not value:
                continue
            normalized = value.strip()
            if normalized and normalized not in merged:
                merged.append(normalized)
        return "\n\n".join(merged) if merged else None

    @staticmethod
    def _with_reasoning_content(
        message: AIMessage | AIMessageChunk,
        reasoning: str | None,
        *,
        preserve_whitespace: bool = False,
    ) -> AIMessage | AIMessageChunk:
        if not reasoning:
            return message

        additional_kwargs = dict(message.additional_kwargs)
        if preserve_whitespace:
            existing = additional_kwargs.get("reasoning_content")
            additional_kwargs["reasoning_content"] = f"{existing}{reasoning}" if isinstance(existing, str) else reasoning
        else:
            additional_kwargs["reasoning_content"] = StandardAPIChatModel._merge_reasoning(
                additional_kwargs.get("reasoning_content"),
                reasoning,
            )
        return message.model_copy(update={"additional_kwargs": additional_kwargs})

    @staticmethod
    def _mask_key(api_key: str) -> str:
        if len(api_key) <= 10:
            return api_key
        return f"{api_key[:6]}...{api_key[-4:]}"

    @staticmethod
    def _is_null_content_response(response_json: dict) -> bool:
        """Detect a broken gateway response where content and tool_calls are both null.

        Some API gateways (e.g. load-balanced proxies with inconsistent backends)
        return HTTP 200 with finish_reason=stop but null content and null tool_calls.
        This is always invalid for a conversational model and should be retried.
        """
        try:
            choices = response_json.get("choices")
            if not choices:
                return False
            message = choices[0].get("message", {})
            content = message.get("content")
            tool_calls = message.get("tool_calls")
            finish_reason = choices[0].get("finish_reason")
            return (
                content is None
                and not tool_calls
                and finish_reason == "stop"
            )
        except Exception:
            return False

    def _current_api_key(self) -> str:
        gw = self._current_base_url_index
        return self._api_keys_per_gateway[gw][self._current_key_indices[gw]]

    def _rotate_to_next_key(self) -> None:
        gw = self._current_base_url_index
        keys = self._api_keys_per_gateway[gw]
        self._current_key_indices[gw] = (self._current_key_indices[gw] + 1) % len(keys)

    def _current_gateway_key_count(self) -> int:
        return len(self._api_keys_per_gateway[self._current_base_url_index])

    def _current_base_url(self) -> str:
        return self._base_urls[self._current_base_url_index]

    def _rotate_to_next_base_url(self) -> None:
        self._current_base_url_index = (self._current_base_url_index + 1) % len(self._base_urls)

    def _reset_gateway_rotation(self) -> None:
        self._current_base_url_index = 0

    def _max_transient_retries_for_status(self, status: int | None) -> int:
        """Extra patience for HTTP 409 (per-account concurrency) without changing other statuses."""
        if status != 409:
            return self._max_transient_retries
        explicit = os.getenv("API_POOL_MAX_409_RETRIES", "").strip()
        if explicit:
            return max(0, int(explicit))
        return max(self._max_transient_retries, 10)

    def _current_transient_retry_delay_seconds(self, attempt: int, *, status: int | None = None) -> float:
        if self._transient_retry_delay_seconds <= 0:
            return 0.0
        base = self._transient_retry_delay_seconds * (2 ** max(0, attempt - 1))
        if status == 409:
            base *= self._transient_409_retry_delay_multiplier
        return base

    def _build_headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _uses_responses_api(self) -> bool:
        return self._wire_api == "responses"

    @classmethod
    def _is_synthetic_provider_error_text(cls, content: str) -> bool:
        normalized = " ".join(content.split())
        return any(normalized.startswith(prefix) for prefix in _SYNTHETIC_PROVIDER_ERROR_PREFIXES)

    @classmethod
    def _should_skip_message_for_upstream_context(cls, message: BaseMessage) -> bool:
        if not isinstance(message, AIMessage):
            return False
        if message.additional_kwargs.get("llm_error"):
            return True
        content = cls._normalize_content(message.content)
        if not content:
            return False
        return cls._is_synthetic_provider_error_text(content)

    def _filter_messages_for_upstream_context(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        filtered = [msg for msg in messages if not self._should_skip_message_for_upstream_context(msg)]
        return filtered or messages

    def _sanitize_compacted_messages_for_tool_protocol(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Fix message lists after tail-slice compaction.

        OpenAI-compatible APIs reject requests where a ``tool``/function_call_output references a
        ``call_id`` that does not appear in any preceding ``assistant`` ``tool_calls``. Compaction
        that keeps only the last *N* non-system messages often slices inside a tool round-trip and
        triggers HTTP 400 (e.g. "No tool call found for function call output with call_id ...").
        """
        if not messages:
            return messages

        msgs = list(messages)
        for _ in range(8):
            changed = False

            assistant_ids: set[str] = set()
            for msg in msgs:
                if isinstance(msg, AIMessage):
                    for tc in msg.tool_calls or []:
                        tid = tc.get("id")
                        if tid is not None and str(tid) != "":
                            assistant_ids.add(str(tid))

            new_msgs: list[BaseMessage] = []
            for msg in msgs:
                if isinstance(msg, ToolMessage):
                    tid = str(msg.tool_call_id or "")
                    if tid and tid in assistant_ids:
                        new_msgs.append(msg)
                    elif tid:
                        logger.warning(
                            "Omitting tool message: no assistant tool_calls for call_id=%s (compaction or history repair)",
                            tid[:24] + ("..." if len(tid) > 24 else ""),
                        )
                        changed = True
                else:
                    new_msgs.append(msg)
            msgs = new_msgs

            trimmed: list[BaseMessage] = []
            i = 0
            while i < len(msgs):
                msg = msgs[i]
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    needed = {str(tc.get("id")) for tc in msg.tool_calls if tc.get("id") is not None}
                    found: set[str] = set()
                    j = i + 1
                    while j < len(msgs):
                        nxt = msgs[j]
                        if isinstance(nxt, HumanMessage):
                            break
                        if isinstance(nxt, ToolMessage) and nxt.tool_call_id:
                            found.add(str(nxt.tool_call_id))
                        j += 1
                    missing = needed - found
                    if missing:
                        changed = True
                        kept_calls = [tc for tc in msg.tool_calls if str(tc.get("id")) in found]
                        logger.warning(
                            "Trimming %d assistant tool_calls missing tool results after compaction",
                            len(missing),
                        )
                        trimmed.append(msg.model_copy(update={"tool_calls": kept_calls}))
                    else:
                        trimmed.append(msg)
                else:
                    trimmed.append(msg)
                i += 1
            msgs = trimmed

            if not changed:
                break

        return msgs

    def _convert_messages(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        payload_messages: list[dict[str, Any]] = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                payload_messages.append({"role": "system", "content": self._normalize_content(msg.content)})
            elif isinstance(msg, HumanMessage):
                payload_messages.append({"role": "user", "content": self._normalize_content(msg.content)})
            elif isinstance(msg, AIMessage):
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": self._normalize_content(msg.content),
                }
                if msg.tool_calls:
                    assistant_message["tool_calls"] = [
                        {
                            "id": tc.get("id"),
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"]) if isinstance(tc["args"], dict) else str(tc["args"]),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                payload_messages.append(assistant_message)
            elif isinstance(msg, ToolMessage):
                payload_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": self._normalize_content(msg.content),
                    }
                )

        return payload_messages

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        converted: list[dict] = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                converted.append(tool)
                continue

            if "name" in tool:
                converted.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {}),
                        },
                    }
                )
        return converted

    def _convert_tools_for_responses(self, tools: list[dict]) -> list[dict]:
        converted: list[dict] = []
        for tool in self._convert_tools(tools):
            function = tool.get("function")
            if not isinstance(function, Mapping):
                continue
            converted.append(
                {
                    "type": "function",
                    "name": str(function.get("name", "") or ""),
                    "description": function.get("description", ""),
                    "parameters": function.get("parameters", {}),
                }
            )
        return converted

    def _convert_messages_to_responses_input(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        input_items: list[dict[str, Any]] = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                input_items.append({"role": "system", "content": self._normalize_content(msg.content)})
            elif isinstance(msg, HumanMessage):
                input_items.append({"role": "user", "content": self._normalize_content(msg.content)})
            elif isinstance(msg, AIMessage):
                content = self._normalize_content(msg.content)
                if content:
                    input_items.append({"role": "assistant", "content": content})
                for tool_call in msg.tool_calls or []:
                    args = tool_call.get("args", {})
                    arguments = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
                    call_id = str(tool_call.get("id", "") or "")
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": call_id,
                            "name": str(tool_call.get("name", "") or ""),
                            "arguments": arguments,
                        }
                    )
            elif isinstance(msg, ToolMessage):
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.tool_call_id,
                        "output": self._normalize_content(msg.content),
                    }
                )

        return input_items

    @staticmethod
    def _count_non_system_messages(messages: list[BaseMessage]) -> int:
        return sum(1 for msg in messages if not isinstance(msg, SystemMessage))

    def _compact_messages_for_payload(self, messages: list[BaseMessage]) -> tuple[list[BaseMessage] | None, int | None]:
        sanitized_messages = self._filter_messages_for_upstream_context(messages)
        non_system_messages = [msg for msg in sanitized_messages if not isinstance(msg, SystemMessage)]
        system_messages = [msg for msg in sanitized_messages if isinstance(msg, SystemMessage)]
        current_count = len(non_system_messages)

        for target in _CONTEXT_COMPACTION_TARGETS:
            if current_count > target:
                compacted = [*system_messages, *non_system_messages[-target:]]
                compacted = self._sanitize_compacted_messages_for_tool_protocol(compacted)
                return compacted, target

        return None, None

    def _compact_messages_for_transient_recovery(self, messages: list[BaseMessage]) -> tuple[list[BaseMessage] | None, int | None]:
        sanitized_messages = self._filter_messages_for_upstream_context(messages)
        non_system_messages = [msg for msg in sanitized_messages if not isinstance(msg, SystemMessage)]
        system_messages = [msg for msg in sanitized_messages if isinstance(msg, SystemMessage)]
        current_count = len(non_system_messages)

        for target in _TRANSIENT_RECOVERY_TARGETS:
            if current_count > target:
                compacted = [*system_messages, *non_system_messages[-target:]]
                compacted = self._sanitize_compacted_messages_for_tool_protocol(compacted)
                return compacted, target

        return None, None

    def _estimate_request_body_chars(self, payload: dict[str, Any]) -> int:
        return len(json.dumps(payload, ensure_ascii=False))

    @staticmethod
    def _truncate_text_for_request_budget(text: str, max_chars: int) -> str:
        normalized = text.strip()
        if len(normalized) <= max_chars:
            return normalized
        if max_chars <= 96:
            return "[Earlier context omitted due to gateway request size limit.]"

        reserved = len("\n\n[...content truncated for gateway size limit...]\n\n")
        available = max(32, max_chars - reserved)
        head = max(16, available // 2)
        tail = max(16, available - head)
        if head + tail >= len(normalized):
            return normalized[:max_chars]

        return (
            f"{normalized[:head]}\n\n"
            "[...content truncated for gateway size limit...]\n\n"
            f"{normalized[-tail:]}"
        )

    def _compact_responses_payload_for_size(self, payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        estimated_chars = self._estimate_request_body_chars(payload)
        if estimated_chars <= self._max_request_body_chars:
            return payload, False

        raw_input = payload.get("input")
        if not isinstance(raw_input, list):
            return payload, False

        compacted = json.loads(json.dumps(payload, ensure_ascii=False))
        input_items = compacted.get("input")
        if not isinstance(input_items, list):
            return payload, False

        def within_budget() -> bool:
            return self._estimate_request_body_chars(compacted) <= self._max_request_body_chars

        def shrink_fields(items: list[dict[str, Any]]) -> bool:
            for item in items:
                if not isinstance(item, dict):
                    continue
                if item.get("role") == "system":
                    continue
                for field in ("content", "output", "arguments"):
                    value = item.get(field)
                    if not isinstance(value, str) or len(value) <= 512:
                        continue
                    for keep_chars in (24_000, 12_000, 6_000, 3_000, 1_500, 512):
                        item[field] = self._truncate_text_for_request_budget(value, keep_chars)
                        if within_budget():
                            return True
                    item[field] = "[Earlier context omitted due to gateway request size limit.]"
                    if within_budget():
                        return True
            return within_budget()

        if shrink_fields(input_items):
            logger.warning(
                "Compacted API pool responses payload from %d to %d chars by truncating historical content",
                estimated_chars,
                self._estimate_request_body_chars(compacted),
            )
            return compacted, True

        system_items = [item for item in input_items if isinstance(item, dict) and item.get("role") == "system"]
        non_system_items = [item for item in input_items if not (isinstance(item, dict) and item.get("role") == "system")]
        for keep_items in (4, 2, 1):
            if len(non_system_items) <= keep_items:
                continue
            candidate_input = [*system_items, *non_system_items[-keep_items:]]
            compacted["input"] = candidate_input
            if within_budget() or shrink_fields(candidate_input):
                logger.warning(
                    "Compacted API pool responses payload from %d to %d chars by keeping the latest %d non-system items",
                    estimated_chars,
                    self._estimate_request_body_chars(compacted),
                    keep_items,
                )
                return compacted, True

        return payload, False

    def _build_chat_completions_payload(
        self,
        messages: list[BaseMessage],
        tools: list[dict] | None = None,
        stop: list[str] | None = None,
        *,
        stream: bool = False,
    ) -> dict[str, Any]:
        messages = self._filter_messages_for_upstream_context(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._convert_messages(messages),
        }

        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        normalized_reasoning_effort = normalize_runtime_reasoning_effort(
            engine="api_pool",
            runtime_model=self.model,
            reasoning_effort=self.reasoning_effort,
        )
        if normalized_reasoning_effort is not None:
            payload["reasoning_effort"] = normalized_reasoning_effort
        if stop:
            payload["stop"] = stop
        if tools:
            payload["tools"] = self._convert_tools(tools)
            payload["tool_choice"] = "auto"
        if self.extra_body:
            payload.update(self.extra_body)
        if stream:
            payload["stream"] = True

        return payload

    def _build_responses_payload(
        self,
        messages: list[BaseMessage],
        tools: list[dict] | None = None,
        stop: list[str] | None = None,
        *,
        stream: bool = False,
    ) -> dict[str, Any]:
        messages = self._filter_messages_for_upstream_context(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "input": self._convert_messages_to_responses_input(messages),
            "store": False,
        }

        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_output_tokens"] = self.max_tokens

        normalized_reasoning_effort = normalize_runtime_reasoning_effort(
            engine="api_pool",
            runtime_model=self.model,
            reasoning_effort=self.reasoning_effort,
        )
        if normalized_reasoning_effort is not None:
            payload["reasoning"] = {"effort": normalized_reasoning_effort}

        if tools:
            payload["tools"] = self._convert_tools_for_responses(tools)
            payload["tool_choice"] = "auto"

        if self.extra_body:
            payload.update(self.extra_body)
        if stream:
            payload["stream"] = True
        compacted_payload, _ = self._compact_responses_payload_for_size(payload)
        return compacted_payload

    def _build_payload(
        self,
        messages: list[BaseMessage],
        tools: list[dict] | None = None,
        stop: list[str] | None = None,
        *,
        stream: bool = False,
    ) -> dict[str, Any]:
        if self._uses_responses_api():
            return self._build_responses_payload(messages, tools=tools, stop=stop, stream=stream)
        return self._build_chat_completions_payload(messages, tools=tools, stop=stop, stream=stream)

    @staticmethod
    def _parse_sse_data_line(line: str) -> dict[str, Any] | None:
        if not line.startswith("data:"):
            return None

        raw_data = line[5:].strip()
        if not raw_data or raw_data == "[DONE]":
            return None

        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError:
            logger.debug("Skipping non-JSON API pool SSE frame: %s", raw_data)
            return None

        return data if isinstance(data, dict) else None

    @staticmethod
    def _extract_stream_tool_call_chunks(delta: dict[str, Any]) -> list[dict[str, Any]]:
        tool_call_chunks: list[dict[str, Any]] = []

        for idx, tool_call in enumerate(delta.get("tool_calls") or []):
            if not isinstance(tool_call, Mapping):
                continue

            function = tool_call.get("function")
            function_mapping = function if isinstance(function, Mapping) else {}
            tool_call_chunks.append(
                {
                    "name": str(function_mapping.get("name", "") or ""),
                    "args": str(function_mapping.get("arguments", "") or ""),
                    "id": tool_call.get("id"),
                    "index": tool_call.get("index", idx),
                    "type": "tool_call_chunk",
                }
            )

        return tool_call_chunks

    def _stream_chunk_from_sse_event(self, data: dict[str, Any]) -> ChatGenerationChunk | None:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return None

        choice = choices[0]
        if not isinstance(choice, Mapping):
            return None

        delta = choice.get("delta")
        finish_reason = choice.get("finish_reason")
        if not isinstance(delta, Mapping):
            delta = {}

        content = self._normalize_content(delta.get("content", "")) if "content" in delta else ""
        tool_call_chunks = self._extract_stream_tool_call_chunks(delta)
        reasoning = self._merge_reasoning(
            self._extract_reasoning_text(delta.get("reasoning_content"), strip_parts=False),
            self._extract_reasoning_text(delta.get("reasoning"), strip_parts=False),
            self._extract_reasoning_text(delta.get("reasoning_details"), strip_parts=False),
        )

        if not content and not tool_call_chunks and not reasoning and finish_reason is not None:
            return None

        message = AIMessageChunk(
            content=content,
            tool_call_chunks=tool_call_chunks,
        )
        message = self._with_reasoning_content(message, reasoning, preserve_whitespace=True)

        generation_info = {"finish_reason": finish_reason} if finish_reason is not None else None
        return ChatGenerationChunk(message=message, generation_info=generation_info)

    @staticmethod
    def _tool_call_chunks_from_response_message(message: Mapping[str, Any]) -> list[dict[str, Any]]:
        tool_call_chunks: list[dict[str, Any]] = []

        for idx, tool_call in enumerate(message.get("tool_calls") or []):
            if not isinstance(tool_call, Mapping):
                continue

            function = tool_call.get("function")
            function_mapping = function if isinstance(function, Mapping) else {}
            raw_arguments = function_mapping.get("arguments", "")
            if isinstance(raw_arguments, str):
                arguments = raw_arguments
            else:
                arguments = json.dumps(raw_arguments, ensure_ascii=False)

            tool_call_chunks.append(
                {
                    "name": str(function_mapping.get("name", "") or ""),
                    "args": arguments,
                    "id": tool_call.get("id"),
                    "index": idx,
                    "type": "tool_call_chunk",
                }
            )

        return tool_call_chunks

    def _stream_chunk_from_response(self, response: dict[str, Any]) -> ChatGenerationChunk | None:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            return None

        choice = choices[0]
        if not isinstance(choice, Mapping):
            return None

        message_payload = choice.get("message")
        if not isinstance(message_payload, Mapping):
            return None

        content = self._normalize_content(message_payload.get("content", ""))
        cleaned_content, inline_reasoning = self._strip_inline_think_tags(content)
        reasoning = self._merge_reasoning(
            self._extract_reasoning_text(message_payload.get("reasoning_content")),
            self._extract_reasoning_text(message_payload.get("reasoning")),
            self._extract_reasoning_text(message_payload.get("reasoning_details")),
            self._extract_reasoning_text(response.get("reasoning")),
            inline_reasoning,
        )
        tool_call_chunks = self._tool_call_chunks_from_response_message(message_payload)
        finish_reason = choice.get("finish_reason")

        if not cleaned_content and not tool_call_chunks and not reasoning and finish_reason is None:
            return None

        message = AIMessageChunk(
            content=cleaned_content,
            tool_call_chunks=tool_call_chunks,
        )
        message = self._with_reasoning_content(message, reasoning)
        generation_info = {"finish_reason": finish_reason} if finish_reason is not None else None
        return ChatGenerationChunk(message=message, generation_info=generation_info)

    @staticmethod
    def _tool_calls_payload_from_langchain(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []

        for tool_call in tool_calls:
            if not isinstance(tool_call, Mapping):
                continue

            args = tool_call.get("args", {})
            arguments = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
            payload.append(
                {
                    "id": tool_call.get("id"),
                    "type": "function",
                    "function": {
                        "name": str(tool_call.get("name", "") or ""),
                        "arguments": arguments,
                    },
                }
            )

        return payload

    @staticmethod
    def _responses_tool_call_chunks_from_output(output: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tool_call_chunks: list[dict[str, Any]] = []
        for idx, item in enumerate(output):
            if not isinstance(item, Mapping) or item.get("type") != "function_call":
                continue
            arguments = item.get("arguments", "")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            tool_call_chunks.append(
                {
                    "name": str(item.get("name", "") or ""),
                    "args": arguments,
                    "id": item.get("call_id") or item.get("id"),
                    "index": idx,
                    "type": "tool_call_chunk",
                }
            )
        return tool_call_chunks

    def _extract_responses_reasoning(self, response: dict[str, Any]) -> str | None:
        output = response.get("output")
        reasoning_parts: list[str] = []
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, Mapping):
                    continue
                if item.get("type") != "reasoning":
                    continue
                extracted = self._merge_reasoning(
                    self._extract_reasoning_text(item.get("summary")),
                    self._extract_reasoning_text(item.get("content")),
                )
                if extracted:
                    reasoning_parts.append(extracted)
        top_level_reasoning = self._extract_reasoning_text(response.get("reasoning"))
        if top_level_reasoning:
            reasoning_parts.append(top_level_reasoning)
        return self._merge_reasoning(*reasoning_parts)

    def _extract_responses_content(self, response: dict[str, Any]) -> str:
        output = response.get("output")
        parts: list[str] = []
        if not isinstance(output, list):
            return ""
        for item in output:
            if not isinstance(item, Mapping) or item.get("type") != "message":
                continue
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, Mapping):
                        continue
                    if part.get("type") in {"output_text", "text"}:
                        text = part.get("text")
                        if isinstance(text, str) and text:
                            parts.append(text)
            elif isinstance(content, str) and content:
                parts.append(content)
        return "\n".join(parts).strip()

    def _stream_chunk_from_responses_response(self, response: dict[str, Any]) -> ChatGenerationChunk | None:
        content = self._extract_responses_content(response)
        cleaned_content, inline_reasoning = self._strip_inline_think_tags(content)
        output = response.get("output")
        output_items = output if isinstance(output, list) else []
        tool_call_chunks = self._responses_tool_call_chunks_from_output(output_items)
        reasoning = self._merge_reasoning(
            self._extract_responses_reasoning(response),
            inline_reasoning,
        )
        if not cleaned_content and not tool_call_chunks and not reasoning:
            return None
        message = AIMessageChunk(
            content=cleaned_content,
            tool_call_chunks=tool_call_chunks,
        )
        message = self._with_reasoning_content(message, reasoning)
        return ChatGenerationChunk(message=message, generation_info=None)

    def _parse_responses_tool_calls(self, response: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        output = response.get("output")
        if not isinstance(output, list):
            return [], []

        tool_calls: list[dict[str, Any]] = []
        invalid_tool_calls: list[dict[str, Any]] = []
        for item in output:
            if not isinstance(item, Mapping) or item.get("type") != "function_call":
                continue
            raw_arguments = item.get("arguments", "{}")
            try:
                parsed_arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            except json.JSONDecodeError as exc:
                invalid_tool_calls.append(
                    {
                        "type": "invalid_tool_call",
                        "name": item.get("name"),
                        "args": str(raw_arguments),
                        "id": item.get("call_id") or item.get("id"),
                        "error": f"Failed to parse tool arguments: {exc}",
                    }
                )
                continue
            tool_calls.append(
                {
                    "name": str(item.get("name", "") or ""),
                    "args": parsed_arguments if isinstance(parsed_arguments, dict) else {},
                    "id": item.get("call_id") or item.get("id", ""),
                    "type": "tool_call",
                }
            )
        return tool_calls, invalid_tool_calls

    async def _request_responses(
        self,
        messages: list[BaseMessage],
        *,
        tools: list[dict] | None = None,
        stop: list[str] | None = None,
    ) -> dict[str, Any]:
        current_messages = messages
        payload = self._build_responses_payload(current_messages, tools=tools, stop=stop)
        max_key_rotations = self._current_gateway_key_count()
        key_rotation_attempts = 0
        transient_retry_attempts = 0
        gateway_rotation_attempts = 0
        last_retryable_error: RuntimeError | TimeoutError | None = None

        while True:
            base_url = self._current_base_url()
            url = f"{base_url}/responses"
            api_key = self._current_api_key()
            headers = self._build_headers(api_key)

            try:
                async with self._create_http_client() as client:
                    response = await client.post(url, headers=headers, json=payload)
                self._reset_gateway_rotation()
                response.raise_for_status()
                return response.json()
            except httpx.TimeoutException as exc:
                last_retryable_error = TimeoutError(
                    f"API pool transient retries exhausted after {transient_retry_attempts + 1} attempts. "
                    f"Last error: request timed out after {self.timeout_seconds}s"
                )
                compacted_messages, target = self._compact_messages_for_transient_recovery(current_messages)
                if compacted_messages is not None:
                    logger.warning(
                        "API pool responses timeout on base_url=%s key=%s; compacting context to latest %d non-system messages before retry",
                        base_url,
                        self._mask_key(api_key),
                        target,
                    )
                    current_messages = compacted_messages
                    payload = self._build_responses_payload(current_messages, tools=tools, stop=stop)
                    continue
                should_rotate_gateway = len(self._base_urls) > 1 and gateway_rotation_attempts < len(self._base_urls) - 1
                if should_rotate_gateway:
                    gateway_rotation_attempts += 1
                    self._rotate_to_next_base_url()
                    transient_retry_attempts += 1
                    logger.warning(
                        "API pool responses request timed out on base_url=%s key=%s; rotating gateway for attempt %d/%d",
                        base_url,
                        self._mask_key(api_key),
                        transient_retry_attempts,
                        self._max_transient_retries,
                    )
                    continue
                should_retry_same_gateway = transient_retry_attempts < self._max_transient_retries
                if should_retry_same_gateway:
                    transient_retry_attempts += 1
                    if self._current_gateway_key_count() > 1:
                        self._rotate_to_next_key()
                        api_key = self._current_api_key()
                    delay_seconds = self._current_transient_retry_delay_seconds(transient_retry_attempts)
                    logger.warning(
                        "API pool responses timeout on base_url=%s key=%s; retrying same gateway attempt %d/%d with key=%s after %.1fs",
                        base_url,
                        self._mask_key(api_key),
                        transient_retry_attempts,
                        self._max_transient_retries,
                        self._mask_key(self._current_api_key()),
                        delay_seconds,
                    )
                    if delay_seconds > 0:
                        await asyncio.sleep(delay_seconds)
                    continue
                raise last_retryable_error from exc
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                detail = self._extract_error_message(exc.response)
                if status == 413:
                    compacted_messages, target = self._compact_messages_for_payload(current_messages)
                    if compacted_messages is not None and target is not None:
                        logger.warning(
                            "API pool responses request hit HTTP 413 on base_url=%s; compacting non-system context from %d to %d messages and retrying",
                            base_url,
                            self._count_non_system_messages(current_messages),
                            target,
                        )
                        current_messages = compacted_messages
                        payload = self._build_responses_payload(current_messages, tools=tools, stop=stop)
                        transient_retry_attempts = 0
                        gateway_rotation_attempts = 0
                        continue
                if self._is_upstream_gateway_block(status, detail):
                    raise RuntimeError(
                        "API pool upstream denied the current egress path or proxy "
                        f"(HTTP {status} on {exc.response.request.url.host}: {detail}). "
                        "This is not a DeerFlow or Docker container failure; switch to a different "
                        "API pool gateway or route this domain via a different network exit."
                    ) from exc
                if status in KEY_ROTATION_STATUS_CODES:
                    last_retryable_error = RuntimeError(
                        self._key_rotation_exhaustion_message(
                            status=status, detail=detail, max_key_rotations=max_key_rotations
                        )
                    )
                    if key_rotation_attempts < max_key_rotations - 1:
                        key_rotation_attempts += 1
                        transient_retry_attempts = 0
                        logger.warning(
                            "Key %s failed with %s on responses base_url=%s, rotating to next key",
                            self._mask_key(api_key),
                            status,
                            base_url,
                        )
                        self._rotate_to_next_key()
                        continue
                    should_failover_gateway = len(self._base_urls) > 1 and gateway_rotation_attempts < len(self._base_urls) - 1
                    if should_failover_gateway:
                        gateway_rotation_attempts += 1
                        logger.warning(
                            "All keys exhausted on responses base_url=%s; failing over to next gateway",
                            base_url,
                        )
                        self._rotate_to_next_base_url()
                        max_key_rotations = self._current_gateway_key_count()
                        key_rotation_attempts = 0
                        transient_retry_attempts = 0
                        continue
                    raise last_retryable_error from exc
                if status in TRANSIENT_STATUS_CODES:
                    last_retryable_error = RuntimeError(
                        f"API pool transient retries exhausted after {transient_retry_attempts + 1} attempts. "
                        f"Last error HTTP {status}: {detail}"
                    )
                    compacted_messages, target = self._compact_messages_for_transient_recovery(current_messages)
                    if compacted_messages is not None:
                        logger.warning(
                            "API pool responses transient HTTP %s on base_url=%s key=%s; compacting context to latest %d non-system messages before retry",
                            status,
                            base_url,
                            self._mask_key(api_key),
                            target,
                        )
                        current_messages = compacted_messages
                        payload = self._build_responses_payload(current_messages, tools=tools, stop=stop)
                        continue
                    should_rotate_gateway = len(self._base_urls) > 1 and gateway_rotation_attempts < len(self._base_urls) - 1
                    if should_rotate_gateway:
                        gateway_rotation_attempts += 1
                        self._rotate_to_next_base_url()
                        transient_retry_attempts += 1
                        logger.warning(
                            "API pool responses transient HTTP %s on base_url=%s key=%s; rotating to next gateway for attempt %d/%d",
                            status,
                            base_url,
                            self._mask_key(api_key),
                            transient_retry_attempts,
                            self._max_transient_retries_for_status(status),
                        )
                        continue
                    cap = self._max_transient_retries_for_status(status)
                    should_retry_same_gateway = transient_retry_attempts < cap
                    if should_retry_same_gateway:
                        transient_retry_attempts += 1
                        if self._current_gateway_key_count() > 1:
                            self._rotate_to_next_key()
                        delay_seconds = self._current_transient_retry_delay_seconds(transient_retry_attempts, status=status)
                        logger.warning(
                            "API pool responses transient HTTP %s on base_url=%s key=%s; retrying same gateway attempt %d/%d with key=%s after %.1fs",
                            status,
                            base_url,
                            self._mask_key(api_key),
                            transient_retry_attempts,
                            cap,
                            self._mask_key(self._current_api_key()),
                            delay_seconds,
                        )
                        if delay_seconds > 0:
                            await asyncio.sleep(delay_seconds)
                        continue
                    raise last_retryable_error from exc
                raise RuntimeError(f"API pool request failed with HTTP {status}: {detail}") from exc
            except httpx.HTTPError as exc:
                last_retryable_error = RuntimeError(
                    f"API pool transient transport retries exhausted after {transient_retry_attempts + 1} attempts. "
                    f"Last error: {exc}"
                )
                compacted_messages, target = self._compact_messages_for_transient_recovery(current_messages)
                if compacted_messages is not None:
                    logger.warning(
                        "API pool responses transport error on base_url=%s key=%s; compacting context to latest %d non-system messages before retry: %s",
                        base_url,
                        self._mask_key(api_key),
                        target,
                        exc,
                    )
                    current_messages = compacted_messages
                    payload = self._build_responses_payload(current_messages, tools=tools, stop=stop)
                    continue
                should_rotate_gateway = len(self._base_urls) > 1 and gateway_rotation_attempts < len(self._base_urls) - 1
                if should_rotate_gateway:
                    gateway_rotation_attempts += 1
                    self._rotate_to_next_base_url()
                    transient_retry_attempts += 1
                    logger.warning(
                        "API pool responses transport error on base_url=%s key=%s; rotating to next gateway for attempt %d/%d: %s",
                        base_url,
                        self._mask_key(api_key),
                        transient_retry_attempts,
                        self._max_transient_retries,
                        exc,
                    )
                    continue
                should_retry_same_gateway = transient_retry_attempts < self._max_transient_retries
                if should_retry_same_gateway:
                    transient_retry_attempts += 1
                    if self._current_gateway_key_count() > 1:
                        self._rotate_to_next_key()
                    delay_seconds = self._current_transient_retry_delay_seconds(transient_retry_attempts)
                    logger.warning(
                        "API pool responses transport error on base_url=%s key=%s; retrying same gateway attempt %d/%d with key=%s after %.1fs: %s",
                        base_url,
                        self._mask_key(api_key),
                        transient_retry_attempts,
                        self._max_transient_retries,
                        self._mask_key(self._current_api_key()),
                        delay_seconds,
                        exc,
                    )
                    if delay_seconds > 0:
                        await asyncio.sleep(delay_seconds)
                    continue
                raise last_retryable_error from exc

    async def _request_chat_completions_via_stream(
        self,
        messages: list[BaseMessage],
        *,
        tools: list[dict] | None = None,
        stop: list[str] | None = None,
        base_url: str,
        api_key: str,
    ) -> dict[str, Any]:
        payload = self._build_payload(messages, tools=tools, stop=stop, stream=True)
        url = f"{base_url}/chat/completions"
        headers = self._build_headers(api_key)
        merged_chunk: ChatGenerationChunk | None = None
        finish_reason: str | None = None
        response_model = self.model

        async with self._create_http_client() as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    data = self._parse_sse_data_line(line)
                    if data is None:
                        continue

                    if isinstance(data.get("model"), str):
                        response_model = data["model"]

                    chunk = self._stream_chunk_from_sse_event(data)
                    if chunk is None:
                        continue

                    chunk_finish_reason = chunk.generation_info.get("finish_reason") if chunk.generation_info else None
                    if isinstance(chunk_finish_reason, str):
                        finish_reason = chunk_finish_reason

                    merged_chunk = chunk if merged_chunk is None else merged_chunk + chunk

        if merged_chunk is None:
            raise RuntimeError("API pool streaming fallback returned no assistant content")

        message_payload: dict[str, Any] = {
            "role": "assistant",
            "content": self._normalize_content(merged_chunk.message.content),
        }
        reasoning = self._extract_reasoning_text(merged_chunk.message.additional_kwargs.get("reasoning_content"))
        if reasoning:
            message_payload["reasoning_content"] = reasoning
        if merged_chunk.message.tool_calls:
            message_payload["tool_calls"] = self._tool_calls_payload_from_langchain(merged_chunk.message.tool_calls)

        return {
            "model": response_model,
            "choices": [
                {
                    "finish_reason": finish_reason or "stop",
                    "message": message_payload,
                }
            ],
            "usage": {},
        }

    @staticmethod
    def _summarize_html_error(response: httpx.Response) -> str:
        host = response.request.url.host if response.request else "upstream"
        if response.status_code in {504, 524}:
            return f"API pool upstream gateway timed out ({host})"
        if response.status_code in {502, 503}:
            return f"API pool upstream is temporarily unavailable ({host})"
        return f"API pool upstream returned an HTML error page ({host})"

    @staticmethod
    def _extract_error_message(response: httpx.Response) -> str:
        content_type = response.headers.get("content-type", "").lower()
        text = response.text.strip()
        if "text/html" in content_type or text.lower().startswith("<!doctype html") or text.lower().startswith("<html"):
            return StandardAPIChatModel._summarize_html_error(response)

        try:
            payload = response.json()
        except ValueError:
            return text or f"HTTP {response.status_code}"

        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str) and message:
                    return message
            message = payload.get("message")
            if isinstance(message, str) and message:
                return message
        return response.text.strip() or f"HTTP {response.status_code}"

    @staticmethod
    def _is_upstream_gateway_block(status_code: int, detail: str) -> bool:
        normalized = detail.lower()
        return status_code == 403 and ("error code: 1010" in normalized or "cloudflare" in normalized)

    @staticmethod
    def _provider_account_error_hint(status: int, detail: str) -> str | None:
        """Human hint when failure is almost certainly billing/quota on the provider side."""
        if status not in {401, 403, 429}:
            return None
        d = (detail or "").lower()
        raw = detail or ""
        if "insufficient account balance" in d or ("insufficient" in d and "balance" in d):
            return (
                "供應商：帳戶餘額不足（Insufficient account balance）。"
                "請至 API 後台充值或改用仍有額度的 Key。"
                "若多把 key 共用同一賬戶，輪換 key 無法繞過欠費。"
            )
        if "no active subscription" in d or ("subscription" in d and "not found" in d):
            return (
                "供應商：無有效訂閱／套餐，請續費或更換有效的 Key。"
            )
        if "餘額不足" in raw or "余额不足" in raw:
            return "供應商：餘額不足，請充值或更換 API Key。"
        # Daily / quota caps from pool gateways (e.g. HTTP 403 "daily usage limit exceeded").
        if (
            "daily usage limit" in d
            or "daily limit" in d
            or ("daily" in d and "quota" in d)
            or (
                "日" in raw
                and (
                    "限額" in raw
                    or "限额" in raw
                    or "上限" in raw
                    or "用量" in raw
                )
            )
        ):
            return (
                "供應商：已達本日用量或請求上限"
                "（並非 DeerFlow 故障）。"
                "請等待額度重置、在供應商後台加購配額"
                "，或在 API_POOL_KEYS 換成其他帳戶的 Key；"
                "同一方案下輪換多把 Key 通常無法繞過帳戶"
                "級每日上限，可暫改 Codex／Claude 等非 API Pool 引擎。"
            )
        if "usage limit" in d and ("exceeded" in d or "reach" in d):
            return (
                "供應商：用量上限已滿。"
                "請等待重置、更換有效 Key，或暫不使用 API Pool 引擎。"
            )
        if status == 429 and ("rate" in d or "too many" in d or "限流" in raw):
            return (
                "供應商：觸發速率或配額限制（429）。"
                "請稍後再試、降低併發，或更換 Key／線路。"
            )
        return None

    @staticmethod
    def _key_rotation_exhaustion_message(*, status: int, detail: str, max_key_rotations: int) -> str:
        base = f"API pool key rotation exhausted after {max_key_rotations} keys. Last error HTTP {status}: {detail}"
        hint = StandardAPIChatModel._provider_account_error_hint(status, detail)
        if hint:
            return f"{base} {hint}"
        return base

    async def _request_chat_completions(
        self,
        messages: list[BaseMessage],
        *,
        tools: list[dict] | None = None,
        stop: list[str] | None = None,
    ) -> dict[str, Any]:
        if self._uses_responses_api():
            return await self._request_responses(messages, tools=tools, stop=stop)

        current_messages = messages
        payload = self._build_payload(current_messages, tools=tools, stop=stop)
        max_key_rotations = self._current_gateway_key_count()
        key_rotation_attempts = 0
        transient_retry_attempts = 0
        gateway_rotation_attempts = 0
        last_retryable_error: RuntimeError | TimeoutError | None = None

        while True:
            base_url = self._current_base_url()
            url = f"{base_url}/chat/completions"
            api_key = self._current_api_key()
            headers = self._build_headers(api_key)

            try:
                async with self._create_http_client() as client:
                    response = await client.post(url, headers=headers, json=payload)
                self._reset_gateway_rotation()
                response.raise_for_status()
                response_json = response.json()
                null_same_key_attempts = 0
                while self._is_null_content_response(response_json):
                    if null_same_key_attempts < self._max_transient_retries:
                        null_same_key_attempts += 1
                        delay_seconds = self._current_transient_retry_delay_seconds(null_same_key_attempts)
                        logger.warning(
                            "Gateway HTTP 200 with null content on base_url=%s key=%s; retrying same key (%d/%d) after %.1fs",
                            base_url,
                            self._mask_key(api_key),
                            null_same_key_attempts,
                            self._max_transient_retries,
                            delay_seconds,
                        )
                        if delay_seconds > 0:
                            await asyncio.sleep(delay_seconds)
                        async with self._create_http_client() as client:
                            response = await client.post(url, headers=headers, json=payload)
                        self._reset_gateway_rotation()
                        response.raise_for_status()
                        response_json = response.json()
                        continue
                    break

                if self._is_null_content_response(response_json):
                    compacted_messages, target = self._compact_messages_for_transient_recovery(current_messages)
                    if compacted_messages is not None:
                        logger.warning(
                            "Gateway null content on base_url=%s key=%s; compacting context to latest %d non-system messages before retry",
                            base_url,
                            self._mask_key(api_key),
                            target,
                        )
                        current_messages = compacted_messages
                        payload = self._build_payload(current_messages, tools=tools, stop=stop)
                        continue

                    last_retryable_error = RuntimeError(
                        f"API pool key rotation exhausted after {max_key_rotations} keys. Last error: gateway returned null content (empty response body)"
                    )
                    if key_rotation_attempts < max_key_rotations - 1:
                        key_rotation_attempts += 1
                        logger.warning(
                            "Key %s returned null content on base_url=%s, rotating to next key",
                            self._mask_key(api_key),
                            base_url,
                        )
                        self._rotate_to_next_key()
                        continue
                    should_failover_gateway = len(self._base_urls) > 1 and gateway_rotation_attempts < len(self._base_urls) - 1
                    if should_failover_gateway:
                        gateway_rotation_attempts += 1
                        logger.warning(
                            "All keys returned null content on base_url=%s; failing over to next gateway",
                            base_url,
                        )
                        self._rotate_to_next_base_url()
                        max_key_rotations = self._current_gateway_key_count()
                        key_rotation_attempts = 0
                        transient_retry_attempts = 0
                        continue
                    logger.error(
                        "All API pool keys returned null content on base_url=%s. Last key=%s",
                        base_url,
                        self._mask_key(api_key),
                    )
                    raise last_retryable_error
                return response_json
            except httpx.TimeoutException as exc:
                last_retryable_error = TimeoutError(
                    f"API pool transient retries exhausted after {transient_retry_attempts + 1} attempts. "
                    f"Last error: request timed out after {self.timeout_seconds}s"
                )
                should_rotate_gateway = len(self._base_urls) > 1 and gateway_rotation_attempts < len(self._base_urls) - 1
                if should_rotate_gateway:
                    gateway_rotation_attempts += 1
                    self._rotate_to_next_base_url()
                    transient_retry_attempts += 1
                    logger.warning(
                        "API pool request timed out on base_url=%s key=%s; rotating gateway for attempt %d/%d",
                        base_url,
                        self._mask_key(api_key),
                        transient_retry_attempts,
                        self._max_transient_retries,
                    )
                    continue

                logger.warning(
                    "API pool non-stream request timed out on base_url=%s before a response; falling back to streamed collection",
                    base_url,
                )
                try:
                    return await self._request_chat_completions_via_stream(
                        current_messages,
                        tools=tools,
                        stop=stop,
                        base_url=base_url,
                        api_key=api_key,
                    )
                except Exception:
                    logger.error("API pool request timed out after %ss", self.timeout_seconds)
                    raise last_retryable_error from exc
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                detail = self._extract_error_message(exc.response)
                if status == 413:
                    compacted_messages, target = self._compact_messages_for_payload(current_messages)
                    if compacted_messages is not None and target is not None:
                        logger.warning(
                            "API pool chat.completions request hit HTTP 413 on base_url=%s; compacting non-system context from %d to %d messages and retrying",
                            base_url,
                            self._count_non_system_messages(current_messages),
                            target,
                        )
                        current_messages = compacted_messages
                        payload = self._build_payload(current_messages, tools=tools, stop=stop)
                        transient_retry_attempts = 0
                        gateway_rotation_attempts = 0
                        continue
                if self._is_upstream_gateway_block(status, detail):
                    logger.error(
                        "API pool upstream access denied by gateway/WAF on base_url=%s. key=%s detail=%s",
                        base_url,
                        self._mask_key(api_key),
                        detail,
                    )
                    raise RuntimeError(
                        "API pool upstream denied the current egress path or proxy "
                        f"(HTTP {status} on {exc.response.request.url.host}: {detail}). "
                        "This is not a DeerFlow or Docker container failure; switch to a different "
                        "API pool gateway or route this domain via a different network exit."
                    ) from exc

                if status in KEY_ROTATION_STATUS_CODES:
                    last_retryable_error = RuntimeError(
                        self._key_rotation_exhaustion_message(
                            status=status, detail=detail, max_key_rotations=max_key_rotations
                        )
                    )
                    if key_rotation_attempts < max_key_rotations - 1:
                        key_rotation_attempts += 1
                        transient_retry_attempts = 0
                        logger.warning(
                            "Key %s failed with %s on base_url=%s, rotating to next key",
                            self._mask_key(api_key),
                            status,
                            base_url,
                        )
                        self._rotate_to_next_key()
                        continue

                    should_failover_gateway = len(self._base_urls) > 1 and gateway_rotation_attempts < len(self._base_urls) - 1
                    if should_failover_gateway:
                        gateway_rotation_attempts += 1
                        logger.warning(
                            "All keys exhausted on base_url=%s; failing over to next gateway",
                            base_url,
                        )
                        self._rotate_to_next_base_url()
                        max_key_rotations = self._current_gateway_key_count()
                        key_rotation_attempts = 0
                        transient_retry_attempts = 0
                        continue

                    logger.error(
                        "All API pool keys were exhausted after HTTP %s on base_url=%s. Last key=%s detail=%s",
                        status,
                        base_url,
                        self._mask_key(api_key),
                        detail,
                    )
                    raise last_retryable_error from exc

                if status in TRANSIENT_STATUS_CODES:
                    last_retryable_error = RuntimeError(
                        f"API pool transient retries exhausted after {transient_retry_attempts + 1} attempts. "
                        f"Last error HTTP {status}: {detail}"
                    )
                    compacted_messages, target = self._compact_messages_for_transient_recovery(current_messages)
                    if compacted_messages is not None:
                        logger.warning(
                            "API pool chat.completions transient HTTP %s on base_url=%s key=%s; compacting context to latest %d non-system messages before retry",
                            status,
                            base_url,
                            self._mask_key(api_key),
                            target,
                        )
                        current_messages = compacted_messages
                        payload = self._build_payload(current_messages, tools=tools, stop=stop)
                        continue

                    should_rotate_gateway = len(self._base_urls) > 1 and gateway_rotation_attempts < len(self._base_urls) - 1
                    if should_rotate_gateway:
                        gateway_rotation_attempts += 1
                        self._rotate_to_next_base_url()
                        transient_retry_attempts += 1
                        logger.warning(
                            "API pool transient HTTP %s on base_url=%s key=%s; rotating to next gateway for attempt %d/%d",
                            status,
                            base_url,
                            self._mask_key(api_key),
                            transient_retry_attempts,
                            self._max_transient_retries_for_status(status),
                        )
                        continue

                    cap = self._max_transient_retries_for_status(status)
                    should_retry_same_gateway = transient_retry_attempts < cap
                    if should_retry_same_gateway:
                        transient_retry_attempts += 1
                        delay_seconds = self._current_transient_retry_delay_seconds(transient_retry_attempts, status=status)
                        logger.warning(
                            "API pool transient HTTP %s on base_url=%s key=%s; retrying same gateway attempt %d/%d after %.1fs",
                            status,
                            base_url,
                            self._mask_key(api_key),
                            transient_retry_attempts,
                            cap,
                            delay_seconds,
                        )
                        if delay_seconds > 0:
                            await asyncio.sleep(delay_seconds)
                        continue

                    logger.error(
                        "API pool transient retries exhausted after HTTP %s on base_url=%s. Last key=%s detail=%s",
                        status,
                        base_url,
                        self._mask_key(api_key),
                        detail,
                    )
                    logger.warning(
                        "API pool non-stream transient HTTP %s on base_url=%s before a response; falling back to streamed collection",
                        status,
                        base_url,
                    )
                    try:
                        return await self._request_chat_completions_via_stream(
                            current_messages,
                            tools=tools,
                            stop=stop,
                            base_url=base_url,
                            api_key=api_key,
                        )
                    except Exception:
                        raise last_retryable_error from exc

                logger.error("API pool request failed (HTTP %s): %s", status, detail)
                raise RuntimeError(f"API pool request failed with HTTP {status}: {detail}") from exc
            except httpx.HTTPError as exc:
                last_retryable_error = RuntimeError(
                    f"API pool transient transport retries exhausted after {transient_retry_attempts + 1} attempts. "
                    f"Last error: {exc}"
                )
                should_rotate_gateway = len(self._base_urls) > 1 and gateway_rotation_attempts < len(self._base_urls) - 1
                if should_rotate_gateway:
                    gateway_rotation_attempts += 1
                    self._rotate_to_next_base_url()
                    transient_retry_attempts += 1
                    logger.warning(
                        "API pool transport error on base_url=%s key=%s; rotating to next gateway for attempt %d/%d: %s",
                        base_url,
                        self._mask_key(api_key),
                        transient_retry_attempts,
                        self._max_transient_retries,
                        exc,
                    )
                    continue

                should_retry_same_gateway = transient_retry_attempts < self._max_transient_retries
                if should_retry_same_gateway:
                    transient_retry_attempts += 1
                    logger.warning(
                        "API pool transport error on base_url=%s key=%s; retrying same gateway attempt %d/%d: %s",
                        base_url,
                        self._mask_key(api_key),
                        transient_retry_attempts,
                        self._max_transient_retries,
                        exc,
                    )
                    if self._transient_retry_delay_seconds > 0:
                        await asyncio.sleep(self._transient_retry_delay_seconds)
                    continue

                logger.warning(
                    "API pool non-stream transport failed on base_url=%s before a response; falling back to streamed collection: %s",
                    base_url,
                    exc,
                )
                try:
                    return await self._request_chat_completions_via_stream(
                        current_messages,
                        tools=tools,
                        stop=stop,
                        base_url=base_url,
                        api_key=api_key,
                    )
                except Exception:
                    logger.error("API pool transport retries exhausted on base_url=%s: %s", base_url, exc)
                    raise last_retryable_error from exc

        if last_retryable_error is not None:
            raise last_retryable_error
        raise RuntimeError("API pool request failed without a captured exception")

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        tools = kwargs.get("tools")
        if self._uses_responses_api():
            response = await self._request_responses(messages, tools=tools, stop=stop)
            chunk = self._stream_chunk_from_responses_response(response)
            if chunk is not None:
                yield chunk
                return
            raise RuntimeError("API pool responses request returned no assistant content")

        current_messages = messages
        payload = self._build_payload(current_messages, tools=tools, stop=stop, stream=True)
        max_key_rotations = self._current_gateway_key_count()
        key_rotation_attempts = 0
        transient_retry_attempts = 0
        gateway_rotation_attempts = 0
        last_retryable_error: RuntimeError | TimeoutError | None = None

        while True:
            base_url = self._current_base_url()
            url = f"{base_url}/chat/completions"
            api_key = self._current_api_key()
            headers = self._build_headers(api_key)
            yielded_partial_output = False

            try:
                async with self._create_http_client() as client:
                    async with client.stream("POST", url, headers=headers, json=payload) as response:
                        self._reset_gateway_rotation()
                        response_is_error = getattr(response, "is_error", None)
                        if response_is_error is None:
                            status_code = getattr(response, "status_code", None)
                            response_is_error = isinstance(status_code, int) and status_code >= 400
                        if response_is_error:
                            await response.aread()
                            response.raise_for_status()

                        async for line in response.aiter_lines():
                            if not line:
                                continue

                            data = self._parse_sse_data_line(line)
                            if data is None:
                                continue

                            chunk = self._stream_chunk_from_sse_event(data)
                            if chunk is None:
                                continue

                            yielded_partial_output = True
                            yield chunk
                return
            except httpx.TimeoutException as exc:
                if yielded_partial_output:
                    raise TimeoutError(
                        f"API pool streaming response timed out after partial output from {base_url}"
                    ) from exc

                last_retryable_error = TimeoutError(
                    f"API pool transient retries exhausted after {transient_retry_attempts + 1} attempts. "
                    f"Last error: request timed out after {self.timeout_seconds}s"
                )
                should_rotate_gateway = len(self._base_urls) > 1 and gateway_rotation_attempts < len(self._base_urls) - 1
                if should_rotate_gateway:
                    gateway_rotation_attempts += 1
                    self._rotate_to_next_base_url()
                    transient_retry_attempts += 1
                    logger.warning(
                        "API pool stream timed out on base_url=%s key=%s; rotating gateway for attempt %d/%d",
                        base_url,
                        self._mask_key(api_key),
                        transient_retry_attempts,
                        self._max_transient_retries,
                    )
                    continue

                should_retry_same_gateway = transient_retry_attempts < self._max_transient_retries
                if should_retry_same_gateway:
                    transient_retry_attempts += 1
                    logger.warning(
                        "API pool streaming timeout on base_url=%s key=%s; retrying same gateway attempt %d/%d",
                        base_url,
                        self._mask_key(api_key),
                        transient_retry_attempts,
                        self._max_transient_retries,
                    )
                    if self._transient_retry_delay_seconds > 0:
                        await asyncio.sleep(self._transient_retry_delay_seconds)
                    continue

                logger.warning(
                    "API pool streaming timed out on base_url=%s before any output; falling back to non-stream response",
                    base_url,
                )
                response = await self._request_chat_completions(current_messages, tools=tools, stop=stop)
                chunk = self._stream_chunk_from_response(response)
                if chunk is not None:
                    yield chunk
                    return
                logger.error("API pool streaming fallback returned no chunk after timeout")
                raise last_retryable_error from exc
            except httpx.HTTPStatusError as exc:
                try:
                    await exc.response.aread()
                except Exception:
                    logger.debug("Failed to pre-read streaming error response body", exc_info=True)
                status = exc.response.status_code
                detail = self._extract_error_message(exc.response)

                if self._is_upstream_gateway_block(status, detail):
                    logger.error(
                        "API pool upstream access denied by gateway/WAF on base_url=%s. key=%s detail=%s",
                        base_url,
                        self._mask_key(api_key),
                        detail,
                    )
                    raise RuntimeError(
                        "API pool upstream denied the current egress path or proxy "
                        f"(HTTP {status} on {exc.response.request.url.host}: {detail}). "
                        "This is not a DeerFlow or Docker container failure; switch to a different "
                        "API pool gateway or route this domain via a different network exit."
                    ) from exc

                if status in KEY_ROTATION_STATUS_CODES:
                    last_retryable_error = RuntimeError(
                        self._key_rotation_exhaustion_message(
                            status=status, detail=detail, max_key_rotations=max_key_rotations
                        )
                    )
                    if key_rotation_attempts < max_key_rotations - 1:
                        key_rotation_attempts += 1
                        transient_retry_attempts = 0
                        logger.warning(
                            "Key %s failed with %s on streaming base_url=%s, rotating to next key",
                            self._mask_key(api_key),
                            status,
                            base_url,
                        )
                        self._rotate_to_next_key()
                        continue

                    should_failover_gateway = len(self._base_urls) > 1 and gateway_rotation_attempts < len(self._base_urls) - 1
                    if should_failover_gateway:
                        gateway_rotation_attempts += 1
                        logger.warning(
                            "All keys exhausted on streaming base_url=%s; failing over to next gateway",
                            base_url,
                        )
                        self._rotate_to_next_base_url()
                        max_key_rotations = self._current_gateway_key_count()
                        key_rotation_attempts = 0
                        transient_retry_attempts = 0
                        continue

                    logger.error(
                        "All API pool keys were exhausted after HTTP %s on streaming base_url=%s. Last key=%s detail=%s",
                        status,
                        base_url,
                        self._mask_key(api_key),
                        detail,
                    )
                    raise last_retryable_error from exc

                if status in TRANSIENT_STATUS_CODES:
                    last_retryable_error = RuntimeError(
                        f"API pool transient retries exhausted after {transient_retry_attempts + 1} attempts. "
                        f"Last error HTTP {status}: {detail}"
                    )
                    compacted_messages, target = self._compact_messages_for_transient_recovery(current_messages)
                    if compacted_messages is not None:
                        logger.warning(
                            "API pool streaming transient HTTP %s on base_url=%s key=%s; compacting context to latest %d non-system messages before retry",
                            status,
                            base_url,
                            self._mask_key(api_key),
                            target,
                        )
                        current_messages = compacted_messages
                        payload = self._build_payload(current_messages, tools=tools, stop=stop, stream=True)
                        continue

                    should_rotate_gateway = len(self._base_urls) > 1 and gateway_rotation_attempts < len(self._base_urls) - 1
                    if should_rotate_gateway:
                        gateway_rotation_attempts += 1
                        self._rotate_to_next_base_url()
                        transient_retry_attempts += 1
                        logger.warning(
                            "API pool streaming transient HTTP %s on base_url=%s key=%s; rotating to next gateway for attempt %d/%d",
                            status,
                            base_url,
                            self._mask_key(api_key),
                            transient_retry_attempts,
                            self._max_transient_retries_for_status(status),
                        )
                        continue

                    cap = self._max_transient_retries_for_status(status)
                    should_retry_same_gateway = transient_retry_attempts < cap
                    if should_retry_same_gateway:
                        transient_retry_attempts += 1
                        delay_seconds = self._current_transient_retry_delay_seconds(transient_retry_attempts, status=status)
                        logger.warning(
                            "API pool streaming transient HTTP %s on base_url=%s key=%s; retrying same gateway attempt %d/%d after %.1fs",
                            status,
                            base_url,
                            self._mask_key(api_key),
                            transient_retry_attempts,
                            cap,
                            delay_seconds,
                        )
                        if delay_seconds > 0:
                            await asyncio.sleep(delay_seconds)
                        continue

                    logger.warning(
                        "API pool streaming transient HTTP %s on base_url=%s before any output; falling back to non-stream response",
                        status,
                        base_url,
                    )
                    response = await self._request_chat_completions(current_messages, tools=tools, stop=stop)
                    chunk = self._stream_chunk_from_response(response)
                    if chunk is not None:
                        yield chunk
                        return
                    logger.error("API pool streaming fallback returned no chunk after HTTP %s", status)
                    raise last_retryable_error from exc

                logger.error("API pool streaming request failed (HTTP %s): %s", status, detail)
                raise RuntimeError(f"API pool request failed with HTTP {status}: {detail}") from exc
            except httpx.HTTPError as exc:
                if yielded_partial_output:
                    raise RuntimeError(f"API pool streaming transport error after partial output: {exc}") from exc

                last_retryable_error = RuntimeError(
                    f"API pool transient transport retries exhausted after {transient_retry_attempts + 1} attempts. "
                    f"Last error: {exc}"
                )
                should_rotate_gateway = len(self._base_urls) > 1 and gateway_rotation_attempts < len(self._base_urls) - 1
                if should_rotate_gateway:
                    gateway_rotation_attempts += 1
                    self._rotate_to_next_base_url()
                    transient_retry_attempts += 1
                    logger.warning(
                        "API pool streaming transport error on base_url=%s key=%s; rotating to next gateway for attempt %d/%d: %s",
                        base_url,
                        self._mask_key(api_key),
                        transient_retry_attempts,
                        self._max_transient_retries,
                        exc,
                    )
                    continue

                should_retry_same_gateway = transient_retry_attempts < self._max_transient_retries
                if should_retry_same_gateway:
                    transient_retry_attempts += 1
                    logger.warning(
                        "API pool streaming transport error on base_url=%s key=%s; retrying same gateway attempt %d/%d: %s",
                        base_url,
                        self._mask_key(api_key),
                        transient_retry_attempts,
                        self._max_transient_retries,
                        exc,
                    )
                    if self._transient_retry_delay_seconds > 0:
                        await asyncio.sleep(self._transient_retry_delay_seconds)
                    continue

                logger.warning(
                    "API pool streaming transport failed on base_url=%s before any output; falling back to non-stream response: %s",
                    base_url,
                    exc,
                )
                response = await self._request_chat_completions(messages, tools=tools, stop=stop)
                chunk = self._stream_chunk_from_response(response)
                if chunk is not None:
                    yield chunk
                    return
                logger.error("API pool streaming fallback returned no chunk after transport error")
                raise last_retryable_error from exc

        if last_retryable_error is not None:
            raise last_retryable_error
        raise RuntimeError("API pool streaming request failed without a captured exception")

    def _parse_tool_calls(self, message: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        tool_calls: list[dict[str, Any]] = []
        invalid_tool_calls: list[dict[str, Any]] = []

        for tool_call in message.get("tool_calls", []) or []:
            function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
            raw_arguments = function.get("arguments", "{}")
            try:
                parsed_arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            except json.JSONDecodeError as exc:
                invalid_tool_calls.append(
                    {
                        "type": "invalid_tool_call",
                        "name": function.get("name"),
                        "args": str(raw_arguments),
                        "id": tool_call.get("id"),
                        "error": f"Failed to parse tool arguments: {exc}",
                    }
                )
                continue

            tool_calls.append(
                {
                    "name": function.get("name", ""),
                    "args": parsed_arguments if isinstance(parsed_arguments, dict) else {},
                    "id": tool_call.get("id", ""),
                    "type": "tool_call",
                }
            )

        return tool_calls, invalid_tool_calls


    def _parse_response(self, response: dict[str, Any]) -> ChatResult:
        if self._uses_responses_api():
            usage = response.get("usage", {})
            content = self._extract_responses_content(response)
            cleaned_content, inline_reasoning = self._strip_inline_think_tags(content)
            tool_calls, invalid_tool_calls = self._parse_responses_tool_calls(response)
            reasoning_content = self._merge_reasoning(
                self._extract_responses_reasoning(response),
                inline_reasoning,
            )

            additional_kwargs: dict[str, Any] = {}
            if reasoning_content:
                additional_kwargs["reasoning_content"] = reasoning_content

            ai_message = AIMessage(
                content=cleaned_content,
                tool_calls=tool_calls,
                invalid_tool_calls=invalid_tool_calls,
                additional_kwargs=additional_kwargs,
                response_metadata={
                    "model": response.get("model", self.model),
                    "usage": usage,
                    "finish_reason": response.get("status"),
                },
            )

            return ChatResult(
                generations=[ChatGeneration(message=ai_message)],
                llm_output={
                    "token_usage": {
                        "prompt_tokens": usage.get("input_tokens", 0),
                        "completion_tokens": usage.get("output_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    },
                    "model_name": response.get("model", self.model),
                },
            )

        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message", {}) if isinstance(choice, dict) else {}

        content = self._normalize_content(message.get("content", ""))
        cleaned_content, inline_reasoning = self._strip_inline_think_tags(content)
        tool_calls, invalid_tool_calls = self._parse_tool_calls(message)
        usage = response.get("usage", {})
        reasoning_content = self._merge_reasoning(
            self._extract_reasoning_text(message.get("reasoning_content")),
            self._extract_reasoning_text(message.get("reasoning")),
            self._extract_reasoning_text(message.get("reasoning_details")),
            self._extract_reasoning_text(response.get("reasoning")),
            inline_reasoning,
        )

        additional_kwargs: dict[str, Any] = {}
        if reasoning_content:
            additional_kwargs["reasoning_content"] = reasoning_content

        ai_message = AIMessage(
            content=cleaned_content,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
            additional_kwargs=additional_kwargs,
            response_metadata={
                "model": response.get("model", self.model),
                "usage": usage,
                "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
            },
        )

        return ChatResult(
            generations=[ChatGeneration(message=ai_message)],
            llm_output={
                "token_usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                "model_name": response.get("model", self.model),
            },
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools = kwargs.get("tools")
        response = asyncio.run(self._request_chat_completions(messages, tools=tools, stop=stop))
        return self._parse_response(response)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools = kwargs.get("tools")
        response = await self._request_chat_completions(messages, tools=tools, stop=stop)
        return self._parse_response(response)

    def bind_tools(self, tools: list, **kwargs: Any) -> Any:
        from langchain_core.runnables import RunnableBinding
        from langchain_core.tools import BaseTool
        from langchain_core.utils.function_calling import convert_to_openai_function

        formatted_tools = []
        for tool in tools:
            if isinstance(tool, BaseTool):
                try:
                    fn = convert_to_openai_function(tool)
                    formatted_tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": fn["name"],
                                "description": fn.get("description", ""),
                                "parameters": fn.get("parameters", {}),
                            },
                        }
                    )
                except Exception:
                    formatted_tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": {"type": "object", "properties": {}},
                            },
                        }
                    )
            elif isinstance(tool, dict):
                if "function" in tool:
                    formatted_tools.append(tool)
                elif "name" in tool:
                    formatted_tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": tool["name"],
                                "description": tool.get("description", ""),
                                "parameters": tool.get("parameters", {}),
                            },
                        }
                    )

        return RunnableBinding(bound=self, kwargs={"tools": formatted_tools}, **kwargs)

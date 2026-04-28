"""Custom Claude provider using Claude Code CLI directly since OAuth is blocked by API."""

import logging
import subprocess
import json
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

logger = logging.getLogger(__name__)

class ClaudeCliChatModel(BaseChatModel):
    """LangChain chat model using official Claude Code CLI underneath."""

    model: str = "claude-cli-4-6"

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "claude-cli"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Build prompt from messages
        prompt_parts = []
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if isinstance(msg, SystemMessage):
                prompt_parts.append(f"System: {content}")
            elif isinstance(msg, HumanMessage):
                prompt_parts.append(f"User: {content}")
            elif isinstance(msg, AIMessage):
                prompt_parts.append(f"Assistant: {content}")

        full_prompt = "\n".join(prompt_parts)

        # Execute claude-code CLI
        logger.info("Executing claude-code via CLI provider...")
        try:
            result = subprocess.run(
                ["npx", "-y", "@anthropic-ai/claude-code", "-p", full_prompt],
                capture_output=True,
                text=True,
                timeout=300
            )
            output = result.stdout.strip()
            if not output:
                output = f"CLI Error: {result.stderr.strip()}"
        except Exception as e:
            output = f"Execution exception: {str(e)}"

        message = AIMessage(content=output)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: list, **kwargs: Any) -> Any:
        return self

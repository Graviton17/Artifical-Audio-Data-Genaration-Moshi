"""OpenAI implementation of :class:`BaseLLM`.

Uses the ``openai`` SDK (``pip install openai``), which exposes the native
chat-completions API. The API key is read from the ``api_key`` argument or
the ``OPENAI_API_KEY`` environment variable.
"""

from __future__ import annotations

import os
from typing import Any

from .base_llm import BaseLLM, LLMError, LLMResponse, Message

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised only without the dep
    OpenAI = None


class OpenAILLM(BaseLLM):
    """Chat completions backed by OpenAI's hosted models."""

    def __init__(
        self,
        model: str = "gpt-4.1",
        *,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ) -> None:
        if OpenAI is None:
            raise ImportError("openai is not installed. Run `pip install openai`.")
        super().__init__(
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise LLMError("No OpenAI API key found. Pass api_key= or set OPENAI_API_KEY.")
        self._client = OpenAI(api_key=key)

    def _complete(self, messages: list[Message], **overrides: Any) -> LLMResponse:
        params = self._resolved(overrides)

        kwargs: dict[str, Any] = {
            "model": self.model,
            # OpenAI uses {"role", "content"} directly; "system" role is native.
            "messages": messages,
            "temperature": params["temperature"],
        }
        if params["max_tokens"] is not None:
            kwargs["max_tokens"] = params["max_tokens"]
        if overrides.get("response_format"):
            kwargs["response_format"] = overrides["response_format"]

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        return LLMResponse(
            text=choice.message.content or "",
            model=self.model,
            usage=self._usage(response),
            raw=response,
        )

    @staticmethod
    def _usage(response: Any) -> dict[str, int]:
        usage = getattr(response, "usage", None)
        if not usage:
            return {}
        return {
            "input": getattr(usage, "prompt_tokens", 0) or 0,
            "output": getattr(usage, "completion_tokens", 0) or 0,
        }

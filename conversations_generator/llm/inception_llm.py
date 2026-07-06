"""Inception Labs implementation of :class:`BaseLLM`.

Calls the OpenAI-style ``/v1/chat/completions`` endpoint at
``https://api.inceptionlabs.ai`` directly via ``requests`` (Inception Labs has
no dedicated Python SDK). The API key is read from the ``api_key`` argument or
``INCEPTION_API_KEY`` in ``conversations_generator/config.json``.
"""

from __future__ import annotations

from typing import Any, Iterator

import requests

from ..configuration_reader import get as config_get
from .base_llm import BaseLLM, LLMError, LLMResponse, Message

_API_URL = "https://api.inceptionlabs.ai/v1/chat/completions"


class InceptionLLM(BaseLLM):
    """Chat completions backed by Inception Labs's hosted models (e.g. Mercury)."""

    def __init__(
        self,
        model: str = "mercury-2",
        *,
        api_key: str | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        reasoning_effort: str | None = "low",
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        timeout: float = 120.0,
    ) -> None:
        super().__init__(
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        key = api_key or config_get("INCEPTION_API_KEY")
        if not key:
            raise LLMError(
                "No Inception Labs API key found. Pass api_key= or set "
                "INCEPTION_API_KEY in conversations_generator/config.json."
            )
        self._headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

    def _complete(self, messages: list[Message], **overrides: Any) -> LLMResponse:
        params = self._resolved(overrides)

        payload: dict[str, Any] = {
            "model": self.model,
            # Inception/OpenAI use {"role", "content"} directly; "system" role is native.
            "messages": messages,
            "temperature": params["temperature"],
        }
        if params["max_tokens"] is not None:
            payload["max_tokens"] = params["max_tokens"]
        reasoning_effort = overrides.get("reasoning_effort", self.reasoning_effort)
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        if overrides.get("response_format"):
            payload["response_format"] = overrides["response_format"]

        response = requests.post(
            _API_URL,
            headers=self._headers,
            json=payload,
            timeout=self.timeout,
        )
        if not response.ok:
            raise LLMError(
                f"Inception Labs API request failed ({response.status_code}): {response.text}"
            )
        data = response.json()
        choice = data["choices"][0]

        return LLMResponse(
            text=choice["message"]["content"] or "",
            model=self.model,
            usage=self._usage(data),
            raw=data,
        )

    def _complete_stream(self, messages: list[Message], **overrides: Any) -> Iterator[str]:
        params = self._resolved(overrides)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": params["temperature"],
            "stream": True,
        }
        if params["max_tokens"] is not None:
            payload["max_tokens"] = params["max_tokens"]
        reasoning_effort = overrides.get("reasoning_effort", self.reasoning_effort)
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        if overrides.get("response_format"):
            payload["response_format"] = overrides["response_format"]

        response = requests.post(
            _API_URL,
            headers=self._headers,
            json=payload,
            timeout=self.timeout,
            stream=True,
        )
        if not response.ok:
            raise LLMError(
                f"Inception Labs API request failed ({response.status_code}): {response.text}"
            )
        # The SSE stream carries UTF-8, but the server sends no charset, so
        # requests would otherwise guess ISO-8859-1 and mangle Devanagari.
        response.encoding = "utf-8"
        yield from self._iter_sse_content_tracked(
            response.iter_lines(decode_unicode=True)
        )

    @staticmethod
    def _usage(data: dict[str, Any]) -> dict[str, int]:
        usage = data.get("usage")
        if not usage:
            return {}
        return {
            "input": usage.get("prompt_tokens", 0) or 0,
            "output": usage.get("completion_tokens", 0) or 0,
        }

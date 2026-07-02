"""Krutrim Cloud implementation of :class:`BaseLLM`.

Calls the OpenAI-style ``/v1/chat/completions`` endpoint at
``https://cloud.olakrutrim.com`` directly via ``requests`` (Krutrim has no
dedicated Python SDK). The API key is read from the ``api_key`` argument or
the ``KRUTRIM_API_KEY`` environment variable.
"""

from __future__ import annotations

import os
from typing import Any

import requests

from .base_llm import BaseLLM, LLMError, LLMResponse, Message

_API_URL = "https://cloud.olakrutrim.com/v1/chat/completions"


class KrutrimLLM(BaseLLM):
    """Chat completions backed by Krutrim Cloud's hosted models (e.g. Gemma-4)."""

    def __init__(
        self,
        model: str = "gemma-4-26B-A4B-it",
        *,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
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
        self.timeout = timeout
        key = api_key or os.getenv("KRUTRIM_API_KEY")
        if not key:
            raise LLMError(
                "No Krutrim API key found. Pass api_key= or set KRUTRIM_API_KEY."
            )
        self._headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

    def _complete(self, messages: list[Message], **overrides: Any) -> LLMResponse:
        params = self._resolved(overrides)

        payload: dict[str, Any] = {
            "model": self.model,
            # Krutrim/OpenAI use {"role", "content"} directly; "system" role is native.
            "messages": messages,
            "temperature": params["temperature"],
        }
        if params["max_tokens"] is not None:
            payload["max_tokens"] = params["max_tokens"]
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
                f"Krutrim API request failed ({response.status_code}): {response.text}"
            )
        data = response.json()
        choice = data["choices"][0]

        return LLMResponse(
            text=choice["message"]["content"] or "",
            model=self.model,
            usage=self._usage(data),
            raw=data,
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

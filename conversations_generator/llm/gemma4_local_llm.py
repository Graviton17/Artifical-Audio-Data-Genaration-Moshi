"""Self-hosted Gemma-4 implementation of :class:`BaseLLM`.

Talks to a local ``llama-server`` (OpenAI-compatible ``/v1/chat/completions``),
exposed publicly through an ngrok tunnel, directly via ``requests``. Unlike the
hosted providers, the base URL is **configurable** — the free-tier ngrok tunnel
mints a new URL every restart — so it's read from ``GEMMA4_LOCAL_BASE_URL`` in
``conversations_generator/config.json`` (falling back to the ``base_url``
argument). The bearer token comes from ``GEMMA4_LOCAL_API_KEY``.

Gemma-4 is a *reasoning* model: it spends tokens thinking before answering, so
``max_tokens`` defaults generously here — too small a budget truncates the reply
mid-thought (``finish_reason: "length"`` with no final answer).
"""

from __future__ import annotations

from typing import Any, Iterator

import requests

from ..configuration_reader import get as config_get
from .base_llm import BaseLLM, LLMError, LLMResponse, Message

# Used only if neither the base_url argument nor config supplies one. This
# ngrok URL is ephemeral and will be wrong after the tunnel restarts — set
# GEMMA4_LOCAL_BASE_URL in config.json to the current URL.
_DEFAULT_BASE_URL = "https://squirarchal-attemptable-mammie.ngrok-free.dev"
_COMPLETIONS_PATH = "/v1/chat/completions"


class Gemma4LocalLLM(BaseLLM):
    """Chat completions backed by a self-hosted Gemma-4 ``llama-server``."""

    def __init__(
        self,
        model: str = "gemma-4",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = 2048,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        timeout: float = 300.0,
    ) -> None:
        super().__init__(
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )
        self.timeout = timeout

        base = (base_url or config_get("GEMMA4_LOCAL_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")
        self._api_url = f"{base}{_COMPLETIONS_PATH}"

        key = api_key or config_get("GEMMA4_LOCAL_API_KEY")
        if not key:
            raise LLMError(
                "No Gemma-4 local API key found. Pass api_key= or set "
                "GEMMA4_LOCAL_API_KEY in conversations_generator/config.json."
            )
        self._headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            # ngrok's free tier serves a browser interstitial unless this header
            # is present; skip it so the API responds with JSON, not HTML.
            "ngrok-skip-browser-warning": "true",
        }

    def _complete(self, messages: list[Message], **overrides: Any) -> LLMResponse:
        params = self._resolved(overrides)

        payload: dict[str, Any] = {
            "model": self.model,
            # llama-server/OpenAI use {"role", "content"} directly; "system" role is native.
            "messages": messages,
            "temperature": params["temperature"],
        }
        if params["max_tokens"] is not None:
            payload["max_tokens"] = params["max_tokens"]
        if overrides.get("response_format"):
            payload["response_format"] = overrides["response_format"]

        response = requests.post(
            self._api_url,
            headers=self._headers,
            json=payload,
            timeout=self.timeout,
        )
        if not response.ok:
            raise LLMError(
                f"Gemma-4 local API request failed ({response.status_code}): {response.text}"
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
        if overrides.get("response_format"):
            payload["response_format"] = overrides["response_format"]

        response = requests.post(
            self._api_url,
            headers=self._headers,
            json=payload,
            timeout=self.timeout,
            stream=True,
        )
        if not response.ok:
            raise LLMError(
                f"Gemma-4 local API request failed ({response.status_code}): {response.text}"
            )
        # The SSE stream carries UTF-8, but the server sends no charset, so
        # requests would otherwise guess ISO-8859-1 and mangle non-ASCII text.
        response.encoding = "utf-8"
        yield from self._iter_sse_content(
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

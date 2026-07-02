"""Google Gemini implementation of :class:`BaseLLM`.

Uses the unified ``google-genai`` SDK (``pip install google-genai``). The API key is
read from the ``api_key`` argument or the ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``
environment variables.
"""

from __future__ import annotations

import os
from typing import Any

from .base_llm import BaseLLM, LLMError, LLMResponse, Message

try:  # Imported lazily-ish so the base package works without the SDK installed.
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - exercised only without the dep
    genai = None
    genai_types = None


class GeminiLLM(BaseLLM):
    """Chat completions backed by Google Gemini."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        *,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = 65536,
        thinking_budget: int | None = 8192,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        timeout: float = 600.0,
    ) -> None:
        if genai is None:
            raise ImportError(
                "google-genai is not installed. Run `pip install google-genai`."
            )
        super().__init__(
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )
        self.thinking_budget = thinking_budget
        self.timeout = timeout
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise LLMError(
                "No Gemini API key found. Pass api_key= or set GEMINI_API_KEY."
            )
        self._client = genai.Client(api_key=key)

    def _complete(self, messages: list[Message], **overrides: Any) -> LLMResponse:
        params = self._resolved(overrides)
        system_instruction, contents = self._split_messages(messages)

        mime_type = overrides.get("response_mime_type")
        if not mime_type and overrides.get("response_format", {}).get("type") == "json_object":
            mime_type = "application/json"

        # Cap Gemini's internal "thinking" phase so it doesn't spin forever.
        thinking_config = None
        if self.thinking_budget is not None:
            thinking_config = genai_types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
            )

        config = genai_types.GenerateContentConfig(
            temperature=params["temperature"],
            max_output_tokens=params["max_tokens"],
            system_instruction=system_instruction or None,
            # Ask Gemini to emit raw JSON when the caller wants it (generate_json).
            response_mime_type=mime_type,
            thinking_config=thinking_config,
        )

        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        return LLMResponse(
            text=response.text or "",
            model=self.model,
            usage=self._usage(response),
            raw=response,
        )

    # ------------------------------------------------------------------ #
    # Message translation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_messages(messages: list[Message]) -> tuple[str, list[Any]]:
        """Split our generic messages into (system_instruction, gemini_contents).

        Gemini takes system text separately and uses "user"/"model" roles for turns.
        """
        system_parts: list[str] = []
        contents: list[Any] = []
        for msg in messages:
            role = msg["role"]
            text = msg["content"]
            if role == "system":
                system_parts.append(text)
            else:
                gemini_role = "model" if role == "assistant" else "user"
                contents.append(
                    genai_types.Content(
                        role=gemini_role,
                        parts=[genai_types.Part.from_text(text=text)],
                    )
                )
        return "\n\n".join(system_parts), contents

    @staticmethod
    def _usage(response: Any) -> dict[str, int]:
        meta = getattr(response, "usage_metadata", None)
        if not meta:
            return {}
        return {
            "input": getattr(meta, "prompt_token_count", 0) or 0,
            "output": getattr(meta, "candidates_token_count", 0) or 0,
        }

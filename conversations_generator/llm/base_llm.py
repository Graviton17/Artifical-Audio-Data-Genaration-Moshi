"""Provider-agnostic LLM interface used by the conversation-generation agents.

Agents depend only on :class:`BaseLLM`; concrete providers (Gemini, etc.) live in
sibling modules and are selected at construction time. This keeps agent logic free
of any vendor SDK details.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# A chat message. `role` is one of "system", "user", "assistant".
Message = dict[str, str]


@dataclass
class LLMResponse:
    """Normalized result returned by every provider."""

    text: str
    model: str
    # Token counts if the provider reports them: {"input": int, "output": int}.
    usage: dict[str, int] = field(default_factory=dict)
    # The untouched provider response, for debugging / provider-specific needs.
    raw: Any = None


class LLMError(RuntimeError):
    """Raised when a provider call fails after exhausting retries."""


class BaseLLM(ABC):
    """Base class for chat-completion LLMs.

    Subclasses implement :meth:`_complete`, which performs a single call against
    the provider. The public methods here add retries, a simple system-prompt
    convenience, and JSON parsing on top of that primitive.
    """

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    # ------------------------------------------------------------------ #
    # Public API used by agents
    # ------------------------------------------------------------------ #
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        **overrides: Any,
    ) -> str:
        """Single-turn helper: send one user prompt, return the text reply."""
        messages: list[Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **overrides).text

    def chat(self, messages: list[Message], **overrides: Any) -> LLMResponse:
        """Multi-turn completion with retry/backoff around the provider call."""
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._complete(messages, **overrides)
            except Exception as err:  # noqa: BLE001 - normalized into LLMError below
                last_err = err
                if attempt == self.max_retries:
                    break
                time.sleep(self.retry_backoff ** (attempt - 1))
        raise LLMError(
            f"{type(self).__name__} failed after {self.max_retries} attempts: {last_err}"
        ) from last_err

    def generate_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        **overrides: Any,
    ) -> Any:
        """Like :meth:`generate` but parse the reply as JSON.

        Tolerates responses wrapped in ```json ... ``` fences.
        """
        text = self.generate(prompt, system=system, **overrides)
        return self._parse_json(text)

    # ------------------------------------------------------------------ #
    # Subclass contract
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _complete(self, messages: list[Message], **overrides: Any) -> LLMResponse:
        """Perform one provider call. Overrides may include `temperature`,
        `max_tokens`, or provider-specific keys."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _resolved(self, overrides: dict[str, Any]) -> dict[str, Any]:
        """Merge per-call overrides over instance defaults."""
        return {
            "temperature": overrides.get("temperature", self.temperature),
            "max_tokens": overrides.get("max_tokens", self.max_tokens),
        }

    @staticmethod
    def _parse_json(text: str) -> Any:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            # Strip a leading ```json / ``` fence and the trailing ```.
            cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned
            cleaned = cleaned.rsplit("```", 1)[0].strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as err:
            raise LLMError(f"Model did not return valid JSON: {err}\n---\n{text}") from err

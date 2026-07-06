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
from typing import Any, Callable, Iterator


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
        temperature: float = 0.3,
        max_tokens: int | None = None,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._last_usage: dict[str, int] = {}
        self._last_model: str = model

    @property
    def last_usage(self) -> dict[str, int]:
        """Token counts from the most recent successful provider call."""
        return dict(self._last_usage)

    def _store_response(self, response: LLMResponse) -> LLMResponse:
        """Remember usage/model from a provider response for downstream tracking."""
        self._last_usage = dict(response.usage or {})
        self._last_model = response.model or self.model
        return response

    # ------------------------------------------------------------------ #
    # Public API used by agents
    # ------------------------------------------------------------------ #
    def generate(
        self,
        prompt: str,
        **overrides: Any,
    ) -> str:
        """Single-turn generation: prompt → text reply.

        The caller is responsible for combining any system instructions and
        dynamic content into ``prompt`` before calling this method.
        Retries with exponential backoff on transient provider errors.
        """
        messages: list[Message] = [{"role": "user", "content": prompt}]

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._store_response(self._complete(messages, **overrides)).text
            except Exception as err:  # noqa: BLE001 - normalized into LLMError below
                last_err = err
                if attempt == self.max_retries:
                    break
                time.sleep(self.retry_backoff ** (attempt - 1))
        raise LLMError(
            f"{type(self).__name__} failed after {self.max_retries} attempts: {last_err}"
        ) from last_err

    def generate_stream(
        self,
        prompt: str,
        on_chunk: Callable[[str], None] | None = None,
        **overrides: Any,
    ) -> str:
        """Like :meth:`generate`, but streams text chunks as they arrive.

        ``on_chunk`` is called with each incremental piece of text the
        provider emits (e.g. to print it live to the terminal). Returns the
        full concatenated text once the stream ends. Retries the whole call
        on transient failures, same as :meth:`generate`.
        """
        messages: list[Message] = [{"role": "user", "content": prompt}]

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                chunks: list[str] = []
                for chunk in self._complete_stream(messages, **overrides):
                    if not chunk:
                        continue
                    chunks.append(chunk)
                    if on_chunk is not None:
                        on_chunk(chunk)
                text = "".join(chunks)
                # Streaming providers often omit usage; keep prior counts if unset.
                if not self._last_usage:
                    self._last_model = self.model
                return text
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
        **overrides: Any,
    ) -> Any:
        """Like :meth:`generate` but parse the reply as JSON.

        Tolerates responses wrapped in ```json ... ``` fences. Malformed JSON
        (the model occasionally emits a garbled/truncated object) is retried
        with the same backoff as transport-level failures, since it's just as
        transient — a fresh call to the same model+prompt usually parses fine.
        """
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            text = self.generate(prompt, **overrides)
            try:
                return self._parse_json(text)
            except LLMError as err:
                last_err = err
                if attempt == self.max_retries:
                    break
                time.sleep(self.retry_backoff ** (attempt - 1))
        raise last_err  # type: ignore[misc]

    def generate_json_stream(
        self,
        prompt: str,
        on_chunk: Callable[[str], None] | None = None,
        **overrides: Any,
    ) -> Any:
        """Like :meth:`generate_json`, but streams the raw text live as it arrives.

        Useful so JSON-mode calls (e.g. topic generation) still show visible,
        real-time progress in the terminal instead of going silent until the
        whole response lands.
        """
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            text = self.generate_stream(prompt, on_chunk=on_chunk, **overrides)
            try:
                return self._parse_json(text)
            except LLMError as err:
                last_err = err
                if attempt == self.max_retries:
                    break
                time.sleep(self.retry_backoff ** (attempt - 1))
        raise last_err  # type: ignore[misc]

    # ------------------------------------------------------------------ #
    # Subclass contract
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _complete(self, messages: list[Message], **overrides: Any) -> LLMResponse:
        """Perform one provider call. Overrides may include `temperature`,
        `max_tokens`, or provider-specific keys."""
        raise NotImplementedError

    def _complete_stream(self, messages: list[Message], **overrides: Any) -> Iterator[str]:
        """Perform one provider call, yielding text chunks as they arrive.

        Providers that support real streaming override this. The default
        falls back to a single non-streamed call and yields the whole text
        as one chunk, so streaming is always safe to call.
        """
        yield self._complete(messages, **overrides).text

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
    def _iter_sse_content(lines: Iterator[bytes | str]) -> Iterator[str]:
        """Parse an OpenAI-style ``text/event-stream`` body into content deltas.

        Used by the REST-based providers (Krutrim, Sarvam, Inception Labs) that
        have no SDK and stream raw ``data: {...}`` / ``data: [DONE]`` lines.

        Lines must be **raw bytes** (``response.iter_lines()`` with no
        ``decode_unicode``), decoded here explicitly as UTF-8. ``requests``'
        ``decode_unicode=True`` decodes using its guessed/apparent encoding —
        which for these APIs is often not UTF-8 — and silently mangles
        non-ASCII text (e.g. Devanagari) into mojibake.
        """
        for line in lines:
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="replace")
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = event.get("choices") or []
            if choices:
                delta = (choices[0].get("delta") or {}).get("content")
                if delta:
                    yield delta

    def _iter_sse_content_tracked(
        self,
        lines: Iterator[bytes | str],
    ) -> Iterator[str]:
        """Like :meth:`_iter_sse_content`, but records ``usage`` on ``self``."""
        for line in lines:
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="replace")
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            usage = event.get("usage")
            if usage:
                self._last_usage = {
                    "input": usage.get("prompt_tokens", 0) or 0,
                    "output": usage.get("completion_tokens", 0) or 0,
                }
            choices = event.get("choices") or []
            if choices:
                delta = (choices[0].get("delta") or {}).get("content")
                if delta:
                    yield delta

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

"""Provider-agnostic LLM interface used by the conversation-generation agents.

Agents depend only on :class:`BaseLLM`; concrete providers (Gemini, etc.) live in
sibling modules and are selected at construction time. This keeps agent logic free
of any vendor SDK details.
"""

from __future__ import annotations

import json
import threading
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


@dataclass
class _ModelUsage:
    """Running token totals for a single model."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
        }


class TokenUsageTracker:
    """Process-wide accumulator of token usage, keyed by model name.

    Every :class:`BaseLLM` call records the provider-reported ``usage`` here, so
    the runner can dump a per-model token summary at the end of a run. Streaming
    calls that don't report usage simply contribute nothing.
    """

    def __init__(self) -> None:
        self._by_model: dict[str, _ModelUsage] = {}
        # Guards the dict against concurrent updates from parallel workers.
        self._lock = threading.Lock()

    def record(self, model: str, usage: dict[str, int] | None) -> None:
        if not usage:
            return
        with self._lock:
            entry = self._by_model.setdefault(model, _ModelUsage())
            entry.input_tokens += int(usage.get("input", 0) or 0)
            entry.output_tokens += int(usage.get("output", 0) or 0)
            entry.calls += 1

    def as_dict(self) -> dict[str, Any]:
        """Serializable summary: per-model totals plus a grand total."""
        with self._lock:
            models = {model: usage.as_dict() for model, usage in self._by_model.items()}
            total = {
                "input_tokens": sum(u.input_tokens for u in self._by_model.values()),
                "output_tokens": sum(u.output_tokens for u in self._by_model.values()),
                "total_tokens": sum(u.total_tokens for u in self._by_model.values()),
                "calls": sum(u.calls for u in self._by_model.values()),
            }
        return {"models": models, "total": total}

    def reset(self) -> None:
        with self._lock:
            self._by_model.clear()


# Single shared tracker for the whole process. Agents call through BaseLLM, which
# records into this; the runner reads it to write output/metadata.json.
TOKEN_USAGE = TokenUsageTracker()


class LLMError(RuntimeError):
    """Raised when a provider call fails after exhausting retries."""


class APILimitError(RuntimeError):
    """Raised when a provider returns a rate-limit or quota error.

    Not a subclass of :class:`LLMError` so pipeline retry loops that catch
    transient LLM failures do not swallow it — the runner terminates and saves
    the checkpoint instead.
    """


def is_api_limit_error(err: Exception) -> bool:
    """Return True when ``err`` indicates an API rate-limit or quota exhaustion."""
    if isinstance(err, APILimitError):
        return True

    for attr in ("status_code", "status"):
        code = getattr(err, attr, None)
        if code == 429:
            return True

    err_type = type(err).__name__.lower()
    if "ratelimit" in err_type or "rate_limit" in err_type:
        return True

    msg = str(err).lower()
    if " 429" in msg or "(429)" in msg or "status_code=429" in msg:
        return True

    keywords = (
        "rate limit",
        "rate_limit",
        "ratelimit",
        "too many requests",
        "quota exceeded",
        "insufficient_quota",
        "resource exhausted",
        "resource_exhausted",
        "billing hard limit",
        "exceeded your current quota",
    )
    return any(keyword in msg for keyword in keywords)


def _reraise_if_api_limit(err: Exception) -> None:
    """Re-raise API-limit errors immediately (no further retries)."""
    if is_api_limit_error(err):
        if isinstance(err, APILimitError):
            raise err
        raise APILimitError(str(err)) from err


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
        # Token usage from the most recent streaming call. Providers populate this
        # inside ``_complete_stream`` (the streamed text yields no return value),
        # and ``generate_stream`` records it into ``TOKEN_USAGE`` afterwards.
        # Backed by thread-local storage so parallel workers sharing one LLM
        # instance don't clobber each other's in-flight usage.
        self._stream_usage_local = threading.local()

    @property
    def _last_stream_usage(self) -> dict[str, int]:
        return getattr(self._stream_usage_local, "value", {}) or {}

    @_last_stream_usage.setter
    def _last_stream_usage(self, value: dict[str, int]) -> None:
        self._stream_usage_local.value = value

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
                response = self._complete(messages, **overrides)
                TOKEN_USAGE.record(response.model or self.model, response.usage)
                return response.text
            except Exception as err:  # noqa: BLE001 - normalized into LLMError below
                _reraise_if_api_limit(err)
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
                # Reset so a stream that reports no usage doesn't inherit the
                # previous call's numbers.
                self._last_stream_usage = {}
                for chunk in self._complete_stream(messages, **overrides):
                    if not chunk:
                        continue
                    chunks.append(chunk)
                    if on_chunk is not None:
                        on_chunk(chunk)
                # Record whatever usage the provider captured during the stream.
                TOKEN_USAGE.record(self.model, self._last_stream_usage)
                return "".join(chunks)
            except Exception as err:  # noqa: BLE001 - normalized into LLMError below
                _reraise_if_api_limit(err)
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
        response = self._complete(messages, **overrides)
        TOKEN_USAGE.record(response.model or self.model, response.usage)
        yield response.text

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
            if not choices:
                continue
            delta = (choices[0].get("delta") or {}).get("content")
            if delta:
                yield delta

    def _iter_sse_content_with_usage(self, lines: Iterator[bytes | str]) -> Iterator[str]:
        """Like :meth:`_iter_sse_content`, but also capture the stream's usage.

        OpenAI-style streams (Sarvam, Krutrim, Inception) emit a trailing
        ``data: {... "usage": {...}}`` event when the request asks for it
        (``stream_options={"include_usage": true}``). This records that usage
        into ``self._last_stream_usage`` so streamed calls count toward the
        per-model token totals, while yielding content deltas exactly as before.
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
            usage = event.get("usage")
            if usage:
                self._last_stream_usage = {
                    "input": usage.get("prompt_tokens", 0) or 0,
                    "output": usage.get("completion_tokens", 0) or 0,
                }
            choices = event.get("choices") or []
            if not choices:
                continue
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

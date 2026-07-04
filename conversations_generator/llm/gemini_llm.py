"""Google Gemini implementation of :class:`BaseLLM`.

Uses the unified ``google-genai`` SDK (``pip install google-genai``). The API key is
read from the ``api_key`` argument or ``GEMINI_API_KEY`` /
``GOOGLE_API_KEY`` in ``conversations_generator/config.json``.
"""

from __future__ import annotations

from typing import Any, Literal

from ..configuration_reader import get as config_get
from .base_llm import BaseLLM, LLMError, LLMResponse, Message

try:  # Imported lazily-ish so the base package works without the SDK installed.
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - exercised only without the dep
    genai = None
    genai_types = None


ThinkingLevel = Literal["minimal", "low", "medium", "high"]


class GeminiLLM(BaseLLM):
    """Chat completions backed by Google Gemini."""

    def __init__(
        self,
        model: str = "gemini-3.5-flash",
        *,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = 65536,
        thinking_level: ThinkingLevel | None = "medium",
        thinking_budget: int | None = None,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        timeout: float = 600.0,
    ) -> None:
        """
        Notes on defaults (Gemini 3.x):

        - ``temperature``: Google no longer recommends overriding this for
          Gemini 3.x models (can cause looping/degraded output on long
          generations). Default is ``None`` -> left unset -> API default (1.0).
          Pass an explicit float only if you've tested it for your use case.
        - ``max_tokens``: this is a shared budget across *thinking + visible
          output*. For long structured generations (e.g. ~100-turn transcript
          JSON), 20k was too tight once thinking tokens are counted. Set to
          the model ceiling (65536) by default so thinking + long transcripts
          in verbose scripts (e.g. Hindi/Devanagari) don't get truncated.
          Lower this only if you've confirmed your transcripts are short and
          want to save cost/latency.
        - ``thinking_level``: replaces the legacy numeric ``thinking_budget``.
          Default ``"medium"`` for this task: multi-turn transcript generation
          needs to track running timestamps, alternating
          interruption/backchannel/overlap patterns, and a coherent emotional
          arc across ~100+ turns -- "low" risks logical drift in later turns.
          Drop to ``"low"``/``"minimal"`` if you've validated that shorter or
          simpler conversations stay coherent without it (cheaper, faster).
          Use ``"high"`` only if you see drift even at "medium".
        - ``thinking_budget``: legacy numeric param, still supported for
          backward compatibility. Do NOT set both ``thinking_level`` and
          ``thinking_budget`` in the same request -- if both are provided,
          ``thinking_level`` wins and a warning is not raised by the SDK, so
          we enforce it here explicitly.
        """
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
        if thinking_level is not None and thinking_budget is not None:
            raise ValueError(
                "Set only one of thinking_level or thinking_budget, not both."
            )
        self.thinking_level = thinking_level
        self.thinking_budget = thinking_budget
        self.timeout = timeout
        key = api_key or config_get("GEMINI_API_KEY") or config_get("GOOGLE_API_KEY")
        if not key:
            raise LLMError(
                "No Gemini API key found. Pass api_key= or set GEMINI_API_KEY "
                "in conversations_generator/config.json."
            )
        self._client = genai.Client(api_key=key)

    def _complete(self, messages: list[Message], **overrides: Any) -> LLMResponse:
        params = self._resolved(overrides)
        system_instruction, contents = self._split_messages(messages)

        mime_type = overrides.get("response_mime_type")
        if not mime_type and overrides.get("response_format", {}).get("type") == "json_object":
            mime_type = "application/json"

        thinking_level = overrides.get("thinking_level", self.thinking_level)
        thinking_budget = overrides.get("thinking_budget", self.thinking_budget)
        thinking_config = self._build_thinking_config(self.model, thinking_level, thinking_budget)

        config = genai_types.GenerateContentConfig(
            # Only pass temperature if explicitly set -- leave unset (API
            # default) otherwise, per Gemini 3.x guidance.
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

        self._raise_if_truncated(response, params["max_tokens"])

        return LLMResponse(
            text=response.text or "",
            model=self.model,
            usage=self._usage(response),
            raw=response,
        )

    # ------------------------------------------------------------------ #
    # Thinking config
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_thinking_config(
        model: str,
        thinking_level: ThinkingLevel | None,
        thinking_budget: int | None,
    ) -> Any:
        # "thinking_level" is a Gemini 3.x-only parameter -- Flash models don't
        # support it and error out if it's sent. Fall back to thinking_budget
        # (if set) or no thinking_config at all for those models.
        if thinking_level is not None and "flash" in model.lower():
            thinking_level = None
        if thinking_level is not None:
            return genai_types.ThinkingConfig(thinking_level=thinking_level)
        if thinking_budget is not None:
            return genai_types.ThinkingConfig(thinking_budget=thinking_budget)
        return None

    # ------------------------------------------------------------------ #
    # Truncation handling
    # ------------------------------------------------------------------ #
    @staticmethod
    def _raise_if_truncated(response: Any, max_tokens: int | None) -> None:
        """Fail loudly and specifically when generation was cut off by the
        token budget, instead of letting downstream JSON parsing choke on a
        confusing 'Expecting value' error with no context.
        """
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return
        finish_reason = getattr(candidates[0], "finish_reason", None)
        finish_reason_name = getattr(finish_reason, "name", str(finish_reason))

        if finish_reason_name == "MAX_TOKENS":
            meta = getattr(response, "usage_metadata", None)
            thoughts = getattr(meta, "thoughts_token_count", None) if meta else None
            output = getattr(meta, "candidates_token_count", None) if meta else None
            raise LLMError(
                "Gemini generation was truncated by max_output_tokens "
                f"(configured max_tokens={max_tokens}, "
                f"thinking_tokens={thoughts}, output_tokens={output}). "
                "Increase max_tokens, lower thinking_level, or reduce the "
                "requested output length (e.g. fewer turns per call)."
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
            "thinking": getattr(meta, "thoughts_token_count", 0) or 0,
            "output": getattr(meta, "candidates_token_count", 0) or 0,
        }
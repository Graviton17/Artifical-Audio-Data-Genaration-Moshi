"""Base class for the conversation-generation agents."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from ..configuration_reader import get_agent_temperature
from ..llm import BaseLLM, GroqLLM
from ..logger import Logger
from ..prompts import resolve_system_prompt
from ..usage_tracker import record_agent_call


class BaseAgent(ABC):
    """Common wiring shared by every agent."""

    prompt_name: str | None = None
    # Short label used in usage reports (defaults to class name sans "Agent").
    agent_name: str | None = None
    # Key into config.json's "AGENT_TEMPERATURES" section. When set, this agent's
    # LLM calls default to that temperature (unless the caller passes one
    # explicitly), so each stage's sampling temperature is managed centrally.
    temperature_key: str | None = None

    def __init__(self, llm: BaseLLM | None = None) -> None:
        if not self.prompt_name:
            raise ValueError(f"{type(self).__name__} must set a class-level prompt_name.")
        # Default provider is Groq; pass any BaseLLM to use a different one.
        self.llm = llm if llm is not None else GroqLLM()
        self.langfuse_prompt = resolve_system_prompt(self.prompt_name)

    @property
    def usage_agent_name(self) -> str:
        if self.agent_name:
            return self.agent_name
        name = type(self).__name__
        if name.endswith("Agent"):
            name = name[: -len("Agent")]
        return name

    def _apply_agent_temperature(self, overrides: dict[str, Any]) -> None:
        """Inject this agent's configured temperature unless one was passed in."""
        if self.temperature_key and "temperature" not in overrides:
            overrides["temperature"] = get_agent_temperature(self.temperature_key)

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the agent's task and return its output."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Convenience wrappers that prepend this agent's system prompt
    # ------------------------------------------------------------------ #
    def _generate(
        self,
        prompt: str,
        system_vars: dict[str, Any] | None = None,
        *,
        stream: bool = False,
        stream_label: str | None = None,
        **overrides: Any,
    ) -> str:
        system = self.langfuse_prompt.compile(**(system_vars or {}))
        full_prompt = f"{system}\n\n{prompt}"
        self._apply_agent_temperature(overrides)
        start = time.perf_counter()
        try:
            if not stream:
                text = self.llm.generate(full_prompt, **overrides)
            else:
                Logger.stream_start(stream_label or f"{type(self).__name__} — live output")
                try:
                    text = self.llm.generate_stream(
                        full_prompt, on_chunk=Logger.stream_chunk, **overrides
                    )
                finally:
                    Logger.stream_end()
        finally:
            record_agent_call(
                agent=self.usage_agent_name,
                model=getattr(self.llm, "_last_model", self.llm.model),
                usage=self.llm.last_usage,
                duration_sec=time.perf_counter() - start,
            )
        return text

    def _generate_json(
        self,
        prompt: str,
        system_vars: dict[str, Any] | None = None,
        *,
        stream: bool = False,
        stream_label: str | None = None,
        **overrides: Any,
    ) -> Any:
        system = self.langfuse_prompt.compile(**(system_vars or {}))
        full_prompt = f"{system}\n\n{prompt}"
        self._apply_agent_temperature(overrides)
        start = time.perf_counter()
        try:
            if not stream:
                result = self.llm.generate_json(full_prompt, **overrides)
            else:
                Logger.stream_start(stream_label or f"{type(self).__name__} — live output")
                try:
                    result = self.llm.generate_json_stream(
                        full_prompt, on_chunk=Logger.stream_chunk, **overrides
                    )
                finally:
                    Logger.stream_end()
        finally:
            record_agent_call(
                agent=self.usage_agent_name,
                model=getattr(self.llm, "_last_model", self.llm.model),
                usage=self.llm.last_usage,
                duration_sec=time.perf_counter() - start,
            )
        return result

"""Base class for the conversation-generation agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..llm import BaseLLM, GroqLLM
from ..logger import Logger
from ..prompts import resolve_system_prompt


class BaseAgent(ABC):
    """Common wiring shared by every agent."""

    prompt_name: str | None = None

    def __init__(self, llm: BaseLLM | None = None) -> None:
        if not self.prompt_name:
            raise ValueError(f"{type(self).__name__} must set a class-level prompt_name.")
        # Default provider is Groq; pass any BaseLLM to use a different one.
        self.llm = llm if llm is not None else GroqLLM()
        self.langfuse_prompt = resolve_system_prompt(self.prompt_name)

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
        if not stream:
            return self.llm.generate(full_prompt, **overrides)

        Logger.stream_start(stream_label or f"{type(self).__name__} — live output")
        try:
            return self.llm.generate_stream(full_prompt, on_chunk=Logger.stream_chunk, **overrides)
        finally:
            Logger.stream_end()

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
        if not stream:
            return self.llm.generate_json(full_prompt, **overrides)

        Logger.stream_start(stream_label or f"{type(self).__name__} — live output")
        try:
            return self.llm.generate_json_stream(full_prompt, on_chunk=Logger.stream_chunk, **overrides)
        finally:
            Logger.stream_end()

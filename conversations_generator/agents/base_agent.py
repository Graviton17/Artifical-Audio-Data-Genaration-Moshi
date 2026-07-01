"""Base class for the conversation-generation agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..llm import BaseLLM, GroqLLM
from ..prompts import resolve_system_prompt


class BaseAgent(ABC):
    """Common wiring shared by every agent."""

    prompt_name: str | None = None

    def __init__(self, llm: BaseLLM | None = None) -> None:
        if not self.prompt_name:
            raise ValueError(f"{type(self).__name__} must set a class-level prompt_name.")
        # Default provider is Groq; pass any BaseLLM to use a different one.
        self.llm = llm if llm is not None else GroqLLM()
        self.system_prompt = resolve_system_prompt(self.prompt_name)

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the agent's task and return its output."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Convenience wrappers that prepend this agent's system prompt
    # ------------------------------------------------------------------ #
    def _build_full_prompt(self, prompt: str) -> str:
        """Combine the static system prompt with the dynamic per-call prompt."""
        return f"{self.system_prompt}\n\n{prompt}"

    def _generate(self, prompt: str, **overrides: Any) -> str:
        return self.llm.generate(self._build_full_prompt(prompt), **overrides)

    def _generate_json(self, prompt: str, **overrides: Any) -> Any:
        return self.llm.generate_json(self._build_full_prompt(prompt), **overrides)

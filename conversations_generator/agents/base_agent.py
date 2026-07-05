"""Base class for the conversation-generation agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .. import settings
from ..llm import BaseLLM, GroqLLM
from ..prompts import resolve_system_prompt


class BaseAgent(ABC):
    """Common wiring shared by every agent."""

    prompt_name: str | None = None
    # Key into settings.AGENT_TEMPERATURES. When set, this agent's LLM calls
    # default to that stage's configured temperature (unless the caller passes
    # one explicitly), so each stage's sampling temperature is managed centrally.
    temperature_key: str | None = None

    def __init__(self, llm: BaseLLM | None = None) -> None:
        if not self.prompt_name:
            raise ValueError(f"{type(self).__name__} must set a class-level prompt_name.")
        # Default provider is Groq; pass any BaseLLM to use a different one.
        self.llm = llm if llm is not None else GroqLLM()
        self.langfuse_prompt = resolve_system_prompt(self.prompt_name)

    def _apply_agent_temperature(self, overrides: dict[str, Any]) -> None:
        """Inject this agent's configured temperature unless one was passed in."""
        if self.temperature_key and "temperature" not in overrides:
            overrides["temperature"] = settings.get_agent_temperature(self.temperature_key)

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the agent's task and return its output."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Convenience wrappers that prepend this agent's system prompt
    # ------------------------------------------------------------------ #
    def _generate(self, prompt: str, system_vars: dict[str, Any] | None = None, **overrides: Any) -> str:
        system = self.langfuse_prompt.compile(**(system_vars or {}))
        full_prompt = f"{system}\n\n{prompt}"
        self._apply_agent_temperature(overrides)
        return self.llm.generate(full_prompt, **overrides)

    def _generate_json(self, prompt: str, system_vars: dict[str, Any] | None = None, **overrides: Any) -> Any:
        system = self.langfuse_prompt.compile(**(system_vars or {}))
        full_prompt = f"{system}\n\n{prompt}"
        self._apply_agent_temperature(overrides)
        return self.llm.generate_json(full_prompt, **overrides)

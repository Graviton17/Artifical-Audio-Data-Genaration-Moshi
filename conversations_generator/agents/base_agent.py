"""Base class for the conversation-generation agents.

An agent pairs a fixed *system prompt* (its role and instructions) with a
provider-agnostic :class:`BaseLLM`. Subclasses set :attr:`system_prompt` and
implement :meth:`run`. The LLM defaults to Groq, so agents work out of the box
given a ``GROQ_API_KEY``; inject any other :class:`BaseLLM` to switch providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..llm import BaseLLM, GroqLLM


class BaseAgent(ABC):
    """Common wiring shared by every agent.

    Subclasses provide a :attr:`system_prompt` describing their role and
    implement :meth:`run`. Use :meth:`_generate` / :meth:`_generate_json` to call
    the LLM with this agent's system prompt already applied.
    """

    # Role-specific instructions; overridden by each concrete agent.
    system_prompt: str = ""

    def __init__(self, llm: BaseLLM | None = None) -> None:
        # Default provider is Groq; pass any BaseLLM to use a different one.
        self.llm = llm if llm is not None else GroqLLM()

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the agent's task and return its output."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Convenience wrappers that always apply this agent's system prompt
    # ------------------------------------------------------------------ #
    def _generate(self, prompt: str, **overrides: Any) -> str:
        return self.llm.generate(prompt, system=self.system_prompt, **overrides)

    def _generate_json(self, prompt: str, **overrides: Any) -> Any:
        return self.llm.generate_json(prompt, system=self.system_prompt, **overrides)

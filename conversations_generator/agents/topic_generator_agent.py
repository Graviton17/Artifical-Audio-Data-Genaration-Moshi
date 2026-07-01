"""Agent that proposes conversation topics, one at a time."""

from __future__ import annotations

from typing import Any

from ..llm import BaseLLM
from .base_agent import BaseAgent


class TopicGeneratorAgent(BaseAgent):
    """Generate conversation topics one per call, avoiding past repeats."""

    prompt_name = "topic-generator-agent"

    def __init__(self, llm: BaseLLM | None = None) -> None:
        super().__init__(llm)
        # Topics produced so far this session; fed back to avoid repeats.
        self.history: list[dict[str, str]] = []

    def run(
        self,
        *,
        language: str = "Hinglish",
        domain: str | None = None,
        agent_emotion: str | None = None,
        user_emotion: str | None = None,
        extra_guidance: str | None = None,
        **overrides: Any,
    ) -> dict[str, str]:
        """Generate the next single topic and append it to :attr:`history`."""
        prompt = self._build_prompt(
            language=language,
            domain=domain,
            agent_emotion=agent_emotion,
            user_emotion=user_emotion,
            extra_guidance=extra_guidance,
        )

        overrides.setdefault("response_format", {"type": "json_object"})
        topic = self._normalize(self._generate_json(prompt, **overrides))
        self.history.append(topic)
        return topic

    def reset(self) -> None:
        """Forget previously generated topics."""
        self.history.clear()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _build_prompt(
        self,
        *,
        language: str,
        domain: str | None,
        agent_emotion: str | None,
        user_emotion: str | None,
        extra_guidance: str | None,
    ) -> str:
        lines = [
            "Generate ONE new conversation topic.",
            f"Language of the conversations: {language}.",
        ]
        if domain:
            lines.append(f"Domain / setting: {domain}.")
        if agent_emotion:
            lines.append(f"Agent's emotional tone: {agent_emotion}.")
        if user_emotion:
            lines.append(f"User's emotional tone: {user_emotion}.")
        if extra_guidance:
            lines.append(extra_guidance)

        if self.history:
            already = "\n".join(f"- {t['title']}" for t in self.history)
            lines.append(
                "Topics already generated (do NOT repeat or closely resemble these):\n"
                f"{already}"
            )

        lines.append(
            'Return a single JSON object with keys "title", and "context".'
        )
        return "\n".join(lines)

    @staticmethod
    def _normalize(result: Any) -> dict[str, str]:
        """Coerce the model output into a clean topic dict."""
        # Tolerate the object being wrapped in a single-element list.
        if isinstance(result, list):
            result = result[0] if result else {}
        if not isinstance(result, dict):
            raise ValueError(f"Expected a topic object, got {type(result).__name__}")

        title = str(result.get("title", "")).strip()
        if not title:
            raise ValueError(f"Topic is missing a 'title': {result!r}")
        return {
            "title": title,
            "context": str(result.get("context", "")).strip(),
        }

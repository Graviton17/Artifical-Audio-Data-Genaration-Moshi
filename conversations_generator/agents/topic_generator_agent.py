"""Agent that proposes conversation topics, one at a time."""

from __future__ import annotations

import random
from typing import Any

from ..llm import BaseLLM
from .base_agent import BaseAgent

CONVERSATION_TYPES: list[str] = [
    "Explain",
    "Education",
    "Sales",
    "Inquiry",
    "Day-to-day",
    "Information-based",
    "Life experience",
    "Reasoning-based",
    "Formal address",
    "Interview",
    "Emotion",
]


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
        conversation_type: str | None = None,
        agent_emotion: str | None = None,
        user_emotion: str | None = None,
        agent_accent: str | None = None,
        user_accent: str | None = None,
        gender_pair: str | None = None,
        **overrides: Any,
    ) -> dict[str, str]:
        """Generate the next single topic and append it to :attr:`history`."""
        # Pick a random conversation type if none was explicitly provided.
        chosen_type = conversation_type or random.choice(CONVERSATION_TYPES)

        prompt = self._build_prompt(
            language=language,
            agent_emotion=agent_emotion,
            user_emotion=user_emotion,
            agent_accent=agent_accent,
            user_accent=user_accent,
            gender_pair=gender_pair,
        )

        system_vars = {"conversation_type": chosen_type}

        overrides.setdefault("response_format", {"type": "json_object"})
        topic = self._normalize(
            self._generate_json(
                prompt,
                system_vars=system_vars,
                stream=True,
                stream_label=f"Generating topic ({language})…",
                **overrides,
            )
        )
        topic["conversation_type"] = chosen_type  # track which type was used
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
        agent_emotion: str | None,
        user_emotion: str | None,
        agent_accent: str | None = None,
        user_accent: str | None = None,
        gender_pair: str | None = None,
    ) -> str:
        lines = [
            "Generate ONE new conversation topic.",
            f"Language of the conversations: {language}.",
        ]
        if agent_emotion:
            lines.append(f"Agent's emotional tone: {agent_emotion}.")
        if user_emotion:
            lines.append(f"User's emotional tone: {user_emotion}.")
        if agent_accent:
            lines.append(f"Agent's accent: {agent_accent}.")
        if user_accent:
            lines.append(f"User's accent: {user_accent}.")
        if gender_pair:
            lines.append(f"Agent/user gender pair in sequence where M means Male and F means Female: {gender_pair}.")

        if self.history:
            already = "\n".join(
                f"- {t['title']} (type: {t.get('conversation_type', 'unknown')})"
                for t in self.history
            )
            lines.append(
                "Topics already generated (do NOT repeat, closely resemble, "
                "or use a similar theme/setting/scenario as any of these):\n"
                f"{already}\n"
                "IMPORTANT: Ensure maximum diversity in the themes, settings, and scenarios. "
                "Avoid clustering around any single domain."
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

"""Agent that proposes conversation topics, one at a time.

Each call to :meth:`run` produces a single topic (title + context) for a
two-speaker spoken conversation. The agent remembers everything it has already
generated in :attr:`history` and feeds those titles back into the prompt, so
successive calls yield fresh, non-duplicate topics. The returned topic and its
context are what the downstream ``conversation_generator_agent`` consumes.
"""

from __future__ import annotations

from typing import Any

from ..llm import BaseLLM
from .base_agent import BaseAgent


class TopicGeneratorAgent(BaseAgent):
    """Generate conversation topics one per call, avoiding past repeats."""

    system_prompt = (
        "You are a topic designer for a synthetic speech dataset. You invent "
        "realistic topics for spontaneous, two-person spoken conversations (an "
        "agent and a user talking naturally, as on a phone or voice-assistant "
        "call).\n"
        "\n"
        "Each time you are called you produce EXACTLY ONE topic. You are shown the "
        "topics already generated; your new topic must be clearly different from "
        "all of them (no near-duplicates or overlapping themes).\n"
        "\n"
        "Rules:\n"
        "- The topic must suit everyday spoken dialogue, not an essay or monologue.\n"
        "- Match the requested language, emotional tone, and domain when given.\n"
        "- 'title': short, 3-8 words.\n"
        "- 'context': 2-4 sentences of grounding detail (setting, goal, who the "
        "speakers are, what they discuss) that a writer can use to script the "
        "conversation.\n"
        "- Return ONLY a single JSON object with exactly the keys \"title\" and "
        "\"context\". No prose, no markdown, no code fences."
    )

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
        """Generate the next single topic and append it to :attr:`history`.

        Parameters mirror the corpus profile fields (see
        ``data/corpus_instances.jsonl``) so topics can be steered per combination.
        ``overrides`` pass through to the LLM (e.g. ``temperature``).
        """
        prompt = self._build_prompt(
            language=language,
            domain=domain,
            agent_emotion=agent_emotion,
            user_emotion=user_emotion,
            extra_guidance=extra_guidance,
        )
        # Ask the provider for strict JSON; the prompt already says "json" (a Groq
        # json_object requirement). Callers can override via response_format=.
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

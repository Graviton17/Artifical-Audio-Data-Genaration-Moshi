"""Agent that proposes conversation topics, one at a time."""

from __future__ import annotations

import random
import threading
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

# Everyday-life domains, one picked at random per topic. Seeding each call with a
# concrete area is what actually produces variety: without it, feeding the model
# the list of past topics makes it IMITATE them (few-shot anchoring) and cluster
# on the same "planning / comparing" mould. A random domain steers each topic
# into a genuinely different corner of life instead.
TOPIC_DOMAINS: list[str] = [
    "health, illness, doctor visits, or medicine",
    "cooking, recipes, or a specific dish",
    "a festival, ritual, or religious observance",
    "a travel story or trip that already happened",
    "work, a job, colleagues, or the workplace",
    "neighbours, housing society, or the local community",
    "a hobby, sport, or game",
    "school, college, exams, or studies",
    "a technology or gadget problem",
    "a government office, paperwork, or a document errand",
    "family news, relationships, or a personal milestone",
    "a household repair, appliance, or maintenance issue",
    "weather, seasons, or a natural event",
    "commuting, public transport, or a vehicle",
    "food, a restaurant, or eating out",
    "pets or animals",
    "movies, music, TV, or entertainment",
    "money matters like a bill, salary, or a bank issue",
    "clothes, a wedding, or a social event",
    "fitness, exercise, or daily routine",
    "gardening, plants, or the home garden",
    "a childhood or nostalgic memory",
    "helping someone with a decision or a piece of advice",
    "a misunderstanding, complaint, or dispute to resolve",
]


class TopicGeneratorAgent(BaseAgent):
    """Generate conversation topics one per call, avoiding past repeats."""

    prompt_name = "topic-generator-agent"
    temperature_key = "topic"

    def __init__(self, llm: BaseLLM | None = None) -> None:
        super().__init__(llm)
        # Topics produced so far this session; fed back to avoid repeats.
        self.history: list[dict[str, str]] = []
        # Serialises topic generation so parallel workers never produce the same
        # topic: the whole "read history → generate → append" step is atomic, so
        # each worker sees every topic chosen before it and adds a distinct one.
        self._lock = threading.Lock()

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
        include_numbers: bool = False,
        **overrides: Any,
    ) -> dict[str, str]:
        """Generate the next single topic and append it to :attr:`history`.

        ``include_numbers`` steers whether the topic is chosen so that concrete
        numbers naturally come up in the conversation (see the runner, which
        toggles this per-conversation from ``NUMBER_INCLUSION_PERCENTAGE``).
        """
        overrides.setdefault("response_format", {"type": "json_object"})
        # Hold the lock across the whole read→generate→append so concurrent
        # workers can't pick the same domain/history snapshot and collide. Topic
        # generation is a small slice of the pipeline, so serialising just this
        # step barely dents the parallel speed-up while guaranteeing uniqueness.
        with self._lock:
            # Pick a random conversation type if none was explicitly provided.
            chosen_type = conversation_type or random.choice(CONVERSATION_TYPES)
            # Seed this topic with a random everyday-life domain, avoiding the ones
            # used most recently so consecutive topics land in different areas.
            chosen_domain = self._pick_domain()

            prompt = self._build_prompt(
                language=language,
                agent_emotion=agent_emotion,
                user_emotion=user_emotion,
                agent_accent=agent_accent,
                user_accent=user_accent,
                gender_pair=gender_pair,
                include_numbers=include_numbers,
                domain=chosen_domain,
            )

            system_vars = {"conversation_type": chosen_type}

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
            topic["domain"] = chosen_domain  # track which domain seeded it
            self.history.append(topic)
        return topic

    def _pick_domain(self) -> str:
        """Choose a domain, avoiding those used in the most recent topics.

        Excludes the domains of roughly the last half-list of topics so the same
        area isn't reused back-to-back; falls back to the full list once every
        domain has been used recently.
        """
        window = min(len(TOPIC_DOMAINS) - 1, len(self.history))
        recent = {t.get("domain") for t in self.history[-window:]} if window else set()
        candidates = [d for d in TOPIC_DOMAINS if d not in recent]
        return random.choice(candidates or TOPIC_DOMAINS)

    def reset(self) -> None:
        """Forget previously generated topics."""
        self.history.clear()

    def prime(self, topics: list[str] | list[dict[str, str]]) -> None:
        """Reset history and seed it with previously-generated topics.

        Called at the start of each instance (including on resume from a
        checkpoint) so the "do NOT repeat" list already contains every topic
        produced for that instance in earlier runs — keeping newly generated
        topics distinct instead of restarting from an empty history. Accepts
        either plain title strings or full topic dicts.
        """
        self.history = []
        for topic in topics:
            if isinstance(topic, str):
                title = topic.strip()
                if title:
                    self.history.append({"title": title, "context": ""})
            elif isinstance(topic, dict) and str(topic.get("title", "")).strip():
                self.history.append(
                    {
                        "title": str(topic["title"]).strip(),
                        "context": str(topic.get("context", "")).strip(),
                        "conversation_type": topic.get("conversation_type", "unknown"),
                    }
                )

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
        include_numbers: bool = False,
        domain: str | None = None,
    ) -> str:
        lines = [
            "Generate ONE new conversation topic.",
            f"Language of the conversations: {language}.",
        ]
        if domain:
            lines.append(
                f"Set this conversation firmly in this area of everyday life: {domain}. "
                "Invent a specific, concrete situation within that area — do not drift "
                "into shopping, budgeting, or comparing options unless that area is "
                "itself about money."
            )
        if include_numbers:
            lines.append(
                "This topic MUST be one where concrete numbers naturally come up "
                "and can be discussed with reasoning. Numbers arise in MANY kinds "
                "of situations — do NOT default to prices, budgets, or comparing "
                "costs. Draw the numeric element from a wide range, e.g. dates and "
                "durations, a medicine dosage or schedule, cooking quantities and "
                "timings, a travel itinerary or distances, exam marks or scores, "
                "measurements, ages, a match result, quantities of things, or "
                "phone/order/account numbers. Pick a title and context that invite "
                "specific figures during the conversation WITHOUT making the topic "
                "about shopping, budgeting, or price comparison."
            )
        else:
            lines.append(
                "This topic should flow naturally as a mostly qualitative "
                "discussion — do NOT centre it on statistics or figures; specific "
                "numbers should not be needed to have the conversation."
            )
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
                "IMPORTANT: Ensure maximum diversity in themes, settings, scenarios, "
                "AND title framing. Pick a different domain from the ones above, and "
                "vary the sentence shape of the title — do NOT keep starting titles "
                "with 'Planning', 'Comparing', or 'Choosing', and avoid clustering "
                "around shopping, budgets, or comparing options."
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

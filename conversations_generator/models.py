"""Domain models for the conversation-generation pipeline.

Typed dataclasses that mirror the corpus JSONL schema so the rest of the code
gets attribute access and IDE completion instead of raw-dict spelunking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CorpusInstance:
    """One row from ``corpus_instances.jsonl``.

    Captures every attribute that influences how the pipeline generates a
    conversation: language, emotional tone, accent, and gender for both the
    agent and user sides, plus book-keeping fields for corpus planning.
    """

    corpus_combination_id: int
    language: str
    agent_emotion: str | None = None
    user_emotion: str | None = None
    agent_accent: str | None = None
    user_accent: str | None = None
    gender_pair: str | None = None
    # ── corpus planning fields (optional, not used by the pipeline) ──
    joint_probability: float | None = None
    duration_sec: float | None = None
    duration_hr: float | None = None
    est_conversations_needed: int | None = None

    # ------------------------------------------------------------------ #
    # Factories
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CorpusInstance":
        """Build from a flat dict (e.g. one row of a JSONL DataFrame)."""
        return cls(
            corpus_combination_id=raw["corpus_combination_id"],
            language=raw["language"],
            agent_emotion=raw.get("agent_emotion"),
            user_emotion=raw.get("user_emotion"),
            agent_accent=raw.get("agent_accent"),
            user_accent=raw.get("user_accent"),
            gender_pair=raw.get("gender_pair"),
            joint_probability=raw.get("joint_probability"),
            duration_sec=raw.get("duration_sec"),
            duration_hr=raw.get("duration_hr"),
            est_conversations_needed=raw.get("est_conversations_needed"),
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def to_profile(self) -> dict[str, Any]:
        """Return only the fields that ``ConversationRunner.run()`` expects.

        This is the kwargs dict you unpack into ``runner.run(**instance.to_profile())``.
        ``None`` values are excluded so the agent's defaults kick in.
        """
        profile: dict[str, Any] = {"language": self.language}
        if self.agent_emotion is not None:
            profile["agent_emotion"] = self.agent_emotion
        if self.user_emotion is not None:
            profile["user_emotion"] = self.user_emotion
        if self.agent_accent is not None:
            profile["agent_accent"] = self.agent_accent
        if self.user_accent is not None:
            profile["user_accent"] = self.user_accent
        if self.gender_pair is not None:
            profile["gender_pair"] = self.gender_pair
        return profile

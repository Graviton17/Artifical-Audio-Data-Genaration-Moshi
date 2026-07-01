"""Agent that generates a full multi-turn conversation from a topic.

Takes the title + context produced by :class:`TopicGeneratorAgent` and produces
a list of conversation turns following the schema defined in
``conversation_field_schema.json``.  Each turn is a dict with fields like
``turn_id``, ``speaker``, ``text``, ``emotion``, planned/real timing, overlap
and interruption metadata, etc.

The system prompt is managed in Langfuse under the name
``conversation-generator-agent``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..llm import BaseLLM
from .base_agent import BaseAgent

# ------------------------------------------------------------------ #
# Load the field schema once at module level
# ------------------------------------------------------------------ #
_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "data" / "conversation_field_schema.json"
with open(_SCHEMA_PATH, "r", encoding="utf-8") as _f:
    CONVERSATION_FIELD_SCHEMA: dict[str, Any] = json.load(_f)


class ConversationGeneratorAgent(BaseAgent):
    """Generate a multi-turn conversation from a topic produced by the topic agent.

    The conversation is returned as a list of turn dicts following the
    ``conversation_field_schema.json`` schema.  Each turn includes planned and
    real timing, overlap/interruption metadata, emotion tags, etc.
    """

    prompt_name = "conversation-generator-agent"

    def __init__(self, llm: BaseLLM | None = None) -> None:
        super().__init__(llm)

    def run(
        self,
        *,
        title: str,
        context: str,
        language: str = "Hinglish",
        conversation_type: str | None = None,
        agent_emotion: str | None = None,
        user_emotion: str | None = None,
        agent_accent: str | None = None,
        user_accent: str | None = None,
        gender_pair: str | None = None,
        **overrides: Any,
    ) -> list[dict[str, Any]]:
        """Generate a full conversation for the given topic.

        Parameters
        ----------
        title : str
            Conversation title from :class:`TopicGeneratorAgent`.
        context : str
            Conversation context/description from :class:`TopicGeneratorAgent`.
        language : str
            Language the dialogue should be written in.
        conversation_type : str | None
            Type of conversation (e.g. "Sales", "Inquiry").
        agent_emotion, user_emotion : str | None
            Dominant emotion for each speaker throughout the conversation.
        agent_accent, user_accent : str | None
            Accent style for each speaker.
        gender_pair : str | None
            Gender pair string like "M-F", "M-M", "F-F", "F-M".
        **overrides
            Extra kwargs forwarded to the LLM (temperature, max_tokens, etc.).

        Returns
        -------
        list[dict]
            Ordered list of turn dicts, each matching the conversation field
            schema (turn_id, speaker, text, emotion, timing fields, etc.).
        """
        prompt = self._build_prompt(
            title=title,
            context=context,
            language=language,
            conversation_type=conversation_type,
            agent_emotion=agent_emotion,
            user_emotion=user_emotion,
            agent_accent=agent_accent,
            user_accent=user_accent,
            gender_pair=gender_pair,
        )

        system_vars: dict[str, Any] = {}
        if conversation_type:
            system_vars["conversation_type"] = conversation_type

        overrides.setdefault("response_format", {"type": "json_object"})
        raw_result = self._generate_json(prompt, system_vars=system_vars, **overrides)
        return self._normalize(raw_result)

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #
    def _build_prompt(
        self,
        *,
        title: str,
        context: str,
        language: str,
        conversation_type: str | None,
        agent_emotion: str | None,
        user_emotion: str | None,
        agent_accent: str | None,
        user_accent: str | None,
        gender_pair: str | None,
    ) -> str:
        """Assemble the user-side prompt sent alongside the Langfuse system prompt."""
        lines: list[str] = [
            "Generate a realistic, natural-sounding multi-turn conversation.",
            "",
            "## Topic",
            f"**Title:** {title}",
            f"**Context:** {context}",
            f"**Language:** {language}",
        ]

        if conversation_type:
            lines.append(f"**Conversation type:** {conversation_type}")
        if agent_emotion:
            lines.append(f"**Speaker 1 (agent) emotion:** {agent_emotion}")
        if user_emotion:
            lines.append(f"**Speaker 2 (user) emotion:** {user_emotion}")
        if agent_accent:
            lines.append(f"**Speaker 1 (agent) accent:** {agent_accent}")
        if user_accent:
            lines.append(f"**Speaker 2 (user) accent:** {user_accent}")
        if gender_pair:
            lines.append(
                f"**Gender pair (speaker_1-speaker_2, M=Male, F=Female):** {gender_pair}"
            )
        
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Output normalization
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize(result: Any) -> list[dict[str, Any]]:
        """Coerce the model output into a clean list of turn dicts.

        Accepts either:
        - A dict with a ``"turns"`` key containing a list of turn dicts.
        - A bare list of turn dicts.
        """
        if isinstance(result, dict):
            # Try the expected {"turns": [...]} wrapper.
            if "turns" in result:
                turns = result["turns"]
            else:
                # Maybe the LLM returned a single-key dict with an unusual key.
                values = list(result.values())
                if len(values) == 1 and isinstance(values[0], list):
                    turns = values[0]
                else:
                    raise ValueError(
                        f"Expected a dict with a 'turns' key, got keys: {list(result.keys())}"
                    )
        elif isinstance(result, list):
            turns = result
        else:
            raise ValueError(
                f"Expected a list or dict of turns, got {type(result).__name__}"
            )

        if not turns:
            raise ValueError("The conversation has no turns.")

        # Light validation: every turn must have at least turn_id, speaker, text.
        required_keys = {"turn_id", "speaker", "text"}
        for i, turn in enumerate(turns):
            if not isinstance(turn, dict):
                raise ValueError(f"Turn {i} is not a dict: {type(turn).__name__}")
            missing = required_keys - turn.keys()
            if missing:
                raise ValueError(
                    f"Turn {i} (turn_id={turn.get('turn_id', '?')}) is missing "
                    f"required keys: {missing}"
                )

        return turns

"""Agent that surgically fixes only the flagged turns of a conversation.

Where :class:`~conversations_generator.agents.conversation_generator_agent.ConversationGeneratorAgent`
writes an entire conversation from a topic, this agent does the opposite job:
given an *already generated* conversation plus turn-scoped validation feedback,
it returns corrected versions of **only the flagged turns (and the turns they're
structurally linked to)** — never the whole conversation. This is the
Cursor-style "grep the broken part and edit just that part" flow, which keeps a
small model from re-breaking turns it already got right.

The deterministic surrounding work — deciding which turns are editable, merging
the patch back in, and retiming the tail — lives in
:mod:`conversations_generator.patching.patch_engine`. This class is only the LLM
call plus strict parsing of its patch.

The system prompt is managed in Langfuse under the name
``conversation-fixer-agent`` (see ``data/conversation_fixer_prompt.md`` for the
canonical text to paste there).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..llm import BaseLLM
from ..patching.feedback import AttemptHistory, FeedbackItem, render_feedback
from ..patching.patch_engine import context_window, index_by_id
from .base_agent import BaseAgent

# Timing fields are shown to the fixer (it may need to nudge a backchannel/
# interrupter timestamp), but real_* / error_time are alignment-stage outputs
# that are null at generation time, so we drop them to save tokens.
_HIDDEN_FIELDS = {"real_start_sec", "real_end_sec", "error_time"}

# Minimal per-turn schema every patched turn must still satisfy before we trust
# it enough to merge. The deterministic engine + manual validator re-check the
# rest, so this is deliberately light.
_REQUIRED_KEYS = {"turn_id", "speaker", "text"}


@dataclass
class PatchResult:
    """Parsed output of one fixer call.

    Attributes
    ----------
    patch : dict[str, dict]
        ``turn_id`` -> corrected turn dict, restricted to the turns the fixer was
        allowed to edit (ids outside the allowed set are dropped here already).
    dropped_ids : list[str]
        Turn_ids the model returned that were *not* in the allowed set — i.e.
        turns it tried to edit but wasn't permitted to. Kept for logging so
        over-editing is visible.
    notes : str
        Optional free-text explanation the model returned about its fix.
    """

    patch: dict[str, dict[str, Any]] = field(default_factory=dict)
    dropped_ids: list[str] = field(default_factory=list)
    notes: str = ""


class ConversationFixerAgent(BaseAgent):
    """LLM patcher: rewrites only the flagged turns of a conversation.

    One LLM call per patch round. Sends the fixer the editable turns, a small
    read-only context window around them, the concrete feedback to satisfy, and
    the attempt history, and parses back a patch keyed by ``turn_id``.
    """

    prompt_name = "conversation-fixer-agent"

    def __init__(self, llm: BaseLLM | None = None) -> None:
        super().__init__(llm)

    def run(
        self,
        *,
        turns: list[dict[str, Any]],
        target_ids: list[str],
        feedback: list[FeedbackItem],
        topic: dict[str, str],
        history: AttemptHistory | None = None,
        language: str = "Hinglish",
        conversation_type: str | None = None,
        agent_emotion: str | None = None,
        user_emotion: str | None = None,
        agent_accent: str | None = None,
        user_accent: str | None = None,
        gender_pair: str | None = None,
        **overrides: Any,
    ) -> PatchResult:
        """Produce corrected versions of the ``target_ids`` turns.

        Parameters
        ----------
        turns : list[dict]
            The full current conversation (the fixer sees all of it as context
            but is instructed to only rewrite ``target_ids``).
        target_ids : list[str]
            Turn_ids the fixer is allowed to edit (flagged + linked), from
            :func:`conversations_generator.patching.patch_engine.collect_target_ids`.
        feedback : list[FeedbackItem]
            The concrete, turn-scoped findings to satisfy.
        topic : dict
            Topic the conversation was generated from (title/context) — grounds
            any rewritten text.
        history : AttemptHistory | None
            Cross-round memory so the fixer avoids re-introducing past failures.
        language, *_emotion, *_accent, gender_pair, conversation_type :
            Corpus-instance attributes the rewritten turns must still respect.
        **overrides
            Extra kwargs forwarded to the LLM.

        Returns
        -------
        PatchResult
        """
        if not target_ids:
            return PatchResult()

        conversation_type = conversation_type or topic.get("conversation_type")
        allowed = set(target_ids)

        prompt = self._build_prompt(
            turns=turns,
            target_ids=target_ids,
            feedback=feedback,
            topic=topic,
            history=history,
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

        # Fixing is conservative, not creative — keep variance low.
        overrides.setdefault("temperature", 0.3)
        overrides.setdefault("response_format", {"type": "json_object"})
        raw_result = self._generate_json(prompt, system_vars=system_vars, **overrides)

        from ..logger import Logger
        Logger.debug(f"Fixer LLM Output:\n{json.dumps(raw_result, indent=2, ensure_ascii=False)}")
        return self._normalize(raw_result, allowed)

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #
    def _build_prompt(
        self,
        *,
        turns: list[dict[str, Any]],
        target_ids: list[str],
        feedback: list[FeedbackItem],
        topic: dict[str, str],
        history: AttemptHistory | None,
        language: str,
        conversation_type: str | None,
        agent_emotion: str | None,
        user_emotion: str | None,
        agent_accent: str | None,
        user_accent: str | None,
        gender_pair: str | None,
    ) -> str:
        by_id = index_by_id(turns)
        editable = [self._clean(by_id[tid]) for tid in target_ids if tid in by_id]

        # Read-only neighbours so a rewritten turn still flows, minus the turns
        # already listed as editable.
        ctx_ids = [c for c in context_window(turns, set(target_ids), radius=1) if c not in set(target_ids)]
        context = [self._clean(by_id[c]) for c in ctx_ids if c in by_id]

        lines: list[str] = [
            "You are FIXING an already-generated conversation. Rewrite ONLY the "
            "turns listed under 'Turns you may edit', and return a corrected "
            "version of each one. Do NOT touch any other turn, and do NOT add or "
            "remove turns.",
            "",
            "## Corpus instance requirements (rewritten turns must still match these)",
            f"- language: {language}",
        ]
        if conversation_type:
            lines.append(f"- conversation_type: {conversation_type}")
        if agent_emotion:
            lines.append(f"- speaker_1 (agent) emotion: {agent_emotion}")
        if user_emotion:
            lines.append(f"- speaker_2 (user) emotion: {user_emotion}")
        if agent_accent:
            lines.append(f"- speaker_1 (agent) accent: {agent_accent}")
        if user_accent:
            lines.append(f"- speaker_2 (user) accent: {user_accent}")
        if gender_pair:
            lines.append(f"- gender_pair (speaker_1-speaker_2, M=Male, F=Female): {gender_pair}")

        lines += [
            "",
            "## Topic (for grounding any rewritten text)",
            f"**Title:** {topic.get('title', '')}",
            f"**Context:** {topic.get('context', '')}",
            "",
            "## Issues you must fix",
            render_feedback(feedback) or "(none)",
        ]

        if history is not None and history.records:
            lines += [
                "",
                "## Previous attempts (do NOT reintroduce these problems)",
                history.render(),
            ]
            stuck = history.persistent_turn_ids()
            if stuck:
                lines.append("")
                lines.append(
                    "These turns have failed repeatedly — reconsider them more "
                    f"boldly rather than nudging: {', '.join(stuck)}"
                )

        if context:
            lines += [
                "",
                "## Surrounding turns (READ-ONLY context — do not edit or return these)",
                "```json",
                json.dumps(context, ensure_ascii=False, indent=2),
                "```",
            ]

        lines += [
            "",
            "## Turns you may edit",
            "```json",
            json.dumps(editable, ensure_ascii=False, indent=2),
            "```",
            "",
            "Return ONLY a JSON object of this exact shape (one entry per turn you "
            "edited, each a COMPLETE turn object with every field, keeping the same "
            "turn_id):",
            '{"patched_turns": [ { ...full corrected turn... } ], "notes": "short explanation"}',
        ]
        return "\n".join(lines)

    @staticmethod
    def _clean(turn: dict[str, Any]) -> dict[str, Any]:
        """Drop alignment-only fields before showing a turn to the fixer."""
        return {k: v for k, v in turn.items() if k not in _HIDDEN_FIELDS}

    # ------------------------------------------------------------------ #
    # Output normalization
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize(result: Any, allowed_ids: set[str]) -> PatchResult:
        """Parse the model's JSON into a :class:`PatchResult`.

        Accepts either ``{"patched_turns": [...]}``, a bare list of turns, or a
        ``{turn_id: turn}`` mapping. Turns missing a valid ``turn_id`` / required
        keys are skipped; turns whose id isn't in ``allowed_ids`` are recorded in
        ``dropped_ids`` and not applied.
        """
        notes = ""
        raw_turns: list[Any]

        if isinstance(result, dict) and "patched_turns" in result:
            raw_turns = result.get("patched_turns") or []
            notes = str(result.get("notes", "")).strip()
        elif isinstance(result, list):
            raw_turns = result
        elif isinstance(result, dict):
            # Treat as a {turn_id: turn} mapping, injecting the key as turn_id.
            raw_turns = []
            for tid, turn in result.items():
                if isinstance(turn, dict):
                    turn = {**turn, "turn_id": turn.get("turn_id", tid)}
                    raw_turns.append(turn)
        else:
            raise ValueError(f"Fixer returned unexpected type {type(result).__name__}")

        patch: dict[str, dict[str, Any]] = {}
        dropped: list[str] = []
        for turn in raw_turns:
            if not isinstance(turn, dict):
                continue
            if _REQUIRED_KEYS - turn.keys():
                continue
            tid = turn.get("turn_id")
            if not isinstance(tid, str) or not tid:
                continue
            if tid not in allowed_ids:
                dropped.append(tid)
                continue
            patch[tid] = turn

        return PatchResult(patch=patch, dropped_ids=dropped, notes=notes)

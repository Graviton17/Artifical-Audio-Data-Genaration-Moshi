"""Agent that repairs a conversation with *targeted edits* instead of a full rewrite.

When the LLM agent validator
(:class:`~conversations_generator.agents.conversation_validator_agent.ConversationValidatorAgent`)
fails a conversation, the default repair is to regenerate the whole transcript —
throwing away all the turns that were fine. This agent offers a cheaper path: it
reads the validator's specific issues (each usually tied to a ``turn_id``) and
emits a small **patch** touching only the affected turns, e.g.:

    {"edits": [
      {"turn_id": "t36", "action": "replace",
       "text": "Main quotes dekh kar aapko call karta hoon."},
      {"turn_id": "t8", "action": "delete"}
    ]}

* ``action="replace"`` — fix the ``text`` (and optionally ``emotion`` /
  ``turn_type``) of one turn (a gender-agreement slip, a wrong word, …).
* ``action="delete"`` — drop a turn entirely (e.g. one of several redundant
  backchannels the validator flagged as filler padding).

The patch is applied by
:meth:`~conversations_generator.agents.conversation_formatter_agent.ConversationFormatterAgent.apply_edits`,
which then re-runs the deterministic timing/relationship layout so the edited
conversation still passes manual validation by construction.

Unlike the other agents, this one does **not** use a Langfuse-managed system
prompt — its instructions are self-contained below, so no extra Langfuse prompt
needs to be created for the edit path to work.
"""

from __future__ import annotations

import json
from typing import Any

from ..llm import BaseLLM
from .base_agent import BaseAgent

# Speaker → gender is derived from the corpus ``gender_pair`` string (e.g. "F-M"
# or "FM": speaker_1 then speaker_2). Used to remind the editor which verb/
# adjective gender each speaker must use when fixing agreement issues.
_GENDER_WORDS = {"m": "Male", "f": "Female"}


class _LocalPrompt:
    """Minimal stand-in for a Langfuse prompt object (has a ``.compile``)."""

    def __init__(self, text: str) -> None:
        self.prompt = text

    def compile(self, **_vars: Any) -> str:
        return self.prompt


_SYSTEM_PROMPT = """\
You are a meticulous dialogue EDITOR for two-speaker (speaker_1 = agent,
speaker_2 = user) spoken conversations. You are given an already-written
conversation and a list of specific problems found by a validator. Your job is
to fix ONLY those problems with the smallest possible changes — you are editing,
NOT rewriting.

Hard rules:
- Change ONLY what is needed to resolve the listed issues. Every other turn must
  stay exactly as it is. Do NOT rephrase, "improve", or re-translate untouched
  turns.
- Never invent new turns. You may only REPLACE the text/emotion/turn_type of an
  existing turn, or DELETE an existing turn.
- Keep the SAME language, script, and register as the surrounding dialogue
  (Hindi in Devanagari, Hinglish in Roman, English in English — match what the
  turn already uses).
- Respect gender: a speaker's Hindi/Hinglish verbs and adjectives must agree with
  THAT speaker's gender as given below (e.g. a male speaker says "karta hoon",
  a female speaker says "karti hoon").
- For a flagged interruption fragment (a turn whose text is cut off with "—"),
  keep it a genuine incomplete fragment; do not complete the sentence.
- When the validator complains about excessive/repetitive backchanneling, DELETE
  the weakest, most redundant backchannel turns (turn_type "Backchanneling")
  rather than editing them — but keep at least a few so the flow stays natural.
- Preserve each edited turn's turn_type unless the issue is specifically about the
  turn_type being wrong.

Output format — return ONLY a single JSON object, no prose, no markdown fences:
{"edits": [
  {"turn_id": "<id>", "action": "replace", "text": "<new text>",
   "emotion": "<optional: Neutral|Happy|Sad|Angry>",
   "turn_type": "<optional: Normal|Overlapping|Interruption|Backchanneling>"},
  {"turn_id": "<id>", "action": "delete"}
]}
If nothing genuinely needs changing, return {"edits": []}.
"""


class ConversationEditorAgent(BaseAgent):
    """Produce a minimal edit patch that fixes validator-flagged issues."""

    # Not backed by Langfuse; prompt_name is only used for the base-class check.
    prompt_name = "conversation-editor-agent"
    temperature_key = "editor"

    def __init__(self, llm: BaseLLM | None = None) -> None:
        # Bypass Langfuse: install the self-contained system prompt directly so
        # this agent works without a managed prompt existing.
        from ..llm import GroqLLM

        if not self.prompt_name:
            raise ValueError(f"{type(self).__name__} must set a class-level prompt_name.")
        self.llm = llm if llm is not None else GroqLLM()
        self.langfuse_prompt = _LocalPrompt(_SYSTEM_PROMPT)

    def run(
        self,
        *,
        turns: list[dict[str, Any]],
        issues: list[Any] | None = None,
        feedback: str | None = None,
        language: str = "Hinglish",
        gender_pair: str | None = None,
        agent_emotion: str | None = None,
        user_emotion: str | None = None,
        agent_accent: str | None = None,
        user_accent: str | None = None,
        conversation_type: str | None = None,  # accepted from profile; not needed here
        **overrides: Any,
    ) -> list[dict[str, Any]]:
        """Return a list of edit dicts fixing the flagged ``issues``.

        Parameters
        ----------
        turns : list[dict]
            The current formatted conversation turns.
        issues : list
            Validator issues — each item may be a ``ValidationIssue`` dataclass or
            a plain dict with ``severity`` / ``turn_id`` / ``description``.
        feedback : str | None
            The validator's free-text feedback / regeneration guidance.
        language, gender_pair, agent_emotion, user_emotion, agent_accent, user_accent :
            Corpus-instance constraints, so edits stay consistent with them.

        Returns
        -------
        list[dict]
            Edit patches (possibly empty) for
            :meth:`ConversationFormatterAgent.apply_edits`.
        """
        prompt = self._build_prompt(
            turns=turns,
            issues=issues or [],
            feedback=feedback,
            language=language,
            gender_pair=gender_pair,
            agent_emotion=agent_emotion,
            user_emotion=user_emotion,
            agent_accent=agent_accent,
            user_accent=user_accent,
        )

        overrides.setdefault("response_format", {"type": "json_object"})
        raw = self._generate_json(
            prompt,
            stream=True,
            stream_label="Editing flagged turns (targeted fix)…",
            **overrides,
        )
        return self._extract_edits(raw)

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #
    def _build_prompt(
        self,
        *,
        turns: list[dict[str, Any]],
        issues: list[Any],
        feedback: str | None,
        language: str,
        gender_pair: str | None,
        agent_emotion: str | None,
        user_emotion: str | None,
        agent_accent: str | None,
        user_accent: str | None,
    ) -> str:
        lines: list[str] = [
            "Fix the specific issues listed below by editing ONLY the affected "
            "turns. Return the JSON edit patch described in the system prompt.",
            "",
            "## Conversation constraints",
            f"- language: {language}",
        ]
        for who, gender in self._speaker_genders(gender_pair).items():
            lines.append(f"- {who} gender: {gender}")
        if agent_emotion:
            lines.append(f"- speaker_1 (agent) emotion: {agent_emotion}")
        if user_emotion:
            lines.append(f"- speaker_2 (user) emotion: {user_emotion}")
        if agent_accent:
            lines.append(f"- speaker_1 (agent) accent: {agent_accent}")
        if user_accent:
            lines.append(f"- speaker_2 (user) accent: {user_accent}")

        lines += ["", "## Issues to fix"]
        for issue in issues:
            severity, turn_id, description = self._issue_fields(issue)
            tag = f"[{turn_id}] " if turn_id else "[whole conversation] "
            lines.append(f"- ({severity}) {tag}{description}")
        if feedback:
            lines += ["", "## Validator feedback", feedback]

        lines += [
            "",
            "## Current conversation turns (edit these; keep all others unchanged)",
            "```json",
            json.dumps(self._compact_turns(turns), ensure_ascii=False, indent=2),
            "```",
            "",
            "Return ONLY the single JSON edit patch — no prose, no markdown fences.",
        ]
        return "\n".join(lines)

    @staticmethod
    def _speaker_genders(gender_pair: str | None) -> dict[str, str]:
        """Map speaker_1/speaker_2 to Male/Female from a 'F-M'/'FM' style string."""
        if not gender_pair:
            return {}
        letters = [c for c in str(gender_pair).lower() if c in _GENDER_WORDS]
        out: dict[str, str] = {}
        if len(letters) >= 1:
            out["speaker_1 (agent)"] = _GENDER_WORDS[letters[0]]
        if len(letters) >= 2:
            out["speaker_2 (user)"] = _GENDER_WORDS[letters[1]]
        return out

    @staticmethod
    def _compact_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Send the editor only the semantic fields (timing is rebuilt on apply)."""
        keep = ("turn_id", "speaker", "turn_type", "emotion", "text", "overlaps_with")
        return [{k: t.get(k) for k in keep} for t in turns]

    @staticmethod
    def _issue_fields(issue: Any) -> tuple[str, str | None, str]:
        """Normalize a ValidationIssue dataclass or dict into (severity, turn_id, description)."""
        if isinstance(issue, dict):
            return (
                str(issue.get("severity", "minor")),
                issue.get("turn_id") or None,
                str(issue.get("description", "")).strip(),
            )
        return (
            str(getattr(issue, "severity", "minor")),
            getattr(issue, "turn_id", None) or None,
            str(getattr(issue, "description", "")).strip(),
        )

    # ------------------------------------------------------------------ #
    # Output extraction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_edits(result: Any) -> list[dict[str, Any]]:
        """Pull a clean list of edit dicts out of the model's JSON response."""
        if isinstance(result, dict):
            items = result.get("edits")
            if items is None:
                # Tolerate a bare list under a single key, or a single edit object.
                if "turn_id" in result:
                    items = [result]
                else:
                    values = [v for v in result.values() if isinstance(v, list)]
                    items = values[0] if values else []
        elif isinstance(result, list):
            items = result
        else:
            items = []

        edits: list[dict[str, Any]] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            item = {str(k).lower(): v for k, v in item.items()}
            tid = str(item.get("turn_id", "")).strip()
            if not tid:
                continue
            action = str(item.get("action", "replace")).strip().lower()
            if action not in {"replace", "delete"}:
                action = "replace"
            edit: dict[str, Any] = {"turn_id": tid, "action": action}
            if action == "replace":
                for key in ("text", "emotion", "turn_type"):
                    if item.get(key) is not None:
                        edit[key] = item[key]
            edits.append(edit)
        return edits

"""LLM judge for *formatting faithfulness* — transcript vs formatted turns.

This runs **after the formatter**, on conversations whose *content* has already
been approved by
:class:`~conversations_generator.agents.conversation_content_validator_agent.ConversationContentValidatorAgent`.
Its job is deliberately narrow and different from content validation: it does
NOT re-judge accent, emotion, realism, or naturalness. It only checks that the
formatter faithfully turned the approved transcript into JSON turns:

* every transcript line is present, once, in the same order (nothing dropped,
  added, merged, split, or reordered);
* each turn's ``text`` matches its transcript line (not reworded, translated, or
  "completed" — interrupt ``—`` fragments preserved exactly);
* ``speaker``, ``emotion`` and ``turn_type`` match the line's ``S1/S2`` tag,
  ``(Emotion)`` and ``[tag]``.

Deterministic schema/timing/overlap-symmetry is already covered by
:class:`~conversations_generator.agents.conversation_validator_manual.ConversationValidatorManual`,
so this agent ignores all timing and relational fields. A failure here means the
formatter mangled the conversion, so the caller re-runs the formatter (the
transcript, already good, is left untouched).

The system prompt is managed in Langfuse under the name
``conversation-format-validator-agent``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..llm import BaseLLM
from .base_agent import BaseAgent

VALID_VERDICTS = {"PASS", "FAIL", "NEEDS_REVIEW"}
VALID_SEVERITIES = {"critical", "major", "minor"}

# Timing/relational fields the faithfulness judge doesn't need (recomputed
# deterministically and already checked by ConversationValidatorManual).
_IGNORED_FIELDS = {
    "planned_start_sec", "planned_end_sec", "real_start_sec", "real_end_sec",
    "error_time", "overlaps_with", "overlaps_kind", "interrupted",
    "interrupted_by", "join_ratio",
}


@dataclass
class FormatIssue:
    """A single formatting-faithfulness problem."""

    severity: str            # "critical" | "major" | "minor"
    turn_id: str | None
    description: str


@dataclass
class FormatValidationReport:
    """Structured faithfulness judgement from :class:`ConversationFormatValidatorAgent`."""

    verdict: str = "NEEDS_REVIEW"                # "PASS" | "FAIL" | "NEEDS_REVIEW"
    issues: list[FormatIssue] = field(default_factory=list)
    feedback: str = ""
    raw: Any = None

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"

    def as_feedback(self) -> str:
        """Render this report as re-formatting feedback for the formatter."""
        parts: list[str] = []
        if self.feedback:
            parts.append(self.feedback)
        if self.issues:
            parts.append(
                "\n".join(
                    f"- ({i.severity}) "
                    + (f"[{i.turn_id}] " if i.turn_id else "")
                    + i.description
                    for i in self.issues
                )
            )
        return "\n".join(parts).strip()

    def print(self) -> None:
        icon = {"PASS": "✅", "FAIL": "❌", "NEEDS_REVIEW": "⚠️"}.get(self.verdict, "❓")
        print(f"{icon} Format verdict: {self.verdict}")
        if self.issues:
            print(f"{len(self.issues)} issue(s):")
            for i in self.issues:
                tag = f"[{i.turn_id}] " if i.turn_id else ""
                print(f"  - ({i.severity}) {tag}{i.description}")
        else:
            print("No faithfulness issues flagged.")
        if self.feedback:
            print(f"\nFeedback: {self.feedback}")


class ConversationFormatValidatorAgent(BaseAgent):
    """LLM judge for faithful transcript→JSON conversion.

    One LLM call per conversation. Sends the original transcript and the
    formatter's turns (timing/relational fields stripped) and parses a strict
    JSON verdict (schema defined in the ``conversation-format-validator-agent``
    prompt).
    """

    prompt_name = "conversation-format-validator-agent"
    temperature_key = "validator"

    def __init__(self, llm: BaseLLM | None = None) -> None:
        super().__init__(llm)

    def run(
        self,
        *,
        transcript: str,
        turns: list[dict[str, Any]],
        **overrides: Any,
    ) -> FormatValidationReport:
        """Check that ``turns`` faithfully represent ``transcript``.

        Parameters
        ----------
        transcript : str
            The approved tagged plain-text transcript.
        turns : list[dict]
            The formatter's output turns to check against the transcript.
        **overrides
            Extra kwargs forwarded to the LLM.
        """
        if not transcript or not transcript.strip():
            raise ValueError("Format validator received an empty transcript.")

        prompt = self._build_prompt(transcript=transcript, turns=turns)

        overrides.setdefault("response_format", {"type": "json_object"})
        raw_result = self._generate_json(
            prompt,
            stream=True,
            stream_label="Validating formatting faithfulness…",
            **overrides,
        )
        from ..logger import Logger
        Logger.debug(f"Format validator output:\n{json.dumps(raw_result, indent=2, ensure_ascii=False)}")
        return self._normalize(raw_result)

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #
    def _build_prompt(self, *, transcript: str, turns: list[dict[str, Any]]) -> str:
        lines = [
            "Check that the formatted JSON turns below are a FAITHFUL conversion "
            "of the transcript — same lines, same order, same wording, correct "
            "speaker/emotion/turn_type per tag. Judge conversion fidelity ONLY; "
            "do not judge dialogue quality, accent, or realism.",
            "",
            "## Source transcript (tagged plain text — the ground truth)",
            transcript,
            "",
            "## Formatted turns to check (timing/relational fields stripped)",
            "```json",
            json.dumps(self._strip_fields(turns), ensure_ascii=False, indent=2),
            "```",
            "",
            "Return ONLY the single JSON object described in the system prompt.",
        ]
        return "\n".join(lines)

    @staticmethod
    def _strip_fields(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep only the fields relevant to a faithful-conversion check."""
        return [{k: v for k, v in t.items() if k not in _IGNORED_FIELDS} for t in turns]

    # ------------------------------------------------------------------ #
    # Output normalization
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize(result: Any) -> FormatValidationReport:
        if isinstance(result, list):
            result = result[0] if result else {}
        if not isinstance(result, dict):
            raise ValueError(f"Expected a validation object, got {type(result).__name__}")

        result = {str(k).lower(): v for k, v in result.items()}

        verdict = str(result.get("verdict", "NEEDS_REVIEW")).strip().upper()
        if verdict not in VALID_VERDICTS:
            verdict = "NEEDS_REVIEW"

        issues: list[FormatIssue] = []
        for raw_issue in result.get("issues", []) or []:
            if not isinstance(raw_issue, dict):
                continue
            raw_issue = {str(k).lower(): v for k, v in raw_issue.items()}
            severity = str(raw_issue.get("severity", "minor")).strip().lower()
            if severity not in VALID_SEVERITIES:
                severity = "minor"
            description = str(raw_issue.get("description", "")).strip()
            if not description:
                continue
            issues.append(
                FormatIssue(
                    severity=severity,
                    turn_id=raw_issue.get("turn_id") or None,
                    description=description,
                )
            )

        return FormatValidationReport(
            verdict=verdict,
            issues=issues,
            feedback=str(result.get("feedback", "")).strip(),
            raw=result,
        )

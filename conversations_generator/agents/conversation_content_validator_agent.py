"""LLM judge for the *content* of a generated conversation transcript.

This runs **before the formatter**, directly on the tagged plain-text transcript
produced by :class:`~conversations_generator.agents.conversation_generator_agent.ConversationGeneratorAgent`.
It judges the two things a machine can't check mechanically:

1. **Corpus-instance fit** — does the dialogue actually reflect the instance it
   was generated for: ``language``, ``agent_emotion``, ``user_emotion``,
   ``agent_accent``, ``user_accent``, ``gender_pair``, ``conversation_type``,
   and relevance to the topic?
2. **Realism** — does it read like a real spoken conversation (natural phrasing,
   believable code-mixing, coherent turn-taking, no robotic repetition)?

It gates the pipeline: the transcript is only handed to the formatter once this
judge returns ``PASS``. If it fails, the generator regenerates with the feedback.
Because it runs on the raw transcript, no timing/schema noise is in front of the
model — it judges dialogue only. Faithfulness of the *formatting* is a separate
concern, checked afterwards by
:class:`~conversations_generator.agents.conversation_format_validator_agent.ConversationFormatValidatorAgent`.

The system prompt is managed in Langfuse under the name
``conversation-content-validator-agent``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..llm import BaseLLM
from .base_agent import BaseAgent

VALID_VERDICTS = {"PASS", "FAIL", "NEEDS_REVIEW"}
VALID_SEVERITIES = {"critical", "major", "minor"}

# Keys the judge scores inside `corpus_field_matches`.
CORPUS_FIELDS = [
    "language",
    "agent_emotion",
    "user_emotion",
    "agent_accent",
    "user_accent",
    "gender_pair",
    "conversation_type",
    "topic_relevance",
]


@dataclass
class ContentIssue:
    """A single content problem the judge flagged."""

    severity: str            # "critical" | "major" | "minor"
    turn_ref: str | None     # e.g. a quoted snippet or line number, else None
    description: str


@dataclass
class ContentValidationReport:
    """Structured content judgement from :class:`ConversationContentValidatorAgent`."""

    verdict: str = "NEEDS_REVIEW"                        # "PASS" | "FAIL" | "NEEDS_REVIEW"
    realism_score: float = 0.0                           # 0-10, "does it feel real"
    corpus_match_score: float = 0.0                      # 0-10, fit to corpus instance
    corpus_field_matches: dict[str, bool] = field(default_factory=dict)
    strengths: list[str] = field(default_factory=list)
    issues: list[ContentIssue] = field(default_factory=list)
    feedback: str = ""                                   # free-text regen guidance
    raw: Any = None

    @property
    def mismatched_fields(self) -> list[str]:
        """Corpus fields the judge says the transcript does NOT satisfy."""
        return [k for k, v in self.corpus_field_matches.items() if not v]

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"

    def as_feedback(self) -> str:
        """Render this report as regeneration feedback for the generator."""
        parts: list[str] = []
        if self.feedback:
            parts.append(self.feedback)
        if self.mismatched_fields:
            parts.append("Corpus fields NOT satisfied: " + ", ".join(self.mismatched_fields))
        if self.issues:
            parts.append(
                "\n".join(
                    f"- ({i.severity}) {i.description}"
                    + (f" [{i.turn_ref}]" if i.turn_ref else "")
                    for i in self.issues
                )
            )
        return "\n".join(parts).strip()

    def print(self) -> None:
        icon = {"PASS": "✅", "FAIL": "❌", "NEEDS_REVIEW": "⚠️"}.get(self.verdict, "❓")
        print(f"{icon} Content verdict: {self.verdict}")
        print(f"   Corpus match score: {self.corpus_match_score:.1f}/10")
        print(f"   Realism score:      {self.realism_score:.1f}/10")
        if self.corpus_field_matches:
            print("Corpus field matches:")
            for k, v in self.corpus_field_matches.items():
                print(f"  {'✔' if v else '✘'} {k}")
        if self.issues:
            print(f"{len(self.issues)} issue(s):")
            for i in self.issues:
                tag = f"[{i.turn_ref}] " if i.turn_ref else ""
                print(f"  - ({i.severity}) {tag}{i.description}")
        if self.feedback:
            print(f"\nFeedback: {self.feedback}")


class ConversationContentValidatorAgent(BaseAgent):
    """LLM judge for corpus-fit + realism of a plain-text conversation transcript.

    One LLM call per transcript. Sends the corpus-instance requirements, the
    topic, and the raw tagged transcript, and parses a strict JSON verdict
    (schema defined in the ``conversation-content-validator-agent`` prompt).
    """

    prompt_name = "conversation-content-validator-agent"
    temperature_key = "validator"

    def __init__(self, llm: BaseLLM | None = None) -> None:
        super().__init__(llm)

    def run(
        self,
        *,
        transcript: str,
        topic: dict[str, str],
        language: str = "Hinglish",
        conversation_type: str | None = None,
        agent_emotion: str | None = None,
        user_emotion: str | None = None,
        agent_accent: str | None = None,
        user_accent: str | None = None,
        gender_pair: str | None = None,
        **overrides: Any,
    ) -> ContentValidationReport:
        """Judge a transcript against the corpus instance it was generated for.

        Parameters
        ----------
        transcript : str
            The tagged plain-text transcript from ``ConversationGeneratorAgent``.
        topic : dict
            The topic dict (``title`` + ``context``) the conversation came from.
        language, agent_emotion, user_emotion, agent_accent, user_accent, gender_pair :
            The corpus instance's *required* attributes (usually
            ``**instance.to_profile()``).
        conversation_type : str | None
            Defaults to ``topic.get("conversation_type")`` when not given.
        **overrides
            Extra kwargs forwarded to the LLM.
        """
        if not transcript or not transcript.strip():
            raise ValueError("Content validator received an empty transcript.")

        conversation_type = conversation_type or topic.get("conversation_type")

        prompt = self._build_prompt(
            transcript=transcript,
            topic=topic,
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
        raw_result = self._generate_json(
            prompt,
            system_vars=system_vars,
            stream=True,
            stream_label="Validating transcript content…",
            **overrides,
        )
        from ..logger import Logger
        Logger.debug(f"Content validator output:\n{json.dumps(raw_result, indent=2, ensure_ascii=False)}")
        return self._normalize(raw_result)

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #
    def _build_prompt(
        self,
        *,
        transcript: str,
        topic: dict[str, str],
        language: str,
        conversation_type: str | None,
        agent_emotion: str | None,
        user_emotion: str | None,
        agent_accent: str | None,
        user_accent: str | None,
        gender_pair: str | None,
    ) -> str:
        lines: list[str] = [
            "Judge the transcript below against the corpus instance it was "
            "generated for, and how real it sounds.",
            "",
            "## Required attributes (the transcript MUST match these)",
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
            "## Topic",
            f"**Title:** {topic.get('title', '')}",
            f"**Context:** {topic.get('context', '')}",
            "",
            "## Transcript to judge (tagged plain text, one turn per line)",
            transcript,
            "",
            "Return ONLY the single JSON object described in the system prompt.",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Output normalization
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize(result: Any) -> ContentValidationReport:
        if isinstance(result, list):
            result = result[0] if result else {}
        if not isinstance(result, dict):
            raise ValueError(f"Expected a validation object, got {type(result).__name__}")

        result = {str(k).lower(): v for k, v in result.items()}

        verdict = str(result.get("verdict", "NEEDS_REVIEW")).strip().upper()
        if verdict not in VALID_VERDICTS:
            verdict = "NEEDS_REVIEW"

        field_matches_raw = result.get("corpus_field_matches", {}) or {}
        if isinstance(field_matches_raw, dict):
            field_matches_raw = {str(k).lower(): v for k, v in field_matches_raw.items()}
        corpus_field_matches = {k: bool(field_matches_raw.get(k, False)) for k in CORPUS_FIELDS}

        issues: list[ContentIssue] = []
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
                ContentIssue(
                    severity=severity,
                    turn_ref=(raw_issue.get("turn_ref") or raw_issue.get("turn_id") or None),
                    description=description,
                )
            )

        return ContentValidationReport(
            verdict=verdict,
            realism_score=_clamp_score(result.get("realism_score")),
            corpus_match_score=_clamp_score(result.get("corpus_match_score")),
            corpus_field_matches=corpus_field_matches,
            strengths=[str(s).strip() for s in (result.get("strengths") or []) if str(s).strip()],
            issues=issues,
            feedback=str(result.get("feedback", "")).strip(),
            raw=result,
        )


def _clamp_score(value: Any, default: float = 0.0, lo: float = 0.0, hi: float = 10.0) -> float:
    """Best-effort float coercion, clamped to the [lo, hi] scoring range."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, score))

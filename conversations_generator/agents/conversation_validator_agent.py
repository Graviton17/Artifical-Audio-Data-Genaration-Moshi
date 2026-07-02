"""LLM-based agent that judges corpus-fit and realism of a generated conversation.

Unlike :class:`~conversations_generator.agents.conversation_validator_manual.ConversationValidatorManual`
(which does deterministic, no-LLM timing/overlap/schema checks), this agent
makes one LLM call to judge the things that can't be checked mechanically:

1. **Corpus-instance fit** — does the generated conversation actually reflect
   the corpus instance it was generated for: ``language``, ``agent_emotion``,
   ``user_emotion``, ``agent_accent``, ``user_accent``, ``gender_pair``,
   ``conversation_type``, and relevance to the given topic?
2. **Realism / naturalness** — does it read like a real spoken conversation:
   natural phrasing and code-mixing (for Hinglish etc.), coherent turn-taking,
   emotionally consistent dialogue, no repetition/robotic filler, and
   overlaps/interruptions/backchannels that make sense given the *content*
   (not just the timestamps, which ``ConversationValidatorManual`` already
   checks).

The system prompt is managed in Langfuse under the name
``conversation-validator-agent``.

Typical usage
-------------
    from conversations_generator.agents import ConversationValidatorAgent

    validator = ConversationValidatorAgent()
    report = validator.run(
        turns=turns,
        topic=topic,
        **instance.to_profile(),
    )
    report.print()
    if not report.passed:
        ...  # regenerate, log, discard, escalate, etc.

It composes naturally as a fourth pipeline stage after
``ConversationRunner``'s existing topic -> conversation -> manual-validation
stages: run manual validation first (cheap, deterministic, catches schema/
timing bugs), then run this agent only on conversations that already pass
manual validation (saves LLM calls on obviously-broken output).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..llm import BaseLLM
from .base_agent import BaseAgent

VALID_VERDICTS = {"PASS", "FAIL", "NEEDS_REVIEW"}
VALID_SEVERITIES = {"critical", "major", "minor"}

# Keys the agent is asked to score inside `corpus_field_matches`.
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

# Fields dropped from the transcript before it's sent to the judge model —
# irrelevant to a content/realism judgement and just burns tokens. Manual
# timing correctness is already covered by ConversationValidatorManual.
_TIMING_FIELDS = {"planned_start_sec", "planned_end_sec", "real_start_sec", "real_end_sec", "error_time"}


@dataclass
class ValidationIssue:
    """A single problem the judge model flagged."""

    severity: str            # "critical" | "major" | "minor"
    turn_id: str | None
    description: str


@dataclass
class AgentValidationReport:
    """Structured judgement returned by :class:`ConversationValidatorAgent`."""

    verdict: str = "NEEDS_REVIEW"                       # "PASS" | "FAIL" | "NEEDS_REVIEW"
    realism_score: float = 0.0                          # 0-10, "does it feel real"
    corpus_match_score: float = 0.0                     # 0-10, fit to corpus instance
    corpus_field_matches: dict[str, bool] = field(default_factory=dict)
    strengths: list[str] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)
    feedback: str = ""                                  # free-text summary / regen guidance
    raw: Any = None                                      # untouched parsed LLM JSON

    # ------------------------------------------------------------------ #
    # Convenience properties
    # ------------------------------------------------------------------ #
    @property
    def critical_issues(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "critical"]

    @property
    def major_issues(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "major"]

    @property
    def minor_issues(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "minor"]

    @property
    def has_critical_issues(self) -> bool:
        return bool(self.critical_issues)

    @property
    def mismatched_fields(self) -> list[str]:
        """Corpus fields the judge model says the conversation does NOT satisfy."""
        return [k for k, v in self.corpus_field_matches.items() if not v]

    @property
    def passed(self) -> bool:
        """True only if the model returned a PASS verdict."""
        return self.verdict == "PASS"

    # ------------------------------------------------------------------ #
    # Pretty-printing (mirrors ConversationValidatorManual.ValidationReport)
    # ------------------------------------------------------------------ #
    def print(self) -> None:
        icon = {"PASS": "✅", "FAIL": "❌", "NEEDS_REVIEW": "⚠️"}.get(self.verdict, "❓")
        print(f"{icon} Verdict: {self.verdict}")
        print(f"   Corpus match score: {self.corpus_match_score:.1f}/10")
        print(f"   Realism score:      {self.realism_score:.1f}/10")

        if self.corpus_field_matches:
            print("Corpus field matches:")
            for k, v in self.corpus_field_matches.items():
                print(f"  {'✔' if v else '✘'} {k}")

        if self.strengths:
            print("Strengths:")
            for s in self.strengths:
                print(f"  + {s}")

        if self.issues:
            print(f"{len(self.issues)} issue(s):")
            for i in self.issues:
                tag = f"[{i.turn_id}] " if i.turn_id else ""
                print(f"  - ({i.severity}) {tag}{i.description}")
        else:
            print("No issues flagged.")

        if self.feedback:
            print(f"\nFeedback: {self.feedback}")


class ConversationValidatorAgent(BaseAgent):
    """LLM judge for corpus fit + realism of a generated conversation.

    One LLM call per conversation. Sends the corpus-instance requirements,
    the topic, and the transcript (timing fields stripped) to the model, and
    parses back a strict JSON verdict (see ``conversation-validator-agent``
    prompt in Langfuse for the exact schema demanded of the model).
    """

    prompt_name = "conversation-validator-agent"

    def __init__(self, llm: BaseLLM | None = None) -> None:
        super().__init__(llm)

    def run(
        self,
        *,
        turns: list[dict[str, Any]],
        topic: dict[str, str],
        language: str = "Hinglish",
        conversation_type: str | None = None,
        agent_emotion: str | None = None,
        user_emotion: str | None = None,
        agent_accent: str | None = None,
        user_accent: str | None = None,
        gender_pair: str | None = None,
        **overrides: Any,
    ) -> AgentValidationReport:
        """Judge a generated conversation against the corpus instance it was meant for.

        Parameters
        ----------
        turns : list[dict]
            Conversation turns to validate — output of
            ``ConversationGeneratorAgent.run`` / ``ConversationRunner.generate_conversation``.
        topic : dict
            The topic dict the conversation was generated from (``title`` +
            ``context``, and usually ``conversation_type``).
        language, agent_emotion, user_emotion, agent_accent, user_accent, gender_pair :
            The corpus instance's *required* attributes, i.e. what the
            conversation is supposed to look like. Typically passed as
            ``**instance.to_profile()``.
        conversation_type : str | None
            Defaults to ``topic.get("conversation_type")`` if not given.
        **overrides
            Extra kwargs forwarded to the LLM (temperature, max_tokens, etc.).

        Returns
        -------
        AgentValidationReport
        """
        conversation_type = conversation_type or topic.get("conversation_type")

        prompt = self._build_prompt(
            turns=turns,
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

        # Judging should be low-variance / deterministic-ish, not creative.
        overrides.setdefault("temperature", 0.2)
        overrides.setdefault("response_format", {"type": "json_object"})
        raw_result = self._generate_json(prompt, system_vars=system_vars, **overrides)
        from ..logger import Logger
        Logger.debug(f"Validator LLM Output:\n{json.dumps(raw_result, indent=2)}")
        return self._normalize(raw_result)

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #
    def _build_prompt(
        self,
        *,
        turns: list[dict[str, Any]],
        topic: dict[str, str],
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
            "Validate the conversation below against the corpus instance it was "
            "supposed to be generated for, and judge how real it feels.",
            "",
            "## Corpus instance requirements (what the conversation MUST match)",
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
            "## Topic the conversation was generated from",
            f"**Title:** {topic.get('title', '')}",
            f"**Context:** {topic.get('context', '')}",
            "",
            "## Conversation transcript to validate (timing fields stripped)",
            "```json",
            json.dumps(self._strip_timing(turns), ensure_ascii=False, indent=2),
            "```",
            "",
            "Return ONLY the single JSON object described in the system prompt — "
            "no prose, no markdown fences.",
        ]
        return "\n".join(lines)

    @staticmethod
    def _strip_timing(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop numeric timing fields the content/realism judge doesn't need."""
        return [{k: v for k, v in t.items() if k not in _TIMING_FIELDS} for t in turns]

    # ------------------------------------------------------------------ #
    # Output normalization
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize(result: Any) -> AgentValidationReport:
        """Coerce the model's JSON output into a clean :class:`AgentValidationReport`."""
        if isinstance(result, list):
            result = result[0] if result else {}
        if not isinstance(result, dict):
            raise ValueError(f"Expected a validation object, got {type(result).__name__}")

        # Normalize all keys to lowercase to avoid case-sensitivity bugs from LLM JSON
        result = {str(k).lower(): v for k, v in result.items()}

        verdict = str(result.get("verdict", "NEEDS_REVIEW")).strip().upper()
        if verdict not in VALID_VERDICTS:
            verdict = "NEEDS_REVIEW"

        field_matches_raw = result.get("corpus_field_matches", {}) or {}
        if isinstance(field_matches_raw, dict):
            field_matches_raw = {str(k).lower(): v for k, v in field_matches_raw.items()}
        corpus_field_matches = {
            k: bool(field_matches_raw.get(k, False)) for k in CORPUS_FIELDS
        }

        issues: list[ValidationIssue] = []
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
                ValidationIssue(
                    severity=severity,
                    turn_id=raw_issue.get("turn_id") or None,
                    description=description,
                )
            )

        return AgentValidationReport(
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
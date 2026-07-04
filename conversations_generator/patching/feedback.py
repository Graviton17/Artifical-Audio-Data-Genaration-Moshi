"""Structured validation feedback + attempt history for targeted patching.

The runner used to hand the conversation generator a single free-text
``feedback`` blob and ask it to regenerate the *whole* conversation. That works
for large models but small models re-break untouched turns while fixing one.

This module turns both validators' output into a list of :class:`FeedbackItem`
records that each carry a concrete ``turn_id`` (the "grep coordinates" the
:mod:`conversations_generator.patching.patch_engine` uses to locate exactly
which turns to edit), plus an :class:`AttemptHistory` that remembers what was
tried and what feedback came back on each round, so the fixer has memory across
rounds and doesn't oscillate between two failing states.

Nothing here calls an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # avoid import cycles at runtime; only needed for type hints
    from ..agents.conversation_validator_agent import AgentValidationReport
    from ..agents.conversation_validator_manual import ValidationReport

# turn_id placeholders the validators use for issues that are NOT tied to one
# concrete turn (whole-conversation duration, missing-field-before-id-known,
# etc.). Targets built from these can't be patched turn-by-turn.
NON_TURN_IDS = {"<conversation>", "<file>", "<missing>", "", None}


@dataclass
class FeedbackItem:
    """One actionable validation finding, normalised across both validators.

    Parameters
    ----------
    turn_id : str | None
        The turn the finding is about, or ``None`` / a ``<...>`` placeholder for
        conversation-level findings that aren't tied to a single turn.
    source : str
        Which validator produced it: ``"manual"`` (deterministic timing/schema)
        or ``"agent"`` (LLM realism/corpus judge).
    severity : str
        Normalised severity: ``"error"`` / ``"warning"`` for manual findings,
        ``"critical"`` / ``"major"`` / ``"minor"`` for agent findings.
    message : str
        Human/LLM-readable description of what's wrong (verbatim from the
        validator, which already spells out the offending field and values).
    blocking : bool
        Whether this finding, on its own, means the conversation is not
        acceptable yet (manual errors and agent critical/major issues).
    """

    turn_id: str | None
    source: str
    severity: str
    message: str
    blocking: bool

    @property
    def is_turn_scoped(self) -> bool:
        """True when this finding points at a real, patchable turn_id."""
        return self.turn_id not in NON_TURN_IDS

    def render(self) -> str:
        """One-line rendering used inside the fixer prompt."""
        where = f"[{self.turn_id}]" if self.is_turn_scoped else "[conversation]"
        return f"- ({self.source}/{self.severity}) {where}: {self.message}"


# ---------------------------------------------------------------------------- #
# Builders: validator reports -> FeedbackItem list
# ---------------------------------------------------------------------------- #
def from_manual_report(report: "ValidationReport") -> list[FeedbackItem]:
    """Convert a manual :class:`ValidationReport` into feedback items.

    Manual ``ERROR`` issues are blocking; ``WARNING`` issues (duration window,
    overlap-count minimums) are advisory and passed along non-blocking so the
    fixer can improve them opportunistically without them failing the round.
    """
    items: list[FeedbackItem] = []
    for issue in report.issues:
        is_error = issue.level == "ERROR"
        items.append(
            FeedbackItem(
                turn_id=issue.turn_id,
                source="manual",
                severity="error" if is_error else "warning",
                message=issue.message,
                blocking=is_error,
            )
        )
    return items


def from_agent_report(report: "AgentValidationReport") -> list[FeedbackItem]:
    """Convert an LLM :class:`AgentValidationReport` into feedback items.

    ``critical`` and ``major`` issues are treated as blocking; ``minor`` issues
    are advisory. The report's free-text ``feedback`` summary (if any) is added
    as a single conversation-level, non-blocking note for extra context.
    """
    items: list[FeedbackItem] = []
    for issue in report.issues:
        blocking = issue.severity in {"critical", "major"}
        items.append(
            FeedbackItem(
                turn_id=issue.turn_id,
                source="agent",
                severity=issue.severity,
                message=issue.description,
                blocking=blocking,
            )
        )
    if report.feedback:
        items.append(
            FeedbackItem(
                turn_id=None,
                source="agent",
                severity="minor",
                message=report.feedback,
                blocking=False,
            )
        )
    return items


def blocking_items(items: list[FeedbackItem]) -> list[FeedbackItem]:
    """Subset of feedback that must be fixed for the conversation to pass."""
    return [i for i in items if i.blocking]


def render_feedback(items: list[FeedbackItem]) -> str:
    """Render a feedback list as a prompt-ready bulleted block."""
    return "\n".join(i.render() for i in items)


# ---------------------------------------------------------------------------- #
# Attempt history — the fixer's cross-round memory
# ---------------------------------------------------------------------------- #
@dataclass
class AttemptRecord:
    """A single round in the generate/patch loop."""

    round_no: int
    phase: str                       # "generation" | "patch" | "regeneration"
    target_turn_ids: list[str]       # turns this round tried to fix (empty for generation)
    feedback: list[FeedbackItem]     # feedback that came back after this round

    def summarize(self) -> str:
        """Compact, human-readable one-block summary of this round."""
        header = f"Round {self.round_no} ({self.phase})"
        if self.target_turn_ids:
            header += f" — tried to fix turns: {', '.join(self.target_turn_ids)}"
        blocking = blocking_items(self.feedback)
        if not blocking:
            return f"{header}\n  result: all blocking issues cleared."
        lines = [f"{header}\n  resulting blocking issues:"]
        lines += [f"    {i.render()}" for i in blocking]
        return "\n".join(lines)


@dataclass
class AttemptHistory:
    """Accumulates a compact trail of every generation/patch round.

    Fed to :class:`~conversations_generator.agents.conversation_fixer_agent.ConversationFixerAgent`
    so it can see what was already tried and which issues persisted, and avoid
    oscillating between two broken states (fixing turn A in a way that re-breaks
    turn B, round after round).
    """

    records: list[AttemptRecord] = field(default_factory=list)

    def record(
        self,
        *,
        round_no: int,
        phase: str,
        feedback: list[FeedbackItem],
        target_turn_ids: list[str] | None = None,
    ) -> None:
        self.records.append(
            AttemptRecord(
                round_no=round_no,
                phase=phase,
                target_turn_ids=list(target_turn_ids or []),
                feedback=list(feedback),
            )
        )

    def render(self, max_rounds: int = 4) -> str:
        """Render the most recent ``max_rounds`` rounds for the fixer prompt."""
        if not self.records:
            return "(no previous attempts)"
        recent = self.records[-max_rounds:]
        return "\n\n".join(r.summarize() for r in recent)

    def persistent_turn_ids(self) -> list[str]:
        """Turn_ids that have shown up in blocking feedback more than once.

        These are the "stuck" turns the fixer keeps failing to satisfy; the
        prompt highlights them so the model pays extra attention (or rethinks
        the whole turn instead of nudging it again).
        """
        seen: dict[str, int] = {}
        for rec in self.records:
            for item in blocking_items(rec.feedback):
                if item.is_turn_scoped and item.turn_id is not None:
                    seen[item.turn_id] = seen.get(item.turn_id, 0) + 1
        return sorted(tid for tid, n in seen.items() if n > 1)

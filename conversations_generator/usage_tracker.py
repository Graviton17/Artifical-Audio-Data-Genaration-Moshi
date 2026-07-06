"""Per-agent LLM call tracking: timing, token usage, and stage context.

Each :class:`ConversationRunner.run` creates a fresh :class:`UsageTracker` that
collects every LLM call from topic generation through validation iterations.
Results are written to ``metadata.txt`` and the conversation JSON payload.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

# Stage / attempt context set by the runner around each agent call so parallel
# workers don't cross-contaminate records.
_current_stage: ContextVar[str] = ContextVar("usage_stage", default="unknown")
_current_attempt: ContextVar[int | None] = ContextVar("usage_attempt", default=None)
_active_tracker: ContextVar["UsageTracker | None"] = ContextVar("usage_tracker", default=None)


@dataclass
class AgentCallRecord:
    """One LLM invocation attributed to a pipeline agent."""

    agent: str
    stage: str
    attempt: int | None
    model: str
    input_tokens: int
    output_tokens: int
    duration_sec: float

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "stage": self.stage,
            "attempt": self.attempt,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "duration_sec": round(self.duration_sec, 3),
        }


@dataclass
class UsageTracker:
    """Accumulates LLM usage for one full pipeline run (including retries)."""

    records: list[AgentCallRecord] = field(default_factory=list)

    def record_call(
        self,
        *,
        agent: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration_sec: float = 0.0,
        stage: str | None = None,
        attempt: int | None = None,
    ) -> None:
        self.records.append(
            AgentCallRecord(
                agent=agent,
                stage=stage if stage is not None else _current_stage.get(),
                attempt=attempt if attempt is not None else _current_attempt.get(),
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_sec=duration_sec,
            )
        )

    def totals(self) -> dict[str, int | float]:
        input_t = sum(r.input_tokens for r in self.records)
        output_t = sum(r.output_tokens for r in self.records)
        return {
            "calls": len(self.records),
            "input_tokens": input_t,
            "output_tokens": output_t,
            "total_tokens": input_t + output_t,
            "duration_sec": round(sum(r.duration_sec for r in self.records), 3),
        }

    def by_agent(self) -> dict[str, dict[str, int | float]]:
        """Aggregate usage grouped by agent name."""
        grouped: dict[str, dict[str, int | float]] = {}
        for rec in self.records:
            bucket = grouped.setdefault(
                rec.agent,
                {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "duration_sec": 0.0,
                },
            )
            bucket["calls"] = int(bucket["calls"]) + 1
            bucket["input_tokens"] = int(bucket["input_tokens"]) + rec.input_tokens
            bucket["output_tokens"] = int(bucket["output_tokens"]) + rec.output_tokens
            bucket["total_tokens"] = int(bucket["total_tokens"]) + rec.total_tokens
            bucket["duration_sec"] = float(bucket["duration_sec"]) + rec.duration_sec
        for bucket in grouped.values():
            bucket["duration_sec"] = round(float(bucket["duration_sec"]), 3)
        return grouped

    def to_dict(self) -> dict[str, Any]:
        return {
            "totals": self.totals(),
            "by_agent": self.by_agent(),
            "calls": [r.to_dict() for r in self.records],
        }

    def format_metadata_lines(self) -> list[str]:
        """Human-readable lines for metadata.txt."""
        totals = self.totals()
        lines = [
            "## LLM usage",
            f"total_calls: {totals['calls']}",
            f"total_input_tokens: {totals['input_tokens']}",
            f"total_output_tokens: {totals['output_tokens']}",
            f"total_tokens: {totals['total_tokens']}",
            f"total_llm_duration_sec: {totals['duration_sec']}",
            "",
            "### By agent",
        ]
        for agent, stats in sorted(self.by_agent().items()):
            lines.append(
                f"{agent}: calls={stats['calls']}, "
                f"in={stats['input_tokens']}, out={stats['output_tokens']}, "
                f"total={stats['total_tokens']}, "
                f"duration_sec={stats['duration_sec']}"
            )
        lines += ["", "### Per call (chronological)"]
        for rec in self.records:
            attempt = f", attempt={rec.attempt}" if rec.attempt is not None else ""
            lines.append(
                f"- [{rec.stage}{attempt}] {rec.agent} ({rec.model}): "
                f"in={rec.input_tokens}, out={rec.output_tokens}, "
                f"duration_sec={rec.duration_sec:.3f}"
            )
        lines.append("")
        return lines


class usage_context:
    """Context manager to set stage/attempt for the active tracker."""

    def __init__(
        self,
        tracker: UsageTracker,
        *,
        stage: str,
        attempt: int | None = None,
    ) -> None:
        self.tracker = tracker
        self.stage = stage
        self.attempt = attempt
        self._stage_token: Any = None
        self._attempt_token: Any = None
        self._tracker_token: Any = None

    def __enter__(self) -> UsageTracker:
        self._tracker_token = _active_tracker.set(self.tracker)
        self._stage_token = _current_stage.set(self.stage)
        self._attempt_token = _current_attempt.set(self.attempt)
        return self.tracker

    def __exit__(self, *args: Any) -> None:
        if self._attempt_token is not None:
            _current_attempt.reset(self._attempt_token)
        if self._stage_token is not None:
            _current_stage.reset(self._stage_token)
        if self._tracker_token is not None:
            _active_tracker.reset(self._tracker_token)


def record_agent_call(
    *,
    agent: str,
    model: str,
    usage: dict[str, int] | None,
    duration_sec: float,
) -> None:
    """Record one LLM call if a tracker is active (no-op otherwise)."""
    tracker = _active_tracker.get()
    if tracker is None:
        return
    usage = usage or {}
    tracker.record_call(
        agent=agent,
        model=model,
        input_tokens=int(usage.get("input", 0) or 0),
        output_tokens=int(usage.get("output", 0) or 0),
        duration_sec=duration_sec,
    )

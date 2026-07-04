"""Manual (deterministic) timing/overlap validator for generated conversations.

Unlike :class:`~conversations_generator.agents.conversation_validator_agent.ConversationValidatorAgent`
(which will use an LLM to judge conversational quality), this validator does
no LLM calls at all. It mechanically checks that the overlap/interruption/
backchannel relationships a turn declares (via ``overlaps_with``,
``overlaps_kind``, ``interrupted``, ``interrupted_by``) are actually
consistent with its own ``planned_start_sec`` / ``planned_end_sec`` (or, once
alignment has run, ``real_start_sec`` / ``real_end_sec``) timestamps.

Typical usage
-------------
    from conversations_generator.agents.conversation_validator_manual import (
        ConversationValidatorManual,
    )

    validator = ConversationValidatorManual()
    report = validator.validate(turns)  # turns = list[dict], e.g. runner output
    report.print()
    if report.has_errors:
        raise ValueError("Generated conversation failed manual validation.")

Can also be run standalone against a JSONL file of turns:

    python -m conversations_generator.agents.conversation_validator_manual conversation.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------ #
# Schema constants (mirrors conversation_field_schema.json)
# ------------------------------------------------------------------ #
VALID_TURN_TYPES = {"Normal", "Overlapping", "Interruption", "Backchanneling"}
VALID_OVERLAP_KINDS = {"Overlapping", "Interruption", "Backchanneling", None}
VALID_EMOTIONS = {"Neutral", "Happy", "Sad", "Angry"}
VALID_SPEAKERS = {"speaker_1", "speaker_2"}

REQUIRED_FIELDS = {
    "turn_id", "speaker", "text", "emotion",
    "planned_start_sec", "planned_end_sec",
    "real_start_sec", "real_end_sec", "error_time",
    "turn_type", "overlaps_with", "overlaps_kind",
    "interrupted", "interrupted_by",
}

DEFAULT_TOLERANCE = 0.05           # seconds of slack allowed at boundaries
DEFAULT_MIN_DURATION_SEC = 240     # 4 min
DEFAULT_MAX_DURATION_SEC = 480     # 8 min
DEFAULT_MIN_OVERLAP_COUNT = 2      # per non-Normal turn_type


@dataclass
class Issue:
    """A single validation finding."""

    level: str      # "ERROR" or "WARNING"
    turn_id: str
    message: str


@dataclass
class ValidationReport:
    """Collects issues produced by a single :meth:`ConversationValidatorManual.validate` call."""

    issues: list[Issue] = field(default_factory=list)
    turn_type_counts: dict[str, int] = field(default_factory=dict)
    duration_sec: float | None = None

    def error(self, turn_id: str, message: str) -> None:
        self.issues.append(Issue("ERROR", turn_id, message))

    def warn(self, turn_id: str, message: str) -> None:
        self.issues.append(Issue("WARNING", turn_id, message))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "ERROR"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "WARNING"]

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    def print(self) -> None:
        if self.turn_type_counts:
            total = sum(self.turn_type_counts.values()) or 1
            print("Turn-type distribution:")
            for k, v in self.turn_type_counts.items():
                print(f"  {k:<15} {v:>4}  ({v / total * 100:5.1f}%)")
        if self.duration_sec is not None:
            print(f"Total duration: {self.duration_sec:.1f}s ({self.duration_sec / 60:.1f} min)")

        if not self.issues:
            print("✅ No issues found. All timing/overlap relationships are consistent.")
            return
        for i in self.errors:
            print(f"❌ ERROR   [{i.turn_id}] {i.message}")
        for i in self.warnings:
            print(f"⚠️  WARNING [{i.turn_id}] {i.message}")
        print(f"\n{len(self.errors)} error(s), {len(self.warnings)} warning(s).")


class ConversationValidatorManual:
    """Deterministic QA pass over a generated conversation's turn list.

    Validates schema correctness and the actual timing consistency of every
    overlap / interruption / backchannel relationship the LLM declared,
    against the timestamps it also generated (or, post-alignment, against
    ``real_start_sec`` / ``real_end_sec``).

    Parameters
    ----------
    tolerance : float
        Seconds of slack allowed when comparing turn boundaries, since
        planned timestamps are LLM estimates rather than audio-precise values.
    min_duration_sec, max_duration_sec : float
        Acceptable total-conversation-length window (default: 5-15 min).
    min_overlap_count : int
        Minimum number of Overlapping / Interruption / Backchanneling turns
        expected somewhere in the conversation.
    """

    def __init__(
        self,
        tolerance: float = DEFAULT_TOLERANCE,
        min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
        max_duration_sec: float = DEFAULT_MAX_DURATION_SEC,
        min_overlap_count: int = DEFAULT_MIN_OVERLAP_COUNT,
    ) -> None:
        self.tolerance = tolerance
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.min_overlap_count = min_overlap_count

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def validate(self, turns: list[dict[str, Any]], time_field: str = "planned") -> ValidationReport:
        """Run all checks against a list of turn dicts and return a report.

        Parameters
        ----------
        turns : list[dict]
            Turn objects following ``conversation_field_schema.json``
            (e.g. the ``turns`` list returned by ``ConversationRunner.run()``).
        time_field : {"planned", "real"}
            Which timestamp pair to validate against. Use ``"planned"`` right
            after LLM generation, and ``"real"`` once WhisperX alignment has
            filled ``real_start_sec`` / ``real_end_sec``.
        """
        report = ValidationReport()
        self._check_schema(turns, report)
        self._check_overlap_pairs(turns, report, time_field)
        self._check_conversation_level(turns, report, time_field)
        return report

    def validate_file(self, path: str | Path, time_field: str = "planned") -> ValidationReport:
        """Convenience wrapper: load a JSONL file of turns, then :meth:`validate`."""
        return self.validate(self._load_turns(path), time_field=time_field)

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_turns(path: str | Path) -> list[dict[str, Any]]:
        turns = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    turns.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_no} of {path}: {e}") from e
        return turns

    @staticmethod
    def _get_times(turn: dict[str, Any], time_field: str) -> tuple[float | None, float | None]:
        return turn.get(f"{time_field}_start_sec"), turn.get(f"{time_field}_end_sec")

    # ------------------------------------------------------------------ #
    # Structural / schema-level checks
    # ------------------------------------------------------------------ #
    def _check_schema(self, turns: list[dict[str, Any]], report: ValidationReport) -> None:
        seen_ids: set[str] = set()
        for t in turns:
            tid = t.get("turn_id", "<missing>")

            missing = REQUIRED_FIELDS - t.keys()
            if missing:
                report.error(tid, f"Missing required field(s): {sorted(missing)}")
                continue

            if tid in seen_ids:
                report.error(tid, "Duplicate turn_id.")
            seen_ids.add(tid)

            if t["speaker"] not in VALID_SPEAKERS:
                report.error(tid, f"Invalid speaker '{t['speaker']}'.")
            if t["emotion"] not in VALID_EMOTIONS:
                report.error(tid, f"Invalid emotion '{t['emotion']}'.")
            if t["turn_type"] not in VALID_TURN_TYPES:
                report.error(tid, f"Invalid turn_type '{t['turn_type']}'.")
            if t["overlaps_kind"] not in VALID_OVERLAP_KINDS:
                report.error(tid, f"Invalid overlaps_kind '{t['overlaps_kind']}'.")

            ps, pe = t.get("planned_start_sec"), t.get("planned_end_sec")
            if ps is None or pe is None:
                report.error(tid, "planned_start_sec/planned_end_sec must not be null.")
            elif pe <= ps:
                report.error(tid, f"planned_end_sec ({pe}) must be > planned_start_sec ({ps}).")

            if (t["overlaps_with"] is None) != (t["overlaps_kind"] is None):
                report.error(tid, "overlaps_with and overlaps_kind must both be set or both be null.")

            if t["interrupted"] and not t["interrupted_by"]:
                report.error(tid, "interrupted is true but interrupted_by is null.")
            if not t["interrupted"] and t["interrupted_by"]:
                report.error(tid, "interrupted_by is set but interrupted is false.")

    # ------------------------------------------------------------------ #
    # Timing / overlap relationship checks
    # ------------------------------------------------------------------ #
    def _check_overlap_pairs(
        self, turns: list[dict[str, Any]], report: ValidationReport, time_field: str,
    ) -> None:
        by_id = {t["turn_id"]: t for t in turns if "turn_id" in t}
        checked_pairs: set[frozenset[str]] = set()
        tol = self.tolerance

        for t in turns:
            tid = t.get("turn_id")
            ref_id = t.get("overlaps_with")
            kind = t.get("overlaps_kind")
            if not ref_id:
                continue  # plain Normal turn, nothing to check

            pair_key = frozenset({tid, ref_id})
            if pair_key in checked_pairs:
                continue  # already validated from the other side of this pair
            checked_pairs.add(pair_key)

            ref = by_id.get(ref_id)
            if ref is None:
                report.error(tid, f"overlaps_with references unknown turn_id '{ref_id}'.")
                continue

            if ref.get("overlaps_with") != tid:
                report.error(
                    tid,
                    f"Asymmetric overlap: '{tid}' points to '{ref_id}', but "
                    f"'{ref_id}'.overlaps_with = '{ref.get('overlaps_with')}'.",
                )
            if ref.get("overlaps_kind") != kind:
                report.error(
                    tid,
                    f"overlaps_kind mismatch with '{ref_id}': '{kind}' vs '{ref.get('overlaps_kind')}'.",
                )

            a_start, a_end = self._get_times(ref, time_field)
            b_start, b_end = self._get_times(t, time_field)
            if None in (a_start, a_end, b_start, b_end):
                report.warn(tid, f"Cannot verify timing against '{ref_id}' — {time_field}_* times missing.")
                continue

            if a_start <= b_start:
                first_id, first_s, first_e = ref_id, a_start, a_end
                second_id, second_s, second_e = tid, b_start, b_end
            else:
                first_id, first_s, first_e = tid, b_start, b_end
                second_id, second_s, second_e = ref_id, a_start, a_end

            overlaps_in_time = (first_s < second_s) and (second_s < first_e + tol)

            if kind == "Backchanneling":
                self._check_backchannel(
                    ref, ref_id, t, tid, first_id, first_s, first_e,
                    second_id, second_s, second_e, report,
                )
            elif kind == "Interruption":
                self._check_interruption(t, tid, ref, ref_id, time_field, report)
            elif kind == "Overlapping":
                self._check_collision(
                    t, ref, tid, ref_id, first_id, first_s, first_e,
                    second_id, second_s, second_e, overlaps_in_time, report,
                )

    def _check_backchannel(
        self, ref, ref_id, t, tid, first_id, first_s, first_e,
        second_id, second_s, second_e, report: ValidationReport,
    ) -> None:
        # The host is whichever turn starts first; the backchannel is expected
        # to sit inside it. (If the LLM wrote it the other way round, this
        # will correctly flag it as not nested.)
        tol = self.tolerance
        host_id, host_s, host_e = first_id, first_s, first_e
        sub_id, sub_s, sub_e = second_id, second_s, second_e

        if not (host_s - tol <= sub_s and sub_e <= host_e + tol):
            report.error(
                tid,
                f"Backchanneling '{sub_id}' [{sub_s}, {sub_e}] is not fully "
                f"inside host '{host_id}' [{host_s}, {host_e}].",
            )

        host_turn = ref if host_id == ref_id else t
        if host_turn.get("interrupted"):
            report.error(host_id, "Host of a Backchanneling turn should not be marked interrupted.")

    def _check_interruption(self, t, tid, ref, ref_id, time_field: str, report: ValidationReport) -> None:
        tol = self.tolerance
        if t.get("interrupted") and t.get("interrupted_by") == ref_id:
            victim, victim_id, interrupter, interrupter_id = t, tid, ref, ref_id
        elif ref.get("interrupted") and ref.get("interrupted_by") == tid:
            victim, victim_id, interrupter, interrupter_id = ref, ref_id, t, tid
        else:
            report.error(tid, "Interruption pair has no turn correctly marked interrupted=true/interrupted_by.")
            return

        v_s, v_e = self._get_times(victim, time_field)
        i_s, i_e = self._get_times(interrupter, time_field)
        if v_s is None or i_s is None:
            report.warn(tid, "Cannot verify interruption timing — timestamps missing.")
            return

        if not (v_s < i_s < v_e + tol):
            report.error(
                interrupter_id,
                f"Interrupter must start ({i_s}) inside victim '{victim_id}' span "
                f"[{v_s}, {v_e}]. Condition violated.",
            )
        if not (i_e > v_e - tol):
            report.warn(
                interrupter_id,
                f"Interrupter end ({i_e}) does not extend past victim end ({v_e}) — "
                f"unusual for a genuine interruption.",
            )
        if interrupter.get("interrupted"):
            report.error(interrupter_id, "Interrupter turn should not itself be marked interrupted.")

    def _check_collision(
        self, t, ref, tid, ref_id, first_id, first_s, first_e,
        second_id, second_s, second_e, overlaps_in_time: bool, report: ValidationReport,
    ) -> None:
        if not overlaps_in_time:
            report.error(
                tid,
                f"Overlapping pair '{first_id}' [{first_s}, {first_e}] and "
                f"'{second_id}' [{second_s}, {second_e}] do not actually intersect "
                f"(need: {first_id}.end > {second_id}.start).",
            )
        gap = second_s - first_s
        if gap > 3.0:
            report.warn(
                tid,
                f"Overlapping/collision start-time gap is {gap:.2f}s — real collisions are "
                f"usually near-simultaneous (<3s apart).",
            )
        if t.get("interrupted") or ref.get("interrupted"):
            report.warn(
                tid,
                "Overlapping (collision) turns are not usually marked interrupted; "
                "consider using turn_type/overlaps_kind 'Interruption' instead.",
            )

    # ------------------------------------------------------------------ #
    # Conversation-level checks
    # ------------------------------------------------------------------ #
    def _check_conversation_level(
        self, turns: list[dict[str, Any]], report: ValidationReport, time_field: str,
    ) -> None:
        if not turns:
            report.error("<file>", "No turns found.")
            return

        starts, ends = [], []
        for t in turns:
            s, e = self._get_times(t, time_field)
            if s is not None:
                starts.append(s)
            if e is not None:
                ends.append(e)

        if starts and ends:
            duration = max(ends) - min(starts)
            report.duration_sec = duration
            if not (self.min_duration_sec <= duration <= self.max_duration_sec):
                report.warn(
                    "<conversation>",
                    f"Total duration is {duration:.1f}s ({duration / 60:.1f} min) — outside "
                    f"the target {self.min_duration_sec / 60:.0f}-{self.max_duration_sec / 60:.0f} "
                    f"min range.",
                )

        counts = {k: 0 for k in VALID_TURN_TYPES}
        for t in turns:
            tt = t.get("turn_type")
            if tt in counts:
                counts[tt] += 1
        report.turn_type_counts = counts

        for kind in ("Overlapping", "Interruption", "Backchanneling"):
            if counts[kind] == 0:
                report.warn("<conversation>", f"No '{kind}' turns present — required minimum is {self.min_overlap_count}.")
            elif counts[kind] < self.min_overlap_count:
                report.warn(
                    "<conversation>",
                    f"Only {counts[kind]} '{kind}' turn(s) present — recommended minimum is {self.min_overlap_count}.",
                )

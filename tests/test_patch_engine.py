"""Unit tests for the deterministic patching engine (no LLM).

Covers the three mechanical guarantees the fixer relies on:
  * grep  — target expansion to flagged + structurally-linked turns
  * merge — untouched turns stay identical; disallowed edits are dropped
  * retime — the tail slides only when a fix changes a turn's duration
"""

from __future__ import annotations

import copy

from conversations_generator.patching import (
    AttemptHistory,
    FeedbackItem,
    apply_patch,
    collect_target_ids,
    context_window,
    linked_ids,
    merge_patch,
    reflow_tail,
)


def _turn(tid, start, end, **extra):
    base = {
        "turn_id": tid,
        "speaker": "speaker_1",
        "text": f"text {tid}",
        "emotion": "Neutral",
        "planned_start_sec": start,
        "planned_end_sec": end,
        "turn_type": "Normal",
        "overlaps_with": None,
        "overlaps_kind": None,
        "interrupted": False,
        "interrupted_by": None,
    }
    base.update(extra)
    return base


def _sample_conversation():
    return [
        _turn("t1", 0.0, 5.0),
        _turn("t2", 5.0, 10.0),
        _turn(
            "t3", 8.0, 12.0,
            turn_type="Interruption", overlaps_with="t2", overlaps_kind="Interruption",
        ),
        _turn("t4", 12.0, 18.0),
        _turn("t5", 18.0, 22.0),
    ]


# ----------------------------- grep --------------------------------------- #
def test_linked_ids_reads_both_relationship_fields():
    t = _turn("t3", 8, 12, overlaps_with="t2", interrupted_by="t9")
    assert linked_ids(t) == {"t2", "t9"}


def test_collect_targets_expands_to_linked_partner_both_directions():
    turns = _sample_conversation()
    # Flag only t3 (the interrupter). Its partner t2 must be pulled in too.
    assert collect_target_ids(turns, {"t3"}) == ["t2", "t3"]
    # Flag only t2 (the victim). The turn pointing AT it (t3) must be pulled in.
    assert collect_target_ids(turns, {"t2"}) == ["t2", "t3"]


def test_collect_targets_ignores_unknown_ids():
    turns = _sample_conversation()
    assert collect_target_ids(turns, {"does-not-exist"}) == []


def test_context_window_includes_neighbours_not_edited():
    turns = _sample_conversation()
    ctx = context_window(turns, {"t3"}, radius=1)
    assert ctx == ["t2", "t3", "t4"]


# ----------------------------- merge -------------------------------------- #
def test_apply_patch_replaces_only_target_and_keeps_others_identical():
    turns = _sample_conversation()
    original = copy.deepcopy(turns)
    patch = {"t2": _turn("t2", 5.0, 10.0, text="FIXED", emotion="Happy")}

    new_turns, applied = apply_patch(turns, patch, allowed_ids={"t2"})

    assert applied == ["t2"]
    assert new_turns[1]["text"] == "FIXED"
    assert new_turns[1]["emotion"] == "Happy"
    # Every other turn is byte-identical.
    for i in (0, 2, 3, 4):
        assert new_turns[i] == original[i]
    # Inputs are not mutated.
    assert turns == original


def test_apply_patch_drops_disallowed_ids():
    turns = _sample_conversation()
    patch = {
        "t2": _turn("t2", 5.0, 10.0, text="allowed"),
        "t5": _turn("t5", 18.0, 22.0, text="sneaky over-edit"),
    }
    new_turns, applied = apply_patch(turns, patch, allowed_ids={"t2"})
    assert applied == ["t2"]
    assert new_turns[1]["text"] == "allowed"
    assert new_turns[4]["text"] == "text t5"  # unchanged


def test_apply_patch_never_renumbers_turn_id():
    turns = _sample_conversation()
    # Model returns a turn under key t2 but with a wrong internal turn_id.
    patch = {"t2": _turn("WRONG", 5.0, 10.0, text="x")}
    new_turns, _ = apply_patch(turns, patch, allowed_ids={"t2"})
    assert new_turns[1]["turn_id"] == "t2"


# ----------------------------- retime ------------------------------------- #
def test_reflow_no_shift_when_duration_unchanged():
    turns = _sample_conversation()
    merged = copy.deepcopy(turns)
    merged[1]["text"] = "changed text only"  # same timing
    out = reflow_tail(turns, merged, applied_ids=["t2"])
    assert [t["planned_start_sec"] for t in out] == [0.0, 5.0, 8.0, 12.0, 18.0]


def test_reflow_slides_tail_when_edited_turn_grows():
    turns = _sample_conversation()
    merged = copy.deepcopy(turns)
    # t2 extended by 2s (10.0 -> 12.0); t3 overlaps t2 so it's edited together.
    merged[1]["planned_end_sec"] = 12.0
    merged[2]["planned_start_sec"] = 10.0
    merged[2]["planned_end_sec"] = 14.0
    out = reflow_tail(turns, merged, applied_ids=["t2", "t3"])
    by = {t["turn_id"]: t for t in out}
    # Region end went 12.0 -> 14.0 (delta +2). Turns entirely after slide by +2.
    assert by["t4"]["planned_start_sec"] == 14.0
    assert by["t4"]["planned_end_sec"] == 20.0
    assert by["t5"]["planned_start_sec"] == 20.0
    assert by["t5"]["planned_end_sec"] == 24.0
    # Edited turns keep the fixer's timing; earlier turn untouched.
    assert by["t1"]["planned_start_sec"] == 0.0
    assert by["t2"]["planned_end_sec"] == 12.0


def test_merge_patch_end_to_end():
    turns = _sample_conversation()
    patch = {"t4": _turn("t4", 12.0, 20.0, text="longer turn")}  # +2s duration
    out, applied = merge_patch(turns, patch, allowed_ids={"t4"})
    by = {t["turn_id"]: t for t in out}
    assert applied == ["t4"]
    assert by["t4"]["text"] == "longer turn"
    assert by["t5"]["planned_start_sec"] == 20.0  # slid from 18.0 by +2
    assert by["t5"]["planned_end_sec"] == 24.0


# ------------------------- attempt history -------------------------------- #
def test_history_flags_persistently_failing_turns():
    history = AttemptHistory()
    fb1 = [FeedbackItem("t3", "manual", "error", "bad interruption", blocking=True)]
    fb2 = [FeedbackItem("t3", "manual", "error", "still bad", blocking=True)]
    history.record(round_no=1, phase="generation", feedback=fb1)
    history.record(round_no=2, phase="patch", feedback=fb2, target_turn_ids=["t3"])
    assert history.persistent_turn_ids() == ["t3"]
    assert "Round 2 (patch)" in history.render()

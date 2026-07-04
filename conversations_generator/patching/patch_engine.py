"""Deterministic patching engine — the "grep + merge + retime" core.

This module contains *no* LLM calls. It answers three mechanical questions that
surround the fixer LLM:

1. **grep** — given feedback, *which* turns should be edited? That's the
   flagged turns plus the turns they are structurally linked to (an overlap
   partner via ``overlaps_with`` or an interruption partner via
   ``interrupted_by``), because those relationships can only be made consistent
   by editing both sides. See :func:`collect_target_ids`.

2. **merge** — given the fixer's patch (corrected turns keyed by ``turn_id``),
   splice them back into the full conversation so every *untouched* turn stays
   byte-for-byte identical. Turn_ids outside the allowed target set are dropped,
   so a model that over-edits can't silently rewrite the rest of the
   conversation. See :func:`apply_patch`.

3. **retime** — if a fix changed a turn's duration, the turns *after* it need to
   slide by the same delta so the timeline stays contiguous. This is done
   deterministically in code (small models are bad at timeline arithmetic). See
   :func:`reflow_tail`.

The single entry point most callers want is :func:`merge_patch`, which does
2 then 3 and returns the new conversation.
"""

from __future__ import annotations

import copy
from typing import Any

# Boundary slack (seconds) when deciding which turns count as "after" the edited
# region. Mirrors the manual validator's tolerance so the two agree on adjacency.
DEFAULT_TOLERANCE = 0.05


def index_by_id(turns: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map ``turn_id`` -> turn dict (last one wins on duplicates)."""
    return {t["turn_id"]: t for t in turns if isinstance(t, dict) and "turn_id" in t}


# ---------------------------------------------------------------------------- #
# 1. grep — which turns to edit
# ---------------------------------------------------------------------------- #
def linked_ids(turn: dict[str, Any]) -> set[str]:
    """Turn_ids this turn is structurally tied to and can't be fixed without.

    A turn's overlap/interruption consistency depends on its partner, so if we
    edit one side we must be allowed to edit the other.
    """
    out: set[str] = set()
    for key in ("overlaps_with", "interrupted_by"):
        ref = turn.get(key)
        if isinstance(ref, str) and ref:
            out.add(ref)
    return out


def collect_target_ids(
    turns: list[dict[str, Any]],
    flagged_ids: set[str],
) -> list[str]:
    """Expand flagged turn_ids to the full editable set (flagged + linked).

    For every flagged turn we add the turns it references *and* any turns that
    reference it — the relationship is symmetric and both endpoints may need to
    move for the pair to become consistent. Only ids that actually exist in
    ``turns`` are returned, in conversation order for stable prompts/tests.
    """
    by_id = index_by_id(turns)
    targets: set[str] = {tid for tid in flagged_ids if tid in by_id}

    for tid in list(targets):
        targets |= {ref for ref in linked_ids(by_id[tid]) if ref in by_id}

    # ...and the reverse direction: any turn pointing *at* a flagged turn.
    for t in turns:
        tid = t.get("turn_id")
        if tid in by_id and (linked_ids(t) & targets):
            targets.add(tid)

    order = {t.get("turn_id"): i for i, t in enumerate(turns)}
    return sorted(targets, key=lambda x: order.get(x, 1_000_000))


def context_window(
    turns: list[dict[str, Any]],
    target_ids: set[str],
    radius: int = 1,
) -> list[str]:
    """Turn_ids to *show* the fixer as read-only context (targets + neighbours).

    Gives the model the immediately-surrounding turns so a rewritten turn still
    flows naturally, without letting it edit them.
    """
    context: set[str] = set()
    for i, t in enumerate(turns):
        if t.get("turn_id") in target_ids:
            lo, hi = max(0, i - radius), min(len(turns), i + radius + 1)
            for j in range(lo, hi):
                if "turn_id" in turns[j]:
                    context.add(turns[j]["turn_id"])
    order = {t.get("turn_id"): i for i, t in enumerate(turns)}
    return sorted(context, key=lambda x: order.get(x, 1_000_000))


# ---------------------------------------------------------------------------- #
# 2. merge — splice the patch back in
# ---------------------------------------------------------------------------- #
def apply_patch(
    turns: list[dict[str, Any]],
    patch: dict[str, dict[str, Any]],
    allowed_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Replace patched turns by ``turn_id``, leaving every other turn untouched.

    Parameters
    ----------
    turns : list[dict]
        The current full conversation.
    patch : dict[str, dict]
        ``turn_id`` -> corrected turn dict, as returned by the fixer.
    allowed_ids : set[str] | None
        If given, patch entries whose id is not in this set are ignored — this
        enforces the "only edit flagged + linked turns" guarantee deterministically,
        no matter what the model returns.

    Returns
    -------
    (new_turns, applied_ids)
        The merged conversation (a deep copy; inputs are not mutated) and the
        list of turn_ids that were actually applied.
    """
    applied: list[str] = []
    new_turns: list[dict[str, Any]] = []
    for t in turns:
        tid = t.get("turn_id")
        replacement = patch.get(tid) if tid is not None else None
        if replacement is not None and (allowed_ids is None or tid in allowed_ids):
            merged = copy.deepcopy(replacement)
            merged["turn_id"] = tid  # never let a patch renumber a turn
            new_turns.append(merged)
            applied.append(tid)
        else:
            new_turns.append(copy.deepcopy(t))
    return new_turns, applied


# ---------------------------------------------------------------------------- #
# 3. retime — slide the tail if an edit changed a turn's duration
# ---------------------------------------------------------------------------- #
def reflow_tail(
    original: list[dict[str, Any]],
    merged: list[dict[str, Any]],
    applied_ids: list[str],
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[dict[str, Any]]:
    """Shift turns after the edited region so the timeline stays contiguous.

    If the fix changed the end time of the edited region (e.g. a turn got longer
    or shorter), every turn that originally started at/after the region's end is
    slid by that same delta. Turns *inside* or *before* the region keep whatever
    timing the fixer produced. This preserves all gaps and overlap offsets in the
    untouched tail — it moves as one rigid block — which is safe and predictable,
    unlike asking a small model to recompute the whole timeline.

    A no-op when the region's end time is unchanged (the common case for schema /
    metadata / relationship fixes that don't alter durations).
    """
    if not applied_ids:
        return merged

    orig_by_id = index_by_id(original)
    new_by_id = index_by_id(merged)

    def end_of(by_id: dict[str, dict[str, Any]], tid: str) -> float | None:
        t = by_id.get(tid)
        val = t.get("planned_end_sec") if t else None
        return float(val) if isinstance(val, (int, float)) else None

    old_ends = [e for tid in applied_ids if (e := end_of(orig_by_id, tid)) is not None]
    new_ends = [e for tid in applied_ids if (e := end_of(new_by_id, tid)) is not None]
    if not old_ends or not new_ends:
        return merged

    region_end_before = max(old_ends)
    region_end_after = max(new_ends)
    delta = region_end_after - region_end_before
    if abs(delta) <= tolerance:
        return merged  # duration unchanged — nothing to slide

    applied_set = set(applied_ids)
    for t in merged:
        tid = t.get("turn_id")
        if tid in applied_set:
            continue  # edited turns keep the fixer's timing
        start = t.get("planned_start_sec")
        # Only slide turns that lie entirely after the edited region.
        if isinstance(start, (int, float)) and start >= region_end_before - tolerance:
            for key in ("planned_start_sec", "planned_end_sec"):
                val = t.get(key)
                if isinstance(val, (int, float)):
                    t[key] = round(val + delta, 3)
    return merged


def merge_patch(
    turns: list[dict[str, Any]],
    patch: dict[str, dict[str, Any]],
    allowed_ids: set[str] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Apply a patch and reflow the tail in one call.

    Returns the new conversation and the list of turn_ids actually applied.
    """
    merged, applied = apply_patch(turns, patch, allowed_ids=allowed_ids)
    merged = reflow_tail(turns, merged, applied, tolerance=tolerance)
    return merged, applied

"""Targeted, Cursor-style patching of generated conversations.

Instead of regenerating a whole conversation every time validation fails (which
makes small models re-break turns they'd already got right), this package lets
the pipeline fix *only* the turns the validators flagged:

* :mod:`.feedback`     — normalise both validators' output into ``turn_id``-tagged
  :class:`FeedbackItem`s and keep an :class:`AttemptHistory` across rounds.
* :mod:`.patch_engine` — deterministically grep the turns to edit, merge the
  fixer's patch back in, and retime the tail. No LLM calls.

The LLM that actually rewrites the flagged turns lives in
:class:`conversations_generator.agents.conversation_fixer_agent.ConversationFixerAgent`.
"""

from __future__ import annotations

from .feedback import (
    AttemptHistory,
    AttemptRecord,
    FeedbackItem,
    blocking_items,
    from_agent_report,
    from_manual_report,
    render_feedback,
)
from .patch_engine import (
    apply_patch,
    collect_target_ids,
    context_window,
    index_by_id,
    linked_ids,
    merge_patch,
    reflow_tail,
)

__all__ = [
    "AttemptHistory",
    "AttemptRecord",
    "FeedbackItem",
    "apply_patch",
    "blocking_items",
    "collect_target_ids",
    "context_window",
    "from_agent_report",
    "from_manual_report",
    "index_by_id",
    "linked_ids",
    "merge_patch",
    "reflow_tail",
    "render_feedback",
]

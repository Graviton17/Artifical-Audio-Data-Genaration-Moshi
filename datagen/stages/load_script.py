"""Stage [1]: parse + validate the input conversation JSON.

Input schema (explicit-overlap form)::

    {
      "conversation_id": "current_affairs_01",
      "language": "gu",
      "speakers": {"user1": {"voice": ...}, "user2": {"voice": ...}},
      "turns": [
        {"speaker": "user1", "text": "...", "gap": 0.3},
        {"speaker": "user2", "text": "...", "overlap": 0.5}
      ]
    }

Validation drops the whole conversation (rather than crashing the run) if it is
malformed, so one bad file does not abort a batch.
"""

from __future__ import annotations

import json

from ..models import ConversationScript, GenContext, Turn
from .base import Stage


class LoadScriptStage(Stage):
    def _run(self, ctx: GenContext) -> GenContext:
        try:
            raw = json.loads(ctx.script_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            ctx.drop(f"load: cannot parse JSON ({exc})")
            return ctx

        conv_id = str(raw.get("conversation_id") or ctx.script_path.stem)
        language = str(raw.get("language") or "")
        speakers = raw.get("speakers") or {}
        raw_turns = raw.get("turns") or []

        if not isinstance(speakers, dict) or len(speakers) != 2:
            ctx.drop("load: 'speakers' must define exactly two speakers")
            return ctx
        if not raw_turns:
            ctx.drop("load: no turns")
            return ctx

        speaker_ids = set(speakers.keys())
        turns: list[Turn] = []
        for i, t in enumerate(raw_turns):
            spk = t.get("speaker")
            text = (t.get("text") or "").strip()
            if spk not in speaker_ids:
                ctx.drop(f"load: turn {i} speaker {spk!r} not in {sorted(speaker_ids)}")
                return ctx
            if not text:
                ctx.drop(f"load: turn {i} has empty text")
                return ctx
            gap = float(t.get("gap", 0.0) or 0.0)
            overlap = float(t.get("overlap", 0.0) or 0.0)
            if gap and overlap:
                # mutually exclusive; prefer overlap and warn
                self.log.warning("turn %d has both gap and overlap; using overlap", i)
                gap = 0.0
            turns.append(Turn(speaker=spk, text=text, gap=gap, overlap=overlap))

        ctx.script = ConversationScript(
            conversation_id=conv_id,
            language=language,
            speakers={k: (v or {}) for k, v in speakers.items()},
            turns=turns,
        )
        ctx.metadata["num_turns"] = len(turns)
        return ctx

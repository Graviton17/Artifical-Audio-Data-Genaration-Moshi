"""Scheduling strategy: assign each synthesized clip a global timeline offset.

Turn placement is conversational: each turn is positioned relative to the end of
the *previous* turn. A ``gap`` pushes it later (silence); an ``overlap`` pulls it
earlier so both speakers talk at once -- which is fine because they live on
separate channels, so overlap never mixes acoustically.

``BaseScheduler`` is the swappable interface; :class:`OverlapScheduler` is the
default explicit-control implementation matching the input JSON schema.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import SynthClip
from ..utils.logging import get_logger

log = get_logger("scheduler")


class BaseScheduler(ABC):
    @abstractmethod
    def schedule(self, clips: list[SynthClip]) -> float:
        """Set ``clip.start``/``clip.end`` (seconds) on each clip in place.

        Returns the total timeline duration (seconds).
        """


class OverlapScheduler(BaseScheduler):
    def __init__(
        self,
        default_gap_sec: float = 0.25,
        max_overlap_sec: float = 3.0,
        lead_silence_sec: float = 0.1,
    ):
        self.default_gap = default_gap_sec
        self.max_overlap = max_overlap_sec
        self.lead_silence = lead_silence_sec

    def schedule(self, clips: list[SynthClip]) -> float:
        prev_end = self.lead_silence
        total = 0.0
        for i, clip in enumerate(clips):
            dur = clip.audio.duration if clip.audio is not None else 0.0
            turn = clip.turn

            if i == 0:
                start = self.lead_silence
            elif turn.overlap and turn.overlap > 0:
                overlap = min(turn.overlap, self.max_overlap)
                start = max(0.0, prev_end - overlap)
            else:
                gap = turn.gap if turn.gap and turn.gap > 0 else self.default_gap
                start = prev_end + gap

            clip.start = round(start, 3)
            clip.end = round(start + dur, 3)
            prev_end = clip.end
            total = max(total, clip.end)

        return round(total, 3)

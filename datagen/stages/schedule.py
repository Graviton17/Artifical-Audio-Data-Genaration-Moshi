"""Stage [4]: place clips on the global timeline and build per-speaker tracks.

The scheduler assigns each clip a global ``start``/``end`` (honouring gaps and
overlaps). We then:

  * shift each clip's word timings by its global ``start`` so the alignments live
    on the conversation timeline, and
  * render one mono track per speaker: a silent buffer of the full conversation
    length with each of that speaker's clips written at its offset.

Because each speaker has its own track, overlapping turns never mix acoustically
-- exactly what Moshi full-duplex needs.
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..models import AudioBuffer, GenContext
from ..strategies.scheduler import BaseScheduler
from .base import Stage


class ScheduleStage(Stage):
    def __init__(self, scheduler: BaseScheduler, config: Config):
        super().__init__()
        self.scheduler = scheduler
        self.config = config

    def _run(self, ctx: GenContext) -> GenContext:
        script = ctx.script
        assert script is not None
        sr = ctx.sample_rate

        total = self.scheduler.schedule(ctx.clips)
        if total <= 0:
            ctx.drop("schedule: zero-length timeline")
            return ctx

        n_total = int(round(total * sr))
        tracks: dict[str, np.ndarray] = {
            spk: np.zeros(n_total, dtype=np.float32) for spk in script.speaker_ids()
        }

        for clip in ctx.clips:
            # shift clip-relative word times onto the global timeline
            for w in clip.words:
                w.start = round(w.start + clip.start, 3)
                w.end = round(w.end + clip.start, 3)

            track = tracks[clip.turn.speaker]
            a = int(round(clip.start * sr))
            seg = clip.audio.channel(0)
            b = min(n_total, a + len(seg))
            if b > a:
                # additive mix in case the same speaker's clips ever touch (they
                # normally don't); clip to track bounds.
                track[a:b] += seg[: b - a]

        ctx.tracks = {spk: AudioBuffer(arr[None, :], sr) for spk, arr in tracks.items()}
        ctx.metadata["duration"] = total
        return ctx

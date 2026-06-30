"""Stage [3]: forced-align the KNOWN text of each clip -> word timings.

No ASR: the words are exactly the script text; the aligner only computes timing.
Timings are clip-relative here; :mod:`datagen.stages.schedule` shifts them onto
the global timeline. Each word is tagged with its turn speaker id.
"""

from __future__ import annotations

from ..config import Config
from ..models import GenContext
from ..strategies.aligner import BaseAligner
from .base import Stage


class AlignStage(Stage):
    def __init__(self, aligner: BaseAligner, config: Config):
        super().__init__()
        self.aligner = aligner
        self.config = config

    def _run(self, ctx: GenContext) -> GenContext:
        script = ctx.script
        assert script is not None
        language = script.language or self.config.aligner.language

        for clip in ctx.clips:
            words = self.aligner.align(clip.audio, clip.turn.text, language)
            for w in words:
                w.speaker = clip.turn.speaker
            clip.words = words
        return ctx

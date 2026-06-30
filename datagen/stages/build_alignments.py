"""Stage [6]: build the moshi ``alignments`` list for each stereo variant.

The interleaver in moshi-finetune reads ``[[word, [start, end], label], ...]``,
both speakers interleaved by time, with the agent labelled SPEAKER_MAIN and the
other speaker SPEAKER_OTHER. The label depends on which speaker is the agent in
the variant, so this is computed per variant (the words/timings are shared).
"""

from __future__ import annotations

from ..config import Config
from ..models import GenContext, Word
from .base import Stage


class BuildAlignmentsStage(Stage):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def _run(self, ctx: GenContext) -> GenContext:
        # all words on the global timeline, sorted by start then end
        words: list[Word] = [w for clip in ctx.clips for w in clip.words]
        words.sort(key=lambda w: (w.start, w.end))

        main_label = self.config.labels.main_label
        user_label = self.config.labels.user_label

        for v in ctx.variants:
            v.alignments = [
                [
                    w.text,
                    [w.start, w.end],
                    main_label if w.speaker == v.main_speaker else user_label,
                ]
                for w in words
            ]
        return ctx

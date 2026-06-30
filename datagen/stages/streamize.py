"""Stage [5]: build stereo variants from the per-speaker tracks.

Moshi expects stereo with channel 0 = agent/main, channel 1 = the other speaker.
With channel-swap augmentation we emit both orderings (user1=agent and
user2=agent) so the model is not biased to one position -- this is the "jumble
Moshi as user1 / user2" the real pipeline also does.
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..models import AudioBuffer, GenContext, StreamVariant
from .base import Stage


class StreamizeStage(Stage):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def _run(self, ctx: GenContext) -> GenContext:
        script = ctx.script
        assert script is not None
        spk_ids = script.speaker_ids()
        if len(spk_ids) != 2:
            ctx.drop("streamize: need exactly two speakers")
            return ctx
        spk_a, spk_b = spk_ids
        sr = ctx.sample_rate

        orderings = [(spk_a, spk_b)]
        if self.config.augment.channel_swap:
            orderings.append((spk_b, spk_a))

        for main, user in orderings:
            stereo = np.stack(
                [ctx.tracks[main].channel(0), ctx.tracks[user].channel(0)], axis=0
            )
            ctx.variants.append(
                StreamVariant(
                    name=f"{ctx.stem}__{main}_agent",
                    main_speaker=main,
                    user_speaker=user,
                    stereo=AudioBuffer(samples=stereo, sample_rate=sr),
                )
            )
        return ctx

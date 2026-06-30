"""Stage [2]: synthesize each turn with the TTS strategy.

Each turn -> one :class:`SynthClip` holding a mono :class:`AudioBuffer` resampled
to the project sample rate. Results are cached by content hash so editing one
turn only re-synthesizes that turn.
"""

from __future__ import annotations

from ..config import Config
from ..models import GenContext, SynthClip
from ..strategies.tts import BaseTTS
from ..utils.audio_io import peak_normalize, resample
from ..utils.cache import Cache
from ..models import AudioBuffer
from .base import Stage


class SynthesizeStage(Stage):
    def __init__(self, tts: BaseTTS, config: Config, cache: Cache):
        super().__init__()
        self.tts = tts
        self.config = config
        self.cache = cache

    def _run(self, ctx: GenContext) -> GenContext:
        script = ctx.script
        assert script is not None
        language = script.language or self.config.tts.default_language
        target_sr = ctx.sample_rate

        clips: list[SynthClip] = []
        for turn in script.turns:
            voice = (script.speakers.get(turn.speaker) or {}).get("voice")
            key = Cache.key(
                model=self.config.tts.model_id,
                lang=language,
                voice=voice,
                text=turn.text,
                sr=target_sr,
            )
            buf = self.cache.get_audio(key)
            if buf is None:
                buf = self.tts.synthesize(turn.text, voice, language)
                buf = resample(buf, target_sr)
                buf = AudioBuffer(peak_normalize(buf.channel(0), self.config.cleanup.peak_limit_db), target_sr)
                self.cache.put_audio(key, buf)

            if buf.num_samples == 0:
                ctx.drop(f"synthesize: empty audio for turn {turn.text!r}")
                return ctx
            clips.append(SynthClip(turn=turn, audio=buf))

        ctx.clips = clips
        return ctx

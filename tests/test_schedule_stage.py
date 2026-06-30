"""ScheduleStage: per-speaker tracks (clean separation) + global word shifting."""

from pathlib import Path

import numpy as np

from datagen.config import Config
from datagen.models import AudioBuffer, ConversationScript, GenContext, SynthClip, Turn, Word
from datagen.stages.schedule import ScheduleStage
from datagen.strategies.scheduler import OverlapScheduler


def _tone(dur, sr=24000):
    return AudioBuffer((0.3 * np.ones(int(dur * sr))).astype(np.float32), sr)


def _ctx():
    ctx = GenContext(script_path=Path("x"), sample_rate=24000, cache_dir=".", out_dir=".")
    ctx.script = ConversationScript(
        conversation_id="c", language="gu",
        speakers={"user1": {}, "user2": {}}, turns=[],
    )
    c1 = SynthClip(turn=Turn("user1", "hi"), audio=_tone(1.0))
    c1.words = [Word("hi", 0.0, 0.5, "user1")]
    c2 = SynthClip(turn=Turn("user2", "yo", overlap=0.5), audio=_tone(1.0))
    c2.words = [Word("yo", 0.0, 0.5, "user2")]
    ctx.clips = [c1, c2]
    return ctx


def _stage():
    return ScheduleStage(OverlapScheduler(lead_silence_sec=0.0, default_gap_sec=0.25), Config())


def test_words_shifted_to_global_timeline():
    ctx = _stage()(_ctx())
    # user2 clip starts at 0.5 (overlap), so its word "yo" shifts by 0.5
    yo = [w for c in ctx.clips for w in c.words if w.text == "yo"][0]
    assert yo.start == 0.5 and yo.end == 1.0


def test_tracks_are_speaker_separated():
    ctx = _stage()(_ctx())
    sr = ctx.sample_rate
    t1 = ctx.tracks["user1"].channel(0)
    t2 = ctx.tracks["user2"].channel(0)
    # user1 active 0.0-1.0, silent after 1.0; user2 active 0.5-1.5
    assert np.any(np.abs(t1[: int(1.0 * sr)]) > 0)
    assert np.allclose(t1[int(1.0 * sr):], 0.0)
    assert np.allclose(t2[: int(0.5 * sr)], 0.0)
    assert np.any(np.abs(t2[int(0.5 * sr):]) > 0)


def test_tracks_equal_length():
    ctx = _stage()(_ctx())
    assert ctx.tracks["user1"].num_samples == ctx.tracks["user2"].num_samples

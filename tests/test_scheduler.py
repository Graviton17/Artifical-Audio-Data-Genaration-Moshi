"""Timeline placement: gaps push turns later, overlaps pull them earlier."""

import numpy as np

from datagen.models import AudioBuffer, SynthClip, Turn
from datagen.strategies.scheduler import OverlapScheduler


def _clip(speaker, dur, sr=24000, gap=0.0, overlap=0.0):
    n = int(dur * sr)
    return SynthClip(
        turn=Turn(speaker=speaker, text="x", gap=gap, overlap=overlap),
        audio=AudioBuffer(np.zeros(n, dtype=np.float32), sr),
    )


def test_first_turn_starts_at_lead_silence():
    sch = OverlapScheduler(default_gap_sec=0.25, lead_silence_sec=0.1)
    clips = [_clip("user1", 1.0)]
    total = sch.schedule(clips)
    assert clips[0].start == 0.1
    assert clips[0].end == 1.1
    assert total == 1.1


def test_default_gap_applied_between_turns():
    sch = OverlapScheduler(default_gap_sec=0.25, lead_silence_sec=0.0)
    clips = [_clip("user1", 1.0), _clip("user2", 1.0)]
    sch.schedule(clips)
    assert clips[1].start == 1.25  # prev end 1.0 + gap 0.25


def test_explicit_gap_overrides_default():
    sch = OverlapScheduler(default_gap_sec=0.25, lead_silence_sec=0.0)
    clips = [_clip("user1", 1.0), _clip("user2", 1.0, gap=0.5)]
    sch.schedule(clips)
    assert clips[1].start == 1.5


def test_overlap_pulls_turn_earlier():
    sch = OverlapScheduler(lead_silence_sec=0.0)
    clips = [_clip("user1", 2.0), _clip("user2", 1.0, overlap=0.5)]
    sch.schedule(clips)
    # second starts 0.5s before first ends -> overlap on timeline
    assert clips[1].start == 1.5
    assert clips[1].start < clips[0].end


def test_overlap_is_clamped_to_max():
    sch = OverlapScheduler(lead_silence_sec=0.0, max_overlap_sec=1.0)
    clips = [_clip("user1", 2.0), _clip("user2", 1.0, overlap=5.0)]
    sch.schedule(clips)
    assert clips[1].start == 1.0  # clamped: 2.0 - min(5.0, 1.0)


def test_start_never_negative():
    sch = OverlapScheduler(lead_silence_sec=0.0, max_overlap_sec=10.0)
    clips = [_clip("user1", 1.0), _clip("user2", 1.0, overlap=5.0)]
    sch.schedule(clips)
    assert clips[1].start == 0.0

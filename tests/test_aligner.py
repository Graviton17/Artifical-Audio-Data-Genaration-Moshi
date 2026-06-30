"""Heuristic aligner: covers every word, stays within the clip, ordered, no overlap."""

import numpy as np

from datagen.models import AudioBuffer
from datagen.strategies.aligner import HeuristicAligner


def _audio(dur, sr=24000):
    return AudioBuffer(np.zeros(int(dur * sr), dtype=np.float32), sr)


def test_one_word_per_token():
    words = HeuristicAligner().align(_audio(2.0), "one two three", "gu")
    assert [w.text for w in words] == ["one", "two", "three"]


def test_times_within_clip_and_ordered():
    dur = 3.0
    words = HeuristicAligner().align(_audio(dur), "alpha beta gamma delta", "gu")
    assert all(0.0 <= w.start < w.end <= dur for w in words)
    starts = [w.start for w in words]
    assert starts == sorted(starts)
    # words don't overlap each other
    for a, b in zip(words, words[1:]):
        assert a.end <= b.start + 1e-6


def test_empty_text_or_zero_duration():
    assert HeuristicAligner().align(_audio(1.0), "", "gu") == []
    assert HeuristicAligner().align(_audio(0.0), "hello", "gu") == []


def test_longer_words_get_more_time():
    words = HeuristicAligner(inter_word_gap=0.0).align(_audio(4.0), "a wwwwwwww", "gu")
    short, long = words
    assert (long.end - long.start) > (short.end - short.start)

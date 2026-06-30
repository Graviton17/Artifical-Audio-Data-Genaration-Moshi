"""Alignment merge: time-sorted, both speakers, labels flip per variant."""

from pathlib import Path

from datagen.config import Config
from datagen.models import GenContext, StreamVariant, SynthClip, Turn, Word
from datagen.stages.build_alignments import BuildAlignmentsStage


def _ctx():
    ctx = GenContext(script_path=Path("x"), sample_rate=24000, cache_dir=".", out_dir=".")
    c1 = SynthClip(turn=Turn("user1", "a b"))
    c1.words = [Word("a", 0.0, 0.5, "user1"), Word("b", 0.5, 1.0, "user1")]
    c2 = SynthClip(turn=Turn("user2", "c"))
    c2.words = [Word("c", 0.7, 1.2, "user2")]  # overlaps b in time
    ctx.clips = [c1, c2]
    ctx.variants = [
        StreamVariant(name="v__user1_agent", main_speaker="user1", user_speaker="user2"),
        StreamVariant(name="v__user2_agent", main_speaker="user2", user_speaker="user1"),
    ]
    return ctx


def test_alignments_sorted_by_time_with_both_speakers():
    ctx = BuildAlignmentsStage(Config())(_ctx())
    al = ctx.variants[0].alignments
    starts = [a[1][0] for a in al]
    assert starts == sorted(starts)
    assert [a[0] for a in al] == ["a", "b", "c"]  # time order: 0.0, 0.5, 0.7


def test_labels_flip_between_variants():
    ctx = BuildAlignmentsStage(Config())(_ctx())
    v1, v2 = ctx.variants[0].alignments, ctx.variants[1].alignments
    # same words + timings
    assert [x[:2] for x in v1] == [x[:2] for x in v2]
    # every label flipped
    assert all(a[2] != b[2] for a, b in zip(v1, v2))
    # in user1-agent variant, user1's word "a" is MAIN
    assert v1[0] == ["a", [0.0, 0.5], "SPEAKER_MAIN"]
    assert v2[0] == ["a", [0.0, 0.5], "SPEAKER_OTHER"]

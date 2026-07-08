"""Unit tests for per-speaker audio selection rules."""

import pytest

from voice_collection.models import AudioRef, AudioSample, Gender, Language, SelectionTier
from voice_collection.processing import SpeakerAudioSelector


def _sample(speaker_id: str, duration: float, index: int = 0) -> AudioSample:
    return AudioSample(
        speaker_id=speaker_id,
        language=Language.ENGLISH,
        gender=Gender.MALE,
        duration_seconds=duration,
        transcript=f"utt-{index}",
        audio_ref=AudioRef(raw_bytes=None),
        source_dataset="test",
        source_index=index,
    )


def test_primary_picks_shortest_at_or_above_target():
    selector = SpeakerAudioSelector(target_duration_seconds=10.0, min_acceptable_duration_seconds=5.0)
    selector.offer(_sample("spk_a", 12.0, 0))
    selector.offer(_sample("spk_a", 10.5, 1))
    selector.offer(_sample("spk_a", 15.0, 2))

    pick = selector.selected["spk_a"]
    assert pick.tier is SelectionTier.PRIMARY
    assert pick.sample.duration_seconds == 10.5
    assert pick.sample.source_index == 1


def test_fallback_picks_longest_between_min_and_target():
    selector = SpeakerAudioSelector(target_duration_seconds=10.0, min_acceptable_duration_seconds=5.0)
    selector.offer(_sample("spk_b", 5.0, 0))
    selector.offer(_sample("spk_b", 7.5, 1))
    selector.offer(_sample("spk_b", 6.0, 2))

    pick = selector.selected["spk_b"]
    assert pick.tier is SelectionTier.FALLBACK
    assert pick.sample.duration_seconds == 7.5


def test_discards_speaker_when_all_below_minimum():
    selector = SpeakerAudioSelector(target_duration_seconds=10.0, min_acceptable_duration_seconds=5.0)
    selector.offer(_sample("spk_c", 4.9, 0))
    selector.offer(_sample("spk_c", 2.0, 1))

    assert "spk_c" not in selector.selected
    assert selector.seen_speaker_count == 1
    assert selector.discarded_speaker_ids == ["spk_c"]


def test_primary_beats_fallback_for_same_speaker():
    selector = SpeakerAudioSelector(target_duration_seconds=10.0, min_acceptable_duration_seconds=5.0)
    selector.offer(_sample("spk_d", 8.0, 0))
    selector.offer(_sample("spk_d", 10.0, 1))

    pick = selector.selected["spk_d"]
    assert pick.tier is SelectionTier.PRIMARY
    assert pick.sample.duration_seconds == 10.0


def test_invalid_duration_bounds_raise():
    with pytest.raises(ValueError, match="cannot exceed"):
        SpeakerAudioSelector(target_duration_seconds=5.0, min_acceptable_duration_seconds=10.0)

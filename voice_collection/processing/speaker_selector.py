"""Picks exactly one instance per speaker: the core filtering rule.

Rule (see the project README for the full rationale):

1. If a speaker has at least one instance >= ``target_duration_seconds``
   (default 10s), keep the *shortest* one that qualifies -- closest to the
   target without carrying more audio than necessary.
2. Otherwise, if a speaker has at least one instance >=
   ``min_acceptable_duration_seconds`` (default 5s), keep the *longest* one
   below the target -- the closest available approximation.
3. Otherwise the speaker is discarded entirely (every instance is too short).

Runs online over a streamed dataset: only the current best candidate per
speaker is held in memory (cheap even for undecoded audio -- see
``AudioRef`` in ``models.py``), so this scales to datasets with far more
instances than would fit in memory at once.
"""
from __future__ import annotations

from ..models import AudioSample, SelectionTier, SpeakerSelection


class SpeakerAudioSelector:
    """Online (streaming-friendly) implementation of the selection rule above."""

    def __init__(self, target_duration_seconds: float, min_acceptable_duration_seconds: float) -> None:
        if min_acceptable_duration_seconds > target_duration_seconds:
            raise ValueError("min_acceptable_duration_seconds cannot exceed target_duration_seconds")
        self._target = target_duration_seconds
        self._min_acceptable = min_acceptable_duration_seconds
        self._best: dict[str, SpeakerSelection] = {}
        self._seen_speakers: set[str] = set()

    def offer(self, sample: AudioSample) -> None:
        """Consider one instance; updates the speaker's best pick if it wins."""
        self._seen_speakers.add(sample.speaker_id)

        rank = self._rank(sample.duration_seconds)
        if rank is None:
            return  # below the minimum acceptable duration -- never a candidate

        tier, _ = rank
        current = self._best.get(sample.speaker_id)
        if current is None or self._beats(sample.duration_seconds, current.sample.duration_seconds):
            self._best[sample.speaker_id] = SpeakerSelection(sample=sample, tier=tier)

    def _rank(self, duration: float) -> tuple[SelectionTier, float] | None:
        if duration >= self._target:
            return SelectionTier.PRIMARY, duration
        if duration >= self._min_acceptable:
            return SelectionTier.FALLBACK, duration
        return None

    def _beats(self, candidate_duration: float, current_duration: float) -> bool:
        """True if ``candidate_duration`` should replace ``current_duration``."""
        candidate_rank = self._rank(candidate_duration)
        current_rank = self._rank(current_duration)
        assert candidate_rank is not None and current_rank is not None

        candidate_tier, _ = candidate_rank
        current_tier, _ = current_rank
        if candidate_tier is not current_tier:
            return candidate_tier is SelectionTier.PRIMARY  # primary always beats fallback

        if candidate_tier is SelectionTier.PRIMARY:
            return candidate_duration < current_duration  # closest to target from above
        return candidate_duration > current_duration  # closest to target from below

    @property
    def selected(self) -> dict[str, SpeakerSelection]:
        return dict(self._best)

    @property
    def seen_speaker_count(self) -> int:
        return len(self._seen_speakers)

    @property
    def discarded_speaker_ids(self) -> list[str]:
        return sorted(self._seen_speakers - self._best.keys())

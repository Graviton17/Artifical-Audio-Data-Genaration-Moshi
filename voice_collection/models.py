"""Domain models for the voice-collection pipeline.

Typed dataclasses (and small enums) that flow between the dataset sources,
the speaker-selection strategy, the exporter and the storage backends, so the
rest of the code gets attribute access instead of raw-dict spelunking —
mirrors ``conversations_generator/models.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Language(str, Enum):
    """Corpus languages this service currently understands."""

    ENGLISH = "english"
    HINDI = "hindi"

    def __str__(self) -> str:  # nicer argparse --help / error text than repr
        return self.value


class Gender(str, Enum):
    """Normalised speaker gender used for the on-disk / bucket folder layout."""

    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_raw(cls, value: Any) -> "Gender":
        """Normalise a dataset-specific gender label (``"Male"``, ``"F"``, ...)."""
        if value is None:
            return cls.UNKNOWN
        text = str(value).strip().lower()
        if text in {"m", "male"}:
            return cls.MALE
        if text in {"f", "female"}:
            return cls.FEMALE
        if text in {"", "nan", "none", "unknown"}:
            return cls.UNKNOWN
        return cls.OTHER


class SelectionTier(str, Enum):
    """Which selection rule a speaker's exported clip satisfied.

    ``PRIMARY`` -- at least one instance was >= the target duration (10s by
    default); the *shortest* such instance is kept (closest to the target
    without carrying more audio than necessary).

    ``FALLBACK`` -- no instance reached the target, but at least one reached
    the minimum acceptable duration (5s by default); the *longest* such
    instance is kept (closest to the target from below).
    """

    PRIMARY = "primary_ge_target"
    FALLBACK = "fallback_below_target"

    def __str__(self) -> str:
        return self.value


@dataclass
class AudioRef:
    """A lazy, undecoded reference to one audio instance's bytes.

    Kept undecoded (see ``Audio(decode=False)`` in the ``sources/`` adapters)
    so streaming through thousands of instances per speaker stays cheap;
    only the instance that ends up winning a speaker's selection is ever
    decoded (see ``processing/audio_codec.py``).
    """

    raw_bytes: bytes | None
    path_hint: str | None = None


@dataclass
class AudioSample:
    """One (speaker, instance) row, normalised across every dataset source."""

    speaker_id: str
    language: Language
    gender: Gender
    duration_seconds: float
    transcript: str
    audio_ref: AudioRef
    source_dataset: str
    source_index: int
    extra_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpeakerSelection:
    """The winning instance for one speaker, plus which rule picked it."""

    sample: AudioSample
    tier: SelectionTier


@dataclass
class PipelineReport:
    """Summary of one language's run, written to ``_manifests/<language>_summary.json``."""

    language: Language
    speakers_seen: int = 0
    speakers_selected: int = 0
    speakers_discarded: int = 0
    primary_tier_count: int = 0
    fallback_tier_count: int = 0
    total_selected_duration_seconds: float = 0.0
    gender_breakdown: dict[str, int] = field(default_factory=dict)
    uploaded_file_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language.value,
            "speakers_seen": self.speakers_seen,
            "speakers_selected": self.speakers_selected,
            "speakers_discarded": self.speakers_discarded,
            "primary_tier_count": self.primary_tier_count,
            "fallback_tier_count": self.fallback_tier_count,
            "total_selected_duration_seconds": round(self.total_selected_duration_seconds, 2),
            "gender_breakdown": self.gender_breakdown,
            "uploaded_file_count": self.uploaded_file_count,
        }

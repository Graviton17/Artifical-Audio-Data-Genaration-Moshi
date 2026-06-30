"""Domain models passed between pipeline stages (Data Transfer Objects).

Framework-agnostic by design (no torch / transformers types leak in here) so the
models import cleanly and can be unit-tested without the heavy ML deps installed.
Shares the AudioBuffer / Word / StreamVariant vocabulary with Data-Processing-Moshi
so the two pipelines speak the same language.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class AudioBuffer:
    """A waveform plus its sample rate.

    ``samples`` is always 2-D ``(channels, num_samples)`` so mono and stereo are
    handled uniformly (same convention as Data-Processing-Moshi).
    """

    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        if self.samples.ndim == 1:
            self.samples = self.samples[None, :]
        if self.samples.ndim != 2:
            raise ValueError(
                f"AudioBuffer expects (channels, samples), got shape {self.samples.shape}"
            )

    @property
    def num_channels(self) -> int:
        return self.samples.shape[0]

    @property
    def num_samples(self) -> int:
        return self.samples.shape[1]

    @property
    def duration(self) -> float:
        return self.num_samples / self.sample_rate

    def channel(self, idx: int) -> np.ndarray:
        return self.samples[idx]


@dataclass
class Word:
    """A transcribed word with timing. ``start``/``end`` are seconds.

    During alignment these are clip-relative; the schedule stage shifts them onto
    the global conversation timeline.
    """

    text: str
    start: float
    end: float
    speaker: str = ""  # filled with the script speaker id (user1 / user2)


@dataclass
class Turn:
    """One scripted utterance from the input JSON."""

    speaker: str            # "user1" | "user2"
    text: str
    gap: float = 0.0        # silence after the previous turn before this one starts
    overlap: float = 0.0    # this turn starts this many sec BEFORE the previous turn ends


@dataclass
class ConversationScript:
    """The parsed + validated input JSON for one conversation."""

    conversation_id: str
    language: str
    speakers: dict[str, dict[str, Any]]   # {"user1": {"voice": ...}, "user2": {...}}
    turns: list[Turn]

    def speaker_ids(self) -> list[str]:
        return list(self.speakers.keys())


@dataclass
class SynthClip:
    """A synthesized turn: audio + the words it contains (timed once placed)."""

    turn: Turn
    audio: AudioBuffer | None = None
    words: list[Word] = field(default_factory=list)  # clip-relative until scheduled
    start: float = 0.0   # global timeline offset (sec), set by the schedule stage
    end: float = 0.0     # start + audio.duration


@dataclass
class StreamVariant:
    """One channel-assignment of the conversation (Moshi expects stereo: L=agent).

    The channel-swap augmentation produces two variants: one with user1 as the
    agent (left/main channel) and one with user2 as the agent.
    """

    name: str
    main_speaker: str          # -> left channel (channel 0), labelled SPEAKER_MAIN
    user_speaker: str          # -> right channel (channel 1), labelled SPEAKER_OTHER
    stereo: AudioBuffer | None = None
    alignments: list[list[Any]] = field(default_factory=list)  # interleaver format
    out_wav: Path | None = None
    out_json: Path | None = None


@dataclass
class GenContext:
    """Mutable state threaded through every stage for a single conversation."""

    script_path: Path
    sample_rate: int
    cache_dir: Path
    out_dir: Path

    script: ConversationScript | None = None
    clips: list[SynthClip] = field(default_factory=list)
    # per-speaker mono track laid out on the global timeline (silence elsewhere)
    tracks: dict[str, AudioBuffer] = field(default_factory=dict)
    variants: list[StreamVariant] = field(default_factory=list)

    dropped: bool = False
    drop_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def stem(self) -> str:
        if self.script is not None:
            return self.script.conversation_id
        return self.script_path.stem

    def drop(self, reason: str) -> None:
        self.dropped = True
        self.drop_reason = reason

"""Typed configuration loaded from YAML.

Mirrors Data-Processing-Moshi/dataprep/config.py: nested dataclasses so the rest
of the code gets attribute access and IDE completion instead of dict spelunking.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class TTSConfig:
    # `name` selects the concrete BaseTTS implementation in factory.py.
    name: str = "indic_mio"
    # Wired verbatim as requested. Swap if it does not resolve on HuggingFace.
    model_id: str = "SPRINGLab/Indic-Mio"
    default_language: str = "gu"
    target_sr: int = 24000


@dataclass
class AlignerConfig:
    # `forced` -> CTC/MMS forced alignment of the KNOWN text (no ASR).
    # `heuristic` -> deterministic char/phoneme-proportional timing.
    name: str = "forced"
    model: str = "MahmoudAshraf/mms-300m-1130-forced-aligner"
    language: str = "gu"
    fallback_heuristic: bool = True


@dataclass
class ScheduleConfig:
    default_gap_sec: float = 0.25
    max_overlap_sec: float = 3.0
    lead_silence_sec: float = 0.1


@dataclass
class AugmentConfig:
    channel_swap: bool = True


@dataclass
class LabelsConfig:
    main_label: str = "SPEAKER_MAIN"
    user_label: str = "SPEAKER_OTHER"


@dataclass
class CleanupConfig:
    loudness_lufs: float = -23.0
    peak_limit_db: float = -1.0


@dataclass
class Config:
    sample_rate: int = 24000
    device: str = "cuda"
    input_dir: str = "conversations"
    out_dir: str = "dataset"
    cache_dir: str = ".cache"
    tts: TTSConfig = field(default_factory=TTSConfig)
    aligner: AlignerConfig = field(default_factory=AlignerConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    labels: LabelsConfig = field(default_factory=LabelsConfig)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        valid = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in raw.items():
            if key not in valid:
                raise ValueError(f"Unknown config key: {key!r}")
            field_type = cls.__dataclass_fields__[key].default_factory  # type: ignore[attr-defined]
            if isinstance(value, dict) and field_type is not None and field_type not in (dict,):
                kwargs[key] = field_type(**value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

"""Factory: build a fully-wired :class:`Pipeline` from a :class:`Config`.

The single place that knows which concrete strategy implements each interface, so
swapping the TTS model / aligner / scheduler is a one-line change here (or a new
entry in the registries below). Mirrors Data-Processing-Moshi/dataprep/factory.py.
"""

from __future__ import annotations

from typing import Callable

from .config import Config
from .pipeline import Pipeline
from .stages import (
    AlignStage,
    BuildAlignmentsStage,
    LoadScriptStage,
    ScheduleStage,
    StreamizeStage,
    SynthesizeStage,
    WriteOutputsStage,
)
from .strategies import (
    BaseAligner,
    BaseScheduler,
    BaseTTS,
    ForcedAligner,
    HeuristicAligner,
    IndicMioTTS,
    OverlapScheduler,
)
from .utils.cache import Cache

# --- registries: name (from config) -> constructor -------------------------------
# Add a new TTS model by dropping a BaseTTS subclass in strategies/tts.py and
# registering it here. Nothing else changes.
TTS_REGISTRY: dict[str, Callable[[Config], BaseTTS]] = {
    "indic_mio": lambda c: IndicMioTTS(
        model_id=c.tts.model_id,
        device=c.device,
        default_language=c.tts.default_language,
    ),
}

ALIGNER_REGISTRY: dict[str, Callable[[Config], BaseAligner]] = {
    "forced": lambda c: ForcedAligner(
        model=c.aligner.model,
        device=c.device,
        fallback_heuristic=c.aligner.fallback_heuristic,
    ),
    "heuristic": lambda c: HeuristicAligner(),
}


def build_tts(config: Config) -> BaseTTS:
    try:
        return TTS_REGISTRY[config.tts.name](config)
    except KeyError:
        raise ValueError(
            f"Unknown tts.name {config.tts.name!r}; choices: {sorted(TTS_REGISTRY)}"
        )


def build_aligner(config: Config) -> BaseAligner:
    try:
        return ALIGNER_REGISTRY[config.aligner.name](config)
    except KeyError:
        raise ValueError(
            f"Unknown aligner.name {config.aligner.name!r}; choices: {sorted(ALIGNER_REGISTRY)}"
        )


def build_scheduler(config: Config) -> BaseScheduler:
    return OverlapScheduler(
        default_gap_sec=config.schedule.default_gap_sec,
        max_overlap_sec=config.schedule.max_overlap_sec,
        lead_silence_sec=config.schedule.lead_silence_sec,
    )


def build_pipeline(config: Config) -> Pipeline:
    cache = Cache(config.cache_dir)
    tts = build_tts(config)
    aligner = build_aligner(config)
    scheduler = build_scheduler(config)

    stages = [
        LoadScriptStage(),
        SynthesizeStage(tts, config, cache),
        AlignStage(aligner, config),
        ScheduleStage(scheduler, config),
        StreamizeStage(config),
        BuildAlignmentsStage(config),
        WriteOutputsStage(config),
    ]
    return Pipeline(stages)

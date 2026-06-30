"""Alignment strategy: synthesized audio + KNOWN text -> word timings.

We never re-transcribe (no ASR guessing). The words are exactly what the script
provided; the aligner only solves *timing*. This is what makes the pipeline
robust: text is fixed, only ``[start, end]`` is computed.

``BaseAligner`` is the swappable interface. Two implementations:

* :class:`ForcedAligner`   -- CTC/MMS forced alignment (multilingual, Indic-capable).
* :class:`HeuristicAligner`-- deterministic char-proportional timing; no model.
                              Used as the fallback when forced alignment errors.

Returned ``Word`` times are clip-relative; the schedule stage shifts them onto
the global conversation timeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..models import AudioBuffer, Word
from ..utils.audio_io import to_mono
from ..utils.logging import get_logger

log = get_logger("aligner")


class BaseAligner(ABC):
    @abstractmethod
    def align(self, audio: AudioBuffer, text: str, language: str) -> list[Word]:
        """Return ``[Word(text, start, end), ...]`` covering ``text`` (clip-relative)."""


class HeuristicAligner(BaseAligner):
    """Distribute the clip duration across words proportionally to token length.

    Deterministic and dependency-free. Word weight = len(word) + 1 (a constant for
    the inter-word gap), so longer words get proportionally more time. A small gap
    is inserted between words. Good enough as a fallback and for smoke-tests.
    """

    def __init__(self, inter_word_gap: float = 0.02):
        self.inter_word_gap = inter_word_gap

    def align(self, audio: AudioBuffer, text: str, language: str) -> list[Word]:
        words = text.split()
        total = audio.duration if audio is not None else 0.0
        if not words or total <= 0:
            return []
        weights = np.array([len(w) + 1 for w in words], dtype=np.float64)
        gaps = self.inter_word_gap * (len(words) - 1)
        speech = max(total - gaps, total * 0.5)  # never let gaps eat >half the clip
        durations = speech * (weights / weights.sum())

        out: list[Word] = []
        t = 0.0
        for w, d in zip(words, durations):
            start = t
            end = min(total, t + d)
            out.append(Word(text=w, start=round(start, 3), end=round(end, 3)))
            t = end + self.inter_word_gap
        return out


class ForcedAligner(BaseAligner):
    """CTC/MMS forced alignment of the known text via ``ctc-forced-aligner``.

    Falls back to :class:`HeuristicAligner` per-clip if alignment errors and
    ``fallback_heuristic`` is set.
    """

    MODEL_SR = 16000

    def __init__(
        self,
        model: str = "MahmoudAshraf/mms-300m-1130-forced-aligner",
        device: str = "cuda",
        fallback_heuristic: bool = True,
    ):
        self.model_name = model
        self.device = device
        self.fallback = HeuristicAligner() if fallback_heuristic else None
        self._model: Any = None
        self._tokenizer: Any = None
        self._mod: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import ctc_forced_aligner as cfa

        self._mod = cfa
        self._model, self._tokenizer = cfa.load_alignment_model(
            self.device,
            self.model_name,
            dtype="float16" if self.device.startswith("cuda") else "float32",
        )

    def align(self, audio: AudioBuffer, text: str, language: str) -> list[Word]:
        try:
            return self._align(audio, text, language)
        except Exception as exc:  # noqa: BLE001 - robustness is the whole point
            log.warning("forced alignment failed (%s); using heuristic fallback", exc)
            if self.fallback is None:
                raise
            return self.fallback.align(audio, text, language)

    def _align(self, audio: AudioBuffer, text: str, language: str) -> list[Word]:
        from ..utils.audio_io import resample

        self._load()
        cfa = self._mod
        buf = resample(audio, self.MODEL_SR) if audio.sample_rate != self.MODEL_SR else audio
        wav = to_mono(buf).astype(np.float32)

        emissions, stride = cfa.generate_emissions(self._model, wav, batch_size=1)
        tokens_starred, text_starred = cfa.preprocess_text(
            text, romanize=True, language=language
        )
        segments, scores, blank = cfa.get_alignments(
            emissions, tokens_starred, self._tokenizer
        )
        spans = cfa.get_spans(tokens_starred, segments, blank)
        results = cfa.postprocess_results(text_starred, spans, stride, scores)

        out: list[Word] = []
        for r in results:
            out.append(
                Word(text=r["text"], start=round(float(r["start"]), 3), end=round(float(r["end"]), 3))
            )
        return out

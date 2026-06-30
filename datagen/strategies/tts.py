"""TTS strategy: text + voice + language -> a mono :class:`AudioBuffer`.

``BaseTTS`` is the swappable interface (the name stays ``BaseTTS`` per the design
requirement). ``factory.build_tts`` is the single place that picks the concrete
implementation from ``config.tts.name``, so adding a new model (IndicF5, Parler,
...) is a one-line registry change there -- nothing else in the pipeline cares
which model produced the audio.

``IndicMioTTS`` is the first concrete implementation, wired to the model id
``SPRINGLab/Indic-Mio`` from config. Heavy deps (torch/transformers) are imported
lazily inside ``_load`` so this module imports cleanly for unit tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..models import AudioBuffer


class BaseTTS(ABC):
    """Abstract text-to-speech model. Do not rename: concrete models subclass this."""

    #: native output sample rate of the model (overridden by subclasses)
    native_sr: int = 24000

    @abstractmethod
    def synthesize(self, text: str, voice: str | None, language: str) -> AudioBuffer:
        """Synthesize ``text`` in ``language`` with the given ``voice``.

        Returns a mono :class:`AudioBuffer`. ``voice`` is model-specific: a speaker
        id, a style/description prompt, or a reference-audio path. Implementations
        should tolerate ``voice=None`` by using a sensible default voice.
        """


class IndicMioTTS(BaseTTS):
    """SPRINGLab Indic TTS (model id from config; default ``SPRINGLab/Indic-Mio``).

    The id is wired exactly as requested. If it does not resolve on HuggingFace,
    set ``tts.model_id`` in config.yaml to a verified one (e.g.
    ``ai4bharat/indic-parler-tts``, ``ai4bharat/IndicF5``,
    ``SPRINGLab/F5-Hindi-24KHz``) -- because everything talks to ``BaseTTS``, that
    is the only change needed.
    """

    native_sr = 24000

    def __init__(
        self,
        model_id: str = "SPRINGLab/Indic-Mio",
        device: str = "cuda",
        default_language: str = "gu",
    ):
        self.model_id = model_id
        self.device = device
        self.default_language = default_language
        self._model: Any = None
        self._tokenizer: Any = None
        self._processor: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: F401  (ensures torch is present; device move below)
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        # Generic transformers loading path. Many Indic TTS checkpoints expose a
        # processor (Parler/VITS-style) or a plain tokenizer; we try processor
        # first and fall back to a tokenizer. trust_remote_code covers custom
        # model classes shipped in the repo.
        try:
            self._processor = AutoProcessor.from_pretrained(
                self.model_id, trust_remote_code=True
            )
        except Exception:
            self._processor = None
        if self._processor is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=True
            )
        self._model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = self._model.to(self.device).eval()

    def synthesize(self, text: str, voice: str | None, language: str) -> AudioBuffer:
        import torch

        self._load()
        lang = language or self.default_language

        with torch.no_grad():
            if self._processor is not None:
                # Description/voice-prompt style models (e.g. Parler-like). When a
                # voice/description is given it conditions the speaker; otherwise a
                # neutral description is used.
                description = voice or f"A clear natural voice speaking in {lang}."
                enc = self._processor(text=description, return_tensors="pt").to(self.device)
                prompt = self._processor(text=text, return_tensors="pt").to(self.device)
                out = self._model.generate(
                    input_ids=enc.input_ids,
                    attention_mask=getattr(enc, "attention_mask", None),
                    prompt_input_ids=prompt.input_ids,
                    prompt_attention_mask=getattr(prompt, "attention_mask", None),
                )
            else:
                enc = self._tokenizer(text, return_tensors="pt").to(self.device)
                out = self._model.generate(**enc)

        wav = self._to_numpy(out)
        return AudioBuffer(samples=wav, sample_rate=self.native_sr)

    @staticmethod
    def _to_numpy(out: Any) -> np.ndarray:
        """Coerce a model output into a float32 mono waveform array."""
        import torch

        if isinstance(out, dict):
            # common keys across HF TTS heads
            for k in ("waveform", "audio", "audios", "waveforms", "sequences"):
                if k in out:
                    out = out[k]
                    break
        if isinstance(out, (list, tuple)):
            out = out[0]
        if isinstance(out, torch.Tensor):
            arr = out.detach().to(torch.float32).cpu().numpy()
        else:
            arr = np.asarray(out, dtype=np.float32)
        return arr.reshape(-1).astype(np.float32)  # mono, 1-D (AudioBuffer adds channel axis)

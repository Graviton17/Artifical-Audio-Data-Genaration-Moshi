"""Stage [7]: write stereo WAV + transcript JSON per variant.

The JSON schema is identical to Data-Processing-Moshi's WriteOutputsStage so the
synthetic and real datasets are drop-in interchangeable for moshi-finetune:

  * ``alignments``            -- moshi format ``[[word, [start, end], label], ...]``
  * ``segments``              -- turn-level ``[{speaker, start, end}, ...]``
  * ``transcript_by_speaker`` -- words split per channel label
  * ``speakers``              -- which script speaker maps to main / user
  * ``purity``                -- crosstalk is 0.0 / pass by construction (clean synth)

``synthetic`` metadata is added so consumers can tell the source apart if needed.
"""

from __future__ import annotations

import json

from ..config import Config
from ..models import GenContext
from ..utils.audio_io import save_wav
from .base import Stage


class WriteOutputsStage(Stage):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def _run(self, ctx: GenContext) -> GenContext:
        main_label = self.config.labels.main_label
        user_label = self.config.labels.user_label

        # turn-level segments on the global timeline (shared across variants)
        segments = [
            {"speaker": clip.turn.speaker, "start": clip.start, "end": clip.end}
            for clip in ctx.clips
        ]

        for v in ctx.variants:
            if v.stereo is None:
                continue
            wav_path = ctx.out_dir / f"{v.name}.wav"
            save_wav(wav_path, v.stereo)
            v.out_wav = wav_path

            payload = {
                # --- consumed by moshi-finetune's interleaver ---
                "alignments": v.alignments,
                # --- inspection / downstream extras (ignored by the interleaver) ---
                "segments": segments,
                "transcript_by_speaker": {
                    main_label: [[t, ts] for t, ts, lbl in v.alignments if lbl == main_label],
                    user_label: [[t, ts] for t, ts, lbl in v.alignments if lbl == user_label],
                },
                "speakers": {"main": v.main_speaker, "user": v.user_speaker},
                "purity": {
                    "main_crosstalk": 0.0,
                    "user_crosstalk": 0.0,
                    "pass": True,
                    "quarantined": False,
                },
                "synthetic": {
                    "source": "datagen",
                    "tts_model": self.config.tts.model_id,
                    "language": ctx.script.language if ctx.script else None,
                },
            }

            json_path = wav_path.with_suffix(".json")
            tmp = json_path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            tmp.rename(json_path)
            v.out_json = json_path

        return ctx

"""Content-addressed cache for synthesized clips.

Synthesis is the expensive step. We key each clip by a hash of
(model_id, language, voice, text, target_sr) so re-running a conversation after
editing only one turn re-synthesizes that turn. Clips are stored as .npy + a
sidecar sample-rate file under ``<cache_dir>/tts/``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..models import AudioBuffer


class Cache:
    def __init__(self, cache_dir: str | Path):
        self.root = Path(cache_dir)

    def _tts_dir(self) -> Path:
        d = self.root / "tts"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def key(**parts) -> str:
        blob = json.dumps(parts, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def get_audio(self, key: str) -> AudioBuffer | None:
        npy = self._tts_dir() / f"{key}.npy"
        meta = self._tts_dir() / f"{key}.json"
        if not (npy.exists() and meta.exists()):
            return None
        samples = np.load(npy)
        sr = int(json.loads(meta.read_text())["sample_rate"])
        return AudioBuffer(samples=samples, sample_rate=sr)

    def put_audio(self, key: str, buf: AudioBuffer) -> None:
        np.save(self._tts_dir() / f"{key}.npy", buf.samples)
        (self._tts_dir() / f"{key}.json").write_text(
            json.dumps({"sample_rate": buf.sample_rate})
        )

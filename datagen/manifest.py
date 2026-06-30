"""Build the moshi-finetune ``.jsonl`` manifest from produced outputs.

Each line: ``{"path": "<relative wav>", "duration": <seconds>}``. Only WAVs that
have a sibling ``.json`` transcript are included. Identical format to
Data-Processing-Moshi/dataprep/manifest.py so the datasets are interchangeable.
"""

from __future__ import annotations

import json
from pathlib import Path


def _duration(path: str) -> float:
    try:
        import sphn

        samples, sr = sphn.read(path)
        import numpy as np

        return np.asarray(samples).shape[-1] / sr
    except Exception:
        import soundfile as sf

        info = sf.info(path)
        return info.frames / info.samplerate


def build_manifest(out_dir: str | Path, manifest_name: str = "dataset.jsonl") -> Path:
    out_dir = Path(out_dir)
    wavs = sorted(p for p in out_dir.glob("*.wav") if p.with_suffix(".json").exists())

    manifest_path = out_dir / manifest_name
    if not wavs:
        manifest_path.write_text("")
        return manifest_path

    with open(manifest_path, "w") as f:
        for wav in wavs:
            try:
                dur = _duration(str(wav))
                rel = wav.relative_to(out_dir)
                f.write(json.dumps({"path": str(rel), "duration": float(dur)}) + "\n")
            except Exception:
                continue

    return manifest_path

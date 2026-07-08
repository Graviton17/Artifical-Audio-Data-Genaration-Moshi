"""Decode / resample / write the single winning clip per speaker.

Kept deliberately tiny: every other dataset instance is discarded by
``speaker_selector`` while still undecoded, so this module only ever runs
once per *exported* speaker, never once per instance in the source dataset.
"""
from __future__ import annotations

import io
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from ..models import AudioRef


class AudioProcessingError(RuntimeError):
    """Raised when a selected audio instance cannot be decoded or written."""


def decode_audio(audio_ref: AudioRef) -> tuple[np.ndarray, int]:
    """Decode ``audio_ref`` into a float32 waveform + its native sample rate."""
    if audio_ref.raw_bytes:
        data, sample_rate = sf.read(io.BytesIO(audio_ref.raw_bytes), dtype="float32", always_2d=False)
        return data, sample_rate
    if audio_ref.path_hint and Path(audio_ref.path_hint).exists():
        data, sample_rate = sf.read(audio_ref.path_hint, dtype="float32", always_2d=False)
        return data, sample_rate
    raise AudioProcessingError("Audio sample has neither embedded bytes nor a resolvable local path.")


def to_mono(data: np.ndarray) -> np.ndarray:
    return data.mean(axis=1) if data.ndim > 1 else data


def resample(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return data
    divisor = gcd(orig_sr, target_sr)
    up, down = target_sr // divisor, orig_sr // divisor
    return resample_poly(data, up, down).astype(np.float32)


def write_wav(path: Path, data: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sample_rate, subtype="PCM_16")


def export_audio(audio_ref: AudioRef, target_sample_rate: int, output_path: Path) -> float:
    """Decode, downmix to mono, resample, and write ``output_path``.

    Returns the exported clip's final duration in seconds (post-resample),
    which is what gets recorded in the speaker's ``metadata.json``.
    """
    try:
        data, orig_sr = decode_audio(audio_ref)
        data = to_mono(data)
        data = resample(data, orig_sr, target_sample_rate)
        write_wav(output_path, data, target_sample_rate)
    except AudioProcessingError:
        raise
    except Exception as err:  # noqa: BLE001 - normalize decode/IO failures
        raise AudioProcessingError(f"Failed to export audio to {output_path}: {err}") from err
    return len(data) / float(target_sample_rate)

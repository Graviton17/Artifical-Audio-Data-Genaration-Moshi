"""Audio I/O helpers: save stereo WAV, resample, simple loudness/peak normalize.

Kept dependency-light: soundfile for writing, a small linear resampler so the
core does not hard-require torchaudio/librosa. The TTS / aligner strategies bring
their own heavy deps; this module stays importable for tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..models import AudioBuffer


def resample(buf: AudioBuffer, target_sr: int) -> AudioBuffer:
    """Resample every channel to ``target_sr`` via linear interpolation.

    Linear interpolation is adequate here: synthesized clips are clean and we only
    nudge sample rates to a common grid before scheduling. For best fidelity, set
    the TTS ``target_sr`` equal to ``sample_rate`` so this is a no-op.
    """
    if buf.sample_rate == target_sr:
        return buf
    ratio = target_sr / buf.sample_rate
    n_out = int(round(buf.num_samples * ratio))
    if n_out <= 0:
        return AudioBuffer(np.zeros((buf.num_channels, 0), dtype=np.float32), target_sr)
    x_old = np.linspace(0.0, 1.0, num=buf.num_samples, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    out = np.stack(
        [np.interp(x_new, x_old, buf.channel(c)).astype(np.float32) for c in range(buf.num_channels)],
        axis=0,
    )
    return AudioBuffer(out, target_sr)


def to_mono(buf: AudioBuffer) -> np.ndarray:
    """Average channels to a single mono track (float32)."""
    if buf.num_channels == 1:
        return buf.channel(0).astype(np.float32)
    return buf.samples.mean(axis=0).astype(np.float32)


def peak_normalize(x: np.ndarray, peak_db: float = -1.0) -> np.ndarray:
    """Scale so the absolute peak hits ``peak_db`` dBFS (no-op on silence)."""
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 1e-9:
        return x
    target = 10.0 ** (peak_db / 20.0)
    return (x * (target / peak)).astype(np.float32)


def save_wav(path: str | Path, buf: AudioBuffer) -> None:
    """Write ``buf`` as a WAV. soundfile expects (frames, channels)."""
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = buf.samples.T  # (channels, samples) -> (samples, channels)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # format is passed explicitly because the ".wav.tmp" suffix is not a
    # recognizable extension for soundfile's format inference.
    sf.write(str(tmp), data, buf.sample_rate, subtype="PCM_16", format="WAV")
    tmp.rename(path)

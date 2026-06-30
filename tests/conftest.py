"""Shared fixtures. No torch/transformers needed: a FakeTTS stands in for a real
model so the whole pipeline is exercised without heavy deps or downloads.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from datagen.models import AudioBuffer
from datagen.strategies.tts import BaseTTS


class FakeTTS(BaseTTS):
    """Deterministic stand-in: a sine tone whose length scales with text length."""

    native_sr = 24000

    def __init__(self, sr: int = 24000):
        self.native_sr = sr

    def synthesize(self, text: str, voice, language: str) -> AudioBuffer:
        dur = max(0.4, 0.05 * len(text))
        n = int(dur * self.native_sr)
        t = np.arange(n) / self.native_sr
        wav = (0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
        return AudioBuffer(wav, self.native_sr)


@pytest.fixture
def fake_tts():
    return FakeTTS()


@pytest.fixture
def sample_script_dict():
    return {
        "conversation_id": "conv_test",
        "language": "gu",
        "speakers": {
            "user1": {"voice": "v1"},
            "user2": {"voice": "v2"},
        },
        "turns": [
            {"speaker": "user1", "text": "hello there friend"},
            {"speaker": "user2", "text": "yes indeed", "gap": 0.3},
            {"speaker": "user1", "text": "good to hear that", "overlap": 0.4},
        ],
    }


@pytest.fixture
def script_path(tmp_path, sample_script_dict):
    p = tmp_path / "conv_test.json"
    p.write_text(json.dumps(sample_script_dict, ensure_ascii=False), encoding="utf-8")
    return p

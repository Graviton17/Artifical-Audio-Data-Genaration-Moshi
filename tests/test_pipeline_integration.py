"""End-to-end: FakeTTS + heuristic aligner -> verify the full Moshi output contract.

Exercises every stage and the runner/manifest without any ML downloads.
"""

import json

import numpy as np
import soundfile as sf

from datagen import factory
from datagen.config import Config
from datagen.runner import process_dir


def _run(tmp_path, script_path, fake_tts):
    factory.TTS_REGISTRY["fake"] = lambda c: fake_tts
    cfg = Config()
    cfg.tts.name = "fake"
    cfg.aligner.name = "heuristic"
    cfg.device = "cpu"
    cfg.out_dir = str(tmp_path / "out")
    cfg.cache_dir = str(tmp_path / "cache")
    manifest = process_dir(cfg, script_path.parent)
    return cfg, manifest


def test_produces_two_swapped_variants(tmp_path, script_path, fake_tts):
    cfg, manifest = _run(tmp_path, script_path, fake_tts)
    out = tmp_path / "out"
    wavs = sorted(p.name for p in out.glob("*.wav"))
    assert wavs == ["conv_test__user1_agent.wav", "conv_test__user2_agent.wav"]


def test_wavs_are_stereo_24k(tmp_path, script_path, fake_tts):
    _run(tmp_path, script_path, fake_tts)
    for name in ("conv_test__user1_agent.wav", "conv_test__user2_agent.wav"):
        info = sf.info(str(tmp_path / "out" / name))
        assert info.channels == 2
        assert info.samplerate == 24000


def test_json_contract_matches_real_pipeline(tmp_path, script_path, fake_tts):
    _run(tmp_path, script_path, fake_tts)
    data = json.loads((tmp_path / "out" / "conv_test__user1_agent.json").read_text())
    assert set(data) >= {
        "alignments", "segments", "transcript_by_speaker", "speakers", "purity",
    }
    assert {a[2] for a in data["alignments"]} == {"SPEAKER_MAIN", "SPEAKER_OTHER"}
    assert data["speakers"] == {"main": "user1", "user": "user2"}
    assert data["purity"]["pass"] is True


def test_manifest_format(tmp_path, script_path, fake_tts):
    _, manifest = _run(tmp_path, script_path, fake_tts)
    lines = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    assert len(lines) == 2
    for entry in lines:
        assert set(entry) == {"path", "duration"}
        assert entry["path"].endswith(".wav")
        assert entry["duration"] > 0


def test_channel_separation_zero_crosstalk(tmp_path, script_path, fake_tts):
    _run(tmp_path, script_path, fake_tts)
    data = json.loads((tmp_path / "out" / "conv_test__user1_agent.json").read_text())
    wav, sr = sf.read(str(tmp_path / "out" / "conv_test__user1_agent.wav"))  # (frames, 2)
    seg = data["segments"]
    # first user1 turn: energy on ch0, silence on ch1
    s0 = seg[0]
    a, b = int(s0["start"] * sr), int(s0["end"] * sr)
    assert np.sqrt(np.mean(wav[a:b, 0] ** 2)) > 0
    assert np.allclose(wav[a:b, 1], 0.0)


def test_swapped_variant_flips_labels(tmp_path, script_path, fake_tts):
    _run(tmp_path, script_path, fake_tts)
    v1 = json.loads((tmp_path / "out" / "conv_test__user1_agent.json").read_text())["alignments"]
    v2 = json.loads((tmp_path / "out" / "conv_test__user2_agent.json").read_text())["alignments"]
    assert [a[:2] for a in v1] == [a[:2] for a in v2]
    assert all(a[2] != b[2] for a, b in zip(v1, v2))


def test_report_written(tmp_path, script_path, fake_tts):
    _run(tmp_path, script_path, fake_tts)
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert report["num_produced"] == 1
    assert report["num_dropped"] == 0

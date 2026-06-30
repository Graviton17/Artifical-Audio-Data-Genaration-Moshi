"""Config loading: nested dataclasses, defaults, and unknown-key rejection."""

import pytest

from datagen.config import Config


def test_defaults():
    c = Config()
    assert c.sample_rate == 24000
    assert c.tts.name == "indic_mio"
    assert c.tts.model_id == "SPRINGLab/Indic-Mio"
    assert c.aligner.name == "forced"
    assert c.augment.channel_swap is True
    assert c.labels.main_label == "SPEAKER_MAIN"


def test_from_dict_nested_override():
    c = Config.from_dict(
        {
            "sample_rate": 16000,
            "tts": {"name": "indic_mio", "model_id": "ai4bharat/IndicF5"},
            "schedule": {"default_gap_sec": 0.5},
        }
    )
    assert c.sample_rate == 16000
    assert c.tts.model_id == "ai4bharat/IndicF5"
    assert c.schedule.default_gap_sec == 0.5
    # untouched nested fields keep defaults
    assert c.labels.user_label == "SPEAKER_OTHER"


def test_unknown_top_level_key_rejected():
    with pytest.raises(ValueError):
        Config.from_dict({"nope": 1})


def test_from_yaml_roundtrip(tmp_path):
    import yaml

    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"device": "cpu", "tts": {"default_language": "hi"}}))
    c = Config.from_yaml(p)
    assert c.device == "cpu"
    assert c.tts.default_language == "hi"

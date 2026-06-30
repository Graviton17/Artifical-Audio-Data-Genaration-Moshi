"""Input-JSON parsing + validation (LoadScriptStage)."""

import json

from datagen.models import GenContext
from datagen.stages.load_script import LoadScriptStage


def _ctx(tmp_path, payload):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return GenContext(script_path=p, sample_rate=24000, cache_dir=tmp_path, out_dir=tmp_path)


def test_valid_script(tmp_path, sample_script_dict):
    ctx = LoadScriptStage()(_ctx(tmp_path, sample_script_dict))
    assert not ctx.dropped
    assert ctx.script.conversation_id == "conv_test"
    assert len(ctx.script.turns) == 3
    assert ctx.script.turns[1].gap == 0.3
    assert ctx.script.turns[2].overlap == 0.4


def test_drop_on_bad_json(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{not json", encoding="utf-8")
    ctx = GenContext(script_path=p, sample_rate=24000, cache_dir=tmp_path, out_dir=tmp_path)
    ctx = LoadScriptStage()(ctx)
    assert ctx.dropped and "load" in ctx.drop_reason


def test_drop_when_not_two_speakers(tmp_path, sample_script_dict):
    sample_script_dict["speakers"] = {"user1": {}}
    ctx = LoadScriptStage()(_ctx(tmp_path, sample_script_dict))
    assert ctx.dropped and "two speakers" in ctx.drop_reason


def test_drop_on_unknown_turn_speaker(tmp_path, sample_script_dict):
    sample_script_dict["turns"][0]["speaker"] = "ghost"
    ctx = LoadScriptStage()(_ctx(tmp_path, sample_script_dict))
    assert ctx.dropped and "ghost" in ctx.drop_reason


def test_drop_on_empty_text(tmp_path, sample_script_dict):
    sample_script_dict["turns"][0]["text"] = "   "
    ctx = LoadScriptStage()(_ctx(tmp_path, sample_script_dict))
    assert ctx.dropped and "empty text" in ctx.drop_reason


def test_gap_and_overlap_mutually_exclusive(tmp_path, sample_script_dict):
    sample_script_dict["turns"][1]["gap"] = 0.5
    sample_script_dict["turns"][1]["overlap"] = 0.5
    ctx = LoadScriptStage()(_ctx(tmp_path, sample_script_dict))
    assert not ctx.dropped
    # overlap wins, gap zeroed
    assert ctx.script.turns[1].overlap == 0.5
    assert ctx.script.turns[1].gap == 0.0

"""Focused tests for the voice-collection pipeline orchestration."""

from __future__ import annotations

import json
import sys
import types

from voice_collection.models import AudioRef, AudioSample, Gender, Language, SelectionTier, SpeakerSelection


# Stub heavy runtime modules so these tests stay local/offline and do not depend
# on scipy, datasets, or Hugging Face SDK imports during test collection.
_exporter_stub = types.ModuleType("voice_collection.exporter")


class _DatasetExporterStub:
    def __init__(self, *args, **kwargs):
        pass


_exporter_stub.DatasetExporter = _DatasetExporterStub
sys.modules.setdefault("voice_collection.exporter", _exporter_stub)

_sources_stub = types.ModuleType("voice_collection.sources")


class _DatasetSourceFactoryStub:
    @staticmethod
    def create(language: str, dataset_config: dict):
        raise AssertionError("DatasetSourceFactory.create should not be called in these tests")


_sources_stub.DatasetSourceFactory = _DatasetSourceFactoryStub
sys.modules.setdefault("voice_collection.sources", _sources_stub)

_storage_stub = types.ModuleType("voice_collection.storage")
_storage_stub.create_storage = lambda: None
sys.modules.setdefault("voice_collection.storage", _storage_stub)

from voice_collection.pipeline import VoiceCollectionPipeline


def _selection(
    speaker_id: str,
    *,
    gender: Gender,
    duration_seconds: float,
    tier: SelectionTier,
) -> SpeakerSelection:
    return SpeakerSelection(
        sample=AudioSample(
            speaker_id=speaker_id,
            language=Language.ENGLISH,
            gender=gender,
            duration_seconds=duration_seconds,
            transcript=f"transcript-{speaker_id}",
            audio_ref=AudioRef(raw_bytes=None),
            source_dataset="test-dataset",
            source_index=0,
        ),
        tier=tier,
    )


class _SelectorStub:
    def __init__(self, seen_speaker_count: int) -> None:
        self.seen_speaker_count = seen_speaker_count


def test_build_report_counts_primary_fallback_and_gender_breakdown():
    selections = {
        "speaker_a": _selection(
            "speaker_a",
            gender=Gender.MALE,
            duration_seconds=10.2,
            tier=SelectionTier.PRIMARY,
        ),
        "speaker_b": _selection(
            "speaker_b",
            gender=Gender.FEMALE,
            duration_seconds=7.5,
            tier=SelectionTier.FALLBACK,
        ),
        "speaker_c": _selection(
            "speaker_c",
            gender=Gender.FEMALE,
            duration_seconds=11.0,
            tier=SelectionTier.PRIMARY,
        ),
    }

    report = VoiceCollectionPipeline._build_report("english", _SelectorStub(seen_speaker_count=5), selections)

    assert report.language is Language.ENGLISH
    assert report.speakers_seen == 5
    assert report.speakers_selected == 3
    assert report.speakers_discarded == 2
    assert report.primary_tier_count == 2
    assert report.fallback_tier_count == 1
    assert report.total_selected_duration_seconds == 28.7
    assert report.gender_breakdown == {"male": 1, "female": 2}


def test_write_report_creates_manifest_json(tmp_path):
    pipeline = VoiceCollectionPipeline.__new__(VoiceCollectionPipeline)
    pipeline.output_root = tmp_path

    report = VoiceCollectionPipeline._build_report(
        "english",
        _SelectorStub(seen_speaker_count=1),
        {
            "speaker_a": _selection(
                "speaker_a",
                gender=Gender.MALE,
                duration_seconds=10.0,
                tier=SelectionTier.PRIMARY,
            )
        },
    )

    pipeline._write_report("english", report)

    summary_path = tmp_path / "_manifests" / "english_summary.json"
    assert summary_path.exists()
    assert json.loads(summary_path.read_text(encoding="utf-8")) == {
        "language": "english",
        "speakers_seen": 1,
        "speakers_selected": 1,
        "speakers_discarded": 0,
        "primary_tier_count": 1,
        "fallback_tier_count": 0,
        "total_selected_duration_seconds": 10.0,
        "gender_breakdown": {"male": 1},
        "uploaded_file_count": 0,
    }


def test_has_complete_local_export_requires_manifest_and_matching_speaker_count(tmp_path):
    pipeline = VoiceCollectionPipeline.__new__(VoiceCollectionPipeline)
    pipeline.output_root = tmp_path
    pipeline.max_speakers = None

    report = VoiceCollectionPipeline._build_report(
        "english",
        _SelectorStub(seen_speaker_count=2),
        {
            "speaker_a": _selection(
                "speaker_a",
                gender=Gender.MALE,
                duration_seconds=10.0,
                tier=SelectionTier.PRIMARY,
            ),
            "speaker_b": _selection(
                "speaker_b",
                gender=Gender.FEMALE,
                duration_seconds=7.5,
                tier=SelectionTier.FALLBACK,
            ),
        },
    )

    assert pipeline._has_complete_local_export("english", report) is False

    for gender, speaker in (("male", "speaker_a"), ("female", "speaker_b")):
        speaker_dir = tmp_path / "english" / gender / speaker
        speaker_dir.mkdir(parents=True)
        (speaker_dir / "audio.wav").write_bytes(b"a")
        (speaker_dir / "metadata.json").write_text("{}", encoding="utf-8")

    assert pipeline._has_complete_local_export("english", report) is True


def test_load_existing_report_round_trips_summary(tmp_path):
    pipeline = VoiceCollectionPipeline.__new__(VoiceCollectionPipeline)
    pipeline.output_root = tmp_path

    manifests_dir = tmp_path / "_manifests"
    manifests_dir.mkdir(parents=True)
    (manifests_dir / "english_summary.json").write_text(
        json.dumps(
            {
                "language": "english",
                "speakers_seen": 4,
                "speakers_selected": 2,
                "speakers_discarded": 2,
                "primary_tier_count": 1,
                "fallback_tier_count": 1,
                "total_selected_duration_seconds": 18.5,
                "gender_breakdown": {"male": 1, "female": 1},
                "uploaded_file_count": 3,
            }
        ),
        encoding="utf-8",
    )

    report = pipeline._load_existing_report("english")

    assert report is not None
    assert report.language is Language.ENGLISH
    assert report.speakers_selected == 2
    assert report.uploaded_file_count == 3

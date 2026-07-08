"""Orchestrates one language's run: stream -> select -> export -> upload.

    from voice_collection.pipeline import VoiceCollectionPipeline

    pipeline = VoiceCollectionPipeline()
    report = pipeline.run("hindi")

Mirrors ``conversations_generator``'s runner/pipeline split: this module only
wires the stages together; each stage's own logic (dataset parsing,
selection, export, upload) lives in its own module and is swappable
independently -- see the factories in ``sources/`` and ``storage/``.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import configuration_reader as config
from .exporter import DatasetExporter
from .logger import Logger
from .models import Language, PipelineReport, SelectionTier
from .processing import SpeakerAudioSelector
from .sources import DatasetSourceFactory
from .storage import create_storage

#: How often (in streamed instances) to print a progress heartbeat.
_PROGRESS_EVERY = 500


class VoiceCollectionPipeline:
    """Runs the fetch -> filter -> export -> upload pipeline for one language."""

    def __init__(self) -> None:
        self.output_root = config.get_local_output_dir()
        self.target_sample_rate = config.get_target_sample_rate()
        self.anonymize_speaker_names = config.get_anonymize_speaker_names()
        self.target_duration = config.get_target_duration_seconds()
        self.min_acceptable_duration = config.get_min_acceptable_duration_seconds()
        self.max_speakers = config.get_max_speakers_per_language()
        self.exporter = DatasetExporter(
            output_root=self.output_root,
            target_sample_rate=self.target_sample_rate,
            anonymize_speaker_names=self.anonymize_speaker_names,
        )

    def run(self, language: str, *, upload: bool = True) -> PipelineReport:
        Logger.step(f"Starting voice-collection pipeline for '{language}'")

        dataset_config = config.get_dataset_config(language)
        dataset_config.setdefault("hf_token", config.get_hf_token())
        source = DatasetSourceFactory.create(language, dataset_config)

        selector = SpeakerAudioSelector(self.target_duration, self.min_acceptable_duration)
        Logger.info(f"Streaming '{dataset_config['repo_id']}' (this can take a while for large datasets)...")
        for count, sample in enumerate(source.stream(), start=1):
            selector.offer(sample)
            if count % _PROGRESS_EVERY == 0:
                Logger.info(
                    f"Processed {count} instance(s) | speakers seen: {selector.seen_speaker_count} | "
                    f"qualified so far: {len(selector.selected)}"
                )

        selections = selector.selected
        if self.max_speakers is not None:
            selections = dict(list(selections.items())[: self.max_speakers])

        Logger.step(f"Exporting {len(selections)} speaker(s) for '{language}'...")
        exported_dirs = self.exporter.export_language(Language(language), selections)
        Logger.success(f"Exported {len(exported_dirs)} speaker folder(s) under {self.output_root / language}")

        report = self._build_report(language, selector, selections)

        if upload:
            report.uploaded_file_count = self._upload(language)
        else:
            Logger.warning(f"Upload skipped for '{language}' (dry-run or UPLOAD_ENABLED=false).")

        self._write_report(language, report)
        Logger.success(f"Finished '{language}': {report.to_dict()}", bold=True)
        return report

    def _upload(self, language: str) -> int:
        Logger.step(f"Uploading '{language}' to HuggingFace bucket...")
        storage = create_storage()
        bucket = config.get_hf_bucket_config()["bucket"]
        remote_prefix = language
        local_root = self.output_root / language
        uploaded = storage.upload_directory(local_root, remote_prefix)
        Logger.success(f"Uploaded {uploaded} file(s) to '{bucket}/{remote_prefix}'.")
        return uploaded

    @staticmethod
    def _build_report(language: str, selector: SpeakerAudioSelector, selections: dict) -> PipelineReport:
        report = PipelineReport(language=Language(language))
        report.speakers_seen = selector.seen_speaker_count
        report.speakers_selected = len(selections)
        report.speakers_discarded = report.speakers_seen - report.speakers_selected
        for selection in selections.values():
            if selection.tier is SelectionTier.PRIMARY:
                report.primary_tier_count += 1
            else:
                report.fallback_tier_count += 1
            report.total_selected_duration_seconds += selection.sample.duration_seconds
            gender_key = selection.sample.gender.value
            report.gender_breakdown[gender_key] = report.gender_breakdown.get(gender_key, 0) + 1
        return report

    def _write_report(self, language: str, report: PipelineReport) -> None:
        manifests_dir = self.output_root / "_manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        path = manifests_dir / f"{language}_summary.json"
        path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        Logger.info(f"Wrote pipeline summary -> {path}")

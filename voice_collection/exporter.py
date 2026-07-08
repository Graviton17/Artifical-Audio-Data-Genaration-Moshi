"""Writes the winning per-speaker clip to ``{language}/{gender}/{speaker}/``.

Folder contract (mirrors the HuggingFace bucket layout, see ``storage/huggingface_storage.py``)::

    <output_root>/
        english/
            male/<speaker_id>/audio.wav
            male/<speaker_id>/metadata.json
            female/...
        hindi/
            male/...
            female/...
        _manifests/<language>_summary.json
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from .models import Language, SpeakerSelection
from .processing.audio_codec import AudioProcessingError, export_audio

logger = logging.getLogger(__name__)


def slugify(value: str) -> str:
    """Filesystem-safe speaker folder name (keeps letters, digits, ``_``/``-``)."""
    value = value.strip().replace(" ", "_")
    return re.sub(r"[^a-zA-Z0-9_-]+", "", value) or "unknown"


class DatasetExporter:
    """Writes selected speakers to ``{output_root}/{language}/{gender}/{speaker}/``."""

    def __init__(
        self,
        output_root: Path,
        target_sample_rate: int,
        anonymize_speaker_names: bool = False,
    ) -> None:
        self.output_root = Path(output_root)
        self.target_sample_rate = target_sample_rate
        self.anonymize_speaker_names = anonymize_speaker_names

    def export_language(self, language: Language, selections: dict[str, SpeakerSelection]) -> list[Path]:
        """Export every winning speaker for ``language``; skips (and logs) failures."""
        exported: list[Path] = []
        anonymized_counters: dict[str, int] = {}

        for speaker_id, selection in sorted(selections.items()):
            gender = selection.sample.gender.value
            folder_name = self._folder_name(gender, speaker_id, anonymized_counters)
            speaker_dir = self.output_root / language.value / gender / folder_name
            audio_path = speaker_dir / "audio.wav"

            try:
                exported_duration = export_audio(selection.sample.audio_ref, self.target_sample_rate, audio_path)
            except AudioProcessingError as err:
                logger.warning("Skipping speaker %s: %s", speaker_id, err)
                continue

            metadata = self._build_metadata(selection, exported_duration, audio_path.name)
            (speaker_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            exported.append(speaker_dir)

        return exported

    def _folder_name(self, gender: str, speaker_id: str, counters: dict[str, int]) -> str:
        if not self.anonymize_speaker_names:
            return slugify(speaker_id)
        counters[gender] = counters.get(gender, 0) + 1
        return f"speaker_{counters[gender]}"

    @staticmethod
    def _build_metadata(selection: SpeakerSelection, exported_duration_seconds: float, audio_file: str) -> dict:
        sample = selection.sample
        return {
            "speaker_id": sample.speaker_id,
            "language": sample.language.value,
            "gender": sample.gender.value,
            "transcript": sample.transcript,
            "duration_seconds": round(exported_duration_seconds, 3),
            "original_duration_seconds": round(sample.duration_seconds, 3),
            "selection_tier": selection.tier.value,
            "source_dataset": sample.source_dataset,
            "source_index": sample.source_index,
            "audio_file": audio_file,
            "metadata": sample.extra_metadata,
        }

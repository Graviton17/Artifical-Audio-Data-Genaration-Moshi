"""IndicVoices-R dataset adapter (used here for Hindi).

Per-language releases (the official ``ai4bharat/indicvoices_r`` repo ships one
folder per language, e.g. ``Hindi/``; some mirrors publish a language as a
standalone dataset, e.g. ``SPRINGLab/IndicVoices-R_Hindi``) share this schema:
``text``, ``verbatim``, ``normalized``, ``speaker_id``, ``scenario``,
``task_name``, ``gender``, ``age_group``, ``job_type``, ``qualification``,
``area``, ``district``, ``state``, ``occupation``, ``duration``, ``audio``,
plus speech-quality metrics (``snr``, ``c50``, ``speaking_rate``, ...).
Unlike Svarah, a ``speaker_id`` is provided directly, so no filename parsing
is needed here.

Several of these columns are ``ClassLabel`` (stored as ints); ``_decode_label``
turns them back into their string values using the dataset's own feature
metadata, whichever config/mirror ends up being loaded.
"""
from __future__ import annotations

import logging
from typing import Any, Iterator

from datasets import Audio, load_dataset

from ..models import AudioRef, AudioSample, Gender, Language
from .base_source import BaseDatasetSource

logger = logging.getLogger(__name__)


class IndicVoicesDatasetSource(BaseDatasetSource):
    """Streams :class:`AudioSample` rows from an IndicVoices-R language split."""

    language = Language.HINDI.value

    def stream(self) -> Iterator[AudioSample]:
        repo_id = self.config["repo_id"]
        split = self.config.get("split", "train")
        config_name = self.config.get("config_name")
        data_dir = self.config.get("data_dir")
        token = self.config.get("hf_token")

        strategies = []
        if config_name:
            strategies.append(
                lambda: load_dataset(repo_id, name=config_name, split=split, streaming=True, token=token)
            )
        if data_dir:
            strategies.append(
                lambda: load_dataset(repo_id, data_dir=data_dir, split=split, streaming=True, token=token)
            )
        strategies.append(lambda: load_dataset(repo_id, split=split, streaming=True, token=token))

        dataset = self._load_with_fallbacks(strategies)
        features = getattr(dataset, "features", {}) or {}

        if "audio" in features:
            try:
                dataset = dataset.cast_column("audio", Audio(decode=False))
            except Exception as err:  # noqa: BLE001 - decoding is an optimisation, not required
                logger.warning("Could not disable audio auto-decode for IndicVoices-R: %s", err)

        filter_column = self.config.get("language_filter_column")
        filter_value = self.config.get("language_filter_value")
        if filter_column and filter_value is not None:
            dataset = dataset.filter(
                lambda row: self._decode_label(features, filter_column, row.get(filter_column)) == filter_value
            )

        for index, row in enumerate(dataset):
            audio_field = row.get("audio") or {}
            transcript = row.get("normalized") or row.get("verbatim") or row.get("text") or ""
            speaker_id = str(row.get("speaker_id") or f"unknown_spk_{index}")
            yield AudioSample(
                speaker_id=speaker_id,
                language=Language.HINDI,
                gender=Gender.from_raw(self._decode_label(features, "gender", row.get("gender"))),
                duration_seconds=float(row.get("duration") or 0.0),
                transcript=transcript,
                audio_ref=AudioRef(raw_bytes=audio_field.get("bytes"), path_hint=audio_field.get("path")),
                source_dataset=repo_id,
                source_index=index,
                extra_metadata={
                    "age_group": self._decode_label(features, "age_group", row.get("age_group")),
                    "job_type": self._decode_label(features, "job_type", row.get("job_type")),
                    "qualification": self._decode_label(features, "qualification", row.get("qualification")),
                    "area": self._decode_label(features, "area", row.get("area")),
                    "district": row.get("district"),
                    "state": self._decode_label(features, "state", row.get("state")),
                    "occupation": row.get("occupation"),
                    "task_name": row.get("task_name"),
                    "scenario": self._decode_label(features, "scenario", row.get("scenario")),
                    "snr": row.get("snr"),
                    "c50": row.get("c50"),
                    "speaking_rate": row.get("speaking_rate"),
                },
            )

    @staticmethod
    def _decode_label(features: dict, column: str, value: Any) -> Any:
        feature = features.get(column)
        if value is not None and hasattr(feature, "int2str"):
            try:
                return feature.int2str(value)
            except Exception:  # noqa: BLE001 - fall back to the raw value
                return value
        return value

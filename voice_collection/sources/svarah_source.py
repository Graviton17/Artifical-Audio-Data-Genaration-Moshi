"""Svarah (English, Indic-accented) dataset adapter.

Schema on the Hub: ``audio_filepath`` (Audio), ``duration`` (float),
``text``, ``gender``, ``age-group``, ``primary_language``,
``native_place_state``, ``native_place_district``, ``highest_qualification``,
``job_category``, ``occupation_domain``.

Svarah's HF release has no explicit ``speaker_id`` column (see
https://github.com/AI4Bharat/Svarah/issues/4), but every ``audio_filepath``
follows ``.../<utterance_id>_<gender-code><n>_chunk_<k>.wav`` (e.g.
``wavs/281474976884635_f3269_chunk_0.wav``) -- the ``<gender-code><n>`` group
(``f3269``, ``m118``, ...) is a stable per-speaker code shared by every chunk
of that speaker's recording, which is exactly the "multiple instances of one
speaker" the selector needs. When a filename doesn't match (unexpected mirror
layout), we fall back to a deterministic fingerprint of the speaker metadata
fields so the pipeline degrades gracefully instead of crashing.
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Iterator

from datasets import Audio, load_dataset

from ..models import AudioRef, AudioSample, Gender, Language
from .base_source import BaseDatasetSource

logger = logging.getLogger(__name__)

_SPEAKER_CODE_PATTERN = re.compile(r"([fm]\d{2,6})_chunk", re.IGNORECASE)

_METADATA_FINGERPRINT_FIELDS = (
    "gender",
    "primary_language",
    "native_place_district",
    "native_place_state",
    "highest_qualification",
    "job_category",
    "occupation_domain",
)


class SvarahDatasetSource(BaseDatasetSource):
    """Streams :class:`AudioSample` rows from ``ai4bharat/Svarah``."""

    language = Language.ENGLISH.value

    def stream(self) -> Iterator[AudioSample]:
        repo_id = self.config["repo_id"]
        split = self.config.get("split", "test")
        token = self.config.get("hf_token")

        dataset = self._load_with_fallbacks(
            [lambda: load_dataset(repo_id, split=split, streaming=True, token=token)]
        )
        try:
            dataset = dataset.cast_column("audio_filepath", Audio(decode=False))
        except Exception as err:  # noqa: BLE001 - decoding is an optimisation, not required
            logger.warning("Could not disable audio auto-decode for Svarah: %s", err)

        for index, row in enumerate(dataset):
            audio_field = row.get("audio_filepath") or {}
            path_hint = audio_field.get("path") or ""
            speaker_id = self._resolve_speaker_id(row, path_hint, index)
            yield AudioSample(
                speaker_id=speaker_id,
                language=Language.ENGLISH,
                gender=Gender.from_raw(row.get("gender")),
                duration_seconds=float(row.get("duration") or 0.0),
                transcript=row.get("text") or "",
                audio_ref=AudioRef(raw_bytes=audio_field.get("bytes"), path_hint=path_hint),
                source_dataset=repo_id,
                source_index=index,
                extra_metadata={
                    "age_group": row.get("age-group"),
                    "primary_language": row.get("primary_language"),
                    "native_place_state": row.get("native_place_state"),
                    "native_place_district": row.get("native_place_district"),
                    "highest_qualification": row.get("highest_qualification"),
                    "job_category": row.get("job_category"),
                    "occupation_domain": row.get("occupation_domain"),
                },
            )

    @staticmethod
    def _resolve_speaker_id(row: dict, path_hint: str, index: int) -> str:
        match = _SPEAKER_CODE_PATTERN.search(path_hint)
        if match:
            return match.group(1).lower()

        fingerprint = "|".join(str(row.get(field, "")) for field in _METADATA_FINGERPRINT_FIELDS)
        if fingerprint.strip("|"):
            digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:10]
            logger.debug(
                "Row %d: no speaker code found in '%s', falling back to metadata fingerprint spk_%s",
                index, path_hint, digest,
            )
            return f"spk_{digest}"
        return f"unknown_spk_{index}"

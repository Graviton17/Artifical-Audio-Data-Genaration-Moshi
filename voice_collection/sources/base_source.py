"""Provider-agnostic speech-dataset interface for the voice-collection pipeline.

The pipeline depends only on :class:`BaseDatasetSource`; concrete datasets
(Svarah for English, IndicVoices-R for Hindi) live in sibling modules and are
selected at construction time via :mod:`voice_collection.sources.factory`.
This keeps the selection/export logic free of any HuggingFace dataset-specific
quirks -- the same way ``conversations_generator`` agents depend only on
``BaseLLM``.

Note: this package is named ``sources`` (not ``datasets``) on purpose, so it
never shadows the third-party ``datasets`` library (``pip install datasets``)
that the concrete adapters import from.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator

from ..models import AudioSample

logger = logging.getLogger(__name__)


class DatasetLoadError(RuntimeError):
    """Raised when a dataset cannot be streamed from the Hub after every fallback."""


class BaseDatasetSource(ABC):
    """Base class for a per-language speech-dataset adapter.

    Subclasses normalise the dataset-specific schema (column names, gender
    encodings, filename conventions, ClassLabel decoding, ...) into a stream
    of :class:`AudioSample` objects. Nothing outside ``sources/`` needs to
    know those quirks.
    """

    #: Language this source produces samples for (see ``models.Language``).
    language: str

    def __init__(self, dataset_config: dict[str, Any]) -> None:
        self.config = dataset_config

    @abstractmethod
    def stream(self) -> Iterator[AudioSample]:
        """Yield every audio instance in the dataset as a normalised sample."""
        raise NotImplementedError

    @staticmethod
    def _load_with_fallbacks(strategies: list[Callable[[], Any]]) -> Any:
        """Try each ``load_dataset(...)`` call in turn; return the first success.

        HuggingFace datasets that ship one folder/config per language (like
        IndicVoices-R) don't always expose the same loading signature across
        mirrors/library versions, so callers pass loader callables in
        priority order instead of hard-coding a single call style.
        """
        last_error: Exception | None = None
        for position, strategy in enumerate(strategies, start=1):
            try:
                dataset = strategy()
                logger.info("Dataset loaded using strategy #%d.", position)
                return dataset
            except Exception as err:  # noqa: BLE001 - normalized into DatasetLoadError below
                logger.debug("Dataset load strategy #%d failed: %s", position, err)
                last_error = err
        raise DatasetLoadError(f"All dataset loading strategies failed: {last_error}") from last_error

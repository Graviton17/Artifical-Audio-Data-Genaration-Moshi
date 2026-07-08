"""Wires a ``--language`` CLI value to a concrete dataset source.

This is the single place that decides which :class:`BaseDatasetSource`
backs each language, mirroring ``conversations_generator/llm/factory.py``'s
provider registry.

    from voice_collection.sources import DatasetSourceFactory

    source = DatasetSourceFactory.create("hindi", dataset_config)
"""
from __future__ import annotations

from ..models import Language
from .base_source import BaseDatasetSource
from .indicvoices_source import IndicVoicesDatasetSource
from .svarah_source import SvarahDatasetSource

# Each language's default adapter. Extend via DatasetSourceFactory.register()
# instead of editing call sites when a new language/dataset is added.
_SOURCE_CLASSES: dict[str, type[BaseDatasetSource]] = {
    Language.ENGLISH.value: SvarahDatasetSource,
    Language.HINDI.value: IndicVoicesDatasetSource,
}


class DatasetSourceFactory:
    """Creates the right :class:`BaseDatasetSource` for a language."""

    @classmethod
    def register(cls, language: str):
        """Class decorator: plug in a new language without editing this file's body.

            @DatasetSourceFactory.register("tamil")
            class TamilDatasetSource(BaseDatasetSource):
                ...
        """

        def decorator(source_cls: type[BaseDatasetSource]) -> type[BaseDatasetSource]:
            _SOURCE_CLASSES[language] = source_cls
            return source_cls

        return decorator

    @staticmethod
    def create(language: str, dataset_config: dict) -> BaseDatasetSource:
        source_cls = _SOURCE_CLASSES.get(language)
        if source_cls is None:
            valid = ", ".join(sorted(_SOURCE_CLASSES))
            raise ValueError(f"No dataset source registered for language '{language}'. Choose from: {valid}")
        return source_cls(dataset_config)

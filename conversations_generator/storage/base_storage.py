"""Provider-agnostic storage interface for generated conversations + checkpoint.

The runner depends only on :class:`BaseStorage`; concrete backends (HuggingFace
today, could be S3/GCS/local later) live in sibling modules and are selected at
construction time. This keeps the pipeline free of any vendor SDK details, the
same way agents depend only on ``BaseLLM``.

Layout contract every backend must honor::

    <bucket root>/
        english/                             # one folder per corpus language
            checkpoint.json                  # resume state for this language
            skipped.json                     # abandoned instances (this language)
            instance_<id>/
                conversation_0001/
                    conversation.json
                    metadata.txt
                    transcript.txt
                conversation_0002/
                    ...
        hinglish/
            ...
        hindi/
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .checkpoint import Checkpoint
from .skipped import SkippedRegistry

SUPPORTED_LANGUAGE_FOLDERS = ("english", "hinglish", "hindi")


class StorageError(RuntimeError):
    """Raised when a storage backend operation fails."""


class BaseStorage(ABC):
    """Base class for conversation + checkpoint storage backends."""

    #: File name of the resume state inside each language folder.
    CHECKPOINT_NAME = "checkpoint.json"

    #: File name of the abandoned-instance registry inside each language folder.
    SKIPPED_NAME = "skipped.json"

    @staticmethod
    def normalize_language(language: str) -> str:
        """Map corpus language labels to bucket folder names."""
        return language.strip().lower()

    @staticmethod
    def language_root(language: str) -> str:
        return BaseStorage.normalize_language(language)

    @staticmethod
    def instance_folder(corpus_combination_id: int) -> str:
        """Folder name that holds every conversation for one instance."""
        return f"instance_{corpus_combination_id:04d}"

    @staticmethod
    def conversation_folder(index: int) -> str:
        """Folder name for one conversation's artifacts."""
        return f"conversation_{index:04d}"

    @staticmethod
    def conversation_name(index: int) -> str:
        """Legacy helper — JSON file name (prefer :meth:`conversation_folder`)."""
        return f"conversation_{index:04d}.json"

    @classmethod
    def conversation_base(
        cls,
        language: str,
        corpus_combination_id: int,
        index: int,
    ) -> str:
        """Prefix path for one conversation directory (no trailing file)."""
        return (
            f"{cls.language_root(language)}/"
            f"{cls.instance_folder(corpus_combination_id)}/"
            f"{cls.conversation_folder(index)}"
        )

    # ------------------------------------------------------------------ #
    # Subclass contract
    # ------------------------------------------------------------------ #
    @abstractmethod
    def save_conversation(
        self,
        corpus_combination_id: int,
        index: int,
        payload: dict[str, Any],
        *,
        language: str,
        metadata_text: str | None = None,
        transcript_text: str | None = None,
    ) -> str:
        """Persist one conversation under its instance folder.

        Returns the path (within the bucket) the conversation was written to.
        """
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, language: str) -> Checkpoint:
        """Read ``<language>/checkpoint.json``; return an empty one if absent."""
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, checkpoint: Checkpoint, language: str) -> None:
        """Write ``<language>/checkpoint.json`` (overwriting any prior state)."""
        raise NotImplementedError

    @abstractmethod
    def load_skipped(self, language: str) -> SkippedRegistry:
        """Read ``<language>/skipped.json``; return an empty registry if absent."""
        raise NotImplementedError

    @abstractmethod
    def save_skipped(self, skipped: SkippedRegistry, language: str) -> None:
        """Write ``<language>/skipped.json`` (overwriting any prior state)."""
        raise NotImplementedError

"""Provider-agnostic storage interface for generated conversations + checkpoint.

The runner depends only on :class:`BaseStorage`; concrete backends (HuggingFace
today, could be S3/GCS/local later) live in sibling modules and are selected at
construction time. This keeps the pipeline free of any vendor SDK details, the
same way agents depend only on ``BaseLLM``.

Layout contract every backend must honor::

    <bucket root>/
        checkpoint.json                     # resume state (root level)
        skipped.json                         # instances abandoned after N failures
        <language>/                          # english / hindi / hinglish
            instance_<id>/                   # one folder per corpus instance
                conversation_0001.json
                conversation_0002.json
                ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .checkpoint import Checkpoint
from .skipped import SkippedRegistry


class StorageError(RuntimeError):
    """Raised when a storage backend operation fails."""


class BaseStorage(ABC):
    """Base class for conversation + checkpoint storage backends."""

    #: File name of the resume state at the bucket root.
    CHECKPOINT_NAME = "checkpoint.json"

    #: File name of the abandoned-instance registry at the bucket root.
    SKIPPED_NAME = "skipped.json"

    #: File name of the per-model token-usage summary at the bucket root.
    TOKEN_USAGE_NAME = "metadata.json"

    @staticmethod
    def language_folder(language: str | None) -> str:
        """Top-level folder grouping conversations by language.

        Normalised to a lowercase, filesystem-safe name (e.g. ``english``,
        ``hindi``, ``hinglish``); a blank/unknown language falls back to
        ``unknown``.
        """
        return (language or "unknown").strip().lower() or "unknown"

    @staticmethod
    def instance_folder(corpus_combination_id: int) -> str:
        """Folder name that holds every conversation for one instance."""
        return f"instance_{corpus_combination_id:04d}"

    @staticmethod
    def conversation_name(index: int) -> str:
        """File name (within an instance folder) for one conversation."""
        return f"conversation_{index:04d}.json"

    # ------------------------------------------------------------------ #
    # Subclass contract
    # ------------------------------------------------------------------ #
    @abstractmethod
    def save_conversation(
        self,
        corpus_combination_id: int,
        index: int,
        payload: dict[str, Any],
    ) -> str:
        """Persist one conversation under its instance folder.

        Returns the path (within the bucket) the conversation was written to.
        """
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self) -> Checkpoint:
        """Read the root ``checkpoint.json``; return an empty one if absent."""
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Write the root ``checkpoint.json`` (overwriting any prior state)."""
        raise NotImplementedError

    @abstractmethod
    def save_token_usage(self, summary: dict[str, Any]) -> None:
        """Write the per-model token-usage summary to the bucket root
        (``metadata.json``), overwriting any prior version."""
        raise NotImplementedError

    @abstractmethod
    def load_skipped(self) -> SkippedRegistry:
        """Read the root ``skipped.json``; return an empty registry if absent."""
        raise NotImplementedError

    @abstractmethod
    def save_skipped(self, skipped: SkippedRegistry) -> None:
        """Write the root ``skipped.json`` (overwriting any prior state)."""
        raise NotImplementedError

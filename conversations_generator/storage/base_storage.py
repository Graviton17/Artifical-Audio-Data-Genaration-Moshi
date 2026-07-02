"""Provider-agnostic storage interface for generated conversations + checkpoint.

The runner depends only on :class:`BaseStorage`; concrete backends (HuggingFace
today, could be S3/GCS/local later) live in sibling modules and are selected at
construction time. This keeps the pipeline free of any vendor SDK details, the
same way agents depend only on ``BaseLLM``.

Layout contract every backend must honor::

    <bucket root>/
        checkpoint.json                     # resume state (root level)
        instance_<id>/                       # one folder per corpus instance
            conversation_0001.json
            conversation_0002.json
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .checkpoint import Checkpoint


class StorageError(RuntimeError):
    """Raised when a storage backend operation fails."""


class BaseStorage(ABC):
    """Base class for conversation + checkpoint storage backends."""

    #: File name of the resume state at the bucket root.
    CHECKPOINT_NAME = "checkpoint.json"

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

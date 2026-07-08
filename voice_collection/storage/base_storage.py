"""Provider-agnostic upload interface for the ``{language}/{gender}/{speaker}``
bucket layout.

The pipeline depends only on :class:`BaseStorage`; concrete backends
(HuggingFace Storage Bucket, local copy) live in sibling modules and are
selected at construction time via :mod:`voice_collection.storage.factory`.
This keeps the pipeline free of any vendor SDK details -- the same pattern
``conversations_generator/storage/base_storage.py`` uses for its bucket.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class StorageError(RuntimeError):
    """Raised when a storage backend operation fails."""


class BaseStorage(ABC):
    """Base class for the voice-collection bucket upload backends."""

    @abstractmethod
    def upload_directory(self, local_root: Path, remote_prefix: str) -> int:
        """Upload every file under ``local_root`` beneath ``remote_prefix``.

        Returns the number of files uploaded.
        """
        raise NotImplementedError

    @abstractmethod
    def list_remote_files(self, remote_prefix: str) -> set[str]:
        """Return file paths that already exist under ``remote_prefix``.

        Paths are returned relative to ``remote_prefix`` so callers can compare
        them directly to ``local_root.rglob(...)`` relative paths.
        """
        raise NotImplementedError

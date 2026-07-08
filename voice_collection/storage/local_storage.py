"""Local-filesystem implementation of :class:`BaseStorage`.

Used for ``STORAGE_PROVIDER=local`` in ``config.json``: copies the export
tree to another local folder instead of talking to the HuggingFace bucket, so the pipeline can be
exercised end-to-end (selection, export, folder layout) without any cloud
credentials -- handy together with ``runner.py --dry-run``.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from .base_storage import BaseStorage

logger = logging.getLogger(__name__)


class LocalStorage(BaseStorage):
    """Copies the export tree to ``destination_root/<remote_prefix>``."""

    def __init__(self, destination_root: Path) -> None:
        self.destination_root = Path(destination_root)

    def upload_directory(self, local_root: Path, remote_prefix: str) -> int:
        local_root = Path(local_root)
        destination = self.destination_root / remote_prefix
        if local_root.resolve() == destination.resolve():
            logger.info("Local storage provider: data already staged at %s", destination)
            return sum(1 for path in local_root.rglob("*") if path.is_file())

        count = 0
        for file_path in local_root.rglob("*"):
            if file_path.is_dir():
                continue
            target = destination / file_path.relative_to(local_root)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, target)
            count += 1

        logger.info("Copied %d file(s) to %s", count, destination)
        return count

    def list_remote_files(self, remote_prefix: str) -> set[str]:
        destination = self.destination_root / remote_prefix
        if not destination.exists():
            return set()
        return {
            path.relative_to(destination).as_posix()
            for path in destination.rglob("*")
            if path.is_file()
        }

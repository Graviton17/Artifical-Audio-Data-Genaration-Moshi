"""Local on-disk cache for small HuggingFace bucket control files.

Checkpoints and skipped registries are tiny JSON blobs. Caching them locally
avoids re-downloading on every run and keeps huggingface_hub progress bars
from flashing during generation startup.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_CACHE_ROOT = Path(__file__).resolve().parent.parent.parent / ".cache" / "hf_control"


class ControlFileCache:
    """Hash-indexed cache for ``checkpoint.json`` / ``skipped.json`` per language."""

    def __init__(self, bucket_id: str) -> None:
        self.bucket_id = bucket_id.replace("/", "__")
        self.root = _CACHE_ROOT / self.bucket_id

    def _lang_dir(self, language: str) -> Path:
        return self.root / language.strip().lower()

    def _data_path(self, language: str, filename: str) -> Path:
        return self._lang_dir(language) / filename

    def _hash_path(self, language: str, filename: str) -> Path:
        return self._lang_dir(language) / f".{filename}.hash"

    def read(self, language: str, filename: str) -> dict[str, Any] | None:
        path = self._data_path(language, filename)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return data if isinstance(data, dict) else None

    def stored_hash(self, language: str, filename: str) -> str | None:
        path = self._hash_path(language, filename)
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8").strip() or None

    def write(self, language: str, filename: str, payload: dict[str, Any], *, xet_hash: str | None = None) -> None:
        lang_dir = self._lang_dir(language)
        lang_dir.mkdir(parents=True, exist_ok=True)
        self._data_path(language, filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if xet_hash:
            self._hash_path(language, filename).write_text(xet_hash, encoding="utf-8")

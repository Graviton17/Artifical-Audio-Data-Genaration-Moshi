"""HuggingFace **Storage Bucket** implementation of :class:`BaseStorage`.

Uses the ``huggingface_hub`` bucket API (requires ``huggingface_hub>=1.5.0``).
Conversations and the resume checkpoint live in a HuggingFace Storage Bucket
(the "bucket"), referenced by an ``hf://buckets/<namespace>/<name>`` path.

The write token is read from the ``api_key`` argument or ``HF_TOKEN`` /
``HUGGINGFACE_TOKEN`` in ``conversations_generator/config.json``. The target
bucket is read from ``bucket`` or ``HF_BUCKET`` in the same config; both the
full ``hf://buckets/ns/name`` form and the short ``ns/name`` form are accepted.

Bucket layout::

    hf://buckets/<ns>/<name>/
        english/
            checkpoint.json
            skipped.json
            instance_0001/
                conversation_0001/
                    conversation.json
                    metadata.txt
                    transcript.txt
        hinglish/
            ...
        hindi/
            ...
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ..configuration_reader import get as config_get
from .base_storage import BaseStorage, StorageError
from .checkpoint import Checkpoint
from .disk_cache import ControlFileCache
from .skipped import SkippedRegistry

# Declared as ``Any`` so the ``None`` import fallback below doesn't make type
# checkers flag the guarded call sites as "Object of type None cannot be called".
batch_bucket_files: Any
create_bucket: Any
download_bucket_files: Any

try:  # Lazy-ish import so the package works without the SDK installed.
    from huggingface_hub import (
        batch_bucket_files,
        create_bucket,
        download_bucket_files,
    )
except ImportError:  # pragma: no cover - exercised only without the dep
    batch_bucket_files = None
    create_bucket = None
    download_bucket_files = None

_BUCKET_PREFIX = "hf://buckets/"

# Substrings used to recognise "the file/bucket isn't there yet" errors, so a
# first-ever run (no checkpoint) is treated as empty rather than fatal.
_NOT_FOUND_HINTS = ("not found", "404", "no such", "does not exist", "notfound")


class HuggingFaceStorage(BaseStorage):
    """Store conversations + checkpoint in a HuggingFace Storage Bucket."""

    def __init__(
        self,
        bucket: str | None = None,
        *,
        api_key: str | None = None,
        private: bool = True,
        create: bool = True,
    ) -> None:
        if batch_bucket_files is None:
            raise ImportError(
                "huggingface_hub>=1.5.0 (with bucket support) is not installed. "
                "Run `pip install -U huggingface_hub`."
            )

        raw = bucket or config_get("HF_BUCKET")
        if not raw:
            raise StorageError(
                "No HuggingFace bucket configured. Pass bucket= or set HF_BUCKET "
                "in conversations_generator/config.json "
                "(e.g. 'hf://buckets/inavlabs/kupe-fdx-text-data')."
            )
        # The bucket API takes the short 'namespace/name' form; accept either.
        self.bucket_id = raw.removeprefix(_BUCKET_PREFIX).strip("/")

        self.token = api_key or config_get("HF_TOKEN") or config_get("HUGGINGFACE_TOKEN")
        if not self.token:
            raise StorageError(
                "No HuggingFace token found. Pass api_key= or set HF_TOKEN "
                "in conversations_generator/config.json "
                "(needs a WRITE token with access to the bucket's namespace)."
            )
        # huggingface_hub reads HF_TOKEN from the environment; make sure it's set
        # even if the token was passed in explicitly.
        os.environ.setdefault("HF_TOKEN", self.token)

        self._tmp = Path(tempfile.mkdtemp(prefix="hf_bucket_"))
        self._disk_cache = ControlFileCache(self.bucket_id)

        if create:
            self._ensure_bucket(private)

    def _suppress_hf_progress(self) -> None:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    def _load_control_json(
        self,
        language: str,
        filename: str,
        *,
        force_remote: bool = False,
    ) -> dict[str, Any] | None:
        """Load a small control JSON from disk cache or HF. ``None`` if absent."""
        self._suppress_hf_progress()
        lang = self.normalize_language(language)
        if not force_remote:
            cached = self._disk_cache.read(lang, filename)
            if cached is not None:
                return cached

        path = f"{self.language_root(lang)}/{filename}"
        local = self._tmp / path.replace("/", "__")
        try:
            download_bucket_files(
                self.bucket_id,
                files=[(path, str(local))],
            )
        except Exception as err:  # noqa: BLE001
            if any(hint in str(err).lower() for hint in _NOT_FOUND_HINTS):
                return None
            raise StorageError(f"Failed to read {path}: {err}") from err

        if not local.exists():
            return None

        with open(local, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            self._disk_cache.write(lang, filename, data)
            return data
        return None

    def preload_language_state(
        self,
        languages: list[str],
        *,
        force_remote: bool = True,
    ) -> tuple[dict[str, Checkpoint], dict[str, SkippedRegistry]]:
        """Sync checkpoint + skipped for each language once at run startup."""
        checkpoints: dict[str, Checkpoint] = {}
        skipped: dict[str, SkippedRegistry] = {}
        for language in languages:
            lang = self.normalize_language(language)
            cp_data = self._load_control_json(
                lang, self.CHECKPOINT_NAME, force_remote=force_remote
            )
            checkpoints[lang] = (
                Checkpoint.from_dict(cp_data) if cp_data else self._init_checkpoint(language)
            )
            sk_data = self._load_control_json(
                lang, self.SKIPPED_NAME, force_remote=force_remote
            )
            skipped[lang] = (
                SkippedRegistry.from_dict(sk_data) if sk_data else self._init_skipped(language)
            )
        return checkpoints, skipped

    def _save_control_json(self, language: str, filename: str, payload: dict[str, Any]) -> None:
        lang = self.normalize_language(language)
        path = f"{self.language_root(lang)}/{filename}"
        self._upload_json(path, payload)
        self._disk_cache.write(lang, filename, payload)

    # ------------------------------------------------------------------ #
    # Path helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def _checkpoint_path(cls, language: str) -> str:
        return f"{cls.language_root(language)}/{cls.CHECKPOINT_NAME}"

    @classmethod
    def _skipped_path(cls, language: str) -> str:
        return f"{cls.language_root(language)}/{cls.SKIPPED_NAME}"

    # ------------------------------------------------------------------ #
    # Bucket lifecycle
    # ------------------------------------------------------------------ #
    def _ensure_bucket(self, private: bool) -> None:
        """Create the bucket if it doesn't exist; no-op if it already does."""
        try:
            create_bucket(self.bucket_id, private=private)
        except Exception as err:  # noqa: BLE001 - normalize below
            msg = str(err).lower()
            if "exist" in msg or "409" in msg or "conflict" in msg:
                return  # already there — fine
            raise StorageError(f"Failed to create bucket {self.bucket_id}: {err}") from err

    # ------------------------------------------------------------------ #
    # Internal upload helper
    # ------------------------------------------------------------------ #
    def _upload_json(self, path_in_bucket: str, payload: dict[str, Any]) -> None:
        """Write ``payload`` to a temp file and upload it to ``path_in_bucket``."""
        local = self._tmp / path_in_bucket.replace("/", "__")
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        try:
            batch_bucket_files(self.bucket_id, add=[(str(local), path_in_bucket)])
        except Exception as err:  # noqa: BLE001
            raise StorageError(f"Failed to upload {path_in_bucket}: {err}") from err
        finally:
            local.unlink(missing_ok=True)

    def _upload_text(self, path_in_bucket: str, text: str) -> None:
        """Upload a plain-text file to ``path_in_bucket``."""
        local = self._tmp / path_in_bucket.replace("/", "__")
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(text, encoding="utf-8")
        try:
            batch_bucket_files(self.bucket_id, add=[(str(local), path_in_bucket)])
        except Exception as err:  # noqa: BLE001
            raise StorageError(f"Failed to upload {path_in_bucket}: {err}") from err
        finally:
            local.unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    # Conversations
    # ------------------------------------------------------------------ #
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
        base = self.conversation_base(language, corpus_combination_id, index)
        path_in_bucket = f"{base}/conversation.json"
        self._upload_json(path_in_bucket, payload)

        if metadata_text is not None:
            self._upload_text(f"{base}/metadata.txt", metadata_text)
        if transcript_text:
            self._upload_text(f"{base}/transcript.txt", transcript_text)

        return f"{_BUCKET_PREFIX}{self.bucket_id}/{path_in_bucket}"

    # ------------------------------------------------------------------ #
    # Checkpoint
    # ------------------------------------------------------------------ #
    def load_checkpoint(self, language: str) -> Checkpoint:
        data = self._load_control_json(language, self.CHECKPOINT_NAME)
        if data is None:
            return self._init_checkpoint(language)
        return Checkpoint.from_dict(data)

    def _init_checkpoint(self, language: str) -> Checkpoint:
        """First-run bootstrap: create an empty checkpoint and upload it."""
        checkpoint = Checkpoint()
        self.save_checkpoint(checkpoint, language)
        return checkpoint

    def save_checkpoint(self, checkpoint: Checkpoint, language: str) -> None:
        self._save_control_json(language, self.CHECKPOINT_NAME, checkpoint.to_dict())

    # ------------------------------------------------------------------ #
    # Skipped registry
    # ------------------------------------------------------------------ #
    def load_skipped(self, language: str) -> SkippedRegistry:
        data = self._load_control_json(language, self.SKIPPED_NAME)
        if data is None:
            return self._init_skipped(language)
        return SkippedRegistry.from_dict(data)

    def _init_skipped(self, language: str) -> SkippedRegistry:
        """First-run bootstrap: create an empty skipped registry and upload it."""
        registry = SkippedRegistry()
        self.save_skipped(registry, language)
        return registry

    def save_skipped(self, skipped: SkippedRegistry, language: str) -> None:
        self._save_control_json(language, self.SKIPPED_NAME, skipped.to_dict())

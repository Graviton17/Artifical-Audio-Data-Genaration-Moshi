"""HuggingFace **Storage Bucket** implementation of :class:`BaseStorage`.

Uses the ``huggingface_hub`` bucket API (requires ``huggingface_hub>=1.5.0``).
Conversations and the resume checkpoint live in a HuggingFace Storage Bucket
(the "bucket"), referenced by an ``hf://buckets/<namespace>/<name>`` path.

The write token is read from the ``api_key`` argument or the ``HF_TOKEN`` /
``HUGGINGFACE_TOKEN`` environment variables (``huggingface_hub`` also picks up
``HF_TOKEN`` automatically). The target bucket is read from ``bucket`` or the
``HF_BUCKET`` environment variable; both the full ``hf://buckets/ns/name`` form
and the short ``ns/name`` form are accepted.

Bucket layout::

    hf://buckets/<ns>/<name>/
        checkpoint.json
        instance_0001/conversation_0001.json
        instance_0001/conversation_0002.json
        instance_0002/conversation_0001.json
        ...
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .base_storage import BaseStorage, StorageError
from .checkpoint import Checkpoint

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

        raw = bucket or os.getenv("HF_BUCKET")
        if not raw:
            raise StorageError(
                "No HuggingFace bucket configured. Pass bucket= or set HF_BUCKET "
                "(e.g. 'hf://buckets/Graviton17/artificial-data-conversation')."
            )
        # The bucket API takes the short 'namespace/name' form; accept either.
        self.bucket_id = raw.removeprefix(_BUCKET_PREFIX).strip("/")

        self.token = api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if not self.token:
            raise StorageError(
                "No HuggingFace token found. Pass api_key= or set HF_TOKEN "
                "(needs a WRITE token with access to the bucket's namespace)."
            )
        # huggingface_hub reads HF_TOKEN from the environment; make sure it's set
        # even if the token was passed in explicitly.
        os.environ.setdefault("HF_TOKEN", self.token)

        self._tmp = Path(tempfile.mkdtemp(prefix="hf_bucket_"))

        if create:
            self._ensure_bucket(private)

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
        local.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
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
    ) -> str:
        path_in_bucket = (
            f"{self.instance_folder(corpus_combination_id)}/"
            f"{self.conversation_name(index)}"
        )
        self._upload_json(path_in_bucket, payload)
        return f"{_BUCKET_PREFIX}{self.bucket_id}/{path_in_bucket}"

    # ------------------------------------------------------------------ #
    # Checkpoint
    # ------------------------------------------------------------------ #
    def load_checkpoint(self) -> Checkpoint:
        local = self._tmp / self.CHECKPOINT_NAME
        try:
            download_bucket_files(
                self.bucket_id,
                files=[(self.CHECKPOINT_NAME, str(local))],
            )
        except Exception as err:  # noqa: BLE001
            if any(hint in str(err).lower() for hint in _NOT_FOUND_HINTS):
                return Checkpoint()  # first run — no checkpoint yet
            raise StorageError(f"Failed to read checkpoint: {err}") from err

        if not local.exists():
            # Some backends silently skip a missing file instead of raising.
            return Checkpoint()
        with open(local, "r", encoding="utf-8") as f:
            return Checkpoint.from_dict(json.load(f))

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        self._upload_json(self.CHECKPOINT_NAME, checkpoint.to_dict())

"""HuggingFace Storage Bucket implementation of :class:`BaseStorage`.

Uses the ``huggingface_hub`` bucket API (requires ``huggingface_hub>=1.5.0``).
The target bucket is read from ``HF_BUCKET`` in ``config.json``; both the full
``hf://buckets/<namespace>/<name>`` form and the short ``<namespace>/<name>``
form are accepted.

Bucket layout::

    hf://buckets/inavlabs/voice_collection/
        english/male/<speaker_id>/audio.wav
        english/male/<speaker_id>/metadata.json
        hindi/female/...
        _manifests/english_summary.json
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from .. import configuration_reader as config
from .base_storage import BaseStorage, StorageError

logger = logging.getLogger(__name__)

_BUCKET_PREFIX = "hf://buckets/"
_UPLOAD_BATCH_SIZE = 100

try:
    from huggingface_hub import HfApi, batch_bucket_files, create_bucket
except ImportError:  # pragma: no cover
    HfApi = None
    batch_bucket_files = None
    create_bucket = None


class HuggingFaceStorage(BaseStorage):
    """Uploads the local export tree to a HuggingFace Storage Bucket."""

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

        raw = bucket or config.get("HF_BUCKET")
        if not raw:
            raise StorageError(
                "No HuggingFace bucket configured. Set HF_BUCKET in voice_collection/config.json "
                "(e.g. 'hf://buckets/inavlabs/voice_collection')."
            )
        self.bucket_id = raw.removeprefix(_BUCKET_PREFIX).strip("/")

        self.token = api_key or config.get_hf_token() or config.get("HUGGINGFACE_TOKEN")
        if not self.token:
            raise StorageError(
                "No HuggingFace token found. Set HF_TOKEN in voice_collection/config.json "
                "(needs a WRITE token with access to the bucket namespace)."
            )
        os.environ.setdefault("HF_TOKEN", self.token)
        self._api = HfApi(token=self.token)

        if create:
            self._ensure_bucket(private)

    def _ensure_bucket(self, private: bool) -> None:
        try:
            create_bucket(self.bucket_id, private=private)
        except Exception as err:  # noqa: BLE001
            msg = str(err).lower()
            if "exist" in msg or "409" in msg or "conflict" in msg:
                return
            raise StorageError(f"Failed to create bucket {self.bucket_id}: {err}") from err

    def upload_directory(self, local_root: Path, remote_prefix: str) -> int:
        local_root = Path(local_root)
        if not local_root.exists():
            raise StorageError(f"Local path does not exist: {local_root}")

        prefix = remote_prefix.strip("/")
        file_paths = sorted(path for path in local_root.rglob("*") if path.is_file())
        if not file_paths:
            logger.warning("No files found under %s; nothing to upload.", local_root)
            return 0

        existing_remote = self.list_remote_files(prefix)
        file_paths = [
            path for path in file_paths if path.relative_to(local_root).as_posix() not in existing_remote
        ]
        if not file_paths:
            logger.info("All files under %s already exist in hf://buckets/%s/%s", local_root, self.bucket_id, prefix)
            return 0

        uploaded = 0
        for start in range(0, len(file_paths), _UPLOAD_BATCH_SIZE):
            batch = file_paths[start : start + _UPLOAD_BATCH_SIZE]
            add_pairs = [
                (str(file_path), f"{prefix}/{file_path.relative_to(local_root).as_posix()}" if prefix else file_path.relative_to(local_root).as_posix())
                for file_path in batch
            ]
            try:
                batch_bucket_files(self.bucket_id, add=add_pairs)
            except Exception as err:  # noqa: BLE001
                raise StorageError(
                    f"Failed to upload batch to hf://buckets/{self.bucket_id}/{prefix or ''}: {err}"
                ) from err
            uploaded += len(batch)
            logger.debug(
                "Uploaded batch of %d file(s) -> hf://buckets/%s/%s",
                len(batch),
                self.bucket_id,
                prefix,
            )

        logger.info(
            "Uploaded %d file(s) to hf://buckets/%s/%s",
            uploaded,
            self.bucket_id,
            prefix or "",
        )
        return uploaded

    def list_remote_files(self, remote_prefix: str) -> set[str]:
        prefix = remote_prefix.strip("/")
        try:
            items = self._api.list_bucket_tree(self.bucket_id, prefix=prefix or None, recursive=True)
        except Exception as err:  # noqa: BLE001
            raise StorageError(
                f"Failed to list files in hf://buckets/{self.bucket_id}/{prefix}: {err}"
            ) from err

        existing: set[str] = set()
        for item in items:
            path = getattr(item, "path", None)
            if not path:
                continue
            if prefix:
                if path == prefix:
                    existing.add(path.rsplit("/", 1)[-1])
                    continue
                if not path.startswith(prefix + "/"):
                    continue
                existing.add(path[len(prefix) + 1 :])
            else:
                existing.add(path)
        return existing

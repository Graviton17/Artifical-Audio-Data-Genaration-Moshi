"""Turns ``STORAGE_PROVIDER`` (config.json) into a concrete storage backend.

    from voice_collection.storage import create_storage

    storage = create_storage()              # provider from config.json
    storage = create_storage("local")       # explicit override (e.g. dry runs)
"""
from __future__ import annotations

from .. import configuration_reader as config
from .base_storage import BaseStorage
from .huggingface_storage import HuggingFaceStorage
from .local_storage import LocalStorage


def create_storage(provider: str | None = None) -> BaseStorage:
    """Build the storage backend named by ``provider`` (default: config.json)."""
    resolved = (provider or config.get_storage_provider()).strip().lower()

    if resolved in {"huggingface", "hf"}:
        hf_cfg = config.get_hf_bucket_config()
        return HuggingFaceStorage(
            bucket=hf_cfg["bucket"],
            api_key=config.get_hf_token(),
            private=hf_cfg["private"],
            create=hf_cfg["create_if_missing"],
        )
    if resolved == "local":
        return LocalStorage(destination_root=config.get_local_output_dir().parent / "uploaded")

    raise ValueError(
        f"Unsupported STORAGE_PROVIDER '{resolved}'. Choose 'huggingface' or 'local'."
    )

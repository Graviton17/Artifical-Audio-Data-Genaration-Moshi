"""Tests for the HuggingFace bucket storage backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from voice_collection.storage.base_storage import StorageError
from voice_collection.storage.huggingface_storage import HuggingFaceStorage
import voice_collection.storage.huggingface_storage as hf_storage


def test_upload_directory_batches_and_preserves_relative_paths(tmp_path, monkeypatch):
    local_root = tmp_path / "export"
    (local_root / "male" / "speaker_1").mkdir(parents=True)
    (local_root / "female" / "speaker_2").mkdir(parents=True)
    (local_root / "male" / "speaker_1" / "audio.wav").write_bytes(b"a")
    (local_root / "male" / "speaker_1" / "metadata.json").write_text("{}", encoding="utf-8")
    (local_root / "female" / "speaker_2" / "audio.wav").write_bytes(b"b")

    uploaded_batches: list[list[tuple[str, str]]] = []

    monkeypatch.setattr(hf_storage, "_UPLOAD_BATCH_SIZE", 2)
    monkeypatch.setattr(hf_storage, "create_bucket", lambda bucket_id, private=True: None)
    monkeypatch.setattr(hf_storage, "HfApi", lambda token=None: type("Api", (), {"list_bucket_tree": lambda *args, **kwargs: []})())
    monkeypatch.setattr(
        hf_storage,
        "batch_bucket_files",
        lambda bucket_id, add: uploaded_batches.append(list(add)),
    )

    storage = HuggingFaceStorage(
        bucket="hf://buckets/inavlabs/voice_collection",
        api_key="hf_test_token",
    )

    uploaded_count = storage.upload_directory(local_root, "english")

    assert uploaded_count == 3
    assert len(uploaded_batches) == 2
    flattened = [pair for batch in uploaded_batches for pair in batch]
    assert [remote for _, remote in flattened] == [
        "english/female/speaker_2/audio.wav",
        "english/male/speaker_1/audio.wav",
        "english/male/speaker_1/metadata.json",
    ]


def test_upload_directory_raises_for_missing_local_root(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_storage, "create_bucket", lambda bucket_id, private=True: None)
    monkeypatch.setattr(hf_storage, "batch_bucket_files", lambda bucket_id, add: None)
    monkeypatch.setattr(hf_storage, "HfApi", lambda token=None: type("Api", (), {"list_bucket_tree": lambda *args, **kwargs: []})())

    storage = HuggingFaceStorage(
        bucket="hf://buckets/inavlabs/voice_collection",
        api_key="hf_test_token",
    )

    with pytest.raises(StorageError, match="Local path does not exist"):
        storage.upload_directory(tmp_path / "missing", "english")


def test_init_requires_bucket_and_token(monkeypatch):
    monkeypatch.setattr(hf_storage, "create_bucket", lambda bucket_id, private=True: None)
    monkeypatch.setattr(hf_storage, "batch_bucket_files", lambda bucket_id, add: None)
    monkeypatch.setattr(hf_storage, "HfApi", lambda token=None: type("Api", (), {"list_bucket_tree": lambda *args, **kwargs: []})())
    monkeypatch.setattr(hf_storage.config, "get", lambda key, default=None: default)
    monkeypatch.setattr(hf_storage.config, "get_hf_token", lambda: None)

    with pytest.raises(StorageError, match="No HuggingFace bucket configured"):
        HuggingFaceStorage(bucket=None, api_key="hf_test_token", create=False)

    with pytest.raises(StorageError, match="No HuggingFace token found"):
        HuggingFaceStorage(bucket="hf://buckets/inavlabs/voice_collection", api_key=None, create=False)


def test_upload_directory_skips_already_uploaded_remote_files(tmp_path, monkeypatch):
    local_root = tmp_path / "export"
    (local_root / "male" / "speaker_1").mkdir(parents=True)
    (local_root / "male" / "speaker_1" / "audio.wav").write_bytes(b"a")
    (local_root / "male" / "speaker_1" / "metadata.json").write_text("{}", encoding="utf-8")

    uploaded_batches: list[list[tuple[str, str]]] = []

    monkeypatch.setattr(hf_storage, "create_bucket", lambda bucket_id, private=True: None)
    monkeypatch.setattr(
        hf_storage,
        "HfApi",
        lambda token=None: type(
            "Api",
            (),
            {"list_bucket_tree": lambda *args, **kwargs: [type("Item", (), {"path": "english/male/speaker_1/audio.wav"})()]},
        )(),
    )
    monkeypatch.setattr(
        hf_storage,
        "batch_bucket_files",
        lambda bucket_id, add: uploaded_batches.append(list(add)),
    )

    storage = HuggingFaceStorage(
        bucket="hf://buckets/inavlabs/voice_collection",
        api_key="hf_test_token",
    )

    uploaded_count = storage.upload_directory(local_root, "english")

    assert uploaded_count == 1
    assert uploaded_batches == [[(str(local_root / "male" / "speaker_1" / "metadata.json"), "english/male/speaker_1/metadata.json")]]

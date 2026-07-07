"""Migrate bucket / local output to per-language folder layout.

New layout::

    <bucket>/
        english/
            checkpoint.json
            skipped.json
            instance_0015/
                conversation_0001/
                    conversation.json
                    metadata.txt
                    transcript.txt
        hinglish/
            ...
        hindi/
            ...

Usage::

    # Reorganize existing bucket root files into english/ (in-place copy + delete)
    python -m conversations_generator.migrate_storage --language english --from-bucket

    # Upload local output/ runs into english/ on the bucket
    python -m conversations_generator.migrate_storage --language english --from-local

    # Both
    python -m conversations_generator.migrate_storage --language english --from-bucket --from-local
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .configuration_reader import apply_to_environ
from .storage.huggingface_storage import HuggingFaceStorage
from .storage.base_storage import BaseStorage
from .token_stats import legacy_krutrim_models, patch_hf_model_metadata

try:
    from huggingface_hub import batch_bucket_files, list_bucket_tree
except ImportError:  # pragma: no cover
    batch_bucket_files = None
    list_bucket_tree = None

_CONV_FILE_RE = re.compile(
    r"^instance_(?P<instance>\d+)/conversation_(?P<index>\d+)"
    r"(?:\.json|_(?P<kind>metadata|transcript)\.txt)$"
)
_ROOT_STATE_FILES = frozenset({BaseStorage.CHECKPOINT_NAME, BaseStorage.SKIPPED_NAME})


def normalize_language(language: str) -> str:
    return language.strip().lower()


def conversation_dest(
    language: str,
    instance_id: int,
    index: int,
    *,
    filename: str,
) -> str:
    """Destination path for one artifact inside the new layout."""
    lang = normalize_language(language)
    inst = BaseStorage.instance_folder(instance_id)
    conv = BaseStorage.conversation_folder(index)
    return f"{lang}/{inst}/{conv}/{filename}"


def _dest_filename(old_path: str) -> str | None:
    """Map a legacy flat path to the filename inside conversation_XXXX/."""
    if old_path.endswith(".json"):
        return "conversation.json"
    if old_path.endswith("_metadata.txt"):
        return "metadata.txt"
    if old_path.endswith("_transcript.txt"):
        return "transcript.txt"
    return None


def _parse_legacy_conversation_path(path: str) -> tuple[int, int, str] | None:
    m = _CONV_FILE_RE.match(path)
    if not m:
        return None
    filename = _dest_filename(path)
    if filename is None:
        return None
    return int(m.group("instance")), int(m.group("index")), filename


def migrate_bucket_layout(
    bucket_id: str,
    language: str,
    *,
    dry_run: bool = False,
) -> int:
    """Copy legacy bucket paths into ``<language>/…`` and delete the old keys."""
    if batch_bucket_files is None or list_bucket_tree is None:
        raise ImportError("huggingface_hub>=1.5.0 is required.")

    lang = normalize_language(language)
    copies: list[tuple[str, str, str, str]] = []
    deletes: list[str] = []

    for item in list_bucket_tree(bucket_id, recursive=True):
        path = getattr(item, "path", None)
        xet_hash = getattr(item, "xet_hash", None)
        if not path or not xet_hash:
            continue

        # Skip paths already under a language folder.
        if path.startswith(f"{lang}/"):
            continue

        if path in _ROOT_STATE_FILES:
            dest = f"{lang}/{path}"
            copies.append(("bucket", bucket_id, xet_hash, dest))
            deletes.append(path)
            continue

        parsed = _parse_legacy_conversation_path(path)
        if parsed is None:
            continue
        instance_id, index, filename = parsed
        dest = conversation_dest(language, instance_id, index, filename=filename)
        copies.append(("bucket", bucket_id, xet_hash, dest))
        deletes.append(path)

    if not copies:
        print(f"No legacy files to migrate for language={lang!r}.")
        return 0

    print(f"Migrating {len(copies)} file(s) into {lang}/ …")
    if dry_run:
        for _, _, _, dest in copies[:10]:
            print(f"  -> {dest}")
        if len(copies) > 10:
            print(f"  … and {len(copies) - 10} more")
        return len(copies)

    # Chunk to stay within API limits.
    chunk = 50
    for i in range(0, len(copies), chunk):
        batch_bucket_files(
            bucket_id,
            copy=copies[i : i + chunk],
            delete=deletes[i : i + chunk],
        )
    print(f"Done — migrated {len(copies)} file(s) under {lang}/.")
    return len(copies)


def _load_local_conversation(run_dir: Path) -> dict[str, Any] | None:
    json_path = run_dir / "conversation.json"
    if not json_path.is_file():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def upload_local_output(
    local_root: Path,
    language: str,
    *,
    dry_run: bool = False,
) -> int:
    """Upload ``output/<run_id>/`` folders that match ``language`` to the bucket."""
    if not local_root.is_dir():
        print(f"Local output directory not found: {local_root}")
        return 0

    lang = normalize_language(language)
    storage = HuggingFaceStorage(create=False)
    uploaded = 0

    for run_dir in sorted(local_root.iterdir()):
        if not run_dir.is_dir():
            continue
        payload = _load_local_conversation(run_dir)
        if not payload:
            continue

        profile = payload.get("profile") or {}
        item_lang = str(profile.get("language", "")).lower()
        if item_lang != lang:
            continue

        corpus_id = int(payload["corpus_combination_id"])
        index = int(payload.get("index") or 1)
        metadata_path = run_dir / "metadata.txt"
        transcript_path = run_dir / "transcript.txt"
        metadata_text = (
            metadata_path.read_text(encoding="utf-8") if metadata_path.is_file() else None
        )
        transcript_text = (
            transcript_path.read_text(encoding="utf-8") if transcript_path.is_file() else None
        )

        dest = conversation_dest(lang, corpus_id, index, filename="conversation.json")
        print(f"Upload {run_dir.name} -> {dest}")
        if dry_run:
            uploaded += 1
            continue

        storage.save_conversation(
            corpus_id,
            index,
            payload,
            language=language,
            metadata_text=metadata_text,
            transcript_text=transcript_text or None,
        )
        uploaded += 1

    print(f"Uploaded {uploaded} local conversation(s) to {lang}/.")
    return uploaded


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate HF bucket / local output to per-language folder layout."
    )
    parser.add_argument(
        "--language",
        required=True,
        choices=("english", "hinglish", "hindi"),
        help="Target language folder name.",
    )
    parser.add_argument(
        "--from-bucket",
        action="store_true",
        help="Reorganize legacy files already in the HF bucket.",
    )
    parser.add_argument(
        "--from-local",
        action="store_true",
        help="Upload matching conversations from local output/.",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "output",
        help="Local output root (default: repo output/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without uploading.",
    )
    parser.add_argument(
        "--patch-legacy-models",
        action="store_true",
        help=(
            "Set generation/validation provider to krutrim on all conversations "
            "already in the language folder (updates JSON + metadata.txt)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    apply_to_environ()
    args = _parse_args(argv)

    if not args.from_bucket and not args.from_local and not args.patch_legacy_models:
        args.from_bucket = True
        args.from_local = True

    storage = HuggingFaceStorage(create=False)
    bucket_id = storage.bucket_id

    if args.from_bucket:
        migrate_bucket_layout(bucket_id, args.language, dry_run=args.dry_run)

    if args.from_local:
        upload_local_output(args.local_dir, args.language, dry_run=args.dry_run)

    if args.patch_legacy_models:
        models = legacy_krutrim_models()
        count = patch_hf_model_metadata(
            args.language,
            models,
            bucket_id=bucket_id,
            dry_run=args.dry_run,
        )
        print(
            f"Patched models on {count} conversation(s) under {args.language}/ "
            f"(generation=krutrim, validation=krutrim, model={models.generation_model})."
        )


if __name__ == "__main__":
    main()

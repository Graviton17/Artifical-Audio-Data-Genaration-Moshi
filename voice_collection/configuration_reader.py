"""Load runtime settings and secrets from ``config.json``.

All secrets (HuggingFace token) and environment-style settings for
this service live in ``voice_collection/config.json`` (gitignored -- copy
``config.json.example`` and fill in real values). Call any of the helpers
below; the file is read once and cached for the lifetime of the process.
Mirrors ``conversations_generator/configuration_reader.py``.

    from voice_collection import configuration_reader as config

    config.apply_to_environ()               # optional -- first get/require also loads
    token = config.require("HF_TOKEN")
    english_cfg = config.get_dataset_config("english")
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
_config: dict[str, Any] | None = None
# Untouched JSON (nested dicts / numbers / null preserved), unlike ``_config``
# which flattens scalars to strings for env-var-style access via get()/require().
_raw_config: dict[str, Any] | None = None

DEFAULT_TARGET_DURATION_SECONDS = 10.0
DEFAULT_MIN_ACCEPTABLE_DURATION_SECONDS = 5.0
DEFAULT_TARGET_SAMPLE_RATE = 16000
DEFAULT_STORAGE_PROVIDER = "huggingface"
DEFAULT_HF_BUCKET = "hf://buckets/inavlabs/voice_collection"


class ConfigurationError(RuntimeError):
    """Raised when ``config.json`` is missing, unreadable, or a required key is absent."""


def load_config(path: str | Path | None = None, *, force_reload: bool = False) -> dict[str, Any]:
    """Load and cache ``config.json``. Subsequent calls return the cached dict.

    Parameters
    ----------
    path :
        Override the default ``voice_collection/config.json`` location.
    force_reload :
        Re-read from disk even if a config is already cached.
    """
    global _config, _raw_config
    if _config is not None and not force_reload and path is None:
        return _config

    config_path = Path(path) if path is not None else _CONFIG_PATH
    if not config_path.is_file():
        example = config_path.with_name("config.json.example")
        raise ConfigurationError(
            f"Config file not found: {config_path}. "
            f"Copy {example.name} to config.json and fill in your HF token "
            "(config.json is gitignored and must not be committed)."
        )

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise ConfigurationError(f"Invalid JSON in {config_path}: {err}") from err

    if not isinstance(data, dict):
        raise ConfigurationError(f"Config root must be a JSON object, got {type(data).__name__}")

    _raw_config = data
    # Coerce scalar values to strings so callers can treat them like env vars.
    # Nested structures (e.g. "DATASETS") are skipped here -- read those via
    # get_raw()/get_dataset_config() instead.
    _config = {
        str(k): ("" if v is None else str(v))
        for k, v in data.items()
        if not isinstance(v, dict)
    }
    return _config


def get_raw(key: str, default: Any = None) -> Any:
    """Return ``key`` from config with its original JSON type (dict/number/etc.)."""
    load_config()
    assert _raw_config is not None
    value = _raw_config.get(key)
    return default if value is None else value


def get(key: str, default: str | None = None) -> str | None:
    """Return ``key`` from config, or ``default`` if missing / empty."""
    cfg = load_config()
    value = cfg.get(key)
    if value is None or value == "":
        return default
    return value


def require(key: str) -> str:
    """Return ``key`` from config, or raise :class:`ConfigurationError` if missing."""
    value = get(key)
    if value is None:
        raise ConfigurationError(f"Missing required config key {key!r} in {_CONFIG_PATH.name}.")
    return value


def apply_to_environ() -> dict[str, Any]:
    """Load config and push HF keys into ``os.environ`` (without overwriting).

    ``huggingface_hub`` reads credentials from the environment; this makes
    ``config.json`` the single source of truth. Explicit ``os.environ`` entries
    always win.
    """
    cfg = load_config()
    for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN"):
        value = cfg.get(key)
        if value:
            os.environ.setdefault(key, value)
    return cfg


# ---------------------------------------------------------------------- #
# Typed helpers for the specific settings this service needs.
# ---------------------------------------------------------------------- #
def get_hf_token() -> str | None:
    return get("HF_TOKEN")


def get_target_duration_seconds(default: float = DEFAULT_TARGET_DURATION_SECONDS) -> float:
    """Minimum clip length (seconds) a speaker's *best* instance should reach.

    Read from ``config.json``'s ``TARGET_DURATION_SECONDS``.
    """
    value = get("TARGET_DURATION_SECONDS", str(default))
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_min_acceptable_duration_seconds(
    default: float = DEFAULT_MIN_ACCEPTABLE_DURATION_SECONDS,
) -> float:
    """Absolute floor (seconds) below which a speaker is discarded entirely.

    Read from ``config.json``'s ``MIN_ACCEPTABLE_DURATION_SECONDS``.
    """
    value = get("MIN_ACCEPTABLE_DURATION_SECONDS", str(default))
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_target_sample_rate(default: int = DEFAULT_TARGET_SAMPLE_RATE) -> int:
    """Sample rate every exported ``audio.wav`` is resampled to."""
    value = get("TARGET_SAMPLE_RATE", str(default))
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def get_storage_provider(default: str = DEFAULT_STORAGE_PROVIDER) -> str:
    """``"huggingface"`` (default) or ``"local"`` (dry runs / offline copy)."""
    return (get("STORAGE_PROVIDER") or default).strip().lower()


def is_upload_enabled(default: bool = True) -> bool:
    value = get_raw("UPLOAD_ENABLED", default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"false", "0", "no"}


def get_local_output_dir(default: str = "./output/voice_collection") -> Path:
    """Local staging directory the pipeline writes ``{language}/{gender}/{speaker}`` into."""
    return Path(get("LOCAL_OUTPUT_DIR", default))


def get_max_speakers_per_language() -> int | None:
    """Optional cap on exported speakers per language (``null`` = no cap)."""
    value = get_raw("MAX_SPEAKERS_PER_LANGUAGE", None)
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_anonymize_speaker_names(default: bool = False) -> bool:
    """When true, speaker folders are renamed ``speaker_1``, ``speaker_2``, ... per gender."""
    value = get_raw("ANONYMIZE_SPEAKER_NAMES", default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def get_log_level(default: str = "INFO") -> str:
    return (get("LOG_LEVEL") or default).strip().upper()


def get_hf_bucket_config() -> dict[str, Any]:
    """HuggingFace Storage Bucket destination settings."""
    private = get_raw("HF_BUCKET_PRIVATE", True)
    if not isinstance(private, bool):
        private = str(private).strip().lower() not in {"false", "0", "no"}

    create_if_missing = get_raw("HF_BUCKET_CREATE", True)
    if not isinstance(create_if_missing, bool):
        create_if_missing = str(create_if_missing).strip().lower() not in {"false", "0", "no"}

    return {
        "bucket": get("HF_BUCKET", DEFAULT_HF_BUCKET),
        "private": private,
        "create_if_missing": create_if_missing,
    }


def get_dataset_config(language: str) -> dict[str, Any]:
    """Return the ``DATASETS.<language>`` block (repo id, split, config, ...)."""
    datasets_cfg = get_raw("DATASETS", {})
    if not isinstance(datasets_cfg, dict) or language not in datasets_cfg:
        raise ConfigurationError(
            f"config.json is missing DATASETS.{language}. "
            f"Copy config.json.example and adjust the '{language}' dataset block."
        )
    return dict(datasets_cfg[language])

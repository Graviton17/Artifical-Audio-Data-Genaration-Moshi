"""Load runtime settings and API keys from ``config.json``.

All secrets and environment-style settings live in
``conversations_generator/config.json``. Call :func:`load_config` (or any of the
helpers below) once at process start; values are cached for the lifetime of the
process. Prefer reading keys through this module instead of sharing a ``.env``.

    from conversations_generator.configuration_reader import get, require, load_config

    load_config()                       # optional — first get/require also loads
    key = require("SARVAM_API_KEY")
    env = get("ENV", "development")
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
_config: dict[str, Any] | None = None
# Untouched JSON (nested dicts / numbers preserved), unlike ``_config`` which
# flattens everything to strings for env-var-style access via ``get``/``require``.
_raw_config: dict[str, Any] | None = None

DEFAULT_TEMPERATURE = 0.3


class ConfigurationError(RuntimeError):
    """Raised when ``config.json`` is missing, unreadable, or a required key is absent."""


def load_config(path: str | Path | None = None, *, force_reload: bool = False) -> dict[str, Any]:
    """Load and cache ``config.json``. Subsequent calls return the cached dict.

    Parameters
    ----------
    path :
        Override the default ``conversations_generator/config.json`` location.
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
            f"Copy {example.name} to config.json and fill in your API keys "
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
    # Nested structures (e.g. "MODELS") are skipped here — read those via
    # get_model()/get_raw() instead, since env-var semantics don't apply to them.
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


def get_temperature(default: float = DEFAULT_TEMPERATURE) -> float:
    """Return the shared generation temperature from ``config.json`` (default 0.3)."""
    value = get_raw("TEMPERATURE", default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_model(provider: str, default: str) -> str:
    """Return the configured model name for ``provider`` (e.g. ``"gemini"``).

    Falls back to ``default`` (the provider class's own default) when
    ``config.json`` has no "MODELS" section or no entry for this provider.
    """
    models = get_raw("MODELS", {})
    if not isinstance(models, dict):
        return default
    value = models.get(provider)
    return str(value) if value else default


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
        raise ConfigurationError(
            f"Missing required config key {key!r} in {_CONFIG_PATH.name}."
        )
    return value


def apply_to_environ() -> dict[str, Any]:
    """Load config and push every key into ``os.environ`` (without overwriting).

    Third-party SDKs that only read environment variables (e.g. ``huggingface_hub``)
    pick up values this way. Explicit ``os.environ`` entries always win.
    """
    cfg = load_config()
    for key, value in cfg.items():
        if value:
            os.environ.setdefault(key, value)
    # Langfuse Python SDK historically reads LANGFUSE_HOST; mirror BASE_URL if set.
    base_url = cfg.get("LANGFUSE_BASE_URL") or cfg.get("LANGFUSE_HOST")
    if base_url:
        os.environ.setdefault("LANGFUSE_HOST", base_url)
        os.environ.setdefault("LANGFUSE_BASE_URL", base_url)
    return cfg

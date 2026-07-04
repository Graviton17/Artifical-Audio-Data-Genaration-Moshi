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
    global _config
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

    # Coerce everything to strings so callers can treat values like env vars.
    _config = {str(k): ("" if v is None else str(v)) for k, v in data.items()}
    return _config


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

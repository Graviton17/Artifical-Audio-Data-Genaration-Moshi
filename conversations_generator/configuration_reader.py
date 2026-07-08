"""Load runtime settings and API keys from ``config.json``.

All secrets and environment-style settings live in
``conversations_generator/config.json``. Call :func:`load_config` (or any of the
helpers below) once at process start; values are cached for the lifetime of the
process. Prefer reading keys through this module instead of sharing a ``.env``.

    from conversations_generator.configuration_reader import get, require, load_config

    load_config()                       # optional — first get/require also loads
    key = require("SARVAM_API_KEY")
    mode = get_mode()                   # "dev" or "prod"
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


def get_mode() -> str:
    """Return the run mode, normalized to ``"dev"`` or ``"prod"``.

    Reads ``config.json``'s ``MODE`` (values ``dev``/``prod`` or
    ``development``/``production`` are all accepted), defaulting to ``"dev"``.
    This single switch drives both prompt source (dev → local prompt files,
    prod → Langfuse) and storage (dev → local only, prod → upload to HuggingFace).
    """
    raw = get("MODE") or "dev"
    return "prod" if raw.strip().lower() in {"prod", "production"} else "dev"


def is_production() -> bool:
    """True when running in production mode (see :func:`get_mode`)."""
    return get_mode() == "prod"


def get_number_inclusion_percentage(default: float = 0.5) -> float:
    """Fraction (0.0–1.0) of conversations that should be number-rich.

    Read from ``config.json``'s "NUMBER_INCLUSION_PERCENTAGE". Each conversation
    independently draws this: on a hit it's generated with concrete numbers and
    their reasoning; otherwise it stays qualitative. Accepts either a fraction
    (``0.5``) or a percentage (``50``); values are clamped to [0, 1].
    """
    value = get_raw("NUMBER_INCLUSION_PERCENTAGE", default)
    try:
        pct = float(value)
    except (TypeError, ValueError):
        return default
    if pct > 1.0:  # tolerate "50" meaning 50%
        pct = pct / 100.0
    return max(0.0, min(1.0, pct))


def get_num_workers(default: int = 1) -> int:
    """Number of parallel worker threads for conversation generation.

    Read from ``config.json``'s ``NUM_WORKERS``. Each worker generates a separate
    conversation for the current instance concurrently, so a multi-hour instance
    finishes faster. Topic generation stays serialised internally so parallel
    workers never produce the same topic. Values below 1 are treated as 1
    (sequential), and non-integer/invalid values fall back to ``default``.
    """
    value = get_raw("NUM_WORKERS", default)
    try:
        workers = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, workers)


def get_max_concurrent_api_calls(default: int = 0) -> int:
    """Maximum number of outbound LLM API calls allowed in flight at once.

    Read from ``config.json``'s ``MAX_CONCURRENT_API_CALLS``. This is
    independent of ``NUM_WORKERS``: many worker threads can check
    instance/language budgets and plan work concurrently (cheap, no network
    call), but actual requests to the LLM provider are throttled to this many
    concurrent calls process-wide — protecting against provider-side rate
    limits when ``NUM_WORKERS`` is set very high (e.g. 1000). ``0`` (or
    unset/invalid) means unlimited — no throttling, the previous behaviour.
    """
    value = get_raw("MAX_CONCURRENT_API_CALLS", default)
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, n)


def get_run_languages() -> list[str] | None:
    """Languages to process on this run, in order, from ``config.json``.

    Reads the ``RUN_LANGUAGES`` key — a list like ``["english"]`` or
    ``["hindi", "english"]`` (a bare string ``"english"`` is also accepted).
    Names are lowercased and de-duplicated while preserving order. Returns
    ``None`` when the key is missing or empty, meaning "process every language"
    (the previous default behaviour).

    Because the checkpoint tracks progress per instance, listing a language that
    was already partly done on an earlier run simply resumes its unfinished
    instances — e.g. run ``["english"]`` first, then ``["hindi", "english"]``
    finishes the remaining English instances and does all of Hindi.
    """
    value = get_raw("RUN_LANGUAGES", None)
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return None
    seen: set[str] = set()
    langs: list[str] = []
    for item in value:
        name = str(item).strip().lower()
        if name and name not in seen:
            seen.add(name)
            langs.append(name)
    return langs or None


def get_agent_temperature(agent: str, default: float | None = None) -> float:
    """Return the configured sampling temperature for a specific agent.

    Reads ``config.json``'s "AGENT_TEMPERATURES" section (keys like ``"topic"``,
    ``"conversation"``, ``"formatter"``, ``"validator"``) so each pipeline stage
    can be tuned centrally. Falls back to ``default`` if given, else the global
    :func:`get_temperature`, when there's no per-agent entry.
    """
    temps = get_raw("AGENT_TEMPERATURES", {})
    value = temps.get(agent) if isinstance(temps, dict) else None
    if value is None:
        return default if default is not None else get_temperature()
    try:
        return float(value)
    except (TypeError, ValueError):
        return default if default is not None else get_temperature()


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
    # W&B keys use entity/project syntax in config.json; wandb_logger splits
    # them before touching os.environ / calling wandb.init().
    _skip_env_keys = frozenset({"WANDB_PROJECT", "WANDB_API_KEY", "WANDB_ENABLED"})
    for key, value in cfg.items():
        if value and key not in _skip_env_keys:
            os.environ.setdefault(key, value)
    # Langfuse Python SDK historically reads LANGFUSE_HOST; mirror BASE_URL if set.
    base_url = cfg.get("LANGFUSE_BASE_URL") or cfg.get("LANGFUSE_HOST")
    if base_url:
        os.environ.setdefault("LANGFUSE_HOST", base_url)
        os.environ.setdefault("LANGFUSE_BASE_URL", base_url)
    return cfg

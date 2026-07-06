"""System prompt resolution: mode-driven, Langfuse or local prompt files.

The source is chosen by ``MODE`` in config.json (see
:func:`configuration_reader.get_mode`):

* **dev** — use the local prompt copies in ``data/prompts/*.md`` directly. No
  network call, so development runs are fast and work offline.
* **prod** — fetch from Langfuse (the managed source of truth), retried a few
  times with backoff. If Langfuse is still unreachable, fall back to the local
  copies so a transient blip can't abort a production run.

Both paths return an object with a ``.compile(**vars)`` method, so callers don't
care which source a prompt came from.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from .configuration_reader import get as config_get
from .configuration_reader import is_production
from .logger import Logger

_client: Any | None = None

# Where the local fallback copies of each system prompt live, and how each
# Langfuse prompt name maps to its file (names don't all match 1:1).
_PROMPTS_DIR = Path(__file__).resolve().parent / "data" / "prompts"
_LOCAL_PROMPT_FILES: dict[str, str] = {
    "topic-generator-agent": "topic-generator-prompt.md",
    "conversation-generator-agent": "conversation-generator-agent.md",
    "conversation-formatter-agent": "conversation-formatter-agent.md",
    "conversation-content-validator-agent": "conversation-content-validator-agent.md",
    "conversation-format-validator-agent": "conversation-format-validator-agent.md",
}

# Retry policy for the (network) Langfuse fetch before falling back to local.
_MAX_FETCH_ATTEMPTS = 3
_RETRY_BACKOFF = 2.0


class PromptResolutionError(RuntimeError):
    """Raised when a system prompt can't be resolved from Langfuse OR locally."""


class LocalPrompt:
    """A prompt loaded from a local ``.md`` file.

    Mirrors the tiny slice of the Langfuse prompt object the agents use: a
    ``.prompt`` string and a ``.compile(**vars)`` that substitutes ``{{var}}``
    placeholders (same mustache-style syntax Langfuse uses).
    """

    def __init__(self, text: str) -> None:
        self.prompt = text

    def compile(self, **variables: Any) -> str:
        text = self.prompt
        for key, value in variables.items():
            # Match {{key}} with optional surrounding whitespace.
            text = re.sub(r"\{\{\s*" + re.escape(key) + r"\s*\}\}", str(value), text)
        return text


def _get_client() -> Any:
    """Lazily build a singleton Langfuse client. Raises if unavailable."""
    global _client
    if _client is not None:
        return _client

    public_key = config_get("LANGFUSE_PUBLIC_KEY")
    secret_key = config_get("LANGFUSE_SECRET_KEY")
    if not (public_key and secret_key):
        raise PromptResolutionError(
            "Langfuse is not configured. Set LANGFUSE_PUBLIC_KEY and "
            "LANGFUSE_SECRET_KEY in conversations_generator/config.json "
            "(and LANGFUSE_BASE_URL if self-hosted)."
        )

    try:
        from langfuse import Langfuse
    except ImportError as err:
        raise PromptResolutionError(
            "langfuse is not installed. Run `pip install langfuse`."
        ) from err

    host = config_get("LANGFUSE_BASE_URL") or config_get("LANGFUSE_HOST")
    kwargs: dict[str, Any] = {
        "public_key": public_key,
        "secret_key": secret_key,
    }
    if host:
        kwargs["host"] = host

    _client = Langfuse(**kwargs)
    return _client


def _load_local_prompt(prompt_name: str) -> LocalPrompt:
    """Load the local ``.md`` copy of ``prompt_name`` from ``data/prompts``."""
    filename = _LOCAL_PROMPT_FILES.get(prompt_name)
    if not filename:
        raise PromptResolutionError(
            f"No local fallback prompt is registered for {prompt_name!r}."
        )
    path = _PROMPTS_DIR / filename
    if not path.is_file():
        raise PromptResolutionError(
            f"Local fallback prompt file not found: {path}."
        )
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise PromptResolutionError(f"Local fallback prompt {path} is empty.")
    return LocalPrompt(text)


def _fetch_from_langfuse(prompt_name: str) -> Any:
    """Fetch a Langfuse text prompt, retrying transient failures with backoff."""
    client = _get_client()  # config/install errors propagate (no point retrying)
    last_err: Exception | None = None
    for attempt in range(1, _MAX_FETCH_ATTEMPTS + 1):
        try:
            prompt = client.get_prompt(prompt_name)
        except Exception as err:  # noqa: BLE001 - normalized/retried below
            last_err = err
            if attempt < _MAX_FETCH_ATTEMPTS:
                Logger.warning(
                    f"Langfuse prompt fetch for {prompt_name!r} failed "
                    f"(attempt {attempt}/{_MAX_FETCH_ATTEMPTS}): {err}. Retrying..."
                )
                time.sleep(_RETRY_BACKOFF ** (attempt - 1))
            continue
        if not isinstance(prompt.prompt, str):
            raise PromptResolutionError(
                f"Langfuse prompt {prompt_name!r} is not a text prompt."
            )
        return prompt
    raise PromptResolutionError(
        f"Could not fetch Langfuse prompt {prompt_name!r} after "
        f"{_MAX_FETCH_ATTEMPTS} attempts: {last_err}"
    ) from last_err


def resolve_system_prompt(prompt_name: str) -> Any:
    """Return the system prompt named ``prompt_name``, per the run ``MODE``.

    * dev  → load the local ``data/prompts`` copy directly (no network).
    * prod → fetch from Langfuse (with retries); if it's unreachable, fall back
      to the local copy so a transient blip can't abort the run.

    Raises :class:`PromptResolutionError` only if the required source(s) all fail.
    """
    if not prompt_name:
        raise PromptResolutionError("prompt_name must be set to resolve a system prompt.")

    # Development: local prompts only — fast, offline, no Langfuse dependency.
    if not is_production():
        return _load_local_prompt(prompt_name)

    # Production: Langfuse is the source of truth, with a local safety net.
    try:
        return _fetch_from_langfuse(prompt_name)
    except PromptResolutionError as langfuse_err:
        try:
            local = _load_local_prompt(prompt_name)
        except PromptResolutionError as local_err:
            raise PromptResolutionError(
                f"Could not resolve prompt {prompt_name!r} from Langfuse "
                f"({langfuse_err}) or locally ({local_err})."
            ) from langfuse_err
        Logger.warning(
            f"Using LOCAL fallback prompt for {prompt_name!r} "
            f"(Langfuse unavailable: {langfuse_err})."
        )
        return local

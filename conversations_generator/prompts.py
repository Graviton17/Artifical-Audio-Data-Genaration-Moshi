"""Langfuse-backed system prompt resolution."""

from __future__ import annotations

from typing import Any

from .configuration_reader import get as config_get

_client: Any | None = None


class PromptResolutionError(RuntimeError):
    """Raised when a system prompt cannot be fetched from Langfuse."""


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


def resolve_system_prompt(prompt_name: str) -> Any:
    """Fetch the Langfuse-managed text prompt named ``prompt_name``.

    Raises :class:`PromptResolutionError` if Langfuse isn't installed, isn't
    configured, or the prompt can't be fetched.
    """
    if not prompt_name:
        raise PromptResolutionError("prompt_name must be set to resolve a system prompt.")

    client = _get_client()
    try:
        prompt = client.get_prompt(prompt_name)
    except Exception as err:  # noqa: BLE001 - normalized below
        raise PromptResolutionError(
            f"Could not fetch Langfuse prompt {prompt_name!r}: {err}"
        ) from err

    if not isinstance(prompt.prompt, str):
        raise PromptResolutionError(
            f"Langfuse prompt {prompt_name!r} is not a text prompt."
        )
    return prompt

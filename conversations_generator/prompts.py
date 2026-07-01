"""Langfuse-backed system prompt resolution."""

from __future__ import annotations

import os
from typing import Any

_client: Any | None = None


class PromptResolutionError(RuntimeError):
    """Raised when a system prompt cannot be fetched from Langfuse."""


def _get_client() -> Any:
    """Lazily build a singleton Langfuse client. Raises if unavailable."""
    global _client
    if _client is not None:
        return _client

    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        raise PromptResolutionError(
            "Langfuse is not configured. Set LANGFUSE_PUBLIC_KEY and "
            "LANGFUSE_SECRET_KEY (and LANGFUSE_HOST if self-hosted)."
        )

    try:
        from langfuse import Langfuse
    except ImportError as err:
        raise PromptResolutionError(
            "langfuse is not installed. Run `pip install langfuse`."
        ) from err

    _client = Langfuse()
    return _client


def resolve_system_prompt(prompt_name: str) -> str:
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
    return prompt.prompt

"""Runtime tunables for the conversation-generation pipeline.

Small knobs that steer *how* conversations are generated — per-stage sampling
temperature, how often a conversation should be number-rich, and the dev/prod
run mode. Sensible defaults live here as code; each can be overridden by an
environment variable (loaded from ``.env`` by
:func:`conversations_generator.loaders.load_env`) so runs are tunable without
editing code. This keeps configuration in one place instead of scattered
hard-coded literals inside the agents.

    from conversations_generator import settings

    temp = settings.get_agent_temperature("topic")      # 0.6 by default
    if random.random() < settings.get_number_inclusion_percentage(): ...
    if settings.is_production(): ...
"""

from __future__ import annotations

import os

# Default sampling temperature per pipeline stage. Topic generation is the only
# creative stage (higher temperature = more varied topics); parsing, judging and
# editing want low-variance, near-deterministic output. Override an individual
# stage with the env var ``AGENT_TEMPERATURE_<STAGE>`` (e.g.
# ``AGENT_TEMPERATURE_TOPIC=0.8``).
AGENT_TEMPERATURES: dict[str, float] = {
    "topic": 0.6,
    "conversation": 0.3,
    "formatter": 0.3,
    "validator": 0.3,
    "editor": 0.3,
}

DEFAULT_TEMPERATURE = 0.3
DEFAULT_NUMBER_INCLUSION_PERCENTAGE = 0.65


def get_agent_temperature(agent: str, default: float | None = None) -> float:
    """Return the sampling temperature for a pipeline stage.

    Looks up ``AGENT_TEMPERATURES[agent]``, letting the env var
    ``AGENT_TEMPERATURE_<AGENT>`` override it. Falls back to ``default`` if
    given, else :data:`DEFAULT_TEMPERATURE`, when the stage is unknown.
    """
    env_value = os.getenv(f"AGENT_TEMPERATURE_{agent.upper()}")
    if env_value is not None:
        try:
            return float(env_value)
        except ValueError:
            pass
    if agent in AGENT_TEMPERATURES:
        return AGENT_TEMPERATURES[agent]
    return default if default is not None else DEFAULT_TEMPERATURE


def get_number_inclusion_percentage(
    default: float = DEFAULT_NUMBER_INCLUSION_PERCENTAGE,
) -> float:
    """Fraction (0.0–1.0) of conversations that should be number-rich.

    Read from the env var ``NUMBER_INCLUSION_PERCENTAGE``. Each conversation
    independently draws against this: on a hit it is generated with concrete
    numbers and their reasoning, otherwise it stays qualitative. Accepts either
    a fraction (``0.65``) or a percentage (``65``); the result is clamped to
    ``[0, 1]``.
    """
    raw = os.getenv("NUMBER_INCLUSION_PERCENTAGE")
    if raw is None:
        return default
    try:
        pct = float(raw)
    except ValueError:
        return default
    if pct > 1.0:  # tolerate "65" meaning 65%
        pct = pct / 100.0
    return max(0.0, min(1.0, pct))


def get_mode() -> str:
    """Return the run mode, normalized to ``"dev"`` or ``"prod"``.

    Reads the env var ``ENV`` (``production``/``prod`` → prod; anything else,
    including unset, → dev). This is the single switch the runner uses to decide
    between a single dev conversation and a full production sweep.
    """
    raw = (os.getenv("ENV") or "dev").strip().lower()
    return "prod" if raw in {"prod", "production"} else "dev"


def is_production() -> bool:
    """True when running in production mode (see :func:`get_mode`)."""
    return get_mode() == "prod"

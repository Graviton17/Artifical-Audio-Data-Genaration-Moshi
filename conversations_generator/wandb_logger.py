"""Weights & Biases run tracking for the conversation-generation pipeline.

Emits token-usage (input/output/cache, per model) and pipeline progress
(conversations accepted/failed, instances completed) metrics to W&B so a run
is visible live on the W&B dashboard instead of only in local metadata.json /
terminal logs.

One W&B run is started per process (see :func:`init_run`, called once from
``runner.main``) and every worker thread logs into it — metrics are
namespaced ``tokens/<model>/<field>`` so W&B's line charts naturally plot one
line per model, plus ``tokens/total/*`` for the run-wide numbers.
``progress/*`` tracks how many conversations/instances this run has worked
through.

Best-effort throughout: any W&B failure (missing package, bad key, network
error) is logged and swallowed so a dashboard problem can never abort a
generation run.

    from conversations_generator import wandb_logger

    wandb_logger.init_run({"mode": "prod", "generation_model": "krutrim"})
    ...
    wandb_logger.log_token_usage(TOKEN_USAGE.as_dict())
    wandb_logger.log_progress()
    ...
    wandb_logger.finish_run()
"""

from __future__ import annotations

import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from .configuration_reader import get as config_get
from .configuration_reader import get_raw
from .logger import Logger

try:
    import wandb
except ImportError:  # pragma: no cover - exercised only without the dep
    wandb = None

# Default project (entity/project form) — used when config.json has no
# "WANDB_PROJECT" entry.
DEFAULT_WANDB_PROJECT = "ml-team-inavlabs/Kupe-FDX-Datagen"

# Fallback API key so the pipeline emits metrics out of the box even before
# config.json is filled in. Override via config.json's "WANDB_API_KEY" to
# point at a different account.
_DEFAULT_WANDB_API_KEY = "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"


class RunStats:
    """Thread-safe counters for conversations/instances processed this run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.conversations_accepted = 0
        self.conversations_failed = 0
        self.instances_completed = 0
        self.instances_total = 0

    def set_total_instances(self, n: int) -> None:
        with self._lock:
            self.instances_total = n

    def record_conversation(self, accepted: bool) -> None:
        with self._lock:
            if accepted:
                self.conversations_accepted += 1
            else:
                self.conversations_failed += 1

    def record_instance_complete(self) -> None:
        with self._lock:
            self.instances_completed += 1

    def as_dict(self) -> dict[str, int]:
        with self._lock:
            return {
                "conversations_accepted": self.conversations_accepted,
                "conversations_failed": self.conversations_failed,
                "conversations_total": self.conversations_accepted + self.conversations_failed,
                "instances_completed": self.instances_completed,
                "instances_total": self.instances_total,
            }


# Single shared counter for the whole process, mirroring TOKEN_USAGE in base_llm.
RUN_STATS = RunStats()

_lock = threading.Lock()
_run: Any = None  # active wandb Run, or None when disabled/unavailable
_step = 0


def _wandb_enabled() -> bool:
    raw = get_raw("WANDB_ENABLED", True)
    if isinstance(raw, str):
        return raw.strip().lower() not in {"0", "false", "no", "off"}
    return bool(raw)


def _split_project(project: str) -> tuple[str | None, str]:
    """Split an ``entity/project`` string; a bare project name has no entity."""
    if "/" in project:
        entity, _, name = project.partition("/")
        return entity or None, name
    return None, project


def is_active() -> bool:
    """Whether a W&B run is currently active (initialized and not finished)."""
    return _run is not None


def init_run(config: dict[str, Any] | None = None) -> None:
    """Start a new W&B run for this process.

    Reads ``WANDB_API_KEY`` / ``WANDB_PROJECT`` from ``config.json``, falling
    back to the built-in project/key so a run is emitted with zero setup.
    Best-effort: any failure (missing package, bad credentials, offline) is
    logged and leaves W&B tracking disabled for the rest of the run rather
    than raising.
    """
    global _run, _step
    if wandb is None:
        Logger.warning(
            "wandb is not installed — skipping W&B tracking. "
            "Run `pip install wandb` (or `conda install -c conda-forge wandb`) to enable it."
        )
        return
    if not _wandb_enabled():
        Logger.info("W&B tracking disabled via config.json (WANDB_ENABLED=false).")
        return

    api_key = config_get("WANDB_API_KEY") or _DEFAULT_WANDB_API_KEY
    project_setting = config_get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT
    entity, project = _split_project(project_setting)

    # wandb.init() reads WANDB_* from the environment. apply_to_environ() may
    # have already set WANDB_PROJECT to the raw "entity/project" form, which
    # wandb rejects — override with the split values before init.
    if api_key:
        os.environ.setdefault("WANDB_API_KEY", api_key)
    os.environ["WANDB_PROJECT"] = project
    if entity:
        os.environ["WANDB_ENTITY"] = entity

    try:
        run_name = f"run-{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}-{uuid.uuid4().hex[:6]}"
        with _lock:
            _run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config=config or {},
                reinit=True,
            )
            _step = 0
        Logger.success(f"W&B run started: {project_setting} / {run_name}", bold=True)
    except Exception as err:  # noqa: BLE001 - a dashboard problem must never break the pipeline
        Logger.warning(f"Failed to start W&B run: {err}")
        _run = None


def _log(data: dict[str, Any]) -> None:
    global _step
    if _run is None or wandb is None:
        return
    try:
        with _lock:
            _step += 1
            wandb.log(data, step=_step)
    except Exception as err:  # noqa: BLE001
        Logger.warning(f"W&B log failed: {err}")


def log_token_usage(summary: dict[str, Any]) -> None:
    """Log total + per-model token usage (input/output/cache) as one W&B step.

    ``summary`` is the dict returned by ``TokenUsageTracker.as_dict()``:
    ``{"models": {model: {...}}, "total": {...}}``. Keys are namespaced
    ``tokens/<model>/<field>`` (input/output/cache/total/calls) so W&B's
    charts plot one line per model when compared over time, alongside
    ``tokens/total/*`` for the run-wide numbers. Progress counters are logged
    on the same step so token and conversation charts share an x-axis.
    """
    if _run is None:
        return
    data: dict[str, Any] = {}
    total = summary.get("total", {})
    data["tokens/total/input"] = total.get("input_tokens", 0)
    data["tokens/total/output"] = total.get("output_tokens", 0)
    data["tokens/total/cache"] = total.get("cache_tokens", 0)
    data["tokens/total/all"] = total.get("total_tokens", 0)
    data["tokens/total/calls"] = total.get("calls", 0)

    for model, usage in summary.get("models", {}).items():
        safe_model = str(model).replace("/", "_").replace(" ", "_")
        data[f"tokens/{safe_model}/input"] = usage.get("input_tokens", 0)
        data[f"tokens/{safe_model}/output"] = usage.get("output_tokens", 0)
        data[f"tokens/{safe_model}/cache"] = usage.get("cache_tokens", 0)
        data[f"tokens/{safe_model}/total"] = usage.get("total_tokens", 0)
        data[f"tokens/{safe_model}/calls"] = usage.get("calls", 0)

    data.update({f"progress/{k}": v for k, v in RUN_STATS.as_dict().items()})
    _log(data)


def log_progress(**extra: Any) -> None:
    """Log conversation/instance progress counters, plus any extra scalars."""
    if _run is None:
        return
    data = {f"progress/{k}": v for k, v in RUN_STATS.as_dict().items()}
    data.update(extra)
    _log(data)


def finish_run() -> None:
    """Finalize the active W&B run, stamping final totals into its summary."""
    global _run
    if _run is None or wandb is None:
        return
    try:
        for key, value in RUN_STATS.as_dict().items():
            wandb.run.summary[key] = value
        wandb.finish()
        Logger.success("W&B run finished.")
    except Exception as err:  # noqa: BLE001
        Logger.warning(f"Failed to finish W&B run cleanly: {err}")
    finally:
        _run = None

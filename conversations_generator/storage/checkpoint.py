"""Checkpoint model tracking per-instance generation progress.

A single :class:`Checkpoint` is persisted at the root of the storage bucket
(``checkpoint.json``). It records, per corpus instance, how many seconds of
conversation have been accepted so far. When a machine dies mid-run, any other
machine can download the checkpoint and resume each instance exactly where the
last one left off instead of regenerating from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class InstanceProgress:
    """How much conversation has been generated for one corpus instance."""

    corpus_combination_id: int
    target_sec: float
    generated_sec: float = 0.0
    conversation_count: int = 0

    @property
    def completed(self) -> bool:
        """True once the accepted duration has reached the instance target."""
        return self.generated_sec >= self.target_sec

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_combination_id": self.corpus_combination_id,
            "target_sec": self.target_sec,
            "generated_sec": self.generated_sec,
            "conversation_count": self.conversation_count,
            "completed": self.completed,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "InstanceProgress":
        return cls(
            corpus_combination_id=raw["corpus_combination_id"],
            target_sec=raw.get("target_sec", 0.0),
            generated_sec=raw.get("generated_sec", 0.0),
            conversation_count=raw.get("conversation_count", 0),
        )


@dataclass
class Checkpoint:
    """Root-level resume state: one :class:`InstanceProgress` per instance.

    Keyed by the string form of ``corpus_combination_id`` so it round-trips
    cleanly through JSON (object keys must be strings).
    """

    instances: dict[str, InstanceProgress] = field(default_factory=dict)
    updated_at: str | None = None

    # ------------------------------------------------------------------ #
    # Progress access / mutation
    # ------------------------------------------------------------------ #
    def get(self, corpus_combination_id: int, target_sec: float) -> InstanceProgress:
        """Return the progress record for an instance, creating it if absent."""
        key = str(corpus_combination_id)
        if key not in self.instances:
            self.instances[key] = InstanceProgress(
                corpus_combination_id=corpus_combination_id,
                target_sec=target_sec,
            )
        return self.instances[key]

    def record(self, progress: InstanceProgress, added_sec: float) -> None:
        """Add one accepted conversation's duration to an instance and stamp time."""
        progress.generated_sec += added_sec
        progress.conversation_count += 1
        self.instances[str(progress.corpus_combination_id)] = progress
        self.updated_at = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {
            "updated_at": self.updated_at,
            "instances": {k: v.to_dict() for k, v in self.instances.items()},
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Checkpoint":
        instances = {
            key: InstanceProgress.from_dict(value)
            for key, value in (raw.get("instances") or {}).items()
        }
        return cls(instances=instances, updated_at=raw.get("updated_at"))

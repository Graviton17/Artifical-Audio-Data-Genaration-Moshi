"""Registry of instances abandoned after too many consecutive failures.

A single :class:`SkippedRegistry` is persisted at the root of the storage bucket
(``skipped.json``), alongside ``checkpoint.json``. When an instance fails
validation ``MAX_CONSECUTIVE_FAILURES`` times in a row, production gives up on it
and records it here (production only — dev runs keep nothing).

Two purposes:

* an **audit trail** of which corpus instances could never be generated (and how
  far they got before being abandoned), and
* a **resume guard**: a machine picking the run back up skips any instance
  already in the registry instead of burning another 10 attempts on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class SkippedInstance:
    """One corpus instance abandoned after repeated validation failures."""

    corpus_combination_id: int
    consecutive_failures: int
    reason: str = ""
    #: Progress reached before the instance was abandoned (may be partial).
    generated_sec: float = 0.0
    target_sec: float = 0.0
    conversation_count: int = 0
    #: A few profile fields for readability when auditing the file by hand.
    language: str | None = None
    gender_pair: str | None = None
    skipped_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_combination_id": self.corpus_combination_id,
            "consecutive_failures": self.consecutive_failures,
            "reason": self.reason,
            "generated_sec": self.generated_sec,
            "target_sec": self.target_sec,
            "conversation_count": self.conversation_count,
            "language": self.language,
            "gender_pair": self.gender_pair,
            "skipped_at": self.skipped_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SkippedInstance":
        return cls(
            corpus_combination_id=raw["corpus_combination_id"],
            consecutive_failures=raw.get("consecutive_failures", 0),
            reason=raw.get("reason", ""),
            generated_sec=raw.get("generated_sec", 0.0),
            target_sec=raw.get("target_sec", 0.0),
            conversation_count=raw.get("conversation_count", 0),
            language=raw.get("language"),
            gender_pair=raw.get("gender_pair"),
            skipped_at=raw.get("skipped_at"),
        )


@dataclass
class SkippedRegistry:
    """Root-level ``skipped.json``: one :class:`SkippedInstance` per abandoned instance.

    Keyed by the string form of ``corpus_combination_id`` so it round-trips
    cleanly through JSON (object keys must be strings).
    """

    instances: dict[str, SkippedInstance] = field(default_factory=dict)
    updated_at: str | None = None

    def __contains__(self, corpus_combination_id: int) -> bool:
        """True if this instance has already been abandoned."""
        return str(corpus_combination_id) in self.instances

    def add(self, skipped: SkippedInstance) -> None:
        """Record (or overwrite) an abandoned instance and stamp the time."""
        skipped.skipped_at = datetime.now(timezone.utc).isoformat()
        self.instances[str(skipped.corpus_combination_id)] = skipped
        self.updated_at = skipped.skipped_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "updated_at": self.updated_at,
            "instances": {k: v.to_dict() for k, v in self.instances.items()},
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SkippedRegistry":
        instances = {
            key: SkippedInstance.from_dict(value)
            for key, value in (raw.get("instances") or {}).items()
        }
        return cls(instances=instances, updated_at=raw.get("updated_at"))

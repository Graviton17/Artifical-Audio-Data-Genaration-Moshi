"""Pipeline orchestrator (Composite + Chain of Responsibility).

A :class:`Pipeline` is just an ordered list of :class:`Stage` objects. Each stage
receives the context the previous one produced; stages no-op on a dropped context
unless they opt in via ``runs_on_dropped``. Same shape as Data-Processing-Moshi.
"""

from __future__ import annotations

from .models import GenContext
from .stages.base import Stage
from .utils.logging import get_logger

log = get_logger("pipeline")


class Pipeline:
    def __init__(self, stages: list[Stage]):
        self.stages = stages

    def run(self, ctx: GenContext) -> GenContext:
        for stage in self.stages:
            ctx = stage(ctx)
        return ctx

    def __repr__(self) -> str:
        return "Pipeline(" + " -> ".join(s.name for s in self.stages) + ")"

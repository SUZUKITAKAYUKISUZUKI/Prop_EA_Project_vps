"""Roadmap generation for APM v1."""
from __future__ import annotations

from src.apm.executive_scheduler import ExecutiveScheduler


class RoadmapEngine:
    def __init__(self, *, scheduler: ExecutiveScheduler | None = None) -> None:
        self._scheduler = scheduler or ExecutiveScheduler()

    def evaluate(self, actions: list[dict]) -> list[dict]:
        return self._scheduler.build(actions)

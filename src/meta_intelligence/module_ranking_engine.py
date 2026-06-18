"""Module trust ranking."""
from __future__ import annotations

from typing import Any


class ModuleRankingEngine:
    def evaluate(self, trust_scores: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        ranked = sorted(
            trust_scores.values(),
            key=lambda item: float(item.get("trust_score") or 0),
            reverse=True,
        )
        rankings: list[dict[str, Any]] = []
        for idx, item in enumerate(ranked, start=1):
            rankings.append(
                {
                    "rank": idx,
                    "module": item.get("module"),
                    "trust_score": item.get("trust_score"),
                    "category": item.get("category"),
                }
            )
        return rankings

    def strongest_weakest(self, rankings: list[dict[str, Any]]) -> dict[str, Any]:
        if not rankings:
            return {"strongest_module": None, "weakest_module": None}
        return {
            "strongest_module": rankings[0],
            "weakest_module": rankings[-1],
        }

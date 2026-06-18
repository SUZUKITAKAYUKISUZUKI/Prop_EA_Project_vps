"""Portfolio Fit reporting for SLM v2."""
from __future__ import annotations

from typing import Any

import numpy as np


class PortfolioFitReport:
    def build(
        self,
        fit_by_strategy: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        if not fit_by_strategy:
            return {
                "highest_fit_strategy": None,
                "lowest_fit_strategy": None,
                "average_fit": 0.0,
                "fit_distribution": {},
                "ranking": [],
            }

        ranking = sorted(
            [
                {
                    "strategy": sid,
                    "portfolio_fit_score": float(data.get("portfolio_fit_score") or 0.0),
                    "correlation": data.get("correlation"),
                    "recovery_impact": data.get("recovery_impact"),
                    "challenge_impact": data.get("challenge_impact"),
                    "health_impact": data.get("health_impact"),
                    "components": data.get("components") or {},
                }
                for sid, data in fit_by_strategy.items()
            ],
            key=lambda r: r["portfolio_fit_score"],
            reverse=True,
        )
        scores = [r["portfolio_fit_score"] for r in ranking]
        highest = ranking[0]
        lowest = ranking[-1]
        distribution = {
            "90+": sum(1 for s in scores if s >= 90),
            "75-89": sum(1 for s in scores if 75 <= s < 90),
            "60-74": sum(1 for s in scores if 60 <= s < 75),
            "40-59": sum(1 for s in scores if 40 <= s < 60),
            "<40": sum(1 for s in scores if s < 40),
        }
        return {
            "highest_fit_strategy": highest["strategy"],
            "lowest_fit_strategy": lowest["strategy"],
            "average_fit": round(float(np.mean(scores)), 1),
            "fit_distribution": distribution,
            "ranking": ranking,
        }

    def scatter_points(self, fit_by_strategy: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        points: list[dict[str, Any]] = []
        for sid, data in fit_by_strategy.items():
            points.append(
                {
                    "strategy": sid,
                    "correlation": float(data.get("correlation") or 0.0),
                    "portfolio_fit_score": float(data.get("portfolio_fit_score") or 0.0),
                }
            )
        return sorted(points, key=lambda p: p["portfolio_fit_score"], reverse=True)

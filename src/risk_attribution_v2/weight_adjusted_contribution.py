"""Weight-adjusted portfolio contribution for PRAE v2."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.risk_attribution_v2.common import normalize_contribution_pct, prepare_trades, strategy_metrics


class WeightAdjustedContributionEngine:
    def analyze(
        self,
        trades: pd.DataFrame,
        *,
        allocation_weights: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        work = prepare_trades(trades)
        if work.empty:
            return []

        weights = allocation_weights or {}
        raw_scores: dict[str, float] = {}
        rows: list[dict[str, Any]] = []

        for code, part in work.groupby("strategy_code", sort=False):
            metrics = strategy_metrics(part["R"])
            weight = float(weights.get(str(code), 0.2))
            dd_denom = max(metrics["max_dd"], 0.1)
            pf = metrics["pf"] if metrics["pf"] < 900 else 3.0
            score = weight * metrics["total_r"] * pf / dd_denom
            raw_scores[str(code)] = max(score, 0.0)
            rows.append(
                {
                    "strategy": str(code),
                    "weight": round(weight * 100.0, 1),
                    **metrics,
                    "raw_score": round(score, 4),
                }
            )

        contrib = normalize_contribution_pct(raw_scores)
        for row in rows:
            row["contribution_pct"] = contrib.get(row["strategy"], 0.0)
        return sorted(rows, key=lambda r: r["contribution_pct"], reverse=True)

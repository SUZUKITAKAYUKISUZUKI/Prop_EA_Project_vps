"""Strategy-level risk contribution for PRAE v2."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.risk_attribution_v2.common import (
    compute_risk_score,
    prepare_trades,
    strategy_metrics,
)


class StrategyRiskEngine:
    def analyze(self, trades: pd.DataFrame) -> list[dict[str, Any]]:
        work = prepare_trades(trades)
        if work.empty:
            return []

        rows: list[dict[str, Any]] = []
        dd_vals: list[float] = []
        ulcer_vals: list[float] = []
        streak_vals: list[int] = []

        for code, part in work.groupby("strategy_code", sort=False):
            metrics = strategy_metrics(part["R"])
            dd_vals.append(metrics["max_dd"])
            ulcer_vals.append(metrics["ulcer"])
            streak_vals.append(metrics["max_losing_streak"])
            rows.append({"strategy": str(code), **metrics})

        dd_scale = max(dd_vals) if dd_vals else 1.0
        ulcer_scale = max(ulcer_vals) if ulcer_vals else 1.0
        streak_scale = max(streak_vals) if streak_vals else 1.0

        for row in rows:
            row["risk_score"] = compute_risk_score(
                max_dd=row["max_dd"],
                ulcer=row["ulcer"],
                max_losing_streak=row["max_losing_streak"],
                dd_scale=dd_scale,
                ulcer_scale=ulcer_scale,
                streak_scale=streak_scale,
            )
        return sorted(rows, key=lambda r: r["risk_score"], reverse=True)

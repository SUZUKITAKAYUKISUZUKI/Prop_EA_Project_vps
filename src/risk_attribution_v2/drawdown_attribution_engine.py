"""Drawdown period attribution for PRAE v2."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.risk_attribution_v2.common import normalize_contribution_pct, prepare_trades


class DrawdownAttributionEngine:
    def analyze(self, trades: pd.DataFrame) -> dict[str, Any]:
        work = prepare_trades(trades)
        if work.empty:
            return {"dd_period": None, "contributors": [], "strategy_contribution": {}}

        r = work["R"].astype(float).to_numpy()
        eq = np.cumsum(r)
        peak = np.maximum.accumulate(eq)
        dd = peak - eq
        trough_i = int(np.argmax(dd))
        peak_i = int(np.argmax(eq[: trough_i + 1])) if trough_i >= 0 else 0
        if trough_i <= peak_i:
            return {"dd_period": None, "contributors": [], "strategy_contribution": {}}

        window = work.iloc[peak_i : trough_i + 1]
        losses = window.loc[window["R"] < 0].copy()
        grouped = losses.groupby("strategy_code")["R"].sum().sort_values()
        loss_map = {str(k): abs(float(v)) for k, v in grouped.items()}
        contrib = normalize_contribution_pct(loss_map)

        contributors = [
            {
                "strategy": strategy,
                "loss_r": round(float(losses.loc[losses["strategy_code"] == strategy, "R"].sum()), 2),
                "contribution_pct": contrib.get(strategy, 0.0),
            }
            for strategy in grouped.index.astype(str)
        ]

        start_ts = work.iloc[peak_i]["timestamp"]
        end_ts = work.iloc[trough_i]["timestamp"]
        return {
            "dd_period": {
                "start": start_ts.strftime("%Y-%m-%d"),
                "end": end_ts.strftime("%Y-%m-%d"),
                "max_dd": round(float(-dd[trough_i]), 2),
            },
            "contributors": contributors,
            "strategy_contribution": contrib,
        }

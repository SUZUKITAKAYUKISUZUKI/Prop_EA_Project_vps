"""Funded-phase scenario simulator."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.adaptive_allocation.allocation_simulator import AllocationSimulator
from src.digital_twin._metrics_common import compute_scenario_metrics


class FundedSimulator:
    def __init__(self) -> None:
        self._allocation = AllocationSimulator()

    def run(
        self,
        trades: pd.DataFrame,
        weights: dict[str, float],
        *,
        health_score: float = 50.0,
    ) -> dict[str, Any]:
        metrics = compute_scenario_metrics(trades, weights, health_score=health_score)
        alloc = self._allocation.evaluate(trades, weights, account_state="funded")
        metrics["pass_rate"] = alloc["pass_rate"]
        metrics["avg_pass_days"] = alloc["pass_days"]
        metrics["prob_recovery"] = max(0.0, min(100.0, metrics["recovery_factor"] * 10.0))
        metrics["prob_ruin"] = min(100.0, metrics["max_dd_pct"])
        return metrics

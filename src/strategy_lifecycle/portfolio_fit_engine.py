"""Portfolio Fit Score engine for SLM v2."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from portfolio_analyzer import build_monthly_r_matrix, compute_correlation_matrix
from src.database.profile_migrations import DASHBOARD_STRATEGY_CODES, SETUP_TYPE_BY_STRATEGY_CODE


class PortfolioFitEngine:
    """Evaluate whether a strategy fits portfolio needs, not just standalone quality."""

    WEIGHTS = {
        "diversification_benefit": 0.30,
        "recovery_contribution": 0.25,
        "challenge_contribution": 0.20,
        "state_stability_contribution": 0.15,
        "dd_reduction_contribution": 0.10,
    }

    def build_correlation_matrix(self, trades: pd.DataFrame | None) -> dict[str, dict[str, float]]:
        if trades is None or trades.empty:
            return {code: {code: 1.0 for code in DASHBOARD_STRATEGY_CODES} for code in DASHBOARD_STRATEGY_CODES}

        work = trades.copy()
        if "strategy" not in work.columns:
            work["strategy"] = work.get("strategy_code", work.get("setup_type"))
        if "R" not in work.columns:
            work["R"] = work.get("profit_r", 0.0)
        if "timestamp" not in work.columns:
            return {code: {code: 1.0 for code in DASHBOARD_STRATEGY_CODES} for code in DASHBOARD_STRATEGY_CODES}

        by_strategy: dict[str, pd.DataFrame] = {}
        for code in DASHBOARD_STRATEGY_CODES:
            setup = SETUP_TYPE_BY_STRATEGY_CODE.get(code, code)
            subset = work[
                work["strategy"].astype(str).isin({code, setup})
                | work.get("setup_type", pd.Series(dtype=str)).astype(str).isin({code, setup})
            ]
            if not subset.empty:
                by_strategy[code] = subset

        if len(by_strategy) < 2:
            return {code: {code: 1.0 for code in DASHBOARD_STRATEGY_CODES} for code in DASHBOARD_STRATEGY_CODES}

        monthly = build_monthly_r_matrix(by_strategy)
        corr_df = compute_correlation_matrix(monthly) if not monthly.empty else pd.DataFrame()
        matrix: dict[str, dict[str, float]] = {}
        for row_code in DASHBOARD_STRATEGY_CODES:
            matrix[row_code] = {}
            for col_code in DASHBOARD_STRATEGY_CODES:
                if corr_df.empty or row_code not in corr_df.index or col_code not in corr_df.columns:
                    matrix[row_code][col_code] = 1.0 if row_code == col_code else 0.0
                else:
                    val = float(corr_df.loc[row_code, col_code])
                    matrix[row_code][col_code] = round(val if np.isfinite(val) else 0.0, 4)
        return matrix

    @staticmethod
    def _norm(value: float, *, cap: float = 100.0) -> float:
        return max(0.0, min(100.0, float(value)))

    def diversification_benefit(
        self,
        strategy_id: str,
        correlation_matrix: dict[str, dict[str, float]],
    ) -> float:
        row = correlation_matrix.get(strategy_id) or {}
        peers = [abs(float(v)) for k, v in row.items() if k != strategy_id]
        if not peers:
            return 50.0
        avg_corr = float(np.mean(peers))
        return round(self._norm((1.0 - avg_corr) * 100.0), 1)

    def recovery_contribution_score(self, recovery_contribution: float) -> float:
        return round(self._norm(100.0 - min(100.0, abs(recovery_contribution) * 4.0)), 1)

    def dd_reduction_score(self, dd_contribution: float) -> float:
        return round(self._norm(100.0 - min(100.0, abs(dd_contribution) * 4.0)), 1)

    def challenge_contribution_score(
        self,
        *,
        pass_rate: float,
        profit_contribution: float,
        pass_rate_impact: float | None = None,
        pass_days_impact: float | None = None,
    ) -> float:
        rate = pass_rate_impact if pass_rate_impact is not None else pass_rate
        speed = pass_days_impact if pass_days_impact is not None else max(0.0, profit_contribution)
        return round(
            self._norm(rate, cap=100.0) * 0.6 + self._norm(speed, cap=50.0) * 0.4,
            1,
        )

    def state_stability_score(
        self,
        *,
        health_score: float,
        health_score_impact: float | None = None,
        profit_contribution: float = 0.0,
        dd_contribution: float = 0.0,
    ) -> float:
        impact = health_score_impact
        if impact is None:
            impact = profit_contribution - dd_contribution
        return round(self._norm(health_score) * 0.5 + self._norm(impact + 50.0, cap=100.0) * 0.5, 1)

    def compute(
        self,
        strategy_id: str,
        metrics: dict[str, Any],
        *,
        correlation_matrix: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, Any]:
        correlation_matrix = correlation_matrix or {}
        components = {
            "diversification_benefit": self.diversification_benefit(strategy_id, correlation_matrix),
            "recovery_contribution": self.recovery_contribution_score(
                float(metrics.get("recovery_contribution") or 0.0)
            ),
            "challenge_contribution": self.challenge_contribution_score(
                pass_rate=float(metrics.get("pass_rate") or 0.0),
                profit_contribution=float(metrics.get("profit_contribution") or 0.0),
                pass_rate_impact=metrics.get("pass_rate_impact"),
                pass_days_impact=metrics.get("pass_days_impact"),
            ),
            "state_stability_contribution": self.state_stability_score(
                health_score=float(metrics.get("health_score") or 50.0),
                health_score_impact=metrics.get("health_score_impact"),
                profit_contribution=float(metrics.get("profit_contribution") or 0.0),
                dd_contribution=float(metrics.get("dd_contribution") or 0.0),
            ),
            "dd_reduction_contribution": self.dd_reduction_score(
                float(metrics.get("dd_contribution") or 0.0)
            ),
        }
        score = round(sum(self.WEIGHTS[k] * components[k] for k in self.WEIGHTS), 1)
        avg_corr = 0.0
        row = correlation_matrix.get(strategy_id) or {}
        peers = [abs(float(v)) for k, v in row.items() if k != strategy_id]
        if peers:
            avg_corr = round(float(np.mean(peers)), 4)

        return {
            "strategy_id": strategy_id,
            "portfolio_fit_score": score,
            "components": components,
            "diversification_score": components["diversification_benefit"],
            "recovery_score": components["recovery_contribution"],
            "challenge_score": components["challenge_contribution"],
            "stability_contribution_score": components["state_stability_contribution"],
            "dd_reduction_score": components["dd_reduction_contribution"],
            "average_correlation": avg_corr,
            "correlation": avg_corr,
            "recovery_impact": components["recovery_contribution"],
            "challenge_impact": components["challenge_contribution"],
            "health_impact": components["state_stability_contribution"],
            "dd_impact": components["dd_reduction_contribution"],
        }

    def compute_all(
        self,
        metrics_by_strategy: dict[str, dict[str, Any]],
        trades: pd.DataFrame | None = None,
    ) -> dict[str, dict[str, Any]]:
        matrix = self.build_correlation_matrix(trades)
        return {
            sid: self.compute(sid, metrics, correlation_matrix=matrix)
            for sid, metrics in metrics_by_strategy.items()
        }

    def average_portfolio_fit(self, fit_by_strategy: dict[str, dict[str, Any]]) -> float:
        scores = [float(v.get("portfolio_fit_score") or 0.0) for v in fit_by_strategy.values()]
        return round(float(np.mean(scores)), 1) if scores else 0.0

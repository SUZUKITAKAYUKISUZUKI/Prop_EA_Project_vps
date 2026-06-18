"""Aggregate qualification metrics from PRAE, WFT, State Analytics, and trades."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from src.database.profile_migrations import DASHBOARD_STRATEGY_CODES
from src.repositories.wft_repository import WFTRepository
from src.strategy_lifecycle.portfolio_fit_engine import PortfolioFitEngine
from src.strategy_lifecycle.strategy_score_engine import StrategyScoreEngine


class QualificationEngine:
    def __init__(
        self,
        *,
        wft: WFTRepository | None = None,
        scorer: StrategyScoreEngine | None = None,
        portfolio_fit: PortfolioFitEngine | None = None,
    ) -> None:
        self._wft = wft or WFTRepository(owns_connection=False)
        self._scorer = scorer or StrategyScoreEngine()
        self._fit = portfolio_fit or PortfolioFitEngine()

    def close(self) -> None:
        self._wft.close()

    def build_all_base_metrics(
        self,
        *,
        prae_v2: dict[str, Any] | None = None,
        state_summary: dict[str, Any] | None = None,
        trades: pd.DataFrame | None = None,
    ) -> dict[str, dict[str, Any]]:
        prae_v2 = prae_v2 or {}
        state_summary = state_summary or {}
        health = float(
            state_summary.get("health_score")
            or (prae_v2.get("health_report") or {}).get("health_score")
            or 50.0
        )
        recovery_events = int(
            state_summary.get("recovery_events")
            or (prae_v2.get("health_report") or {}).get("recovery_events")
            or 0
        )
        metrics: dict[str, dict[str, Any]] = {}
        for strategy_id in DASHBOARD_STRATEGY_CODES:
            metrics[strategy_id] = self._build_base_metrics(
                strategy_id,
                prae_v2=prae_v2,
                health=health,
                recovery_events=recovery_events,
                trades=trades,
            )
        return metrics

    def build_metrics(
        self,
        strategy_id: str,
        *,
        prae_v2: dict[str, Any] | None = None,
        state_summary: dict[str, Any] | None = None,
        trades: pd.DataFrame | None = None,
        all_metrics: dict[str, dict[str, Any]] | None = None,
        fit_bundle: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        prae_v2 = prae_v2 or {}
        state_summary = state_summary or {}
        health = float(
            state_summary.get("health_score")
            or (prae_v2.get("health_report") or {}).get("health_score")
            or 50.0
        )
        recovery_events = int(
            state_summary.get("recovery_events")
            or (prae_v2.get("health_report") or {}).get("recovery_events")
            or 0
        )
        base = self._build_base_metrics(
            strategy_id,
            prae_v2=prae_v2,
            health=health,
            recovery_events=recovery_events,
            trades=trades,
        )
        if fit_bundle is None:
            all_metrics = all_metrics or self.build_all_base_metrics(
                prae_v2=prae_v2,
                state_summary=state_summary,
                trades=trades,
            )
            fit_bundle = self._fit.compute_all(all_metrics, trades)
        fit = fit_bundle.get(strategy_id) or self._fit.compute(strategy_id, base)
        base["portfolio_fit_score"] = float(fit.get("portfolio_fit_score") or 0.0)
        base["portfolio_fit_components"] = fit.get("components") or {}
        base["diversification_score"] = fit.get("diversification_score")
        base["recovery_score"] = fit.get("recovery_score")
        base["challenge_score"] = fit.get("challenge_score")
        base["stability_contribution_score"] = fit.get("stability_contribution_score")
        base["dd_reduction_score"] = fit.get("dd_reduction_score")
        base["average_correlation"] = fit.get("average_correlation")
        base["recovery_impact"] = fit.get("recovery_impact")
        base["challenge_impact"] = fit.get("challenge_impact")
        base["health_impact_score"] = fit.get("health_impact")
        base["score"] = self._scorer.compute(
            {
                "oos_pf": base.get("oos_pf"),
                "pass_rate": base.get("pass_rate"),
                "total_r": base.get("total_r"),
                "stability": base.get("stability"),
                "health_impact": health,
                "portfolio_fit": base["portfolio_fit_score"],
            }
        )
        base["candidate_readiness"] = base["score"] >= 75.0 and base["portfolio_fit_score"] >= 60.0
        return base

    def _build_base_metrics(
        self,
        strategy_id: str,
        *,
        prae_v2: dict[str, Any],
        health: float,
        recovery_events: int,
        trades: pd.DataFrame | None,
    ) -> dict[str, Any]:
        wft = self._load_wft_metrics(strategy_id)
        risk = self._strategy_risk(prae_v2, strategy_id)
        dd_contrib = self._dd_contribution(prae_v2, strategy_id)
        recovery_contrib = self._recovery_contribution(prae_v2, strategy_id)
        profit_contrib = self._profit_contribution(prae_v2, strategy_id)
        trade_count = self._trade_count(strategy_id, trades, risk)
        pf = float(risk.get("pf") or 0.0)
        pass_rate = float(wft.get("pass_rate") or risk.get("win_rate") or 0.0)
        total_r = float(risk.get("total_r") or 0.0)
        max_dd = float(risk.get("max_dd") or 0.0)
        stability = float(wft.get("stability") or max(0.0, 100.0 - float(risk.get("risk_score") or 0.0)))
        return {
            "strategy_id": strategy_id,
            "pf": pf,
            "pass_rate": pass_rate,
            "total_r": total_r,
            "max_dd": max_dd,
            "oos_pf": float(wft.get("oos_pf") or 0.0),
            "oos_months": float(wft.get("oos_months") or 0.0),
            "trade_count": trade_count,
            "wft_complete": bool(wft.get("wft_complete")),
            "bt_complete": trade_count >= 50,
            "risk_score": float(risk.get("risk_score") or 0.0),
            "dd_contribution": dd_contrib,
            "recovery_contribution": recovery_contrib,
            "profit_contribution": profit_contrib,
            "health_score": health,
            "recovery_events": recovery_events,
            "stability": stability,
            "pass_rate_impact": pass_rate,
            "pass_days_impact": max(0.0, profit_contrib),
            "health_score_impact": profit_contrib - dd_contrib,
        }

    def _load_wft_metrics(self, strategy_id: str) -> dict[str, Any]:
        row = self._wft.load_latest_for_strategy(strategy_id)
        if not row:
            return {"oos_pf": 0.0, "oos_months": 0.0, "pass_rate": 0.0, "wft_complete": False, "stability": 0.0}
        wft_id = row["wft_id"]
        windows = self._wft.load_windows(str(wft_id))
        oos_months = 0.0
        if windows:
            starts = [w.get("oos_start") for w in windows if w.get("oos_start")]
            ends = [w.get("oos_end") for w in windows if w.get("oos_end")]
            if starts and ends:
                try:
                    start = min(datetime.fromisoformat(str(s)[:10]) for s in starts)
                    end = max(datetime.fromisoformat(str(e)[:10]) for e in ends)
                    oos_months = max(0.0, (end - start).days / 30.44)
                except ValueError:
                    oos_months = float(len(windows))
        stability = 50.0
        if row.get("stability_json"):
            try:
                import json

                payload = json.loads(row["stability_json"])
                stability = float(payload.get("score") or payload.get("stability") or 50.0)
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        return {
            "wft_id": wft_id,
            "oos_pf": float(row["mean_oos_pf"] or 0.0),
            "pass_rate": float(row["pass_rate"] or 0.0),
            "oos_months": round(oos_months, 1),
            "wft_complete": len(windows) > 0,
            "stability": stability,
        }

    @staticmethod
    def _strategy_risk(prae_v2: dict[str, Any], strategy_id: str) -> dict[str, Any]:
        for row in prae_v2.get("strategy_risk") or []:
            if str(row.get("strategy")) == strategy_id:
                return dict(row)
        return {}

    @staticmethod
    def _dd_contribution(prae_v2: dict[str, Any], strategy_id: str) -> float:
        dd = (prae_v2.get("dd_attribution") or {}).get("strategy_contribution") or {}
        return float(dd.get(strategy_id) or 0.0)

    @staticmethod
    def _recovery_contribution(prae_v2: dict[str, Any], strategy_id: str) -> float:
        total = 0.0
        for event in prae_v2.get("recovery_analysis") or []:
            total += float((event.get("strategy_contribution") or {}).get(strategy_id) or 0.0)
        return total

    @staticmethod
    def _profit_contribution(prae_v2: dict[str, Any], strategy_id: str) -> float:
        for row in prae_v2.get("weight_adjusted_contribution") or []:
            if str(row.get("strategy")) == strategy_id:
                return float(row.get("contribution_pct") or 0.0)
        return 0.0

    @staticmethod
    def _trade_count(
        strategy_id: str,
        trades: pd.DataFrame | None,
        risk: dict[str, Any],
    ) -> int:
        if trades is not None and not trades.empty:
            col = "strategy" if "strategy" in trades.columns else "setup_type"
            if col in trades.columns:
                return int((trades[col].astype(str) == strategy_id).sum())
        return int(risk.get("trade_count") or risk.get("trades") or 0)

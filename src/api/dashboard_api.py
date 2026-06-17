"""Dashboard-facing API — dict responses for PySide6 and web clients."""
from __future__ import annotations

from pathlib import Path

from src.repositories.base import normalize_source_path
from src.services.analytics_service import AnalyticsService
from src.services.portfolio_service import PortfolioService
from src.services.trade_service import TradeService

DEFAULT_PORTFOLIO_SOURCE = "backtest_results/main_abcde_3y.csv"


def _source(source_path: str | Path | None) -> str:
    return normalize_source_path(source_path or DEFAULT_PORTFOLIO_SOURCE)


def get_dashboard_summary(source_path: str | Path | None = None, *, run_id: int | None = None) -> dict:
    src = _source(source_path)
    trade_svc = TradeService()
    analytics = AnalyticsService(trade_svc)
    portfolio = PortfolioService(trade_svc.trades)
    trade_summary = trade_svc.summary(run_id=run_id, source_path=src)
    bt = analytics.bt_summary(run_id=run_id, source_path=src)
    mc_rows = portfolio.portfolio.get_mc_results(source_path=src)
    pass_rate = float(mc_rows[0]["pass_rate"]) if mc_rows else 100.0
    return {
        "pf": round(float(bt.get("pf") or trade_summary["pf"]), 4),
        "total_r": round(float(bt.get("total_r") or trade_summary["total_r"]), 2),
        "max_dd": round(float(bt.get("max_dd") or trade_summary["max_dd"]), 2),
        "pass_rate": pass_rate,
        "win_rate": round(float(bt.get("wr") or trade_summary["win_rate"]), 2),
        "trades": trade_summary["trades"],
        "source": src,
    }


def get_strategy_table(source_path: str | Path | None = None, *, run_id: int | None = None) -> list[dict]:
    src = _source(source_path)
    return PortfolioService().strategy_ranking(run_id=run_id, source_path=src)


def get_wft_summary(source_path: str | Path | None = None, *, run_id: int | None = None) -> list[dict]:
    src = _source(source_path)
    rows = PortfolioService().portfolio.get_wft_results(source_path=src)
    return [
        {
            "window_id": row["window_id"],
            "oos_pf": row.get("oos_pf"),
            "oos_r": row.get("oos_r"),
            "oos_dd": row.get("oos_dd"),
            "pass_flag": row.get("pass_flag"),
        }
        for row in rows
    ]


def get_mc_summary(source_path: str | Path | None = None, *, run_id: int | None = None) -> list[dict]:
    src = _source(source_path)
    rows = PortfolioService().portfolio.get_mc_results(source_path=src)
    return [
        {
            "label": row.get("label"),
            "pass_rate": row.get("pass_rate"),
            "ror": row.get("ror"),
            "avg_pass_days": row.get("avg_pass_days"),
            "max_dd": row.get("max_dd"),
        }
        for row in rows
    ]


def get_portfolio_summary(source_path: str | Path | None = None, *, run_id: int | None = None) -> dict:
    src = _source(source_path)
    return PortfolioService().summary(run_id=run_id, source_path=src)

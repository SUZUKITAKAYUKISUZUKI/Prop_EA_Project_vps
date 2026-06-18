"""Dashboard API for Portfolio Risk Attribution Engine."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.api.dashboard_api import DEFAULT_PORTFOLIO_SOURCE
from src.repositories.base import normalize_source_path
from src.services.risk_attribution_service import RiskAttributionService

_svc: RiskAttributionService | None = None


def _service() -> RiskAttributionService:
    global _svc
    if _svc is None:
        _svc = RiskAttributionService(owns_connections=True)
    return _svc


def _report_payload(
    *,
    source_path: str | Path | None = None,
    run_id: int | None = None,
    profile_id: str | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    svc = _service()
    if refresh:
        return svc.run_attribution(
            source_path=source_path,
            run_id=run_id,
            profile_id=profile_id,
            use_cache=False,
        )
    return svc.ensure_report(source_path=source_path, run_id=run_id, profile_id=profile_id)


def get_risk_attribution(
    source_path: str | Path | None = None,
    *,
    run_id: int | None = None,
    profile_id: str | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    payload = _report_payload(
        source_path=source_path,
        run_id=run_id,
        profile_id=profile_id,
        refresh=refresh,
    )
    report = payload.get("report") or {}
    return {
        "report_id": payload.get("report_id"),
        "profile_id": payload.get("profile_id"),
        "source": normalize_source_path(source_path or DEFAULT_PORTFOLIO_SOURCE),
        "overview": report.get("overview", {}),
        "summary": report.get("summary", {}),
        "drawdown": report.get("drawdown", {}),
        "worst_drawdown": report.get("worst_drawdown", {}),
        "recovery": report.get("recovery", {}),
        "charts": payload.get("charts", {}),
    }


def get_strategy_attribution(
    source_path: str | Path | None = None,
    *,
    run_id: int | None = None,
    profile_id: str | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    report = _report_payload(
        source_path=source_path,
        run_id=run_id,
        profile_id=profile_id,
        refresh=refresh,
    ).get("report", {})
    strategy = report.get("strategy", {})
    rows = []
    for code, stats in (strategy.get("strategies") or {}).items():
        rows.append(
            {
                "strategy": code,
                "weight": stats.get("allocated_weight"),
                "trades": stats.get("strategy_trades"),
                "pf": stats.get("strategy_pf"),
                "total_r": stats.get("strategy_total_r"),
                "dd_contribution": stats.get("strategy_drawdown_contribution"),
            }
        )
    return {
        "rankings": strategy.get("rankings", {}),
        "rows": sorted(rows, key=lambda r: r.get("total_r") or 0, reverse=True),
        "allocation": report.get("allocation", {}),
    }


def get_symbol_attribution(
    source_path: str | Path | None = None,
    *,
    run_id: int | None = None,
    profile_id: str | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    symbol_layer = _report_payload(
        source_path=source_path,
        run_id=run_id,
        profile_id=profile_id,
        refresh=refresh,
    ).get("report", {}).get("symbol", {})
    heatmap = [
        {"symbol": sym, "total_r": data["symbol_total_r"], "pf": data["symbol_pf"]}
        for sym, data in (symbol_layer.get("symbols") or {}).items()
    ]
    return {
        "best_symbol": symbol_layer.get("best_symbol"),
        "worst_symbol": symbol_layer.get("worst_symbol"),
        "heatmap": sorted(heatmap, key=lambda r: r["total_r"], reverse=True),
    }


def get_drawdown_attribution(
    source_path: str | Path | None = None,
    *,
    run_id: int | None = None,
    profile_id: str | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    report = _report_payload(
        source_path=source_path,
        run_id=run_id,
        profile_id=profile_id,
        refresh=refresh,
    ).get("report", {})
    return {
        "current": report.get("drawdown", {}),
        "worst": report.get("worst_drawdown", {}),
        "recovery": report.get("recovery", {}),
    }


def get_profile_attribution(
    source_path: str | Path | None = None,
    *,
    run_id: int | None = None,
    profile_id: str | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    return _report_payload(
        source_path=source_path,
        run_id=run_id,
        profile_id=profile_id,
        refresh=refresh,
    ).get("report", {}).get("profile", {"profiles": {}})


def get_bayes_attribution(
    source_path: str | Path | None = None,
    *,
    run_id: int | None = None,
    profile_id: str | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    return _report_payload(
        source_path=source_path,
        run_id=run_id,
        profile_id=profile_id,
        refresh=refresh,
    ).get("report", {}).get("bayes", {"buckets": {}, "available": False})


def get_session_attribution(
    source_path: str | Path | None = None,
    *,
    run_id: int | None = None,
    profile_id: str | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    return _report_payload(
        source_path=source_path,
        run_id=run_id,
        profile_id=profile_id,
        refresh=refresh,
    ).get("report", {}).get("session", {"sessions": {}})


def close_risk_attribution_api() -> None:
    global _svc
    if _svc is not None:
        _svc.close()
        _svc = None

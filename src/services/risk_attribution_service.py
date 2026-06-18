"""Portfolio Risk Attribution service — orchestrates engine, persistence, and reports."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prae.loaders import normalize_trade_frame
from src.analytics.risk_attribution_engine import (
    PortfolioRiskAttributionEngine,
    generate_all_charts,
)
from src.api.dashboard_api import DEFAULT_PORTFOLIO_SOURCE
from src.repositories.base import PROJECT_ROOT, normalize_source_path
from src.repositories.risk_attribution_repository import RiskAttributionRepository
from src.repositories.trade_repository import TradeRepository
from src.services.profile_service import ProfileService

REPORTS_DIR = PROJECT_ROOT / "reports" / "risk_attribution"


class RiskAttributionService:
    def __init__(
        self,
        *,
        repo: RiskAttributionRepository | None = None,
        trade_repo: TradeRepository | None = None,
        profile_service: ProfileService | None = None,
        owns_connections: bool = False,
    ) -> None:
        self._owns = owns_connections or repo is None
        self._repo = repo or RiskAttributionRepository(owns_connection=self._owns)
        self._trades = trade_repo or TradeRepository(owns_connection=False)
        self._profiles = profile_service

    def close(self) -> None:
        if self._profiles is not None:
            self._profiles.close()
        self._trades.close()
        if self._owns:
            self._repo.close()

    def _load_profile_context(self) -> tuple[str, dict[str, float], float | None]:
        svc = self._profiles or ProfileService()
        ctx = svc.load_active_profile()
        pass_rate: float | None = None
        return ctx.profile_id, ctx.strategy_allocations, pass_rate

    def load_trades(
        self,
        *,
        source_path: str | Path | None = None,
        run_id: int | None = None,
    ) -> tuple[Any, int | None]:
        import pandas as pd

        src = normalize_source_path(source_path or DEFAULT_PORTFOLIO_SOURCE)
        df = self._trades.get_trades(source_path=src, run_id=run_id, as_dataframe=True)
        resolved_run = self._trades._resolve_run_id(run_id=run_id, source_path=src)
        if not isinstance(df, pd.DataFrame) or df.empty:
            empty = pd.DataFrame(
                columns=[
                    "timestamp",
                    "strategy",
                    "setup_type",
                    "pair",
                    "R",
                    "profit_r",
                    "lot_factor",
                ]
            )
            return empty, resolved_run
        return normalize_trade_frame(df, source=Path(src).name), resolved_run

    def _cache_key(self, *, source_path: str, profile_id: str, run_id: int | None) -> str:
        raw = f"{source_path}|{profile_id}|{run_id or 0}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def run_attribution(
        self,
        *,
        source_path: str | Path | None = None,
        run_id: int | None = None,
        profile_id: str | None = None,
        use_cache: bool = True,
        write_json: bool = True,
        write_charts: bool = True,
    ) -> dict[str, Any]:
        src = normalize_source_path(source_path or DEFAULT_PORTFOLIO_SOURCE)
        active_id, allocations, pass_rate = self._load_profile_context()
        pid = profile_id or active_id
        trades, resolved_run = self.load_trades(source_path=src, run_id=run_id)

        cache_key = self._cache_key(source_path=src, profile_id=pid, run_id=resolved_run)
        if use_cache:
            cached = self._repo.get_cache(cache_key)
            if cached and cached.get("source_path") == src:
                return cached

        engine = PortfolioRiskAttributionEngine(
            trades,
            profile_id=pid,
            allocation_weights=allocations,
            pass_rate=pass_rate,
        )
        report = engine.run_full_report()
        overview = report["overview"]

        stamp = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H%M%S")
        report_id = f"risk_report_{stamp}"

        chart_paths: dict[str, str] = {}
        if write_charts:
            chart_paths = generate_all_charts(report, REPORTS_DIR / report_id)

        json_path: Path | None = None
        if write_json:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            json_path = REPORTS_DIR / f"risk_report_{datetime.now(timezone.utc).strftime('%Y_%m_%d')}.json"
            payload = {
                "report_id": report_id,
                "source_path": src,
                "source_run_id": str(resolved_run) if resolved_run else None,
                "profile_id": pid,
                "report": report,
                "charts": chart_paths,
            }
            json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        self._repo.save_report(
            report_id=report_id,
            source_run_id=str(resolved_run) if resolved_run else None,
            profile_id=pid,
            total_r=float(overview["total_r"]),
            total_dd=float(overview["current_dd"]),
            pf=float(overview["pf"]),
            win_rate=float(overview["win_rate"]),
            report_json=report,
        )

        result = {
            "report_id": report_id,
            "source_path": src,
            "source_run_id": resolved_run,
            "profile_id": pid,
            "report": report,
            "charts": chart_paths,
            "json_path": str(json_path) if json_path else None,
        }
        self._repo.set_cache(cache_key, result)
        return result

    def run_from_trades(
        self,
        trades: Any,
        *,
        profile_id: str = "",
        allocation_weights: dict[str, float] | None = None,
        pass_rate: float | None = None,
        source_run_id: str | None = None,
        write_json: bool = True,
        write_charts: bool = True,
    ) -> dict[str, Any]:
        """Run attribution on an in-memory trade frame (PRAE / optimizer integration)."""
        import pandas as pd

        from prae.loaders import normalize_trade_frame

        frame = trades if isinstance(trades, pd.DataFrame) else pd.DataFrame(trades)
        if not frame.empty and "R" not in frame.columns:
            frame = normalize_trade_frame(frame)

        pid = profile_id
        weights = allocation_weights
        if not pid or weights is None:
            active_id, active_weights, _ = self._load_profile_context()
            pid = pid or active_id
            weights = weights if weights is not None else active_weights

        engine = PortfolioRiskAttributionEngine(
            frame,
            profile_id=pid,
            allocation_weights=weights or {},
            pass_rate=pass_rate,
        )
        report = engine.run_full_report()
        overview = report["overview"]
        stamp = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H%M%S")
        report_id = f"risk_report_{stamp}"

        chart_paths: dict[str, str] = {}
        if write_charts:
            chart_paths = generate_all_charts(report, REPORTS_DIR / report_id)

        json_path: Path | None = None
        if write_json:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            json_path = REPORTS_DIR / f"risk_report_{datetime.now(timezone.utc).strftime('%Y_%m_%d')}.json"
            payload = {
                "report_id": report_id,
                "source_run_id": source_run_id,
                "profile_id": pid,
                "report": report,
                "charts": chart_paths,
            }
            json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        self._repo.save_report(
            report_id=report_id,
            source_run_id=source_run_id,
            profile_id=pid,
            total_r=float(overview["total_r"]),
            total_dd=float(overview["current_dd"]),
            pf=float(overview["pf"]),
            win_rate=float(overview["win_rate"]),
            report_json=report,
        )
        return {
            "report_id": report_id,
            "profile_id": pid,
            "report": report,
            "charts": chart_paths,
            "json_path": str(json_path) if json_path else None,
        }

    def get_latest(self, *, profile_id: str | None = None) -> dict[str, Any] | None:
        row = self._repo.get_latest_report(profile_id=profile_id)
        if row is None:
            return None
        return row

    def ensure_report(
        self,
        *,
        source_path: str | Path | None = None,
        run_id: int | None = None,
        profile_id: str | None = None,
    ) -> dict[str, Any]:
        latest = self.get_latest(profile_id=profile_id)
        if latest and latest.get("report"):
            return {
                "report_id": latest["report_id"],
                "profile_id": latest["profile_id"],
                "report": latest["report"],
                "from_cache": True,
            }
        return self.run_attribution(
            source_path=source_path,
            run_id=run_id,
            profile_id=profile_id,
        )

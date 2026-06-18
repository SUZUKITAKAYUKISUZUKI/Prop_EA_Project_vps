"""Persist PRAE / PFOO analytics artifacts to portfolio_os.db."""
from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from src.database.db_manager import DatabaseManager, utc_now_iso
from src.repositories.base import create_default_db_manager


def _export_csv_enabled() -> bool:
    return os.environ.get("ANALYTICS_EXPORT_CSV", "0").strip().lower() in {"1", "true", "yes"}


class AnalyticsWriteService:
    def __init__(self, db: DatabaseManager | None = None, *, owns_connection: bool = False) -> None:
        self._db = db or create_default_db_manager()
        self._owns = owns_connection or db is None

    def close(self) -> None:
        if self._owns:
            self._db.close()

    def create_analytics_run(
        self,
        run_type: str,
        *,
        description: str,
        source: str = "BACKTEST",
        parameters: dict[str, Any] | None = None,
    ) -> int:
        return self._db.insert_run(
            run_type,
            strategy="portfolio",
            description=description,
            parameters=parameters,
            source=source,
        )

    def save_prae_artifacts(
        self,
        *,
        artifact_dir: str,
        phase1: Any,
        phase2: Any,
        phase3: Any,
        phase4: Any,
        monte_carlo: Any,
        recommended: dict[str, float],
    ) -> int:
        run_id = self.create_analytics_run(
            "prae",
            description=artifact_dir,
            parameters={"engine": "prae_v1", "recommended": recommended},
            source="BACKTEST",
        )

        summary = getattr(phase1, "strategy_summary", None)
        if isinstance(summary, pd.DataFrame):
            for idx, row in summary.iterrows():
                label = str(row.get("strategy") or row.get("Strategy") or f"row_{idx}")
                self._db.insert_bt_summary(
                    run_id,
                    pf=_float(row.get("pf") or row.get("PF")),
                    wr=_float(row.get("win_rate") or row.get("WinRate")),
                    total_r=_float(row.get("total_r") or row.get("TotalR")),
                    max_dd=_float(row.get("max_dd") or row.get("MaxDD")),
                    label=label,
                )

        dd = getattr(phase2, "dd_contribution", None)
        if isinstance(dd, pd.DataFrame):
            for _, row in dd.iterrows():
                strategy = str(row.get("strategy") or row.get("Strategy") or "UNKNOWN")
                self._db.insert_risk_attribution(
                    run_id,
                    strategy,
                    contribution_r=_float(row.get("dd_contribution_r") or row.get("contribution_r")),
                    contribution_dd=_float(
                        row.get("dd_contribution_pct")
                        or row.get("contribution_dd")
                        or row.get("DD Contribution %")
                    ),
                    contribution_pf=_float(row.get("pf") or row.get("contribution_pf")),
                )

        top = getattr(phase4, "top_candidates", None)
        if isinstance(top, pd.DataFrame):
            for rank, (_, row) in enumerate(top.iterrows(), start=1):
                allocation = row.get("allocation") or row.get("weights") or {}
                if isinstance(allocation, str):
                    allocation = {"raw": allocation}
                self._db.insert_portfolio_result(
                    run_id,
                    allocation_json=allocation if isinstance(allocation, dict) else {"raw": str(allocation)},
                    pf=_float(row.get("pf")),
                    total_r=_float(row.get("total_r") or row.get("score")),
                    max_dd=_float(row.get("dd") or row.get("max_dd")),
                    pass_rate=_float(row.get("passrate") or row.get("pass_rate")),
                    rank=rank,
                )

        trials = int(getattr(monte_carlo, "trials", 0) or 0)
        if trials:
            self._db.insert_mc_result(
                run_id,
                pass_rate=_float(getattr(monte_carlo, "pass_rate", None)),
                ror=_float(getattr(monte_carlo, "risk_of_ruin", None)),
                avg_pass_days=_float(getattr(monte_carlo, "avg_pass_days", None)),
                max_dd=None,
                label=f"prae_{trials}",
            )

        self._db.register_import(artifact_dir.replace("\\", "/"), run_id, "prae", 1, None)
        self._db.portfolio.commit()
        return run_id

    def save_pfoo_artifacts(
        self,
        *,
        artifact_dir: str,
        result: Any,
    ) -> int:
        run_id = self.create_analytics_run(
            "pfoo",
            description=artifact_dir,
            parameters={"engine": "pfoo", "recommended": getattr(result, "recommended_weights", {})},
            source="BACKTEST",
        )

        top = getattr(getattr(result, "risk_budget", None), "top_candidates", None)
        if isinstance(top, pd.DataFrame):
            for rank, (_, row) in enumerate(top.iterrows(), start=1):
                self._db.insert_portfolio_result(
                    run_id,
                    allocation_json={"weights": row.to_dict()},
                    pf=_float(row.get("pf")),
                    total_r=_float(row.get("total_r") or row.get("score")),
                    max_dd=_float(row.get("dd") or row.get("max_dd")),
                    pass_rate=_float(row.get("passrate") or row.get("pass_rate")),
                    rank=rank,
                )

        mc_map = getattr(result, "monte_carlo", {}) or {}
        for trials, mc in sorted(mc_map.items()):
            self._db.insert_mc_result(
                run_id,
                pass_rate=_float(getattr(mc, "pass_rate", None)),
                ror=_float(getattr(mc, "fail_rate", None)),
                avg_pass_days=_float(getattr(mc, "avg_pass_days", None)),
                max_dd=_float(getattr(mc, "worst_dd", None)),
                label=f"pfoo_{trials}",
            )

        self._db.register_import(artifact_dir.replace("\\", "/"), run_id, "pfoo", len(mc_map), None)
        self._db.portfolio.commit()
        return run_id

    def maybe_export_csv(self, df: pd.DataFrame, path: Any) -> None:
        if _export_csv_enabled() and path is not None:
            p = path if isinstance(path, str) else str(path)
            df.to_csv(p, index=False, encoding="utf-8-sig")


def _float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

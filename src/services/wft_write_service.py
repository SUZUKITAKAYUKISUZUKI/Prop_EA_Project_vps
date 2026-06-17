"""Persist walk-forward test results directly to portfolio_os.db."""
from __future__ import annotations

import os
import uuid
from typing import Any

from src.database.db_manager import utc_now_iso
from src.repositories.base import create_default_db_manager
from src.repositories.wft_repository import WFTRepository
from src.services.bt_wft_records import compute_pf_from_records, record_to_bt_trade


class WFTWriteService:
    def __init__(self, *, wft_repo: WFTRepository | None = None, db=None) -> None:
        if wft_repo is not None:
            self._wft = wft_repo
            self._owns = False
        else:
            self._db = db or create_default_db_manager()
            self._wft = WFTRepository(self._db, owns_connection=False)
            self._owns = db is None

    def close(self) -> None:
        if self._owns:
            self._wft.close()

    def save_wft_result(
        self,
        results: Any,
        *,
        wft_id: str | None = None,
        strategy: str | None = None,
        mirror_legacy: bool = True,
    ) -> str:
        profile = getattr(results, "profile", None)
        config = getattr(results, "config", {}) or {}
        strategy_name = strategy or (getattr(profile, "mode", None) if profile else None) or "portfolio"
        windows = list(getattr(results, "windows", []) or [])

        resolved_wft_id = wft_id or f"wft_{strategy_name}_{uuid.uuid4().hex[:12]}"
        self._wft.save_wft(
            {
                "wft_id": resolved_wft_id,
                "strategy": strategy_name,
                "is_months": int(config.get("is_months", 12)),
                "oos_months": int(config.get("oos_months", 3)),
                "step_months": int(config.get("step_months", 3)),
                "total_windows": len(windows),
                "created_at": utc_now_iso(),
            }
        )

        oos_rs: list[float] = []
        oos_pfs: list[float] = []
        oos_dds: list[float] = []
        pass_flags: list[int] = []
        all_wft_trades: list[dict[str, Any]] = []

        for window in windows:
            window_no = int(getattr(window, "window_id", 0))
            window_key = f"{resolved_wft_id}_w{window_no:04d}"
            oos_metrics = getattr(window, "oos_metrics", None)
            oos_records = [
                r for r in getattr(window, "oos_records", []) if r.get("trade_result") != "NOT_EXECUTED"
            ]
            total_r = float(getattr(oos_metrics, "total_profit_r", 0.0)) if oos_metrics else 0.0
            pf_raw = compute_pf_from_records(oos_records) if oos_records else 0.0
            pf = None if pf_raw == float("inf") else pf_raw
            max_dd = float(getattr(oos_metrics, "max_total_dd_pct", 0.0)) if oos_metrics else 0.0
            pass_flag = 1 if not getattr(oos_metrics, "is_disqualified", False) else 0

            self._wft.save_window(
                {
                    "window_id": window_key,
                    "wft_id": resolved_wft_id,
                    "window_no": window_no,
                    "is_start": _ts_text(getattr(window, "is_start", None)),
                    "is_end": _ts_text(getattr(window, "is_end", None)),
                    "oos_start": _ts_text(getattr(window, "oos_start", None)),
                    "oos_end": _ts_text(getattr(window, "oos_end", None)),
                    "total_r": total_r,
                    "pf": pf,
                    "max_dd": max_dd,
                    "pass_flag": pass_flag,
                }
            )

            for record in oos_records:
                base = record_to_bt_trade(record, run_id=window_key)
                if base is None:
                    continue
                trade_id = f"{window_key}:{base['trade_id']}"
                all_wft_trades.append(
                    {
                        **base,
                        "trade_id": trade_id,
                        "window_id": window_key,
                        "wft_id": resolved_wft_id,
                    }
                )

            oos_rs.append(total_r)
            if pf is not None:
                oos_pfs.append(float(pf))
            oos_dds.append(max_dd)
            pass_flags.append(pass_flag)

        if all_wft_trades:
            self._wft.save_trades(all_wft_trades)

        stability = dict(getattr(results, "stability", {}) or {})
        summary = {
            "wft_id": resolved_wft_id,
            "total_oos_r": sum(oos_rs),
            "mean_oos_pf": (sum(oos_pfs) / len(oos_pfs)) if oos_pfs else None,
            "mean_oos_dd": (sum(oos_dds) / len(oos_dds)) if oos_dds else None,
            "pass_rate": (sum(pass_flags) / len(pass_flags) * 100.0) if pass_flags else None,
            "stability": stability,
            "created_at": utc_now_iso(),
        }
        self._wft.save_summary(summary)

        if mirror_legacy:
            self._mirror_legacy_wft(
                wft_id=resolved_wft_id,
                strategy=strategy_name,
                windows=windows,
                summary=summary,
            )
        return resolved_wft_id

    def _mirror_legacy_wft(
        self,
        *,
        wft_id: str,
        strategy: str,
        windows: list[Any],
        summary: dict[str, Any],
    ) -> None:
        db = self._wft._db
        legacy_run_id = db.insert_run(
            "wft",
            strategy=strategy,
            description=wft_id,
            parameters={"wft_id": wft_id, "native": True},
            source="WFT_OOS",
        )
        for window in windows:
            window_no = int(getattr(window, "window_id", 0))
            oos_metrics = getattr(window, "oos_metrics", None)
            pf_raw = 0.0
            oos_records = [
                r for r in getattr(window, "oos_records", []) if r.get("trade_result") != "NOT_EXECUTED"
            ]
            if oos_records:
                pf_val = compute_pf_from_records(oos_records)
                pf_raw = 0.0 if pf_val == float("inf") else pf_val
            db.insert_wft_result(
                legacy_run_id,
                window_no,
                oos_pf=pf_raw,
                oos_r=float(getattr(oos_metrics, "total_profit_r", 0.0)) if oos_metrics else 0.0,
                oos_dd=float(getattr(oos_metrics, "max_total_dd_pct", 0.0)) if oos_metrics else 0.0,
                pass_flag=0 if getattr(oos_metrics, "is_disqualified", False) else 1,
            )
        db.register_import(wft_id, legacy_run_id, "wft", len(windows), None)


def save_wft_window(results: Any, **kwargs: Any) -> str:
    enabled = os.environ.get("WFT_WRITE_SQLITE", "1").strip().lower() not in {"0", "false", "no"}
    if not enabled:
        return ""
    service = WFTWriteService()
    try:
        return service.save_wft_result(results, **kwargs)
    finally:
        service.close()


def _ts_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text[:19].replace("T", " ")

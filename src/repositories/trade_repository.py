"""Trade data repository — sole source for executed trade history."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.database.db_manager import DatabaseManager
from src.database.data_source import DataSource, normalize_source
from src.repositories.base import create_default_db_manager, normalize_source_path
from src.repositories.run_repository import RunRepository


class TradeRepository:
    def __init__(self, db: DatabaseManager | None = None, *, owns_connection: bool = False) -> None:
        self._db = db or create_default_db_manager()
        self._owns_connection = owns_connection or db is None
        self._runs = RunRepository(self._db)

    def close(self) -> None:
        if self._owns_connection:
            self._db.close()

    def _resolve_run_id(
        self,
        *,
        run_id: int | None = None,
        source_path: str | Path | None = None,
    ) -> int | None:
        if run_id is not None:
            return run_id
        if source_path is None:
            return self._runs.get_latest_run_id("trade")
        rel = normalize_source_path(source_path)
        resolved = self._runs.resolve_run_id(source_path=rel, description=rel)
        if resolved is None:
            resolved = self._runs.resolve_run_id(source_path=Path(rel).name)
        return resolved

    def _build_where(
        self,
        *,
        run_id: int | None,
        strategy: str | None,
        symbol: str | None,
        start: str | None,
        end: str | None,
        data_source: str | None = None,
    ) -> tuple[str, list[Any]]:
        clauses = ["1=1"]
        params: list[Any] = []
        if run_id is not None:
            clauses.append("run_id=?")
            params.append(run_id)
        if data_source:
            clauses.append("source=?")
            params.append(normalize_source(data_source))
        if strategy:
            clauses.append("strategy=?")
            params.append(strategy)
        if symbol:
            clauses.append("symbol=?")
            params.append(symbol)
        if start:
            clauses.append("entry_time>=?")
            params.append(start)
        if end:
            clauses.append("entry_time<=?")
            params.append(end)
        return " AND ".join(clauses), params

    def _rows_to_dataframe(self, rows: list[Any], *, source: str = "") -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        records = [dict(r) for r in rows]
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["entry_time"], errors="coerce")
        df["trade_id"] = df["source_trade_id"].fillna(df["trade_id"].astype(str))
        df["pair"] = df["symbol"]
        df["setup_type"] = df["strategy"]
        df["profit_r"] = pd.to_numeric(df["r_multiple"], errors="coerce").fillna(0.0)
        df["trade_result"] = df["result"]
        df["profit_loss"] = pd.to_numeric(df.get("profit"), errors="coerce")
        df["lot_factor"] = 1.0
        df["sized_result_r"] = df["profit_r"]
        if source:
            df["source_file"] = source
        cols = [
            "trade_id",
            "timestamp",
            "pair",
            "setup_type",
            "strategy",
            "trade_result",
            "profit_r",
            "profit_loss",
            "lot_factor",
            "sized_result_r",
            "entry_price",
            "exit_price",
            "entry_time",
            "exit_time",
            "run_id",
            "source",
        ]
        for col in cols:
            if col not in df.columns:
                df[col] = None
        if "source" in df.columns:
            df["data_source"] = df["source"]
        return df[cols + (["data_source"] if "data_source" in df.columns else []) + (["source_file"] if source else [])].sort_values("timestamp").reset_index(drop=True)

    def get_trade(self, trade_id: int) -> dict[str, Any] | None:
        row = self._db.query("SELECT * FROM trades WHERE trade_id=?", (trade_id,), one=True)
        return dict(row) if row else None

    def get_trades(
        self,
        *,
        run_id: int | None = None,
        source_path: str | Path | None = None,
        strategy: str | None = None,
        symbol: str | None = None,
        start: str | None = None,
        end: str | None = None,
        data_source: DataSource | str | None = None,
        as_dataframe: bool = False,
    ) -> list[dict[str, Any]] | pd.DataFrame:
        resolved_run = self._resolve_run_id(run_id=run_id, source_path=source_path)
        where, params = self._build_where(
            run_id=resolved_run,
            strategy=strategy,
            symbol=symbol,
            start=start,
            end=end,
            data_source=data_source,
        )
        rows = self._db.query(
            f"""
            SELECT * FROM trades
            WHERE {where}
            ORDER BY entry_time ASC, trade_id ASC
            """,
            tuple(params),
        )
        if as_dataframe:
            src = normalize_source_path(source_path) if source_path else ""
            return self._rows_to_dataframe(rows, source=Path(src).name if src else "")
        return [dict(r) for r in rows]

    def get_trades_by_strategy(self, strategy: str, **kwargs: Any) -> pd.DataFrame:
        result = self.get_trades(strategy=strategy, as_dataframe=True, **kwargs)
        assert isinstance(result, pd.DataFrame)
        return result

    def get_trades_by_symbol(self, symbol: str, **kwargs: Any) -> pd.DataFrame:
        result = self.get_trades(symbol=symbol, as_dataframe=True, **kwargs)
        assert isinstance(result, pd.DataFrame)
        return result

    def get_trades_by_period(
        self,
        start: str,
        end: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        result = self.get_trades(start=start, end=end, as_dataframe=True, **kwargs)
        assert isinstance(result, pd.DataFrame)
        return result

    def get_trades_df(
        self,
        *,
        run_id: int | None = None,
        source_path: str | Path | None = None,
        executed_only: bool = True,
    ) -> pd.DataFrame:
        df = self.get_trades(run_id=run_id, source_path=source_path, as_dataframe=True)
        if df.empty:
            path_hint = normalize_source_path(source_path) if source_path else "latest trade run"
            raise FileNotFoundError(
                f"No trades in SQLite for source={path_hint}. Run: python tools/migrate_all.py"
            )
        if executed_only and "trade_result" in df.columns:
            df = df[df["trade_result"].isin(["WIN", "LOSS"]) | df["trade_result"].isna()].copy()
            df = df[df["trade_result"] != "NOT_EXECUTED"]
        return df.sort_values("timestamp").reset_index(drop=True)

    def load_legacy_frame(self, source_path: str | Path) -> pd.DataFrame:
        """Drop-in replacement for pd.read_csv on migrated trade logs."""
        return self.get_trades_df(source_path=source_path, executed_only=False)

    def count_trades(self, run_id: int | None = None) -> int:
        if run_id is None:
            row = self._db.query("SELECT COUNT(*) AS c FROM trades", one=True)
        else:
            row = self._db.query("SELECT COUNT(*) AS c FROM trades WHERE run_id=?", (run_id,), one=True)
        return int(row["c"]) if row else 0

    def get_trade_stats(self, *, run_id: int | None = None, source_path: str | Path | None = None) -> dict[str, float | int]:
        resolved = self._resolve_run_id(run_id=run_id, source_path=source_path)
        if resolved is None:
            return {"trades": 0, "wins": 0, "win_rate": 0.0, "total_r": 0.0, "avg_r": 0.0, "gross_profit_r": 0.0, "gross_loss_r": 0.0}
        row = self._db.query(
            """
            SELECT
                COUNT(*) AS trades,
                SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) AS wins,
                COALESCE(SUM(r_multiple), 0) AS total_r,
                COALESCE(AVG(r_multiple), 0) AS avg_r,
                COALESCE(SUM(CASE WHEN r_multiple > 0 THEN r_multiple ELSE 0 END), 0) AS gross_profit_r,
                COALESCE(SUM(CASE WHEN r_multiple < 0 THEN ABS(r_multiple) ELSE 0 END), 0) AS gross_loss_r
            FROM trades
            WHERE run_id=? AND (result IS NULL OR result != 'NOT_EXECUTED')
            """,
            (resolved,),
            one=True,
        )
        trades = int(row["trades"]) if row else 0
        wins = int(row["wins"] or 0) if row else 0
        win_rate = float(wins / trades * 100.0) if trades else 0.0
        gross_profit = float(row["gross_profit_r"] or 0) if row else 0.0
        gross_loss = float(row["gross_loss_r"] or 0) if row else 0.0
        if gross_loss <= 0:
            pf = float("inf") if gross_profit > 0 else 0.0
        else:
            pf = gross_profit / gross_loss
        return {
            "trades": trades,
            "wins": wins,
            "win_rate": win_rate,
            "total_r": float(row["total_r"] or 0) if row else 0.0,
            "avg_r": float(row["avg_r"] or 0) if row else 0.0,
            "pf": float(pf),
            "run_id": resolved,
        }

    def get_strategy_stats(self, *, run_id: int | None = None, source_path: str | Path | None = None) -> list[dict]:
        resolved = self._resolve_run_id(run_id=run_id, source_path=source_path)
        if resolved is None:
            return []
        rows = self._db.query(
            """
            SELECT
                strategy,
                COUNT(*) AS trades,
                COALESCE(SUM(r_multiple), 0) AS total_r,
                COALESCE(AVG(r_multiple), 0) AS avg_r,
                COALESCE(SUM(CASE WHEN r_multiple > 0 THEN r_multiple ELSE 0 END), 0) AS gross_profit_r,
                COALESCE(SUM(CASE WHEN r_multiple < 0 THEN ABS(r_multiple) ELSE 0 END), 0) AS gross_loss_r
            FROM trades
            WHERE run_id=? AND (result IS NULL OR result != 'NOT_EXECUTED')
            GROUP BY strategy
            ORDER BY total_r DESC
            """,
            (resolved,),
        )
        out: list[dict] = []
        for row in rows:
            gp = float(row["gross_profit_r"] or 0)
            gl = float(row["gross_loss_r"] or 0)
            pf = float(gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)
            out.append(
                {
                    "strategy": row["strategy"],
                    "trades": int(row["trades"]),
                    "total_r": float(row["total_r"] or 0),
                    "avg_r": float(row["avg_r"] or 0),
                    "pf": pf,
                }
            )
        return out

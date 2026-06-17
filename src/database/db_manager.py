"""Unified SQLite writer/reader for Portfolio OS and market data."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from src.database.schema import create_market_schema, create_portfolio_schema
from src.database.data_source import (
    FEATURE_LOG_SCHEMA_VERSION,
    PORTFOLIO_DB_SCHEMA_VERSION,
    infer_source_from_run_type,
    normalize_source,
    resolve_feature_schema_version,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class DatabaseManager:
    """Context-manager aware SQLite access for portfolio_os.db and market_data.db."""

    def __init__(
        self,
        portfolio_path: str | Path,
        market_path: str | Path | None = None,
        *,
        journal_mode: str = "WAL",
        synchronous: str = "NORMAL",
    ) -> None:
        self.portfolio_path = Path(portfolio_path)
        self.market_path = Path(market_path) if market_path else None
        self.journal_mode = journal_mode
        self.synchronous = synchronous
        self._portfolio_conn: sqlite3.Connection | None = None
        self._market_conn: sqlite3.Connection | None = None

    def __enter__(self) -> DatabaseManager:
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        self.close()

    def connect(self) -> None:
        self.portfolio_path.parent.mkdir(parents=True, exist_ok=True)
        self._portfolio_conn = sqlite3.connect(self.portfolio_path)
        self._portfolio_conn.row_factory = sqlite3.Row
        create_portfolio_schema(
            self._portfolio_conn,
            journal_mode=self.journal_mode,
            synchronous=self.synchronous,
        )
        if self.market_path is not None:
            self.market_path.parent.mkdir(parents=True, exist_ok=True)
            self._market_conn = sqlite3.connect(self.market_path)
            self._market_conn.row_factory = sqlite3.Row
            create_market_schema(
                self._market_conn,
                journal_mode=self.journal_mode,
                synchronous=self.synchronous,
            )

    def close(self) -> None:
        if self._portfolio_conn is not None:
            self._portfolio_conn.close()
            self._portfolio_conn = None
        if self._market_conn is not None:
            self._market_conn.close()
            self._market_conn = None

    def commit(self) -> None:
        if self._portfolio_conn is not None:
            self._portfolio_conn.commit()
        if self._market_conn is not None:
            self._market_conn.commit()

    def rollback(self) -> None:
        if self._portfolio_conn is not None:
            self._portfolio_conn.rollback()
        if self._market_conn is not None:
            self._market_conn.rollback()

    @property
    def portfolio(self) -> sqlite3.Connection:
        if self._portfolio_conn is None:
            raise RuntimeError("DatabaseManager is not connected")
        return self._portfolio_conn

    @property
    def market(self) -> sqlite3.Connection:
        if self._market_conn is None:
            raise RuntimeError("market_data connection is not configured")
        return self._market_conn

    @contextmanager
    def transaction(self, *, market: bool = False) -> Iterator[sqlite3.Connection]:
        conn = self.market if market else self.portfolio
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def query(
        self,
        sql: str,
        params: tuple | dict | list | None = None,
        *,
        market: bool = False,
        one: bool = False,
    ) -> list[sqlite3.Row] | sqlite3.Row | None:
        conn = self.market if market else self.portfolio
        cur = conn.execute(sql, params or ())
        if one:
            return cur.fetchone()
        return cur.fetchall()

    def insert_run(
        self,
        run_type: str,
        *,
        strategy: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        created_at: str | None = None,
        source: str | None = None,
        schema_version: int | None = None,
    ) -> int:
        resolved_source = normalize_source(
            source or infer_source_from_run_type(run_type, description)
        )
        cur = self.portfolio.execute(
            """
            INSERT INTO runs (run_type, source, schema_version, strategy, created_at, description, parameters_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_type,
                resolved_source,
                schema_version if schema_version is not None else PORTFOLIO_DB_SCHEMA_VERSION,
                strategy,
                created_at or utc_now_iso(),
                description,
                json.dumps(parameters, ensure_ascii=False) if parameters else None,
            ),
        )
        return int(cur.lastrowid)

    def get_run_source(self, run_id: int) -> str:
        row = self.portfolio.execute(
            "SELECT source FROM runs WHERE run_id=? LIMIT 1",
            (run_id,),
        ).fetchone()
        if row and row[0]:
            return str(row[0])
        return "BACKTEST"

    def insert_trade(
        self,
        run_id: int,
        *,
        strategy: str | None = None,
        symbol: str | None = None,
        direction: str | None = None,
        entry_time: str | None = None,
        exit_time: str | None = None,
        entry_price: float | None = None,
        exit_price: float | None = None,
        r_multiple: float | None = None,
        profit: float | None = None,
        result: str | None = None,
        source_trade_id: str | None = None,
        source: str | None = None,
        upsert: bool = True,
    ) -> int:
        trade_source = normalize_source(source or self.get_run_source(run_id))
        sql = """
            INSERT INTO trades (
                run_id, source, strategy, symbol, direction, entry_time, exit_time,
                entry_price, exit_price, r_multiple, profit, result, source_trade_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            run_id,
            trade_source,
            strategy,
            symbol,
            direction,
            entry_time,
            exit_time,
            entry_price,
            exit_price,
            r_multiple,
            profit,
            result,
            source_trade_id if source_trade_id is not None else f"auto:{run_id}:{entry_time}:{symbol}:{strategy}:{r_multiple}:{result}",
        )
        if upsert:
            sql += """
                ON CONFLICT(run_id, source_trade_id) DO UPDATE SET
                    source=excluded.source,
                    strategy=excluded.strategy,
                    symbol=excluded.symbol,
                    direction=excluded.direction,
                    entry_time=excluded.entry_time,
                    exit_time=excluded.exit_time,
                    entry_price=excluded.entry_price,
                    exit_price=excluded.exit_price,
                    r_multiple=excluded.r_multiple,
                    profit=excluded.profit,
                    result=excluded.result
            """
        cur = self.portfolio.execute(sql, params)
        if cur.lastrowid:
            return int(cur.lastrowid)
        row = self.portfolio.execute(
            """
            SELECT trade_id FROM trades
            WHERE run_id=? AND (
                (source_trade_id IS NOT NULL AND source_trade_id=?)
                OR (entry_time=? AND symbol=? AND strategy=? AND r_multiple=? AND result=?)
            )
            LIMIT 1
            """,
            (run_id, source_trade_id, entry_time, symbol, strategy, r_multiple, result),
        ).fetchone()
        return int(row[0]) if row else 0

    def insert_feature(
        self,
        run_id: int,
        feature_json: dict[str, Any] | str,
        *,
        trade_id: int | None = None,
        strategy: str | None = None,
        source_key: str | None = None,
        source: str | None = None,
        schema_version: int | None = None,
        upsert: bool = True,
    ) -> int:
        payload_dict = feature_json if isinstance(feature_json, dict) else None
        payload = feature_json if isinstance(feature_json, str) else json.dumps(feature_json, ensure_ascii=False)
        if source_key is None:
            source_key = f"auto:{run_id}:{hash(payload) & 0xFFFFFFFF}"
        feature_source = normalize_source(source or self.get_run_source(run_id))
        feature_schema_version = (
            schema_version
            if schema_version is not None
            else resolve_feature_schema_version(payload_dict)
        )
        sql = """
            INSERT INTO features (trade_id, run_id, source, schema_version, strategy, feature_json, source_key)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = (trade_id, run_id, feature_source, feature_schema_version, strategy, payload, source_key)
        if upsert:
            sql += """
                ON CONFLICT(run_id, source_key) DO UPDATE SET
                    trade_id=excluded.trade_id,
                    source=excluded.source,
                    schema_version=excluded.schema_version,
                    strategy=excluded.strategy,
                    feature_json=excluded.feature_json
            """
        cur = self.portfolio.execute(sql, params)
        if cur.lastrowid:
            return int(cur.lastrowid)
        if source_key:
            row = self.portfolio.execute(
                "SELECT feature_id FROM features WHERE run_id=? AND source_key=? LIMIT 1",
                (run_id, source_key),
            ).fetchone()
            return int(row[0]) if row else 0
        return 0

    def insert_bt_summary(
        self,
        run_id: int,
        *,
        pf: float | None = None,
        wr: float | None = None,
        total_r: float | None = None,
        max_dd: float | None = None,
        sharpe: float | None = None,
        recovery: float | None = None,
        label: str | None = None,
        upsert: bool = True,
    ) -> int:
        label = label or "aggregate"
        sql = """
            INSERT INTO bt_summary (run_id, pf, wr, total_r, max_dd, sharpe, recovery, label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (run_id, pf, wr, total_r, max_dd, sharpe, recovery, label)
        if upsert:
            sql += """
                ON CONFLICT(run_id, label) DO UPDATE SET
                    pf=excluded.pf,
                    wr=excluded.wr,
                    total_r=excluded.total_r,
                    max_dd=excluded.max_dd,
                    sharpe=excluded.sharpe,
                    recovery=excluded.recovery
            """
        cur = self.portfolio.execute(sql, params)
        return int(cur.lastrowid or 0)

    def insert_wft_result(
        self,
        run_id: int,
        window_id: int,
        *,
        oos_pf: float | None = None,
        oos_r: float | None = None,
        oos_dd: float | None = None,
        pass_flag: int | None = None,
        upsert: bool = True,
    ) -> int:
        sql = """
            INSERT INTO wft_results (run_id, window_id, oos_pf, oos_r, oos_dd, pass_flag)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        params = (run_id, window_id, oos_pf, oos_r, oos_dd, pass_flag)
        if upsert:
            sql += """
                ON CONFLICT(run_id, window_id) DO UPDATE SET
                    oos_pf=excluded.oos_pf,
                    oos_r=excluded.oos_r,
                    oos_dd=excluded.oos_dd,
                    pass_flag=excluded.pass_flag
            """
        cur = self.portfolio.execute(sql, params)
        return int(cur.lastrowid or 0)

    def insert_mc_result(
        self,
        run_id: int,
        *,
        pass_rate: float | None = None,
        ror: float | None = None,
        avg_pass_days: float | None = None,
        max_dd: float | None = None,
        label: str | None = None,
        upsert: bool = True,
    ) -> int:
        sql = """
            INSERT INTO mc_results (run_id, pass_rate, ror, avg_pass_days, max_dd, label)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        params = (run_id, pass_rate, ror, avg_pass_days, max_dd, label)
        if upsert:
            sql += """
                ON CONFLICT(run_id, label) DO UPDATE SET
                    pass_rate=excluded.pass_rate,
                    ror=excluded.ror,
                    avg_pass_days=excluded.avg_pass_days,
                    max_dd=excluded.max_dd
            """
        cur = self.portfolio.execute(sql, params)
        return int(cur.lastrowid or 0)

    def insert_portfolio_result(
        self,
        run_id: int,
        *,
        allocation_json: dict[str, Any] | str | None = None,
        pf: float | None = None,
        total_r: float | None = None,
        max_dd: float | None = None,
        pass_rate: float | None = None,
        rank: int | None = None,
        upsert: bool = True,
    ) -> int:
        payload = None
        if isinstance(allocation_json, dict):
            payload = json.dumps(allocation_json, ensure_ascii=False)
        elif isinstance(allocation_json, str):
            payload = allocation_json
        sql = """
            INSERT INTO portfolio_results (run_id, allocation_json, pf, total_r, max_dd, pass_rate, rank)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = (run_id, payload, pf, total_r, max_dd, pass_rate, rank)
        if upsert and rank is not None:
            sql += """
                ON CONFLICT(run_id, rank) DO UPDATE SET
                    allocation_json=excluded.allocation_json,
                    pf=excluded.pf,
                    total_r=excluded.total_r,
                    max_dd=excluded.max_dd,
                    pass_rate=excluded.pass_rate
            """
        cur = self.portfolio.execute(sql, params)
        return int(cur.lastrowid or 0)

    def insert_risk_attribution(
        self,
        run_id: int,
        strategy: str,
        *,
        contribution_r: float | None = None,
        contribution_dd: float | None = None,
        contribution_pf: float | None = None,
        upsert: bool = True,
    ) -> int:
        sql = """
            INSERT INTO risk_attribution (run_id, strategy, contribution_r, contribution_dd, contribution_pf)
            VALUES (?, ?, ?, ?, ?)
        """
        params = (run_id, strategy, contribution_r, contribution_dd, contribution_pf)
        if upsert:
            sql += """
                ON CONFLICT(run_id, strategy) DO UPDATE SET
                    contribution_r=excluded.contribution_r,
                    contribution_dd=excluded.contribution_dd,
                    contribution_pf=excluded.contribution_pf
            """
        cur = self.portfolio.execute(sql, params)
        return int(cur.lastrowid or 0)

    def insert_candles_batch(
        self,
        rows: list[tuple],
        *,
        ignore_duplicates: bool = True,
    ) -> int:
        verb = "INSERT OR IGNORE" if ignore_duplicates else "INSERT"
        sql = f"""
            {verb} INTO candles (symbol, timeframe, dt, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        cur = self.market.executemany(sql, rows)
        return cur.rowcount

    def register_import(
        self,
        source_path: str,
        run_id: int | None,
        csv_kind: str,
        row_count: int,
        checksum: str | None = None,
    ) -> None:
        self.portfolio.execute(
            """
            INSERT INTO import_registry (source_path, run_id, csv_kind, row_count, imported_at, checksum)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_path) DO UPDATE SET
                run_id=excluded.run_id,
                csv_kind=excluded.csv_kind,
                row_count=excluded.row_count,
                imported_at=excluded.imported_at,
                checksum=excluded.checksum
            """,
            (source_path, run_id, csv_kind, row_count, utc_now_iso(), checksum),
        )

    def get_import_run_id(self, source_path: str) -> int | None:
        row = self.portfolio.execute(
            "SELECT run_id FROM import_registry WHERE source_path=?",
            (source_path,),
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else None

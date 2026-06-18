"""SQLite persistence for Strategy Lifecycle Manager."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.database.db_manager import DatabaseManager, utc_now_iso
from src.repositories.base import create_default_db_manager


class LifecycleRepository:
    def __init__(self, db: DatabaseManager | None = None, *, owns_connection: bool = False) -> None:
        self._db = db or create_default_db_manager()
        self._owns = owns_connection or db is None

    def close(self) -> None:
        if self._owns:
            self._db.close()

    def list_strategies(self, *, active_only: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM strategy_registry"
        params: tuple[Any, ...] = ()
        if active_only:
            sql += " WHERE active = 1"
        sql += " ORDER BY strategy_id ASC"
        return [dict(row) for row in self._db.query(sql, params)]

    def get_strategy(self, strategy_id: str) -> dict[str, Any] | None:
        row = self._db.query(
            "SELECT * FROM strategy_registry WHERE strategy_id=?",
            (strategy_id,),
            one=True,
        )
        return dict(row) if row else None

    def register_strategy(
        self,
        strategy_id: str,
        *,
        strategy_name: str | None = None,
        stage: str = "INCUBATION",
        strategy_version: str = "1.0",
        notes: str | None = None,
    ) -> dict[str, Any]:
        self._db.portfolio.execute(
            """
            INSERT INTO strategy_registry (
                strategy_id, strategy_name, current_stage, strategy_version,
                created_at, score, active, notes
            ) VALUES (?, ?, ?, ?, ?, 0.0, 1, ?)
            ON CONFLICT(strategy_id) DO UPDATE SET
                strategy_name=excluded.strategy_name,
                notes=COALESCE(excluded.notes, strategy_registry.notes)
            """,
            (
                strategy_id,
                strategy_name or strategy_id,
                stage,
                strategy_version,
                utc_now_iso(),
                notes,
            ),
        )
        self._db.portfolio.commit()
        return self.get_strategy(strategy_id) or {}

    def update_stage(
        self,
        strategy_id: str,
        new_stage: str,
        *,
        score: float | None = None,
        portfolio_fit_score: float | None = None,
        promoted_at: str | None = None,
        demoted_at: str | None = None,
        retired_at: str | None = None,
        strategy_version: str | None = None,
        core_strategy: int | None = None,
        diversification_score: float | None = None,
        recovery_score: float | None = None,
        challenge_score: float | None = None,
        stability_contribution_score: float | None = None,
        dd_reduction_score: float | None = None,
    ) -> None:
        fields = ["current_stage = ?"]
        params: list[Any] = [new_stage]
        optional = {
            "score": score,
            "portfolio_fit_score": portfolio_fit_score,
            "promoted_at": promoted_at,
            "demoted_at": demoted_at,
            "retired_at": retired_at,
            "strategy_version": strategy_version,
            "core_strategy": core_strategy,
            "diversification_score": diversification_score,
            "recovery_score": recovery_score,
            "challenge_score": challenge_score,
            "stability_contribution_score": stability_contribution_score,
            "dd_reduction_score": dd_reduction_score,
        }
        for column, value in optional.items():
            if value is not None:
                fields.append(f"{column} = ?")
                params.append(value)
        params.append(strategy_id)
        self._db.portfolio.execute(
            f"UPDATE strategy_registry SET {', '.join(fields)} WHERE strategy_id = ?",
            tuple(params),
        )
        self._db.portfolio.commit()

    def save_evaluation(self, strategy_id: str, metrics: dict[str, Any], *, stage: str | None = None) -> None:
        current = stage or metrics.get("current_stage")
        core = 1 if str(current).upper() == "CORE" else None
        self.update_stage(
            strategy_id,
            str(current),
            score=float(metrics.get("score") or 0.0),
            portfolio_fit_score=float(metrics.get("portfolio_fit_score") or 0.0),
            strategy_version=str(metrics.get("strategy_version") or "1.0"),
            core_strategy=core,
            diversification_score=_maybe_float(metrics.get("diversification_score")),
            recovery_score=_maybe_float(metrics.get("recovery_score")),
            challenge_score=_maybe_float(metrics.get("challenge_score")),
            stability_contribution_score=_maybe_float(metrics.get("stability_contribution_score")),
            dd_reduction_score=_maybe_float(metrics.get("dd_reduction_score")),
        )

    def log_transition(
        self,
        *,
        strategy_id: str,
        old_stage: str | None,
        new_stage: str,
        reason: str,
        score: float | None = None,
        portfolio_fit_score: float | None = None,
        pf: float | None = None,
        pass_rate: float | None = None,
        max_dd: float | None = None,
        oos_pf: float | None = None,
        strategy_version: str | None = None,
        diversification_score: float | None = None,
        recovery_score: float | None = None,
        challenge_score: float | None = None,
        stability_contribution_score: float | None = None,
        dd_reduction_score: float | None = None,
    ) -> int:
        cur = self._db.portfolio.execute(
            """
            INSERT INTO strategy_lifecycle_history (
                timestamp, strategy_id, old_stage, new_stage, reason,
                score, pf, pass_rate, max_dd, oos_pf, portfolio_fit_score,
                strategy_version, diversification_score, recovery_score,
                challenge_score, stability_contribution_score, dd_reduction_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                strategy_id,
                old_stage,
                new_stage,
                reason,
                score,
                pf,
                pass_rate,
                max_dd,
                oos_pf,
                portfolio_fit_score,
                strategy_version,
                diversification_score,
                recovery_score,
                challenge_score,
                stability_contribution_score,
                dd_reduction_score,
            ),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def list_history(
        self,
        strategy_id: str | None = None,
        *,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        if strategy_id:
            rows = self._db.query(
                """
                SELECT * FROM strategy_lifecycle_history
                WHERE strategy_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (strategy_id, limit),
            )
        else:
            rows = self._db.query(
                """
                SELECT * FROM strategy_lifecycle_history
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
        return [dict(row) for row in rows]

    def last_evaluation_timestamp(self) -> str | None:
        row = self._db.query(
            """
            SELECT timestamp FROM strategy_lifecycle_history
            WHERE reason LIKE 'weekly_evaluation%'
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            one=True,
        )
        return str(row["timestamp"]) if row else None

    def days_since_last_evaluation(self) -> float | None:
        ts = self.last_evaluation_timestamp()
        if not ts:
            return None
        try:
            last = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - last).total_seconds() / 86400.0
        except ValueError:
            return None


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

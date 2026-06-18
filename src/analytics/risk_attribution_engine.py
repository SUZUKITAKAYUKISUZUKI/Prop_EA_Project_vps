"""Portfolio Risk Attribution Engine v1 — multi-dimensional P&L / DD decomposition."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prae.metrics import max_drawdown_r, profit_factor
from prae.phase2_risk import compute_dd_contribution, find_worst_dd_window
from src.database.profile_migrations import DASHBOARD_STRATEGY_CODES, SETUP_TYPE_BY_STRATEGY_CODE

BAYES_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (0.0, 0.2, "0.0-0.2"),
    (0.2, 0.4, "0.2-0.4"),
    (0.4, 0.6, "0.4-0.6"),
    (0.6, 0.8, "0.6-0.8"),
    (0.8, 1.0, "0.8-1.0"),
)

SETUP_TO_CODE: dict[str, str] = {v: k for k, v in SETUP_TYPE_BY_STRATEGY_CODE.items()}


def _win_rate(r: pd.Series) -> float:
    if r.empty:
        return 0.0
    wins = (r > 0).sum()
    return round(float(wins / len(r) * 100.0), 2)


def _strategy_dd(r: pd.Series) -> float:
    return round(max_drawdown_r(r), 2)


def _session_from_hour(hour: int) -> str:
    if 0 <= hour < 8:
        return "ASIA"
    if 8 <= hour < 16:
        return "LONDON"
    return "NY"


def _infer_direction(row: pd.Series) -> str:
    for col in ("direction", "side", "trade_direction"):
        if col in row.index and pd.notna(row[col]):
            val = str(row[col]).strip().upper()
            if val in {"BUY", "LONG"}:
                return "BUY"
            if val in {"SELL", "SHORT"}:
                return "SELL"
    entry = row.get("entry_price")
    exit_p = row.get("exit_price")
    r_val = float(row.get("R", 0.0) or 0.0)
    if pd.notna(entry) and pd.notna(exit_p):
        delta = float(exit_p) - float(entry)
        if delta > 0:
            return "BUY" if r_val >= 0 else "SELL"
        if delta < 0:
            return "SELL" if r_val >= 0 else "BUY"
    return "BUY" if r_val >= 0 else "SELL"


@dataclass
class DrawdownPeriod:
    start: str
    end: str
    recovery_end: str | None
    max_dd: float
    peak_index: int
    trough_index: int
    recovery_index: int | None


class PortfolioRiskAttributionEngine:
    """Decompose portfolio P&L and drawdown across strategy, symbol, session, and profile axes."""

    def __init__(
        self,
        trades: pd.DataFrame,
        *,
        profile_id: str = "",
        allocation_weights: dict[str, float] | None = None,
        pass_rate: float | None = None,
    ) -> None:
        self.profile_id = profile_id or "UNKNOWN"
        self.allocation_weights = allocation_weights or {}
        self.pass_rate = pass_rate
        self.trades = self._prepare(trades)
        self._strategies = self._discover_strategies()

    @staticmethod
    def _prepare(trades: pd.DataFrame) -> pd.DataFrame:
        if trades.empty:
            return trades.copy()
        work = trades.copy()
        if "timestamp" not in work.columns:
            raise ValueError("Trade frame must contain 'timestamp'")
        work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
        work = work.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        if "R" not in work.columns:
            if "profit_r" in work.columns:
                work["R"] = pd.to_numeric(work["profit_r"], errors="coerce").fillna(0.0)
            else:
                raise ValueError("Trade frame must contain 'R' or 'profit_r'")
        work["R"] = pd.to_numeric(work["R"], errors="coerce").fillna(0.0)
        strat_col = "strategy" if "strategy" in work.columns else "setup_type"
        work["strategy"] = work[strat_col].astype(str).str.strip()
        work["setup_type"] = work.get("setup_type", work["strategy"]).astype(str).str.strip()
        work["pair"] = work.get("pair", work.get("symbol", "UNKNOWN")).astype(str).str.upper()
        work["strategy_code"] = work["setup_type"].map(SETUP_TO_CODE).fillna(work["strategy"])
        work["hour"] = work["timestamp"].dt.hour.astype(int)
        work["weekday"] = work["timestamp"].dt.day_name()
        work["month"] = work["timestamp"].dt.strftime("%Y-%m")
        work["session"] = work["hour"].map(_session_from_hour)
        if "direction" not in work.columns:
            work["direction"] = work.apply(_infer_direction, axis=1)
        else:
            work["direction"] = work["direction"].astype(str).str.upper()
        if "profile_id" not in work.columns:
            work["profile_id"] = ""
        work["profile_id"] = work["profile_id"].astype(str)
        if "bayes_probability" in work.columns:
            work["bayes_probability"] = pd.to_numeric(work["bayes_probability"], errors="coerce")
        return work

    def _discover_strategies(self) -> tuple[str, ...]:
        if self.trades.empty:
            return DASHBOARD_STRATEGY_CODES
        codes = sorted(self.trades["strategy_code"].dropna().unique().tolist())
        return tuple(codes) if codes else DASHBOARD_STRATEGY_CODES

    def _equity(self) -> np.ndarray:
        return np.cumsum(self.trades["R"].astype(float).to_numpy())

    def current_drawdown_r(self) -> float:
        eq = self._equity()
        if eq.size == 0:
            return 0.0
        peak = np.maximum.accumulate(eq)
        return round(float(eq[-1] - peak[-1]), 2)

    def portfolio_overview(self) -> dict[str, Any]:
        r = self.trades["R"]
        total_r = round(float(r.sum()), 2)
        pf = round(profit_factor(r), 4) if not r.empty else 0.0
        if pf == float("inf"):
            pf = 999.0
        wr = _win_rate(r)
        max_dd = round(max_drawdown_r(r), 2)
        current_dd = self.current_drawdown_r()
        return {
            "total_r": total_r,
            "pf": pf,
            "win_rate": wr,
            "max_dd": max_dd,
            "current_dd": current_dd,
            "trades": int(len(r)),
            "profile_id": self.profile_id,
        }

    def layer_strategy(self) -> dict[str, Any]:
        if self.trades.empty:
            return {"strategies": {}, "rankings": {}}

        dd_trades = self.trades.copy()
        dd_trades["strategy"] = dd_trades["strategy_code"]
        dd_df = compute_dd_contribution(dd_trades, self._strategies)
        dd_map = {
            str(row["Strategy"]): float(row["DD Contribution %"])
            for _, row in dd_df.iterrows()
        }
        grouped = self.trades.groupby("strategy_code", sort=False)
        strategies: dict[str, Any] = {}
        for code, part in grouped:
            r = part["R"]
            weight = self.allocation_weights.get(str(code), self.allocation_weights.get(part["setup_type"].iloc[0], 0.0))
            strategies[str(code)] = {
                "strategy_total_r": round(float(r.sum()), 2),
                "strategy_pf": round(profit_factor(r), 4) if profit_factor(r) != float("inf") else 999.0,
                "strategy_dd": _strategy_dd(r),
                "strategy_win_rate": _win_rate(r),
                "strategy_trades": int(len(r)),
                "strategy_drawdown_contribution": dd_map.get(str(code), 0.0),
                "strategy_risk_contribution": dd_map.get(str(code), 0.0),
                "allocated_weight": round(float(weight) * 100.0, 1) if weight else None,
            }

        by_r = sorted(strategies.items(), key=lambda kv: kv[1]["strategy_total_r"], reverse=True)
        by_dd = sorted(strategies.items(), key=lambda kv: kv[1]["strategy_drawdown_contribution"], reverse=True)
        return {
            "strategies": strategies,
            "rankings": {
                "top_profit_driver": by_r[0][0] if by_r else None,
                "worst_profit_driver": by_r[-1][0] if by_r else None,
                "top_dd_driver": by_dd[0][0] if by_dd else None,
                "most_efficient": max(
                    strategies.items(),
                    key=lambda kv: (kv[1]["strategy_pf"], kv[1]["strategy_total_r"]),
                )[0]
                if strategies
                else None,
            },
        }

    def layer_symbol(self) -> dict[str, Any]:
        if self.trades.empty:
            return {"symbols": {}, "best_symbol": None, "worst_symbol": None}

        symbols: dict[str, Any] = {}
        for sym, part in self.trades.groupby("pair", sort=False):
            r = part["R"]
            symbols[str(sym)] = {
                "symbol_total_r": round(float(r.sum()), 2),
                "symbol_dd": _strategy_dd(r),
                "symbol_pf": round(profit_factor(r), 4) if profit_factor(r) != float("inf") else 999.0,
                "symbol_winrate": _win_rate(r),
                "symbol_trades": int(len(r)),
            }
        ranked = sorted(symbols.items(), key=lambda kv: kv[1]["symbol_total_r"], reverse=True)
        return {
            "symbols": symbols,
            "best_symbol": ranked[0][0] if ranked else None,
            "worst_symbol": ranked[-1][0] if ranked else None,
        }

    def layer_direction(self) -> dict[str, Any]:
        if self.trades.empty:
            return {"buy_total_r": 0.0, "sell_total_r": 0.0, "buy_pf": 0.0, "sell_pf": 0.0}

        out: dict[str, Any] = {}
        for direction in ("BUY", "SELL"):
            part = self.trades[self.trades["direction"] == direction]
            r = part["R"]
            prefix = direction.lower()
            pf_val = profit_factor(r) if not r.empty else 0.0
            out[f"{prefix}_total_r"] = round(float(r.sum()), 2)
            out[f"{prefix}_pf"] = round(pf_val, 4) if pf_val != float("inf") else 999.0
            out[f"{prefix}_trades"] = int(len(r))
        return out

    def layer_session(self) -> dict[str, Any]:
        if self.trades.empty:
            return {"sessions": {}}

        sessions: dict[str, Any] = {}
        for session in ("ASIA", "LONDON", "NY"):
            part = self.trades[self.trades["session"] == session]
            r = part["R"]
            pf_val = profit_factor(r) if not r.empty else 0.0
            sessions[session] = {
                "session_total_r": round(float(r.sum()), 2),
                "session_pf": round(pf_val, 4) if pf_val != float("inf") else 999.0,
                "session_dd": _strategy_dd(r),
                "session_trades": int(len(r)),
            }
        return {"sessions": sessions}

    def layer_time(self) -> dict[str, Any]:
        if self.trades.empty:
            return {"hours": {}, "weekdays": {}, "months": {}}

        hour_stats = (
            self.trades.groupby("hour")["R"]
            .agg(total_r="sum", trades="count", pf=lambda s: profit_factor(s))
            .reset_index()
        )
        weekday_stats = (
            self.trades.groupby("weekday")["R"]
            .agg(total_r="sum", trades="count")
            .reset_index()
        )
        month_stats = (
            self.trades.groupby("month")["R"]
            .agg(total_r="sum", trades="count")
            .reset_index()
        )
        best_hour_row = hour_stats.sort_values("total_r", ascending=False).head(1)
        worst_hour_row = hour_stats.sort_values("total_r", ascending=True).head(1)
        best_wd = weekday_stats.sort_values("total_r", ascending=False).head(1)
        worst_wd = weekday_stats.sort_values("total_r", ascending=True).head(1)
        return {
            "hours": {
                int(row["hour"]): {
                    "total_r": round(float(row["total_r"]), 2),
                    "trades": int(row["trades"]),
                }
                for _, row in hour_stats.iterrows()
            },
            "weekdays": {
                str(row["weekday"]): {
                    "total_r": round(float(row["total_r"]), 2),
                    "trades": int(row["trades"]),
                }
                for _, row in weekday_stats.iterrows()
            },
            "months": {
                str(row["month"]): {
                    "total_r": round(float(row["total_r"]), 2),
                    "trades": int(row["trades"]),
                }
                for _, row in month_stats.iterrows()
            },
            "best_hour": int(best_hour_row.iloc[0]["hour"]) if not best_hour_row.empty else None,
            "worst_hour": int(worst_hour_row.iloc[0]["hour"]) if not worst_hour_row.empty else None,
            "best_weekday": str(best_wd.iloc[0]["weekday"]) if not best_wd.empty else None,
            "worst_weekday": str(worst_wd.iloc[0]["weekday"]) if not worst_wd.empty else None,
        }

    def layer_profile(self) -> dict[str, Any]:
        if self.trades.empty:
            return {"profiles": {}}

        profiles: dict[str, Any] = {}
        work = self.trades.copy()
        if (work["profile_id"] == "").all():
            work["profile_id"] = self.profile_id
        for pid, part in work.groupby("profile_id", sort=False):
            r = part["R"]
            pf_val = profit_factor(r)
            profiles[str(pid)] = {
                "profile_total_r": round(float(r.sum()), 2),
                "profile_pf": round(pf_val, 4) if pf_val != float("inf") else 999.0,
                "profile_dd": _strategy_dd(r),
                "profile_trades": int(len(r)),
                "profile_pass_rate": self.pass_rate,
            }
        return {"profiles": profiles}

    def layer_bayes(self) -> dict[str, Any]:
        if self.trades.empty or "bayes_probability" not in self.trades.columns:
            return {"buckets": {}, "available": False}

        work = self.trades.dropna(subset=["bayes_probability"]).copy()
        if work.empty:
            return {"buckets": {}, "available": False}

        buckets: dict[str, Any] = {}
        for low, high, label in BAYES_BUCKETS:
            if high >= 1.0:
                mask = (work["bayes_probability"] >= low) & (work["bayes_probability"] <= high)
            else:
                mask = (work["bayes_probability"] >= low) & (work["bayes_probability"] < high)
            part = work[mask]
            r = part["R"]
            pf_val = profit_factor(r) if not r.empty else 0.0
            buckets[label] = {
                "bucket_pf": round(pf_val, 4) if pf_val != float("inf") else 999.0,
                "bucket_wr": _win_rate(r),
                "bucket_total_r": round(float(r.sum()), 2),
                "bucket_dd": _strategy_dd(r),
                "bucket_trades": int(len(r)),
            }
        return {"buckets": buckets, "available": True}

    def layer_allocation(self) -> dict[str, Any]:
        if self.trades.empty:
            return {"allocations": []}

        total_r = float(self.trades["R"].sum()) or 1.0
        realized = self.trades.groupby("strategy_code")["R"].sum()
        rows: list[dict[str, Any]] = []
        for code in self._strategies:
            weight = float(self.allocation_weights.get(code, 0.0))
            r_val = float(realized.get(code, 0.0))
            share = r_val / total_r * 100.0 if total_r else 0.0
            weight_pct = weight * 100.0
            delta = share - weight_pct
            status = "balanced"
            if delta > 5.0:
                status = "overallocated"
            elif delta < -5.0:
                status = "underallocated"
            rows.append(
                {
                    "strategy": code,
                    "allocated_weight": round(weight_pct, 1),
                    "realized_r": round(r_val, 2),
                    "contribution_pct": round(share, 1),
                    "efficiency_delta": round(delta, 1),
                    "status": status,
                }
            )
        return {"allocations": rows}

    def _detect_dd_period(self, *, use_current: bool = False) -> DrawdownPeriod | None:
        if self.trades.empty:
            return None
        eq = self._equity()
        r = self.trades["R"].astype(float).to_numpy()
        peak_arr = np.maximum.accumulate(eq)
        dd = peak_arr - eq

        if use_current and eq[-1] < peak_arr[-1]:
            trough_i = len(eq) - 1
            peak_i = int(np.argmax(eq[: trough_i + 1]))
        else:
            trough_i = int(np.argmax(dd))
            peak_i = int(np.argmax(eq[: trough_i + 1])) if trough_i >= 0 else 0

        if trough_i <= peak_i:
            return None

        peak_eq = eq[peak_i]
        recovery_index: int | None = None
        for i in range(trough_i + 1, len(eq)):
            if eq[i] >= peak_eq:
                recovery_index = i
                break

        start_ts = self.trades.iloc[peak_i]["timestamp"]
        end_ts = self.trades.iloc[trough_i]["timestamp"]
        recovery_end = None
        if recovery_index is not None:
            recovery_end = self.trades.iloc[recovery_index]["timestamp"].strftime("%Y-%m-%d")

        return DrawdownPeriod(
            start=start_ts.strftime("%Y-%m-%d"),
            end=end_ts.strftime("%Y-%m-%d"),
            recovery_end=recovery_end,
            max_dd=round(float(-dd[trough_i]), 2),
            peak_index=peak_i,
            trough_index=trough_i,
            recovery_index=recovery_index,
        )

    def _contributors_in_window(
        self,
        window: pd.DataFrame,
        *,
        dimension: str,
    ) -> list[dict[str, Any]]:
        losses = window.loc[window["R"] < 0].copy()
        if losses.empty:
            return []
        grouped = losses.groupby(dimension)["R"].sum().sort_values()
        total_loss = abs(float(losses["R"].sum())) or 1.0
        contributors: list[dict[str, Any]] = []
        for key, val in grouped.items():
            contributors.append(
                {
                    "dimension": dimension,
                    "key": str(key),
                    "contribution_r": round(float(val), 2),
                    "contribution_pct": round(abs(float(val)) / total_loss * 100.0, 1),
                }
            )
        return sorted(contributors, key=lambda x: x["contribution_r"])

    def analyze_drawdown_period(self, *, use_current: bool = False) -> dict[str, Any]:
        period = self._detect_dd_period(use_current=use_current)
        if period is None:
            return {"dd_period": None, "contributors": [], "top_cause": None}

        window = self.trades.iloc[period.peak_index : period.trough_index + 1].copy()
        if (window["profile_id"] == "").all():
            window["profile_id"] = self.profile_id
        contributors: list[dict[str, Any]] = []
        for dim in ("strategy_code", "pair", "session", "profile_id"):
            contributors.extend(self._contributors_in_window(window, dimension=dim))

        top = contributors[0] if contributors else None
        top_cause = None
        if top:
            top_cause = f"{top['key']}"

        strategy_top = [c for c in contributors if c["dimension"] == "strategy_code"][:1]
        symbol_top = [c for c in contributors if c["dimension"] == "pair"][:1]
        session_top = [c for c in contributors if c["dimension"] == "session"][:1]
        if strategy_top and symbol_top and session_top:
            top_cause = f"{strategy_top[0]['key']} {symbol_top[0]['key']} {session_top[0]['key']}"

        return {
            "dd_period": {
                "start": period.start,
                "end": period.end,
                "recovery_end": period.recovery_end,
                "max_dd": period.max_dd,
            },
            "contributors": contributors[:20],
            "top_cause": top_cause,
        }

    def analyze_recovery(self) -> dict[str, Any]:
        period = self._detect_dd_period(use_current=False)
        if period is None or period.recovery_index is None:
            return {"recovery_leader": None, "recovery_contributors": []}

        recovery_slice = self.trades.iloc[period.trough_index + 1 : period.recovery_index + 1]
        gains = recovery_slice.loc[recovery_slice["R"] > 0]
        if gains.empty:
            return {"recovery_leader": None, "recovery_contributors": []}

        by_strategy = gains.groupby("strategy_code")["R"].sum().sort_values(ascending=False)
        total_gain = float(by_strategy.sum()) or 1.0
        contributors = [
            {
                "strategy": str(code),
                "recovery_r": round(float(val), 2),
                "recovery_pct": round(float(val) / total_gain * 100.0, 1),
            }
            for code, val in by_strategy.items()
        ]
        leader = contributors[0] if contributors else None
        return {
            "recovery_leader": leader,
            "recovery_contributors": contributors,
        }

    def run_full_report(self) -> dict[str, Any]:
        overview = self.portfolio_overview()
        strategy = self.layer_strategy()
        drawdown = self.analyze_drawdown_period(use_current=True)
        recovery = self.analyze_recovery()
        worst_dd = self.analyze_drawdown_period(use_current=False)

        return {
            "overview": overview,
            "strategy": strategy,
            "symbol": self.layer_symbol(),
            "direction": self.layer_direction(),
            "session": self.layer_session(),
            "time": self.layer_time(),
            "profile": self.layer_profile(),
            "bayes": self.layer_bayes(),
            "allocation": self.layer_allocation(),
            "drawdown": drawdown,
            "worst_drawdown": worst_dd,
            "recovery": recovery,
            "summary": {
                "current_dd": overview["current_dd"],
                "top_dd_driver": strategy["rankings"].get("top_dd_driver"),
                "top_profit_driver": strategy["rankings"].get("top_profit_driver"),
                "most_efficient_strategy": strategy["rankings"].get("most_efficient"),
                "top_cause": drawdown.get("top_cause") or worst_dd.get("top_cause"),
                "recovery_leader": (recovery.get("recovery_leader") or {}).get("strategy"),
            },
        }


def _save_chart(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def strategy_r_chart(report: dict[str, Any], path: Path) -> Path | None:
    strategies = report.get("strategy", {}).get("strategies", {})
    if not strategies:
        return None
    labels = list(strategies.keys())
    vals = [strategies[k]["strategy_total_r"] for k in labels]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#27ae60" if v >= 0 else "#c0392b" for v in vals]
    ax.bar(labels, vals, color=colors)
    ax.set_title("Strategy Total R")
    ax.set_ylabel("R")
    return _save_chart(fig, path)


def strategy_dd_chart(report: dict[str, Any], path: Path) -> Path | None:
    strategies = report.get("strategy", {}).get("strategies", {})
    if not strategies:
        return None
    labels = list(strategies.keys())
    vals = [strategies[k]["strategy_drawdown_contribution"] for k in labels]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, vals, color="#c0392b")
    ax.set_title("Strategy DD Contribution %")
    ax.set_ylabel("%")
    return _save_chart(fig, path)


def symbol_r_chart(report: dict[str, Any], path: Path) -> Path | None:
    symbols = report.get("symbol", {}).get("symbols", {})
    if not symbols:
        return None
    ranked = sorted(symbols.items(), key=lambda kv: kv[1]["symbol_total_r"], reverse=True)[:20]
    labels = [k for k, _ in ranked]
    vals = [v["symbol_total_r"] for _, v in ranked]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels, vals, color="#2980b9")
    ax.set_title("Symbol vs R")
    ax.set_xlabel("Total R")
    return _save_chart(fig, path)


def bayes_bucket_chart(report: dict[str, Any], path: Path) -> Path | None:
    buckets = report.get("bayes", {}).get("buckets", {})
    if not buckets:
        return None
    labels = list(buckets.keys())
    vals = [buckets[k]["bucket_pf"] for k in labels]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, vals, color="#8e44ad")
    ax.set_title("Bayes Bucket vs PF")
    ax.set_ylabel("PF")
    return _save_chart(fig, path)


def allocation_efficiency_chart(report: dict[str, Any], path: Path) -> Path | None:
    rows = report.get("allocation", {}).get("allocations", [])
    if not rows:
        return None
    labels = [r["strategy"] for r in rows]
    weights = [r["allocated_weight"] for r in rows]
    realized = [r["contribution_pct"] for r in rows]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, weights, width, label="Weight %", color="#3498db")
    ax.bar(x + width / 2, realized, width, label="Realized %", color="#e67e22")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("Allocation Efficiency")
    ax.legend()
    return _save_chart(fig, path)


def drawdown_contribution_chart(report: dict[str, Any], path: Path) -> Path | None:
    contributors = report.get("drawdown", {}).get("contributors", [])
    strategy_rows = [c for c in contributors if c["dimension"] == "strategy_code"][:8]
    if not strategy_rows:
        strategy_rows = contributors[:8]
    if not strategy_rows:
        return None
    labels = [c["key"] for c in strategy_rows]
    vals = [abs(c["contribution_r"]) for c in strategy_rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, vals, color="#c0392b")
    ax.set_title("Drawdown Contributors (|R|)")
    ax.set_ylabel("R")
    return _save_chart(fig, path)


def generate_all_charts(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_fns = {
        "strategy_r_chart": (strategy_r_chart, "strategy_r_chart.png"),
        "strategy_dd_chart": (strategy_dd_chart, "strategy_dd_chart.png"),
        "symbol_r_chart": (symbol_r_chart, "symbol_r_chart.png"),
        "bayes_bucket_chart": (bayes_bucket_chart, "bayes_bucket_chart.png"),
        "allocation_efficiency_chart": (allocation_efficiency_chart, "allocation_efficiency_chart.png"),
        "drawdown_contribution_chart": (drawdown_contribution_chart, "drawdown_contribution_chart.png"),
    }
    out: dict[str, str] = {}
    for key, (fn, filename) in chart_fns.items():
        result = fn(report, output_dir / filename)
        if isinstance(result, Path) and result.exists():
            out[key] = str(result)
    return out

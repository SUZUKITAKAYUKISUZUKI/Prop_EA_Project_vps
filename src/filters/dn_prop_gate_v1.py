"""
DiNapoli (DN) — Prop Gate V1: EV-tier sizing patterns + two-phase prop simulation.

Pattern B + base_risk 0.60% is the adopted production configuration (2026-06).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from audit.risk_manager import STARTING_EQUITY

BASE_RISK_PCT = 1.0
DN_PROP_GATE_BASE_RISK_PCT = float(os.getenv("DN_PROP_GATE_BASE_RISK_PCT", "0.006"))
DN_PROP_GATE_PATTERN = "B"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "backtest_results" / "models" / "dn_prop_gate_v1.json"
DAILY_DD_LIMIT_PCT = 4.5
TOTAL_DD_LIMIT_PCT = 8.5
PHASE1_TARGET_PCT = 8.0
PHASE2_TARGET_PCT = 6.0
WINDOW_DAYS = 90
WINDOW_STEP_DAYS = 7

TIER_ORDER: tuple[str, ...] = ("Top5", "Top10", "Top20", "Middle", "Low")

SIZING_PATTERNS: dict[str, dict[str, float]] = {
    "A": {"Top5": 1.00, "Top10": 1.00, "Top20": 1.00, "Middle": 1.00, "Low": 1.00},
    "B": {"Top5": 2.00, "Top10": 1.50, "Top20": 1.00, "Middle": 0.75, "Low": 0.50},
    "C": {"Top5": 2.50, "Top10": 2.00, "Top20": 1.25, "Middle": 0.75, "Low": 0.25},
    "D": {"Top5": 3.00, "Top10": 2.00, "Top20": 1.00, "Middle": 0.50, "Low": 0.10},
}


def is_dn_prop_gate_enabled() -> bool:
    explicit = os.getenv("DN_PROP_GATE", "").strip().lower()
    if explicit in ("0", "false", "no", "off"):
        return False
    if explicit in ("1", "true", "yes", "on"):
        return True
    from strategies.dinapoli import is_dinapoli_defense_mode

    return is_dinapoli_defense_mode()


def dn_prop_gate_base_risk_frac() -> float:
    return DN_PROP_GATE_BASE_RISK_PCT


def pattern_b_lot_multiplier(ev_rank_v2: float) -> float:
    return tier_multiplier(float(ev_rank_v2), SIZING_PATTERNS["B"])


def evaluate_dn_prop_gate_sizing(
    row: Mapping[str, Any],
    *,
    ev_rank_v2: float | None = None,
) -> dict[str, Any]:
    rank = float(ev_rank_v2 if ev_rank_v2 is not None else row.get("ev_rank_v2", row.get("ev_rank", 0.0)))
    tier = assign_ev_tier(rank)
    mult = pattern_b_lot_multiplier(rank)
    return {
        "ev_rank_v2": round(rank, 6),
        "tier": tier,
        "lot_multiplier": mult,
        "pattern": DN_PROP_GATE_PATTERN,
        "base_risk_pct": DN_PROP_GATE_BASE_RISK_PCT,
    }


def save_adopted_config(path: Path | str | None = None) -> Path:
    path = Path(path or DEFAULT_CONFIG_PATH)
    payload = {
        "version": "dn_prop_gate_v1",
        "adopted": True,
        "pattern": DN_PROP_GATE_PATTERN,
        "base_risk_pct": DN_PROP_GATE_BASE_RISK_PCT,
        "pipeline": {
            "l0_l2": True,
            "l35_generic_bayes": False,
            "l35_dn_prop_gate": True,
            "l4_gemini": False,
            "l45_l6": True,
            "safety_brakes": [
                "profit_cushion",
                "twin_brake",
                "dd_throttling",
            ],
        },
        "sizing": SIZING_PATTERNS["B"],
        "tiers": {
            "Top5": ">= 0.95",
            "Top10": "0.90 <= x < 0.95",
            "Top20": "0.80 <= x < 0.90",
            "Middle": "0.50 <= x < 0.80",
            "Low": "< 0.50",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def assign_ev_tier(ev_rank_v2: float) -> str:
    r = float(ev_rank_v2)
    if r >= 0.95:
        return "Top5"
    if r >= 0.90:
        return "Top10"
    if r >= 0.80:
        return "Top20"
    if r >= 0.50:
        return "Middle"
    return "Low"


def tier_multiplier(ev_rank_v2: float, pattern: Mapping[str, float]) -> float:
    return float(pattern.get(assign_ev_tier(ev_rank_v2), 1.0))


def apply_sizing(df: pd.DataFrame, pattern: Mapping[str, float]) -> pd.DataFrame:
    out = df.copy()
    out["ev_tier"] = out["ev_rank_v2"].map(assign_ev_tier)
    out["tier_mult"] = out["ev_rank_v2"].map(lambda r: tier_multiplier(r, pattern))
    out["scaled_r"] = pd.to_numeric(out["result_r"], errors="coerce").fillna(0.0) * out["tier_mult"]
    return out


def _pf(series: pd.Series) -> float:
    r = pd.to_numeric(series, errors="coerce").fillna(0.0)
    wins = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    if losses <= 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def max_consecutive(series: pd.Series, *, wins: bool) -> int:
    best = 0
    cur = 0
    for val in pd.to_numeric(series, errors="coerce").fillna(0.0):
        hit = val > 0 if wins else val < 0
        if hit:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def cohort_stats(sub: pd.DataFrame) -> dict[str, Any]:
    if sub.empty:
        return {
            "n": 0,
            "wr": 0.0,
            "pf": 0.0,
            "total_r": 0.0,
            "avg_r": 0.0,
            "max_consec_loss": 0,
            "max_consec_win": 0,
        }
    r = pd.to_numeric(sub["result_r"], errors="coerce").fillna(0.0)
    win_flag = r > 0
    return {
        "n": len(sub),
        "wr": float(win_flag.mean() * 100),
        "pf": _pf(r),
        "total_r": float(r.sum()),
        "avg_r": float(r.mean()),
        "max_consec_loss": max_consecutive(r, wins=False),
        "max_consec_win": max_consecutive(r, wins=True),
    }


def equity_curve_pct_returns(scaled_r: Sequence[float], base_risk_pct: float = BASE_RISK_PCT) -> tuple[np.ndarray, np.ndarray]:
    equity = np.empty(len(scaled_r) + 1, dtype=np.float64)
    equity[0] = STARTING_EQUITY
    rets = np.empty(len(scaled_r), dtype=np.float64)
    for i, r in enumerate(scaled_r):
        pct = base_risk_pct / 100.0 * float(r)
        rets[i] = pct
        equity[i + 1] = equity[i] * (1.0 + pct)
    return equity, rets


def max_drawdown_pct(equity: np.ndarray) -> float:
    peak = equity[0]
    max_dd = 0.0
    for val in equity:
        peak = max(peak, val)
        if peak > 0:
            max_dd = max(max_dd, (peak - val) / peak * 100.0)
    return float(max_dd)


def ulcer_index(equity: np.ndarray) -> float:
    peak = equity[0]
    dd_sq: list[float] = []
    for val in equity:
        peak = max(peak, val)
        dd = (peak - val) / peak * 100.0 if peak > 0 else 0.0
        dd_sq.append(dd * dd)
    return float(np.sqrt(np.mean(dd_sq))) if dd_sq else 0.0


def dd_metrics(scaled_r: Sequence[float], *, span_days: int | None = None) -> dict[str, float]:
    if len(scaled_r) == 0:
        return {
            "max_dd_pct": 0.0,
            "recovery_factor": 0.0,
            "ulcer_index": 0.0,
            "calmar_ratio": 0.0,
            "final_equity": STARTING_EQUITY,
        }
    equity, _ = equity_curve_pct_returns(scaled_r)
    max_dd = max_drawdown_pct(equity)
    total_r = float(np.sum(scaled_r))
    recovery = total_r / max_dd if max_dd > 1e-9 else float("inf")
    ulcer = ulcer_index(equity)
    years = (span_days or 365) / 365.25
    ann = ((equity[-1] / STARTING_EQUITY) ** (1.0 / max(years, 1e-9)) - 1.0) * 100.0
    calmar = ann / max_dd if max_dd > 1e-9 else float("inf")
    return {
        "max_dd_pct": max_dd,
        "recovery_factor": float(recovery) if np.isfinite(recovery) else 999.0,
        "ulcer_index": ulcer,
        "calmar_ratio": float(calmar) if np.isfinite(calmar) else 999.0,
        "final_equity": float(equity[-1]),
    }


def sharpe_sortino(returns: np.ndarray, *, periods_per_year: float = 252.0) -> tuple[float, float]:
    if len(returns) < 2:
        return 0.0, 0.0
    mu = float(np.mean(returns))
    sd = float(np.std(returns, ddof=1))
    sharpe = (mu / sd * np.sqrt(periods_per_year)) if sd > 1e-12 else 0.0
    downside = returns[returns < 0]
    dsd = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    sortino = (mu / dsd * np.sqrt(periods_per_year)) if dsd > 1e-12 else 0.0
    return float(sharpe), float(sortino)


def pattern_metrics(df: pd.DataFrame, pattern: Mapping[str, float]) -> dict[str, Any]:
    sized = apply_sizing(df, pattern)
    scaled = sized["scaled_r"].to_numpy(dtype=np.float64)
    equity, rets = equity_curve_pct_returns(scaled)
    span = 0
    if "timestamp" in sized.columns and len(sized) > 1:
        span = max(1, (sized["timestamp"].max() - sized["timestamp"].min()).days)
    trades_per_year = len(scaled) / max(span / 365.25, 1e-9)
    sharpe, sortino = sharpe_sortino(rets, periods_per_year=max(trades_per_year, 1.0))
    dd = dd_metrics(scaled, span_days=span)
    return {
        "pf": _pf(sized["scaled_r"]),
        "total_r": float(np.sum(scaled)),
        "max_dd_pct": dd["max_dd_pct"],
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": dd["calmar_ratio"],
        "trades": len(sized),
    }


def generate_window_starts(timestamps: pd.Series, window_days: int, step_days: int) -> list[pd.Timestamp]:
    if timestamps.empty:
        return []
    data_start = timestamps.min().normalize()
    data_end = timestamps.max().normalize()
    last_start = data_end - pd.Timedelta(days=window_days - 1)
    if last_start < data_start:
        last_start = data_start
    return list(pd.date_range(data_start, last_start, freq=f"{step_days}D"))


@dataclass(frozen=True)
class PropWindowResult:
    outcome: str
    total_days: float | None
    max_dd_pct: float


def simulate_two_phase_window(
    df: pd.DataFrame,
    window_start: pd.Timestamp,
    *,
    base_risk_pct: float = BASE_RISK_PCT,
    daily_dd_limit_pct: float = DAILY_DD_LIMIT_PCT,
    total_dd_limit_pct: float = TOTAL_DD_LIMIT_PCT,
) -> PropWindowResult:
    """Phase1 +8% reset, Phase2 +6% pass. Uses precomputed scaled_r column."""
    if df.empty or "scaled_r" not in df.columns:
        return PropWindowResult("timeout", None, 0.0)

    start = pd.Timestamp(window_start).normalize()
    equity = float(STARTING_EQUITY)
    phase_start = equity
    peak = equity
    phase = 1
    target_pct = PHASE1_TARGET_PCT
    target_equity = equity * (1.0 + target_pct / 100.0)
    max_dd = 0.0

    current_date: str | None = None
    daily_start = equity
    day_min = equity

    for row in df.itertuples(index=False):
        ts = pd.Timestamp(row.timestamp)
        if ts.normalize() < start:
            continue
        if (ts.normalize() - start).days >= WINDOW_DAYS:
            break

        date_key = ts.strftime("%Y-%m-%d")
        if date_key != current_date:
            current_date = date_key
            daily_start = equity
            day_min = equity

        pct = base_risk_pct / 100.0 * float(row.scaled_r)
        equity *= 1.0 + pct
        day_min = min(day_min, equity)
        peak = max(peak, equity)
        total_dd = (peak - equity) / peak * 100.0 if peak > 0 else 0.0
        max_dd = max(max_dd, total_dd)
        daily_dd = (daily_start - day_min) / daily_start * 100.0 if daily_start > 0 else 0.0

        if daily_dd >= daily_dd_limit_pct or total_dd >= total_dd_limit_pct:
            return PropWindowResult("fail", None, max_dd)

        if equity >= target_equity:
            day_count = float(max(0, (ts.normalize() - start).days))
            if phase == 1:
                phase = 2
                equity = float(STARTING_EQUITY)
                phase_start = equity
                peak = equity
                target_pct = PHASE2_TARGET_PCT
                target_equity = equity * (1.0 + target_pct / 100.0)
                current_date = None
                continue
            return PropWindowResult("pass", day_count, max_dd)

    return PropWindowResult("timeout", None, max_dd)


@dataclass(frozen=True)
class PropSimSummary:
    total_windows: int
    pass_count: int
    fail_count: int
    timeout_count: int
    pass_rate: float
    avg_pass_days: float
    median_pass_days: float
    p25_pass_days: float
    p75_pass_days: float
    max_dd_pct: float


def simulate_prop_windows(df: pd.DataFrame) -> PropSimSummary:
    if df.empty:
        return PropSimSummary(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    starts = generate_window_starts(df["timestamp"], WINDOW_DAYS, WINDOW_STEP_DAYS)
    if not starts:
        starts = [df["timestamp"].min().normalize()]

    pass_days: list[float] = []
    max_dds: list[float] = []
    pass_n = fail_n = timeout_n = 0

    for w_start in starts:
        w_end = w_start + pd.Timedelta(days=WINDOW_DAYS)
        w_df = df[(df["timestamp"] >= w_start) & (df["timestamp"] < w_end)].copy()
        res = simulate_two_phase_window(w_df, w_start)
        max_dds.append(res.max_dd_pct)
        if res.outcome == "pass":
            pass_n += 1
            if res.total_days is not None:
                pass_days.append(res.total_days)
        elif res.outcome == "fail":
            fail_n += 1
        else:
            timeout_n += 1

    total = len(starts)
    arr = np.asarray(pass_days, dtype=np.float64)
    return PropSimSummary(
        total_windows=total,
        pass_count=pass_n,
        fail_count=fail_n,
        timeout_count=timeout_n,
        pass_rate=(pass_n / total * 100.0) if total else 0.0,
        avg_pass_days=float(np.mean(arr)) if len(arr) else 0.0,
        median_pass_days=float(np.median(arr)) if len(arr) else 0.0,
        p25_pass_days=float(np.percentile(arr, 25)) if len(arr) else 0.0,
        p75_pass_days=float(np.percentile(arr, 75)) if len(arr) else 0.0,
        max_dd_pct=float(np.max(max_dds)) if max_dds else 0.0,
    )


def rank_gate_candidates(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Priority: pass_rate desc, avg_pass_days asc, max_dd asc, pf desc, total_r desc."""

    def key(row: dict[str, Any]) -> tuple:
        return (
            -float(row.get("pass_rate", 0.0)),
            float(row.get("avg_pass_days", 999.0)),
            float(row.get("max_dd_pct", 999.0)),
            -float(row.get("pf", 0.0)),
            -float(row.get("total_r", 0.0)),
        )

    return sorted(candidates, key=key)


def gate_success_flags(summary: PropSimSummary, metrics: dict[str, Any]) -> dict[str, bool]:
    return {
        "pass_rate_min": summary.pass_rate > 80.0,
        "avg_days_min": summary.avg_pass_days < 10.0,
        "max_dd_min": metrics.get("max_dd_pct", 999.0) < 5.0,
        "pass_rate_ideal": summary.pass_rate > 90.0,
        "avg_days_ideal": summary.avg_pass_days < 5.0,
        "max_dd_ideal": metrics.get("max_dd_pct", 999.0) < 4.0,
    }

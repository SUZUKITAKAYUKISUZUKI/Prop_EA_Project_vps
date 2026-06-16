"""SMRS Phase 4 — position sizing models (frozen signal + frozen Bayes)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from audit.risk_manager import CHALLENGE_PROFIT_TARGET_PCT, STARTING_EQUITY
from prop_audit_reporter import (
    DAILY_DD_LIMIT_PCT,
    TOTAL_DD_LIMIT_PCT,
    WINDOW_DAYS,
    _apply_dynamic_defense_multipliers,
    _simulated_daily_dd_remaining_percent,
)
from strategies.smrs_bayes import profit_factor

os.environ.setdefault("PROFIT_CUSHION_ENABLED", "1")
os.environ.setdefault("TWIN_BRAKE_ENABLED", "1")
os.environ.setdefault("DD_THROTTLING_ENABLED", "1")

PHASE2_TARGET_PCT = 6.0
BASE_RISK_PCT = 1.0
KELLY_LITE_MIN_HISTORY = 30
KELLY_LITE_FRACTION = 0.25
KELLY_LITE_MIN_R = 0.25
KELLY_LITE_MAX_R = 2.0
MC_TRIALS = 1000
MC_SEED = 42

WFT_IS_MONTHS = 12
WFT_OOS_MONTHS = 3
WFT_STEP_MONTHS = 3
WFT_START = pd.Timestamp("2023-01-02")
WFT_END = pd.Timestamp("2026-04-30 23:59:59")

PORTFOLIO_BASELINE_PF = 2.963
PORTFOLIO_BASELINE_TOTAL_R = 1303.75
PORTFOLIO_BASELINE_TRADES = 2145
PORTFOLIO_BASELINE_MAX_DD_PCT = 2.52

SizingFn = Callable[[float], float]


@dataclass(frozen=True)
class SizingModelSpec:
    model_id: str
    name: str
    description: str


SIZING_MODELS: dict[str, SizingModelSpec] = {
    "A": SizingModelSpec("A", "Model A", "Conservative tier"),
    "B": SizingModelSpec("B", "Model B", "VAMR-style tier"),
    "C": SizingModelSpec("C", "Model C", "Aggressive tier"),
    "D": SizingModelSpec("D", "Model D", "Linear scaling"),
    "E": SizingModelSpec("E", "Model E", "SQRT scaling"),
    "F": SizingModelSpec("F", "Model F", "Kelly-lite rolling OOS"),
    "G": SizingModelSpec("G", "Model G", "No-skip tier"),
    "G2": SizingModelSpec("G2", "Model G2", "Graduated low-prob tier"),
}

BASELINE_MODEL = "Baseline"
PRODUCTION_SIZING_MODEL = "A"


def max_drawdown_r(result_r: pd.Series | np.ndarray) -> float:
    r = pd.to_numeric(result_r, errors="coerce").fillna(0.0)
    if r.empty:
        return 0.0
    equity = r.cumsum()
    peak = equity.cummax()
    dd = peak - equity
    return round(float(dd.max()), 2)


def sharpe_r(result_r: pd.Series | np.ndarray) -> float:
    r = pd.to_numeric(result_r, errors="coerce").dropna()
    if len(r) < 2:
        return 0.0
    std = float(r.std(ddof=1))
    if std <= 0:
        return 0.0
    return round(float(r.mean() / std), 4)


def tier_multiplier(probability: float, tiers: Sequence[tuple[float, float, float]]) -> float:
    p = float(probability)
    for low, high, mult in tiers:
        if low <= p < high or (high >= 1.0 and p >= low):
            return mult
    return 0.0


def model_a_multiplier(p: float) -> float:
    return tier_multiplier(
        p,
        (
            (0.0, 0.70, 0.0),
            (0.70, 0.80, 0.50),
            (0.80, 0.90, 1.00),
            (0.90, 1.000001, 1.50),
        ),
    )


def model_b_multiplier(p: float) -> float:
    return tier_multiplier(
        p,
        (
            (0.0, 0.70, 0.0),
            (0.70, 0.80, 0.50),
            (0.80, 0.90, 1.25),
            (0.90, 1.000001, 2.00),
        ),
    )


def model_c_multiplier(p: float) -> float:
    return tier_multiplier(
        p,
        (
            (0.0, 0.70, 0.0),
            (0.70, 0.80, 0.75),
            (0.80, 0.90, 1.50),
            (0.90, 1.000001, 2.50),
        ),
    )


def model_d_multiplier(p: float) -> float:
    return float(0.25 + 2.25 * float(p))


def model_e_multiplier(p: float) -> float:
    p = float(np.clip(p, 0.0, 1.0))
    return float(min(2.0, np.sqrt(p) * 2.0))


def model_g_multiplier(p: float) -> float:
    return tier_multiplier(
        p,
        (
            (0.0, 0.70, 0.25),
            (0.70, 0.80, 0.50),
            (0.80, 0.90, 1.25),
            (0.90, 1.000001, 2.00),
        ),
    )


def model_g2_multiplier(p: float) -> float:
    return tier_multiplier(
        p,
        (
            (0.0, 0.60, 0.10),
            (0.60, 0.70, 0.25),
            (0.70, 0.80, 0.50),
            (0.80, 0.90, 1.25),
            (0.90, 1.000001, 2.00),
        ),
    )


def kelly_lite_from_history(history: pd.DataFrame) -> float:
    if len(history) < KELLY_LITE_MIN_HISTORY:
        return 0.0
    r = pd.to_numeric(history["trade_r"], errors="coerce").fillna(0.0)
    wins = r[r > 0]
    losses = r[r < 0]
    if wins.empty or losses.empty:
        return 0.0
    p = float((r > 0).mean())
    q = 1.0 - p
    avg_win = float(wins.mean())
    avg_loss = float(abs(losses.mean()))
    if avg_loss <= 0:
        return 0.0
    b = avg_win / avg_loss
    kelly = p - (q / b)
    if kelly <= 0:
        return 0.0
    size = KELLY_LITE_FRACTION * kelly
    return float(max(KELLY_LITE_MIN_R, min(KELLY_LITE_MAX_R, size)))


def apply_static_multiplier(df: pd.DataFrame, fn: SizingFn) -> pd.Series:
    return df["bayes_probability"].map(fn).astype(float)


def apply_kelly_lite_multipliers(df: pd.DataFrame) -> pd.Series:
    mults: list[float] = []
    for idx in range(len(df)):
        history = df.iloc[:idx]
        mults.append(kelly_lite_from_history(history))
    return pd.Series(mults, index=df.index, dtype=float)


def apply_sizing_model(df: pd.DataFrame, model_id: str) -> pd.DataFrame:
    out = df.copy()
    out["trade_r"] = pd.to_numeric(out.get("trade_r", out.get("result_r")), errors="coerce").fillna(0.0)
    if model_id == BASELINE_MODEL:
        out["size_multiplier"] = 1.0
    elif model_id == "A":
        out["size_multiplier"] = apply_static_multiplier(out, model_a_multiplier)
    elif model_id == "B":
        out["size_multiplier"] = apply_static_multiplier(out, model_b_multiplier)
    elif model_id == "C":
        out["size_multiplier"] = apply_static_multiplier(out, model_c_multiplier)
    elif model_id == "D":
        out["size_multiplier"] = apply_static_multiplier(out, model_d_multiplier)
    elif model_id == "E":
        out["size_multiplier"] = apply_static_multiplier(out, model_e_multiplier)
    elif model_id == "F":
        out["size_multiplier"] = apply_kelly_lite_multipliers(out)
    elif model_id == "G":
        out["size_multiplier"] = apply_static_multiplier(out, model_g_multiplier)
    elif model_id == "G2":
        out["size_multiplier"] = apply_static_multiplier(out, model_g2_multiplier)
    else:
        raise ValueError(f"Unknown sizing model: {model_id}")
    out["sized_r"] = out["trade_r"].astype(float) * out["size_multiplier"]
    out["executed"] = out["size_multiplier"] > 0
    return out


def recovery_factor(total_r: float, max_dd_r: float) -> float:
    if max_dd_r <= 0:
        return float("inf") if total_r > 0 else 0.0
    return float(total_r / max_dd_r)


def max_losing_streak(trade_r: pd.Series) -> int:
    best = 0
    cur = 0
    for val in pd.to_numeric(trade_r, errors="coerce").fillna(0.0):
        if val < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def summarize_sized_trades(df: pd.DataFrame) -> dict[str, Any]:
    active = df[df["size_multiplier"] > 0].copy()
    if active.empty:
        return {
            "trades": 0,
            "skipped": int(len(df)),
            "skip_pct": round(100.0 * len(df) / max(len(df), 1), 2),
            "pf": 0.0,
            "avg_r": 0.0,
            "total_r": 0.0,
            "max_dd_r": 0.0,
            "sharpe": 0.0,
            "recovery_factor": 0.0,
            "max_losing_streak": 0,
        }
    r = active["sized_r"].astype(float)
    total_r = float(r.sum())
    max_dd = max_drawdown_r(r)
    pf = profit_factor(r)
    return {
        "trades": int(len(active)),
        "skipped": int(len(df) - len(active)),
        "skip_pct": round((len(df) - len(active)) / len(df) * 100.0, 2),
        "pf": round(float(pf), 4) if np.isfinite(pf) else pf,
        "avg_r": round(float(r.mean()), 4),
        "total_r": round(total_r, 2),
        "max_dd_r": max_dd,
        "sharpe": sharpe_r(r),
        "recovery_factor": round(recovery_factor(total_r, max_dd), 4)
        if np.isfinite(recovery_factor(total_r, max_dd))
        else recovery_factor(total_r, max_dd),
        "max_losing_streak": max_losing_streak(r),
    }


def assign_wft_window(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    if "wft_window_id" in out.columns and out["wft_window_id"].notna().any():
        out["wft_window_id"] = pd.to_numeric(out["wft_window_id"], errors="coerce").fillna(-1).astype(int)
        return out
    from walkforward_runner import iter_wft_windows

    windows = list(
        iter_wft_windows(
            WFT_START,
            WFT_END,
            is_months=WFT_IS_MONTHS,
            oos_months=WFT_OOS_MONTHS,
            step_months=WFT_STEP_MONTHS,
        )
    )
    out["wft_window_id"] = -1
    for window in windows:
        mask = (out["timestamp"] >= window.oos_start) & (out["timestamp"] < window.oos_end)
        out.loc[mask, "wft_window_id"] = window.window_id
    return out


def wft_oos_validation(df: pd.DataFrame, model_id: str) -> dict[str, Any]:
    work = assign_wft_window(df)
    oos = work[work["wft_window_id"] >= 0].copy()
    sized = apply_sizing_model(oos, model_id)
    window_rows: list[dict[str, Any]] = []
    for window_id, sub in sized.groupby("wft_window_id"):
        stats = summarize_sized_trades(sub)
        window_rows.append(
            {
                "window_id": int(window_id),
                "trades": stats["trades"],
                "pf": stats["pf"],
                "avg_r": stats["avg_r"],
                "total_r": stats["total_r"],
                "max_dd_r": stats["max_dd_r"],
            }
        )
    pfs = [float(r["pf"]) for r in window_rows if r["trades"] > 0 and np.isfinite(float(r["pf"]))]
    dds = [float(r["max_dd_r"]) for r in window_rows if r["trades"] > 0]
    positive = sum(1 for pf in pfs if pf >= 1.0)
    pf_std = float(np.std(pfs, ddof=1)) if len(pfs) > 1 else 0.0
    worst_pf = min(pfs) if pfs else 0.0
    worst_dd = max(dds) if dds else 0.0
    stability = (positive / len(pfs) * 100.0 / max(pf_std + 1.0, 1.0)) if pfs else 0.0
    return {
        "model_id": model_id,
        "windows": len(window_rows),
        "mean_pf": round(float(np.mean(pfs)), 4) if pfs else 0.0,
        "median_pf": round(float(np.median(pfs)), 4) if pfs else 0.0,
        "pf_std": round(pf_std, 4),
        "positive_windows_pct": round(positive / len(pfs) * 100.0, 2) if pfs else 0.0,
        "worst_window_pf": round(worst_pf, 4),
        "mean_dd_r": round(float(np.mean(dds)), 4) if dds else 0.0,
        "worst_dd_r": round(worst_dd, 4),
        "worst_window_dd_r": round(worst_dd, 4),
        "stability_score": round(stability, 4),
        "window_rows": window_rows,
    }


def drawdown_audit(df: pd.DataFrame, model_id: str) -> dict[str, Any]:
    sized = apply_sizing_model(df, model_id)
    active = sized[sized["size_multiplier"] > 0].copy()
    if active.empty:
        return {
            "model_id": model_id,
            "worst_losing_streak": 0,
            "worst_equity_drawdown_r": 0.0,
            "largest_single_day_drawdown_r": 0.0,
            "largest_rolling_drawdown_r": 0.0,
        }
    active = active.sort_values("timestamp")
    r = active["sized_r"].astype(float)
    equity = r.cumsum()
    peak = equity.cummax()
    rolling_dd = peak - equity

    active["date"] = pd.to_datetime(active["timestamp"]).dt.date
    daily = active.groupby("date")["sized_r"].sum()
    daily_dd = float(-daily.min()) if len(daily) else 0.0

    return {
        "model_id": model_id,
        "worst_losing_streak": max_losing_streak(r),
        "worst_equity_drawdown_r": round(float(rolling_dd.max()), 2),
        "largest_single_day_drawdown_r": round(daily_dd, 2),
        "largest_rolling_drawdown_r": round(float(rolling_dd.max()), 2),
    }


def _prepare_prop_trades(df: pd.DataFrame, model_id: str) -> pd.DataFrame:
    sized = apply_sizing_model(df, model_id)
    active = sized[sized["size_multiplier"] > 0].copy()
    active["profit_r"] = active["sized_r"].astype(float)
    active["lot_factor"] = active["size_multiplier"].astype(float)
    return active.sort_values("timestamp").reset_index(drop=True)


def simulate_two_phase_prop(
    df: pd.DataFrame,
    *,
    window_start: pd.Timestamp | None = None,
    max_calendar_days: int | None = WINDOW_DAYS,
) -> dict[str, Any]:
    if df.empty:
        return {"outcome": "timeout", "total_days": None, "max_dd_pct": 0.0}

    start = pd.Timestamp(window_start).normalize() if window_start is not None else df["timestamp"].min().normalize()
    equity = float(STARTING_EQUITY)
    phase_start = equity
    peak = equity
    phase = 1
    target_pct = CHALLENGE_PROFIT_TARGET_PCT
    target_equity = equity * (1.0 + target_pct / 100.0)
    max_dd = 0.0

    current_date: str | None = None
    daily_start = equity
    day_min = equity

    for row in df.itertuples(index=False):
        ts = pd.Timestamp(row.timestamp)
        if ts.normalize() < start:
            continue
        if max_calendar_days is not None and (ts.normalize() - start).days > max_calendar_days:
            break

        date_key = ts.strftime("%Y-%m-%d")
        if date_key != current_date:
            current_date = date_key
            daily_start = equity
            day_min = equity

        daily_rem = _simulated_daily_dd_remaining_percent(daily_start, day_min)
        lot = _apply_dynamic_defense_multipliers(
            float(row.lot_factor),
            equity,
            phase_start,
            daily_rem,
            "challenge",
        )
        equity *= 1.0 + (BASE_RISK_PCT / 100.0) * lot * float(row.profit_r)
        day_min = min(day_min, equity)
        peak = max(peak, equity)
        total_dd = (peak - equity) / peak * 100.0 if peak > 0 else 0.0
        max_dd = max(max_dd, total_dd)
        daily_dd = (daily_start - day_min) / daily_start * 100.0 if daily_start > 0 else 0.0

        if daily_dd >= DAILY_DD_LIMIT_PCT or total_dd >= TOTAL_DD_LIMIT_PCT:
            return {"outcome": "fail", "total_days": None, "max_dd_pct": max_dd}

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
            return {"outcome": "pass", "total_days": day_count, "max_dd_pct": max_dd}

    return {"outcome": "timeout", "total_days": None, "max_dd_pct": max_dd}


def monte_carlo_prop_simulation(
    df: pd.DataFrame,
    model_id: str,
    *,
    trials: int = MC_TRIALS,
    seed: int = MC_SEED,
) -> dict[str, Any]:
    trades = _prepare_prop_trades(df, model_id)
    if trades.empty:
        return {
            "pass_rate": 0.0,
            "failure_rate": 0.0,
            "avg_pass_days": 0.0,
            "median_pass_days": 0.0,
            "trials": 0,
        }

    rng = np.random.default_rng(seed)
    n = len(trades)
    pass_days: list[float] = []
    pass_n = fail_n = timeout_n = 0

    for _ in range(trials):
        idx = rng.integers(0, n, size=n)
        sample = trades.iloc[idx].reset_index(drop=True)
        start = pd.Timestamp("2020-01-01")
        sample["timestamp"] = pd.date_range(start, periods=len(sample), freq="D")
        res = simulate_two_phase_prop(sample, window_start=start, max_calendar_days=None)
        if res["outcome"] == "pass":
            pass_n += 1
            if res["total_days"] is not None:
                pass_days.append(float(res["total_days"]))
        elif res["outcome"] == "fail":
            fail_n += 1
        else:
            timeout_n += 1

    arr = np.asarray(pass_days, dtype=np.float64)
    return {
        "trials": trials,
        "pass_rate": round(pass_n / trials * 100.0, 2),
        "failure_rate": round(fail_n / trials * 100.0, 2),
        "timeout_rate": round(timeout_n / trials * 100.0, 2),
        "avg_pass_days": round(float(np.mean(arr)), 2) if len(arr) else 0.0,
        "median_pass_days": round(float(np.median(arr)), 2) if len(arr) else 0.0,
    }


def risk_of_ruin(
    df: pd.DataFrame,
    model_id: str,
    *,
    ruin_dd_pct: float = TOTAL_DD_LIMIT_PCT,
    trials: int = MC_TRIALS,
    seed: int = MC_SEED,
) -> float:
    trades = _prepare_prop_trades(df, model_id)
    if trades.empty:
        return 1.0

    rng = np.random.default_rng(seed + hash(model_id) % 10_000)
    n = len(trades)
    ruin_count = 0

    for _ in range(trials):
        idx = rng.integers(0, n, size=n)
        sample = trades.iloc[idx].reset_index(drop=True)
        equity = float(STARTING_EQUITY)
        peak = equity
        ruined = False
        for row in sample.itertuples(index=False):
            equity *= 1.0 + (BASE_RISK_PCT / 100.0) * float(row.lot_factor) * float(row.profit_r)
            peak = max(peak, equity)
            dd = (peak - equity) / peak * 100.0 if peak > 0 else 0.0
            if dd >= ruin_dd_pct:
                ruined = True
                break
        if ruined:
            ruin_count += 1
    return round(ruin_count / trials, 4)


def portfolio_compatibility_row(model_stats: Mapping[str, Any]) -> dict[str, Any]:
    smrs_total_r = float(model_stats.get("total_r", 0.0))
    smrs_pf = float(model_stats.get("pf", 0.0))
    smrs_dd = float(model_stats.get("max_dd_r", 0.0))
    combined_total_r = PORTFOLIO_BASELINE_TOTAL_R + smrs_total_r
    combined_trades = PORTFOLIO_BASELINE_TRADES + int(model_stats.get("trades", 0))
    combined_pf_est = combined_total_r / max(combined_trades, 1) * (
        PORTFOLIO_BASELINE_PF * PORTFOLIO_BASELINE_TRADES / max(PORTFOLIO_BASELINE_TRADES, 1)
    ) / max(smrs_pf, 0.01)
    dd_impact_pct = round(smrs_dd * BASE_RISK_PCT / max(PORTFOLIO_BASELINE_MAX_DD_PCT, 0.01) * 100.0, 2)
    return {
        "model_id": model_stats.get("model_id"),
        "smrs_standalone_pf": smrs_pf,
        "smrs_contribution_total_r": smrs_total_r,
        "smrs_contribution_trades": int(model_stats.get("trades", 0)),
        "portfolio_baseline_pf": PORTFOLIO_BASELINE_PF,
        "portfolio_baseline_total_r": PORTFOLIO_BASELINE_TOTAL_R,
        "expected_combined_total_r": round(combined_total_r, 2),
        "expected_combined_pf_proxy": round(
            (PORTFOLIO_BASELINE_TOTAL_R + smrs_total_r)
            / max(PORTFOLIO_BASELINE_TRADES + int(model_stats.get("trades", 0)), 1),
            4,
        ),
        "expected_dd_impact_pct_of_baseline": dd_impact_pct,
    }


def overall_score(row: Mapping[str, Any], *, baseline: Mapping[str, Any]) -> float:
    pf = float(row.get("pf", 0.0) or 0.0)
    total_r = float(row.get("total_r", 0.0) or 0.0)
    base_total = float(baseline.get("total_r", 1.0) or 1.0)
    max_dd = float(row.get("max_dd_r", 999.0) or 999.0)
    base_dd = float(baseline.get("max_dd_r", 999.0) or 999.0)
    sharpe = float(row.get("sharpe", 0.0) or 0.0)
    pass_rate = float(row.get("pass_rate", 0.0) or 0.0)
    ror = float(row.get("risk_of_ruin", 1.0) or 1.0)
    stability = float(row.get("stability_score", 0.0) or 0.0)
    skip_pct = float(row.get("skip_pct", 0.0) or 0.0)
    skip_penalty = 0.15 if skip_pct > 70.0 else 0.0
    pf_bonus = 0.10 if pf > float(baseline.get("pf", 0.0)) else 0.0
    total_bonus = 0.10 if total_r > base_total else 0.0
    dd_bonus = 0.05 if max_dd <= base_dd else 0.0
    return (
        min(pf, 30.0) * 0.18
        + (total_r / max(base_total, 1.0)) * 0.18
        + (1.0 / max(max_dd, 0.5)) * 0.12
        + sharpe * 0.10
        + (pass_rate / 100.0) * 0.22
        + (1.0 - ror) * 0.10
        + (stability / 100.0) * 0.05
        + pf_bonus
        + total_bonus
        + dd_bonus
        - skip_penalty
    )


def probability_tier_table(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["bayes_probability"] = pd.to_numeric(work["bayes_probability"], errors="coerce")
    work["trade_r"] = pd.to_numeric(work["trade_r"], errors="coerce").fillna(0.0)
    groups = (
        ("p<0.7", work["bayes_probability"] < 0.7),
        ("0.7≤p<0.8", (work["bayes_probability"] >= 0.7) & (work["bayes_probability"] < 0.8)),
        ("0.8≤p<0.9", (work["bayes_probability"] >= 0.8) & (work["bayes_probability"] < 0.9)),
        ("p≥0.9", work["bayes_probability"] >= 0.9),
    )
    rows: list[dict[str, Any]] = []
    for label, mask in groups:
        sub = work.loc[mask]
        r = sub["trade_r"].astype(float)
        pf = profit_factor(r) if len(sub) else 0.0
        rows.append(
            {
                "group": label,
                "trades": int(len(sub)),
                "wr_pct": round(float((r > 0).mean() * 100.0), 2) if len(sub) else 0.0,
                "pf": round(float(pf), 4) if np.isfinite(float(pf)) else pf,
            }
        )
    return pd.DataFrame(rows)


def render_probability_tier_table(df: pd.DataFrame) -> list[str]:
    table = probability_tier_table(df)
    lines = [
        "| Group | Trades | WR | PF |",
        "|---|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        pf = row["pf"]
        pf_s = "inf" if pf == float("inf") else f"{float(pf):.3f}"
        lines.append(f"| {row['group']} | {int(row['trades']):,} | {row['wr_pct']:.2f}% | {pf_s} |")
    lines.append("")
    return lines


def sizing_table_for_model(model_id: str) -> list[str]:
    tables = {
        "A": [
            "p < 0.70 → SKIP",
            "0.70–0.80 → 0.50R",
            "0.80–0.90 → 1.00R",
            "p ≥ 0.90 → 1.50R",
        ],
        "B": [
            "p < 0.70 → SKIP",
            "0.70–0.80 → 0.50R",
            "0.80–0.90 → 1.25R",
            "p ≥ 0.90 → 2.00R",
        ],
        "C": [
            "p < 0.70 → SKIP",
            "0.70–0.80 → 0.75R",
            "0.80–0.90 → 1.50R",
            "p ≥ 0.90 → 2.50R",
        ],
        "D": ["All trades → 0.25R + 2.25R × probability"],
        "E": ["All trades → min(2.0R, sqrt(probability) × 2.0R)"],
        "F": ["Kelly-lite rolling (min 0.25R, max 2.0R)"],
        "G": [
            "p < 0.70 → 0.25R",
            "0.70–0.80 → 0.50R",
            "0.80–0.90 → 1.25R",
            "p ≥ 0.90 → 2.00R",
        ],
        "G2": [
            "p < 0.60 → 0.10R",
            "0.60–0.70 → 0.25R",
            "0.70–0.80 → 0.50R",
            "0.80–0.90 → 1.25R",
            "p ≥ 0.90 → 2.00R",
        ],
    }
    return tables.get(model_id, [])


def phase4_verdict(scorecard: pd.DataFrame, baseline: Mapping[str, Any]) -> tuple[str, str]:
    candidates = scorecard[scorecard["model_id"] != BASELINE_MODEL].copy()
    if candidates.empty:
        return "NO GO", BASELINE_MODEL

    candidates = candidates.sort_values("overall_score", ascending=False)
    best = candidates.iloc[0]
    best_id = str(best["model_id"])

    beats_pf = float(best["pf"]) > float(baseline["pf"])
    beats_total = float(best["total_r"]) > float(baseline["total_r"])
    lower_dd = float(best["max_dd_r"]) <= float(baseline["max_dd_r"])
    positive_windows = float(best["positive_windows_pct"]) >= 100.0
    pass_ok = float(best["pass_rate"]) >= 99.0
    skip_ok = float(best.get("skip_pct", 0.0)) <= 70.0

    if beats_pf and beats_total and lower_dd and positive_windows and pass_ok and skip_ok:
        return "PHASE5 READY", best_id
    if beats_pf and beats_total and positive_windows and pass_ok:
        return "PHASE5 READY", best_id
    return "NO GO", best_id

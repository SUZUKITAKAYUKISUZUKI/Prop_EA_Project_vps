"""
strategies/vamr_calibration.py — VAMR Phase 3.5 probability calibration utilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from strategies.vamr_features import profit_factor
from strategies.vamr_phase2 import max_drawdown_r

_EPS = 1e-15

BRIER_ACCEPTABLE = 0.20
BRIER_STRONG = 0.15
BRIER_EXCELLENT = 0.10

CAL_ERROR_GOOD = 0.08
CAL_ERROR_WATCH = 0.12

PHASE3_RESULTS_COLUMNS = ("timestamp", "pair", "bayes_probability", "trade_r", "trade_result")


@dataclass
class DecileCalibrationRow:
    bucket: float
    trades: int
    predicted: float
    actual: float

    @property
    def calibration_error(self) -> float:
        return abs(self.predicted - self.actual)


def prepare_calibration_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    work["bayes_probability"] = pd.to_numeric(work["bayes_probability"], errors="coerce")
    work["trade_r"] = pd.to_numeric(work["trade_r"], errors="coerce").fillna(0.0)
    if "trade_result" in work.columns:
        work["is_win"] = pd.to_numeric(work["trade_result"], errors="coerce").fillna(0).astype(int)
    else:
        work["is_win"] = (work["trade_r"] > 0).astype(int)
    work = work.dropna(subset=["bayes_probability"]).sort_values("timestamp").reset_index(drop=True)
    return work


def export_phase3_results(scored_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(scored_df["timestamp"]),
            "pair": scored_df["pair"].astype(str).str.upper(),
            "bayes_probability": pd.to_numeric(scored_df["bayes_probability"], errors="coerce"),
            "trade_r": pd.to_numeric(scored_df["result_r"], errors="coerce").fillna(0.0),
            "trade_result": pd.to_numeric(scored_df.get("target_win", np.where(scored_df["result_r"] > 0, 1, 0)), errors="coerce")
            .fillna(0)
            .astype(int),
        }
    )
    return out.dropna(subset=["bayes_probability"]).sort_values("timestamp").reset_index(drop=True)


def decile_calibration_rows(df: pd.DataFrame) -> list[DecileCalibrationRow]:
    rows: list[DecileCalibrationRow] = []
    for idx in range(10):
        low = idx / 10.0
        high = (idx + 1) / 10.0
        if idx == 9:
            mask = (df["bayes_probability"] >= low) & (df["bayes_probability"] <= high)
        else:
            mask = (df["bayes_probability"] >= low) & (df["bayes_probability"] < high)
        sub = df.loc[mask]
        if sub.empty:
            rows.append(DecileCalibrationRow(bucket=(idx + 1) / 10.0, trades=0, predicted=0.0, actual=0.0))
            continue
        rows.append(
            DecileCalibrationRow(
                bucket=(idx + 1) / 10.0,
                trades=int(len(sub)),
                predicted=round(float(sub["bayes_probability"].mean()), 4),
                actual=round(float(sub["is_win"].mean()), 4),
            )
        )
    return rows


def decile_calibration_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = decile_calibration_rows(df)
    return pd.DataFrame(
        {
            "bucket": [r.bucket for r in rows],
            "trades": [r.trades for r in rows],
            "predicted": [r.predicted for r in rows],
            "actual": [r.actual for r in rows],
        }
    )


def mean_calibration_error(rows: list[DecileCalibrationRow]) -> float:
    active = [r for r in rows if r.trades > 0]
    if not active:
        return 0.0
    return float(np.mean([r.calibration_error for r in active]))


def weighted_calibration_error(rows: list[DecileCalibrationRow]) -> float:
    active = [r for r in rows if r.trades > 0]
    if not active:
        return 0.0
    weights = np.array([r.trades for r in active], dtype=float)
    errors = np.array([r.calibration_error for r in active], dtype=float)
    return float(np.average(errors, weights=weights))


def brier_score(df: pd.DataFrame) -> float:
    probs = df["bayes_probability"].clip(_EPS, 1.0 - _EPS).to_numpy(dtype=float)
    y = df["is_win"].astype(float).to_numpy()
    return float(np.mean((probs - y) ** 2))


def log_loss_score(df: pd.DataFrame) -> float:
    probs = df["bayes_probability"].clip(_EPS, 1.0 - _EPS).to_numpy(dtype=float)
    y = df["is_win"].astype(float).to_numpy()
    return float(-np.mean(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs)))


def brier_interpretation(score: float) -> str:
    if score < BRIER_EXCELLENT:
        return "excellent"
    if score < BRIER_STRONG:
        return "strong"
    if score < BRIER_ACCEPTABLE:
        return "acceptable"
    return "weak"


def bucket_metrics(df: pd.DataFrame, *, low: float, high: float, inclusive_high: bool = False) -> dict[str, Any]:
    if inclusive_high:
        mask = (df["bayes_probability"] >= low) & (df["bayes_probability"] <= high)
    else:
        mask = (df["bayes_probability"] >= low) & (df["bayes_probability"] < high)
    sub = df.loc[mask]
    r = sub["trade_r"].astype(float)
    if sub.empty:
        return {"trades": 0, "wr": 0.0, "pf": 0.0, "avg_r": 0.0, "total_r": 0.0, "max_dd_r": 0.0, "mean_prob": 0.0}
    pf = profit_factor(r)
    return {
        "trades": int(len(sub)),
        "wr": round(float(sub["is_win"].mean() * 100.0), 2),
        "pf": round(float(pf), 4) if np.isfinite(pf) else pf,
        "avg_r": round(float(r.mean()), 4),
        "total_r": round(float(r.sum()), 2),
        "max_dd_r": max_drawdown_r(r),
        "mean_prob": round(float(sub["bayes_probability"].mean()), 4),
    }


def check_decile_monotonicity(rows: list[DecileCalibrationRow], *, min_trades: int = 20) -> tuple[bool, list[str]]:
    active = [r for r in rows if r.trades >= min_trades]
    violations: list[str] = []
    for prev, curr in zip(active, active[1:]):
        if curr.actual + 1e-9 < prev.actual:
            violations.append(
                f"WR decreased from bucket {prev.bucket:.1f} "
                f"(pred={prev.predicted:.3f}, actual={prev.actual:.3f}, n={prev.trades}) "
                f"to {curr.bucket:.1f} (pred={curr.predicted:.3f}, actual={curr.actual:.3f}, n={curr.trades})"
            )
    return len(violations) == 0, violations


def check_metric_monotonicity(df: pd.DataFrame, *, n_buckets: int = 10, min_trades: int = 20) -> tuple[bool, list[str]]:
    violations: list[str] = []
    metrics: list[dict[str, Any]] = []
    for idx in range(n_buckets):
        low = idx / n_buckets
        high = (idx + 1) / n_buckets
        metrics.append(bucket_metrics(df, low=low, high=high, inclusive_high=(idx == n_buckets - 1)))
    active = [m for m in metrics if m["trades"] >= min_trades]
    for name in ("wr", "pf", "avg_r"):
        for prev, curr in zip(active, active[1:]):
            if curr[name] + 1e-9 < prev[name]:
                violations.append(f"{name.upper()} decreased between consecutive deciles (min {min_trades} trades)")
    return len(violations) == 0, violations


def pair_calibration_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair in sorted(df["pair"].dropna().unique()):
        sub = df[df["pair"] == pair]
        deciles = decile_calibration_rows(sub)
        cal_err = mean_calibration_error(deciles)
        high = sub[sub["bayes_probability"] >= 0.80]
        high_pred = float(high["bayes_probability"].mean()) if not high.empty else 0.0
        high_actual = float(high["is_win"].mean()) if not high.empty else 0.0
        rows.append(
            {
                "pair": pair,
                "trades": int(len(sub)),
                "mean_calibration_error": round(cal_err, 4),
                "brier_score": round(brier_score(sub), 4),
                "high_conf_trades": int(len(high)),
                "high_conf_predicted": round(high_pred, 4),
                "high_conf_actual_wr": round(high_actual, 4),
                "high_conf_gap": round(abs(high_pred - high_actual), 4),
                "does_80_mean_80": abs(high_pred - high_actual) <= 0.08 if len(high) >= 20 else False,
            }
        )
    return pd.DataFrame(rows)


def yearly_stability_table(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["year"] = work["timestamp"].dt.year
    rows: list[dict[str, Any]] = []
    for year in sorted(work["year"].unique()):
        sub = work[work["year"] == year]
        r = sub["trade_r"].astype(float)
        pf = profit_factor(r)
        rows.append(
            {
                "year": int(year),
                "trades": int(len(sub)),
                "mean_probability": round(float(sub["bayes_probability"].mean()), 4),
                "actual_wr": round(float(sub["is_win"].mean()), 4),
                "pf": round(float(pf), 4) if np.isfinite(pf) else pf,
                "avg_r": round(float(r.mean()), 4),
                "total_r": round(float(r.sum()), 2),
            }
        )
    return pd.DataFrame(rows)


def high_confidence_audit(df: pd.DataFrame) -> pd.DataFrame:
    thresholds = (0.80, 0.85, 0.90, 0.95)
    rows: list[dict[str, Any]] = []
    for thr in thresholds:
        sub = df[df["bayes_probability"] >= thr]
        stats = bucket_metrics(sub, low=thr, high=1.0, inclusive_high=True) if not sub.empty else bucket_metrics(df, low=1.1, high=1.2)
        rows.append(
            {
                "probability_threshold": thr,
                "trades": stats["trades"],
                "wr_pct": stats["wr"],
                "pf": stats["pf"],
                "avg_r": stats["avg_r"],
                "total_r": stats["total_r"],
                "max_dd_r": stats["max_dd_r"],
                "mean_predicted_probability": stats["mean_prob"],
            }
        )
    return pd.DataFrame(rows)


def simulate_position_sizing(df: pd.DataFrame) -> dict[str, Any]:
    def size_multiplier(prob: float) -> float:
        if prob < 0.50:
            return 0.0
        if prob < 0.70:
            return 0.5
        if prob < 0.85:
            return 1.0
        return 1.5

    baseline_r = df["trade_r"].astype(float)
    sized_r = baseline_r * df["bayes_probability"].map(size_multiplier).astype(float)
    active = sized_r[sized_r != 0.0]
    baseline_pf = profit_factor(baseline_r)
    sized_pf = profit_factor(active) if len(active) else 0.0
    baseline_dd = max_drawdown_r(baseline_r)
    sized_dd = max_drawdown_r(active) if len(active) else 0.0
    baseline_total = float(baseline_r.sum())
    sized_total = float(active.sum())
    n_active = int((sized_r != 0.0).sum())
    n_skipped = int((sized_r == 0.0).sum())
    years = max((df["timestamp"].max() - df["timestamp"].min()).days / 365.25, 1.0)
    return {
        "baseline_trades": int(len(df)),
        "active_trades": n_active,
        "skipped_trades": n_skipped,
        "baseline_pf": round(float(baseline_pf), 4) if np.isfinite(baseline_pf) else baseline_pf,
        "sized_pf": round(float(sized_pf), 4) if np.isfinite(sized_pf) else sized_pf,
        "baseline_total_r": round(baseline_total, 2),
        "sized_total_r": round(sized_total, 2),
        "baseline_max_dd_r": baseline_dd,
        "sized_max_dd_r": sized_dd,
        "dd_reduction_pct": round((1.0 - sized_dd / baseline_dd) * 100.0, 2) if baseline_dd > 0 else 0.0,
        "cagr_proxy_baseline": round(baseline_total / years, 2),
        "cagr_proxy_sized": round(sized_total / years, 2),
    }


def phase35_verdict(
    *,
    brier: float,
    log_loss: float,
    calibration_error: float,
    weighted_cal_error: float,
    monotonic_wr: bool,
    monotonic_metrics: bool,
    pair_table: pd.DataFrame,
    yearly_table: pd.DataFrame,
    sizing: dict[str, Any],
) -> tuple[str, list[str]]:
    notes: list[str] = []
    rebuild_flags = 0
    recal_flags = 0
    cal_error = weighted_cal_error

    if brier >= BRIER_ACCEPTABLE:
        rebuild_flags += 1
        notes.append(f"Brier score {brier:.4f} is above acceptable threshold ({BRIER_ACCEPTABLE}).")
    elif brier >= BRIER_STRONG:
        recal_flags += 1
        notes.append(f"Brier score {brier:.4f} is acceptable but not strong (< {BRIER_STRONG}).")

    if cal_error > CAL_ERROR_WATCH:
        rebuild_flags += 1
        notes.append(f"Trade-weighted calibration error {cal_error:.4f} exceeds watch threshold ({CAL_ERROR_WATCH}).")
    elif cal_error > CAL_ERROR_GOOD:
        recal_flags += 1
        notes.append(f"Trade-weighted calibration error {cal_error:.4f} is moderate (> {CAL_ERROR_GOOD}).")

    if not monotonic_wr or not monotonic_metrics:
        recal_flags += 1
        notes.append("Probability monotonicity violations detected.")

    if not pair_table.empty:
        bad_pairs = pair_table[~pair_table["does_80_mean_80"] & (pair_table["high_conf_trades"] >= 20)]
        if len(bad_pairs) == len(pair_table[pair_table["high_conf_trades"] >= 20]) and len(bad_pairs) > 0:
            rebuild_flags += 1
            notes.append("High-confidence calibration fails across all pairs with sufficient samples.")
        elif len(bad_pairs) > 0:
            recal_flags += 1
            notes.append(f"Pair calibration gaps on: {', '.join(bad_pairs['pair'].astype(str).tolist())}.")

    if not yearly_table.empty and len(yearly_table) >= 3:
        wr_gap = yearly_table["actual_wr"] - yearly_table["mean_probability"]
        if float(wr_gap.abs().max()) > 0.20:
            recal_flags += 1
            notes.append("Year-by-year probability vs WR gap exceeds 20 points in at least one year.")

    if sizing["sized_pf"] < 1.05 or sizing["sized_total_r"] <= 0:
        rebuild_flags += 1
        notes.append("Dynamic sizing simulation underperforms baseline.")

    if rebuild_flags >= 2:
        return "REBUILD BAYES MODEL", notes
    if recal_flags >= 1 or rebuild_flags == 1:
        return "PROCEED TO PHASE4 WITH RECALIBRATION", notes
    return "PROCEED TO PHASE4", notes

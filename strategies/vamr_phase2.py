"""
strategies/vamr_phase2.py — VAMR Phase 2 rule extraction and executable validation.

Fixed-rule filter combinations on POC cohort. No ML / Bayes / optimization.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies.vamr import STRATEGY_FULL_NAME
from strategies.vamr_features import add_derived_features, load_poc_cohort, pf_str, profit_factor
from strategies.var_reversal import ALLOWED_PAIRS
from walkforward_runner import iter_wft_windows

STRATEGY_NAME = STRATEGY_FULL_NAME
DEFAULT_INPUT = PROJECT_ROOT / "backtest_results/logs/var_features_pure_10y.csv"
DEFAULT_CACHE = PROJECT_ROOT / "backtest_results/logs/var_features_pure_10y_enriched.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports/vamr_phase2_summary.md"
DEFAULT_START = pd.Timestamp("2015-01-01")
DEFAULT_END = pd.Timestamp("2026-06-12 23:59:59")
IS_MONTHS = 12
OOS_MONTHS = 3
STEP_MONTHS = 3

GO_PF = 1.80
GO_TRADES = 1000
WATCH_PF = 1.40
WATCH_TRADES = 1000


@dataclass(frozen=True)
class Phase2Thresholds:
    poc_distance_q1_max: float
    va_width_q1_max: float
    rejection_q4_min: float


@dataclass(frozen=True)
class PatternSpec:
    name: str
    use_distance: bool = False
    use_va_width: bool = False
    use_rejection: bool = False
    use_retest_one: bool = False
    use_engulfing: bool = False
    exclude_pin: bool = False


@dataclass
class PatternMetrics:
    name: str
    trades: int = 0
    win_rate: float = 0.0
    pf: float = 0.0
    avg_r: float = 0.0
    total_r: float = 0.0
    max_dd_r: float = 0.0
    sharpe: float = 0.0
    verdict: str = "KILL"


@dataclass
class WftPatternMetrics:
    name: str
    mean_pf: float = 0.0
    median_pf: float = 0.0
    pf_std: float = 0.0
    positive_window_ratio: float = 0.0
    stability_score: float = 0.0
    window_rows: list[dict] = field(default_factory=list)


PATTERN_SPECS: tuple[PatternSpec, ...] = (
    PatternSpec("Base"),
    PatternSpec("Pattern 1", exclude_pin=True),
    PatternSpec("Pattern 2", use_distance=True),
    PatternSpec("Pattern 3", use_distance=True, exclude_pin=True),
    PatternSpec("Pattern 4", use_distance=True, use_rejection=True),
    PatternSpec("Pattern 5", use_distance=True, use_rejection=True, exclude_pin=True),
    PatternSpec("Pattern 6", use_distance=True, use_rejection=True, use_va_width=True),
    PatternSpec("Pattern 7", use_distance=True, use_rejection=True, use_va_width=True, exclude_pin=True),
    PatternSpec("Pattern 8", use_distance=True, use_rejection=True, use_va_width=True, use_retest_one=True),
    PatternSpec("Pattern 9", use_distance=True, use_rejection=True, use_va_width=True, use_retest_one=True, exclude_pin=True),
    PatternSpec(
        "Pattern 10",
        use_distance=True,
        use_rejection=True,
        use_va_width=True,
        use_retest_one=True,
        use_engulfing=True,
    ),
)


def compute_thresholds(poc_cohort: pd.DataFrame) -> Phase2Thresholds:
    va = pd.to_numeric(poc_cohort["value_area_width_atr_ratio"], errors="coerce")
    rej = pd.to_numeric(poc_cohort["rejection_strength"], errors="coerce")
    dist = pd.to_numeric(poc_cohort["abs_price_vs_poc"], errors="coerce")
    return Phase2Thresholds(
        poc_distance_q1_max=float(dist.quantile(0.25)),
        va_width_q1_max=float(va.quantile(0.25)),
        rejection_q4_min=float(rej.quantile(0.75)),
    )


def apply_pattern_mask(df: pd.DataFrame, spec: PatternSpec, thresholds: Phase2Thresholds) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if spec.use_distance:
        mask &= pd.to_numeric(df["abs_price_vs_poc"], errors="coerce") <= thresholds.poc_distance_q1_max
    if spec.use_va_width:
        mask &= pd.to_numeric(df["value_area_width_atr_ratio"], errors="coerce") <= thresholds.va_width_q1_max
    if spec.use_rejection:
        mask &= pd.to_numeric(df["rejection_strength"], errors="coerce") >= thresholds.rejection_q4_min
    if spec.use_retest_one:
        mask &= pd.to_numeric(df["retest_count"], errors="coerce") == 1
    if spec.use_engulfing:
        mask &= df["primary_pa_type"].astype(str) == "ENGULFING"
    if spec.exclude_pin:
        mask &= df["primary_pa_type"].astype(str) != "PIN_BAR"
    return mask.fillna(False)


def max_drawdown_r(result_r: pd.Series) -> float:
    r = pd.to_numeric(result_r, errors="coerce").fillna(0.0)
    if r.empty:
        return 0.0
    equity = r.cumsum()
    peak = equity.cummax()
    dd = peak - equity
    return round(float(dd.max()), 2)


def sharpe_r(result_r: pd.Series) -> float:
    r = pd.to_numeric(result_r, errors="coerce").dropna()
    if len(r) < 2:
        return 0.0
    std = float(r.std(ddof=1))
    if std <= 0:
        return 0.0
    return round(float(r.mean() / std), 4)


def verdict_for(pf: float, avg_r: float, trades: int) -> str:
    if trades >= GO_TRADES and np.isfinite(pf) and pf >= GO_PF and avg_r > 0:
        return "GO"
    if trades >= WATCH_TRADES and np.isfinite(pf) and pf >= WATCH_PF:
        return "WATCH"
    return "KILL"


def evaluate_pattern(df: pd.DataFrame, spec: PatternSpec, thresholds: Phase2Thresholds) -> PatternMetrics:
    filtered = df.loc[apply_pattern_mask(df, spec, thresholds)].copy()
    filtered = filtered.sort_values("timestamp")
    r = pd.to_numeric(filtered["result_r"], errors="coerce").fillna(0.0)
    pf = profit_factor(r)
    avg_r = float(r.mean()) if len(r) else 0.0
    metrics = PatternMetrics(
        name=spec.name,
        trades=int(len(filtered)),
        win_rate=round(float((r > 0).mean() * 100.0), 2) if len(r) else 0.0,
        pf=round(pf, 4) if np.isfinite(pf) else pf,
        avg_r=round(avg_r, 4),
        total_r=round(float(r.sum()), 2),
        max_dd_r=max_drawdown_r(r),
        sharpe=sharpe_r(r),
    )
    metrics.verdict = verdict_for(metrics.pf, metrics.avg_r, metrics.trades)
    return metrics


def run_full_period_validation(
    df: pd.DataFrame,
    thresholds: Phase2Thresholds,
) -> list[PatternMetrics]:
    return [evaluate_pattern(df, spec, thresholds) for spec in PATTERN_SPECS]


def run_wft_validation(
    df: pd.DataFrame,
    thresholds: Phase2Thresholds,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, WftPatternMetrics]:
    windows = list(
        iter_wft_windows(start, end, is_months=IS_MONTHS, oos_months=OOS_MONTHS, step_months=STEP_MONTHS)
    )
    out: dict[str, WftPatternMetrics] = {}
    for spec in PATTERN_SPECS:
        out[spec.name] = WftPatternMetrics(name=spec.name)

    for spec in PATTERN_SPECS:
        wp = out[spec.name]
        for window in windows:
            oos = df[(df["timestamp"] >= window.oos_start) & (df["timestamp"] < window.oos_end)]
            metrics = evaluate_pattern(oos, spec, thresholds)
            wp.window_rows.append(
                {
                    "window_id": window.window_id,
                    "oos_start": window.oos_start.date(),
                    "oos_end": window.oos_end.date(),
                    "trades": metrics.trades,
                    "pf": metrics.pf,
                    "avg_r": metrics.avg_r,
                    "total_r": metrics.total_r,
                    "max_dd_r": metrics.max_dd_r,
                }
            )
        pfs = [float(row["pf"]) for row in wp.window_rows if row["trades"] > 0 and np.isfinite(float(row["pf"]))]
        if pfs:
            wp.mean_pf = round(float(np.mean(pfs)), 4)
            wp.median_pf = round(float(np.median(pfs)), 4)
            wp.pf_std = round(float(np.std(pfs, ddof=1)) if len(pfs) > 1 else 0.0, 4)
            wp.positive_window_ratio = round(
                float(sum(1 for pf in pfs if pf >= 1.0) / len(pfs)),
                4,
            )
            wp.stability_score = round(wp.mean_pf * wp.positive_window_ratio, 4)
    return out


def render_summary(
    *,
    thresholds: Phase2Thresholds,
    full_metrics: list[PatternMetrics],
    wft_metrics: dict[str, WftPatternMetrics],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> str:
    lines = [
        "# VAMR Phase 2 — Rule Extraction & Executable Validation",
        "",
        f"## {STRATEGY_NAME}",
        "",
        f"- Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- Period: {start.date()} .. {end.date()}",
        f"- Pairs: {', '.join(sorted(ALLOWED_PAIRS))}",
        f"- Base cohort: POC target trades only",
        "",
        "## Fixed Thresholds (from Phase1 POC cohort)",
        "",
        f"- Filter A — `abs(price_vs_poc)` in Phase1 Q1 band (<= {thresholds.poc_distance_q1_max:.6f}, cohort 0–25%)",
        f"- Filter B — `value_area_width_atr_ratio <= {thresholds.va_width_q1_max:.6f}` (Q1 max)",
        f"- Filter C — `rejection_strength >= {thresholds.rejection_q4_min:.6f}` (Q4 min)",
        f"- Filter D — `retest_count == 1`",
        f"- Filter E — `primary_pa_type == ENGULFING`",
        f"- Filter F — exclude `PIN_BAR`",
        "",
        "## Overall Ranking (Full Period)",
        "",
        "| Rank | Pattern | Trades | WR | PF | AvgR | TotalR | MaxDD R | Sharpe | Verdict |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    ranked = sorted(full_metrics, key=lambda m: (m.verdict != "GO", m.verdict != "WATCH", -float(m.pf) if np.isfinite(m.pf) else 999))
    for rank, row in enumerate(ranked, start=1):
        lines.append(
            f"| {rank} | {row.name} | {row.trades:,} | {row.win_rate:.2f}% | {pf_str(float(row.pf))} | "
            f"{row.avg_r:.4f} | {row.total_r:.2f} | {row.max_dd_r:.2f} | {row.sharpe:.4f} | {row.verdict} |"
        )
    lines.extend(["", "## WFT Ranking (42 windows, OOS only)", ""])
    lines.extend(
        [
            "| Pattern | Mean PF | Median PF | PF Std | Positive Windows | Stability |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    wft_ranked = sorted(wft_metrics.values(), key=lambda w: w.stability_score, reverse=True)
    for row in wft_ranked:
        lines.append(
            f"| {row.name} | {pf_str(row.mean_pf)} | {pf_str(row.median_pf)} | {row.pf_std:.4f} | "
            f"{row.positive_window_ratio:.2%} | {row.stability_score:.4f} |"
        )

    best = next((m for m in ranked if m.verdict == "GO"), None)
    watch = [m for m in ranked if m.verdict == "WATCH"]
    kill = [m for m in ranked if m.verdict == "KILL"]

    lines.extend(["", "## Final Recommendation", ""])
    lines.extend(["### 採用候補", ""])
    if best:
        lines.append(f"- **{best.name}** — PF {pf_str(float(best.pf))}, Trades {best.trades:,}, TotalR {best.total_r:.2f}")
    else:
        lines.append("- _None (no GO pattern on full-period rules)._")

    lines.extend(["", "### 監視候補", ""])
    if watch:
        for row in watch[:3]:
            lines.append(f"- **{row.name}** — PF {pf_str(float(row.pf))}, Trades {row.trades:,}")
    else:
        lines.append("- _None._")

    lines.extend(["", "### 廃棄候補", ""])
    if kill:
        lines.append(f"- {', '.join(row.name for row in kill if row.name == 'Base') or kill[0].name} and weaker patterns (Verdict=KILL)")
    else:
        lines.append("- _None._")

    lines.extend(["", "### Phase3 Gate", ""])
    if best:
        lines.append("- **Proceed to Phase3 (Bayesian Probability Layer)** for the top GO pattern.")
    else:
        top = ranked[0]
        if top.verdict == "WATCH":
            lines.append("- **Hold** — best pattern is WATCH only; do not enter Phase3 yet.")
        else:
            lines.append("- **VAMR freeze candidate** — no pattern reached GO threshold (PF >= 1.80, Trades >= 1000, AvgR > 0).")

    lines.extend(["", "## WFT Window Detail (top GO/WATCH pattern)", ""])
    focus = best or (watch[0] if watch else ranked[0])
    focus_wft = wft_metrics[focus.name]
    lines.extend(
        [
            f"Focus pattern: **{focus.name}**",
            "",
            "| Window | OOS period | Trades | PF | AvgR | TotalR | MaxDD R |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in focus_wft.window_rows:
        lines.append(
            f"| {row['window_id']} | {row['oos_start']} .. {row['oos_end']} | {row['trades']:,} | "
            f"{pf_str(float(row['pf']))} | {row['avg_r']:.4f} | {row['total_r']:.2f} | {row['max_dd_r']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_phase2(
    *,
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    cache_path: Path = DEFAULT_CACHE,
    start: pd.Timestamp = DEFAULT_START,
    end: pd.Timestamp = DEFAULT_END,
    use_cache: bool = True,
) -> tuple[list[PatternMetrics], dict[str, WftPatternMetrics], Phase2Thresholds]:
    print(f"Loading POC cohort from {input_path} ...")
    cohort = load_poc_cohort(
        input_path,
        start=start,
        end=end,
        enrich=True,
        cache_path=cache_path if use_cache else None,
    )
    print(f"  POC trades: {len(cohort):,}")
    thresholds = compute_thresholds(cohort)
    print(
        "Thresholds: "
        f"distance<={thresholds.poc_distance_q1_max:.4f}, "
        f"va_width<={thresholds.va_width_q1_max:.4f}, "
        f"rejection>={thresholds.rejection_q4_min:.4f}"
    )
    full_metrics = run_full_period_validation(cohort, thresholds)
    for row in full_metrics:
        print(
            f"  {row.name:12s} trades={row.trades:6,} PF={pf_str(float(row.pf)):>6} "
            f"AvgR={row.avg_r:+.4f} verdict={row.verdict}"
        )
    print("Running WFT ...")
    wft_metrics = run_wft_validation(cohort, thresholds, start=start, end=end)
    report = render_summary(
        thresholds=thresholds,
        full_metrics=full_metrics,
        wft_metrics=wft_metrics,
        start=start,
        end=end,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote {output_path}")
    return full_metrics, wft_metrics, thresholds


def main() -> int:
    parser = argparse.ArgumentParser(description="VAMR Phase 2 rule validation")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2026-06-12")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    if not args.input.exists():
        print(f"[ERROR] Missing input: {args.input}", file=sys.stderr)
        return 1
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end + " 23:59:59") if len(args.end) <= 10 else pd.Timestamp(args.end)
    run_phase2(
        input_path=args.input,
        output_path=args.output,
        cache_path=args.cache,
        start=start,
        end=end,
        use_cache=not args.no_cache,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
lgr_bayes_diagnostics.py — LGR Bayes Gate V1 診断レポート生成

モデル改良・閾値変更は行わず、確率分布・Calibration・特徴量寄与を可視化する。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from archive.lgr.lgr_bayes_gate import (
    DEFAULT_MODEL_JSON,
    LGR_BAYES_ALLOW_THRESHOLD,
    LGR_BAYES_CAUTION_THRESHOLD,
    LgrBayesModel,
    encode_lgr_bayes_features,
    load_lgr_bayes_model,
    minutes_bin_label,
    normalize_session_type,
    profit_factor,
)

DEFAULT_FEATURE_LOG = (
    Path(__file__).resolve().parent / "backtest_results" / "logs" / "lgr_bayes_features_3y.csv"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "backtest_results" / "LGR_BAYES_DIAGNOSTICS.md"

_EPS = 1e-15


def prepare_diagnostic_df(df: pd.DataFrame) -> pd.DataFrame:
    """全4050件に統一 outcome（shadow 含む）を付与。"""
    work = df.copy()
    executed = work["trade_result"].isin(["WIN", "LOSS"])
    work["outcome"] = np.where(executed, work["trade_result"], work["shadow_trade_result"])
    work["outcome_r"] = np.where(executed, work["profit_r"], work["shadow_profit_r"])
    work = work[work["outcome"].isin(["WIN", "LOSS"])].copy()
    work["is_win"] = work["outcome"] == "WIN"
    work["bayes_probability"] = pd.to_numeric(work["bayes_probability"], errors="coerce")
    work = work.dropna(subset=["bayes_probability"])
    return work


def _agg_block(sub: pd.DataFrame) -> dict[str, Any]:
    if sub.empty:
        return {
            "count": 0,
            "wins": 0,
            "losses": 0,
            "wr_pct": 0.0,
            "pf": 0.0,
            "total_r": 0.0,
            "avg_r": 0.0,
        }
    wins = int(sub["is_win"].sum())
    losses = len(sub) - wins
    r = sub["outcome_r"].astype(float)
    return {
        "count": len(sub),
        "wins": wins,
        "losses": losses,
        "wr_pct": wins / len(sub) * 100.0,
        "pf": profit_factor(r),
        "total_r": float(r.sum()),
        "avg_r": float(r.mean()),
    }


def _fmt_pf(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.3f}"


def _fmt_r(value: float) -> str:
    return f"{value:+.2f}"


def build_fine_probability_table(df: pd.DataFrame, *, width: float = 0.05) -> str:
    rows: list[str] = []
    for start in np.arange(0.0, 1.0, width):
        end = min(start + width, 1.0)
        if end >= 1.0:
            mask = (df["bayes_probability"] >= start) & (df["bayes_probability"] <= end)
            label = f"{start:.2f}-{end:.2f}"
        else:
            mask = (df["bayes_probability"] >= start) & (df["bayes_probability"] < end)
            label = f"{start:.2f}-{end:.2f}"
        stats = _agg_block(df[mask])
        rows.append(
            f"| {label} | {stats['count']} | {stats['wins']} | {stats['losses']} | "
            f"{stats['wr_pct']:.1f}% | {_fmt_pf(stats['pf'])} | {_fmt_r(stats['total_r'])} |"
        )
    return "\n".join(
        [
            "| Bin | Count | WIN | LOSS | WR | PF | TotalR |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *rows,
        ]
    )


def build_coarse_probability_table(df: pd.DataFrame, *, width: float = 0.10) -> str:
    rows: list[str] = []
    for start in np.arange(0.0, 1.0, width):
        end = min(start + width, 1.0)
        if end >= 1.0:
            mask = (df["bayes_probability"] >= start) & (df["bayes_probability"] <= end)
            label = f"{start:.2f}-{end:.2f}"
        else:
            mask = (df["bayes_probability"] >= start) & (df["bayes_probability"] < end)
            label = f"{start:.2f}-{end:.2f}"
        stats = _agg_block(df[mask])
        rows.append(
            f"| {label} | {stats['count']} | {stats['wr_pct']:.1f}% | "
            f"{stats['avg_r']:+.3f} | {_fmt_pf(stats['pf'])} | {_fmt_r(stats['total_r'])} |"
        )
    return "\n".join(
        [
            "| Prob Range | Count | WR | AvgR | PF | TotalR |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *rows,
        ]
    )


def build_calibration_table(df: pd.DataFrame, *, width: float = 0.10) -> tuple[str, dict[str, float]]:
    rows: list[str] = []
    abs_errors: list[float] = []
    for start in np.arange(0.0, 1.0, width):
        end = min(start + width, 1.0)
        if end >= 1.0:
            mask = (df["bayes_probability"] >= start) & (df["bayes_probability"] <= end)
            label = f"{start:.2f}-{end:.2f}"
        else:
            mask = (df["bayes_probability"] >= start) & (df["bayes_probability"] < end)
            label = f"{start:.2f}-{end:.2f}"
        sub = df[mask]
        if sub.empty:
            rows.append(f"| {label} | — | — | — |")
            continue
        predicted = float(sub["bayes_probability"].mean())
        actual = float(sub["is_win"].mean())
        err = abs(predicted - actual)
        abs_errors.append(err)
        rows.append(
            f"| {label} | {predicted:.3f} | {actual:.3f} | {actual - predicted:+.3f} |"
        )

    probs = df["bayes_probability"].clip(_EPS, 1.0 - _EPS).to_numpy()
    y = df["is_win"].astype(float).to_numpy()
    brier = float(np.mean((probs - y) ** 2))
    log_loss = float(-np.mean(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs)))
    cal_error = float(np.mean(abs_errors)) if abs_errors else 0.0

    table = "\n".join(
        [
            "| Prob Range | Predicted | Actual WR | Error (Actual-Pred) |",
            "| --- | ---: | ---: | ---: |",
            *rows,
            "",
            f"- **Mean Calibration Error** (|Predicted − Actual| per bin): **{cal_error:.4f}**",
            f"- **Brier Score**: **{brier:.4f}**",
            f"- **Log Loss**: **{log_loss:.4f}**",
        ]
    )
    return table, {"calibration_error": cal_error, "brier_score": brier, "log_loss": log_loss}


def build_probability_summary(df: pd.DataFrame) -> tuple[str, dict[str, float]]:
    probs = df["bayes_probability"]
    stats = {
        "max": float(probs.max()),
        "mean": float(probs.mean()),
        "median": float(probs.median()),
        "p95": float(probs.quantile(0.95)),
        "p99": float(probs.quantile(0.99)),
        "min": float(probs.min()),
    }
    allow_count = int((df["bayes_regime"] == "ALLOW").sum()) if "bayes_regime" in df.columns else 0
    threshold_verdict = (
        f"**判定: ALLOW=0 の主因は閾値設計** — max={stats['max']:.4f} < ALLOW閾値 {LGR_BAYES_ALLOW_THRESHOLD:.2f}"
        if stats["max"] < LGR_BAYES_ALLOW_THRESHOLD
        else f"max={stats['max']:.4f} は ALLOW 閾値 {LGR_BAYES_ALLOW_THRESHOLD:.2f} 以上（閾値単独では ALLOW=0 を説明できない）"
    )
    text = "\n".join(
        [
            "| Stat | Value |",
            "| --- | ---: |",
            f"| max(bayes_probability) | {stats['max']:.4f} |",
            f"| mean(bayes_probability) | {stats['mean']:.4f} |",
            f"| median(bayes_probability) | {stats['median']:.4f} |",
            f"| 95 percentile | {stats['p95']:.4f} |",
            f"| 99 percentile | {stats['p99']:.4f} |",
            f"| min(bayes_probability) | {stats['min']:.4f} |",
            "",
            f"- ALLOW count: **{allow_count}**",
            f"- CAUTION threshold: **{LGR_BAYES_CAUTION_THRESHOLD:.2f}**",
            f"- ALLOW threshold: **{LGR_BAYES_ALLOW_THRESHOLD:.2f}**",
            "",
            threshold_verdict,
        ]
    )
    return text, stats


def _attach_feature_bins(df: pd.DataFrame, model: LgrBayesModel) -> pd.DataFrame:
    work = df.copy()
    encoded_rows: list[dict[str, str]] = []
    for row in work.itertuples(index=False):
        encoded_rows.append(
            encode_lgr_bayes_features(
                {
                    "pair": row.pair,
                    "session_type": row.session_type,
                    "minutes_from_session_open": row.minutes_from_session_open,
                    "positive_close_ratio": row.positive_close_ratio,
                    "directional_efficiency": row.directional_efficiency,
                },
                model=model,
            )
        )
    encoded_df = pd.DataFrame(encoded_rows)
    work["pair_bin"] = encoded_df["pair"]
    work["session_bin"] = encoded_df["session_type"]
    work["minutes_bin"] = encoded_df["minutes_from_session_open"]
    work["pcr_bin"] = encoded_df["positive_close_ratio"]
    work["de_bin"] = encoded_df["directional_efficiency"]
    return work


def build_feature_section(df: pd.DataFrame, feature_col: str, title: str) -> str:
    rows: list[str] = []
    for bin_label, sub in df.groupby(feature_col, sort=False):
        stats = _agg_block(sub)
        rows.append(
            f"| {bin_label} | {stats['count']} | {stats['wr_pct']:.1f}% | "
            f"{_fmt_pf(stats['pf'])} | {_fmt_r(stats['total_r'])} |"
        )
    return "\n".join(
        [
            f"## {title}",
            "",
            "| Bin | Count | WR | PF | TotalR |",
            "| --- | --- | ---: | ---: | ---: |",
            *rows,
            "",
        ]
    )


def build_top_n_table(df: pd.DataFrame, n: int) -> str:
    top = df.nlargest(min(n, len(df)), "bayes_probability")
    stats = _agg_block(top)
    return (
        f"| Top {n} | {stats['count']} | {stats['wr_pct']:.1f}% | "
        f"{_fmt_pf(stats['pf'])} | {_fmt_r(stats['total_r'])} | {stats['avg_r']:+.3f} |"
    )


def build_ranking_section(df: pd.DataFrame) -> str:
    rows = [build_top_n_table(df, n) for n in (100, 500, 1000, 2000)]
    bottom = df.nsmallest(min(100, len(df)), "bayes_probability")
    bottom_stats = _agg_block(bottom)
    rows.append(
        f"| Bottom 100 | {bottom_stats['count']} | {bottom_stats['wr_pct']:.1f}% | "
        f"{_fmt_pf(bottom_stats['pf'])} | {_fmt_r(bottom_stats['total_r'])} | {bottom_stats['avg_r']:+.3f} |"
    )
    return "\n".join(
        [
            "| Rank Group | Count | WR | PF | TotalR | AvgR |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *rows,
        ]
    )


def build_shadow_reject_section(df: pd.DataFrame, *, width: float = 0.10) -> str:
    reject = df[df["bayes_regime"] == "REJECT"].copy() if "bayes_regime" in df.columns else df.iloc[0:0]
    rows: list[str] = []
    for start in np.arange(0.0, 1.0, width):
        end = min(start + width, 1.0)
        if end >= 1.0:
            mask = (reject["bayes_probability"] >= start) & (reject["bayes_probability"] <= end)
            label = f"{start:.2f}-{end:.2f}"
        else:
            mask = (reject["bayes_probability"] >= start) & (reject["bayes_probability"] < end)
            label = f"{start:.2f}-{end:.2f}"
        stats = _agg_block(reject[mask])
        rows.append(
            f"| {label} | {stats['count']} | {stats['wr_pct']:.1f}% | "
            f"{_fmt_pf(stats['pf'])} | {_fmt_r(stats['total_r'])} |"
        )
    total_stats = _agg_block(reject)
    header = "\n".join(
        [
            f"Reject population: **{total_stats['count']}** trades | "
            f"WR **{total_stats['wr_pct']:.1f}%** | PF **{_fmt_pf(total_stats['pf'])}** | "
            f"TotalR **{_fmt_r(total_stats['total_r'])}**",
            "",
            "| Range | Count | WR | PF | TotalR |",
            "| --- | ---: | ---: | ---: | ---: |",
            *rows,
        ]
    )
    return header


def infer_root_cause(
    df: pd.DataFrame,
    prob_stats: dict[str, float],
    cal_metrics: dict[str, float],
    model: LgrBayesModel,
) -> str:
    top100 = _agg_block(df.nlargest(100, "bayes_probability"))
    bottom100 = _agg_block(df.nsmallest(100, "bayes_probability"))
    spread_wr = top100["wr_pct"] - bottom100["wr_pct"]
    spread_avg_r = top100["avg_r"] - bottom100["avg_r"]

    threshold_issue = prob_stats["max"] < LGR_BAYES_ALLOW_THRESHOLD
    calibration_issue = cal_metrics["calibration_error"] > 0.08 or cal_metrics["brier_score"] > 0.24
    discrimination_weak = spread_wr < 5.0 or spread_avg_r < 0.05

    feature_spreads: list[tuple[str, float]] = []
    binned = _attach_feature_bins(df, model)
    for col, name in (
        ("pair_bin", "pair"),
        ("session_bin", "session_type"),
        ("minutes_bin", "minutes_from_session_open"),
        ("pcr_bin", "positive_close_ratio"),
        ("de_bin", "directional_efficiency"),
    ):
        wr_by_bin = binned.groupby(col)["is_win"].mean() * 100.0
        if len(wr_by_bin) >= 2:
            feature_spreads.append((name, float(wr_by_bin.max() - wr_by_bin.min())))

    max_feature_spread = max((spread for _, spread in feature_spreads), default=0.0)
    feature_issue = max_feature_spread < 8.0 and discrimination_weak

    causes: list[tuple[str, str, int]] = []
    if threshold_issue:
        causes.append(
            (
                "ALLOW閾値問題",
                f"max P(win)={prob_stats['max']:.4f} が ALLOW 閾値 {LGR_BAYES_ALLOW_THRESHOLD:.2f} 未満。"
                f"全件が CAUTION/REJECT に分類される構造。",
                3,
            )
        )
    if calibration_issue:
        causes.append(
            (
                "確率校正（Calibration）問題",
                f"Mean Calibration Error={cal_metrics['calibration_error']:.4f}, "
                f"Brier={cal_metrics['brier_score']:.4f}, LogLoss={cal_metrics['log_loss']:.4f}。"
                "予測確率と実勝率のズレが大きい。",
                2,
            )
        )
    if discrimination_weak:
        causes.append(
            (
                "モデル識別力問題",
                f"Top100 WR={top100['wr_pct']:.1f}% vs Bottom100 WR={bottom100['wr_pct']:.1f}% "
                f"(spread {spread_wr:+.1f}pp)。確率順位で利益が十分に分離していない。",
                2 if not threshold_issue else 1,
            )
        )
    if feature_issue:
        causes.append(
            (
                "特徴量情報量問題",
                f"最大ビン間 WR spread={max_feature_spread:.1f}pp。"
                "V1 5特徴量では勝敗を十分に説明できていない可能性。",
                1,
            )
        )

    causes.sort(key=lambda item: item[2], reverse=True)
    primary = causes[0][0] if causes else "未判定"

    lines = [
        "## Root Cause Summary",
        "",
        f"**主因判定: {primary}**",
        "",
        "### 判定根拠",
        "",
    ]
    for name, detail, _ in causes:
        lines.append(f"- **{name}**: {detail}")

    lines.extend(
        [
            "",
            "### 参考: 確率順位の分離度",
            "",
            "| Group | Count | WR | AvgR | PF | TotalR |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            f"| Top 100 | {top100['count']} | {top100['wr_pct']:.1f}% | {top100['avg_r']:+.3f} | "
            f"{_fmt_pf(top100['pf'])} | {_fmt_r(top100['total_r'])} |",
            f"| Bottom 100 | {bottom100['count']} | {bottom100['wr_pct']:.1f}% | {bottom100['avg_r']:+.3f} | "
            f"{_fmt_pf(bottom100['pf'])} | {_fmt_r(bottom100['total_r'])} |",
            "",
            "### 特徴量ビン間 WR Spread",
            "",
            "| Feature | Max WR Spread (pp) |",
            "| --- | ---: |",
        ]
    )
    for name, spread in sorted(feature_spreads, key=lambda x: x[1], reverse=True):
        lines.append(f"| {name} | {spread:.1f} |")

    return "\n".join(lines)


def build_lgr_bayes_diagnostics_report(
    df: pd.DataFrame,
    *,
    model: LgrBayesModel | None = None,
) -> str:
    work = prepare_diagnostic_df(df)
    if work.empty:
        return "# LGR Bayes Gate V1 Diagnostics\n\nNo valid rows.\n"

    model = model or load_lgr_bayes_model(DEFAULT_MODEL_JSON)
    binned = _attach_feature_bins(work, model)

    fine_table = build_fine_probability_table(work, width=0.05)
    coarse_table = build_coarse_probability_table(work, width=0.10)
    cal_table, cal_metrics = build_calibration_table(work, width=0.10)
    prob_summary, prob_stats = build_probability_summary(work)
    ranking_table = build_ranking_section(work)
    shadow_table = build_shadow_reject_section(work, width=0.10)
    root_cause = infer_root_cause(work, prob_stats, cal_metrics, model)

    regime_lines = []
    if "bayes_regime" in work.columns:
        for regime in ("ALLOW", "CAUTION", "REJECT"):
            count = int((work["bayes_regime"] == regime).sum())
            regime_lines.append(f"| {regime} | {count} |")

    sections = [
        "# LGR Bayes Gate V1 Diagnostics",
        "",
        f"Source rows: **{len(df)}** | Analyzed (valid outcome + probability): **{len(work)}**",
        "",
        "## Regime Distribution",
        "",
        "| Regime | Count |",
        "| --- | ---: |",
        *regime_lines,
        "",
        "## Probability Distribution (0.05 bins)",
        "",
        fine_table,
        "",
        "## Expected Value by Probability Band (0.10 bins)",
        "",
        coarse_table,
        "",
        "## Calibration",
        "",
        cal_table,
        "",
        "## ALLOW=0 Root Cause — Probability Summary",
        "",
        prob_summary,
        "",
        "## Feature Contribution",
        "",
        build_feature_section(binned, "pair_bin", "pair"),
        build_feature_section(binned, "session_bin", "session_type"),
        build_feature_section(binned, "minutes_bin", "minutes_from_session_open"),
        build_feature_section(binned, "pcr_bin", "positive_close_ratio"),
        build_feature_section(binned, "de_bin", "directional_efficiency"),
        "## Top-N Analysis (bayes_probability descending)",
        "",
        ranking_table,
        "",
        "## Shadow Analysis — REJECT Population by Probability Band",
        "",
        shadow_table,
        "",
        root_cause,
        "",
    ]
    return "\n".join(sections)


def write_lgr_bayes_diagnostics_report(
    feature_log_path: Path | None = None,
    output_path: Path | None = None,
    *,
    model_json: Path | None = None,
) -> Path:
    log_path = Path(feature_log_path or DEFAULT_FEATURE_LOG)
    out_path = Path(output_path or DEFAULT_OUTPUT)
    df = pd.read_csv(log_path)
    model = load_lgr_bayes_model(Path(model_json or DEFAULT_MODEL_JSON))
    report = build_lgr_bayes_diagnostics_report(df, model=model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    return out_path


__all__ = [
    "build_lgr_bayes_diagnostics_report",
    "prepare_diagnostic_df",
    "write_lgr_bayes_diagnostics_report",
]

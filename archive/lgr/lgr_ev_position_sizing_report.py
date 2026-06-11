"""
lgr_ev_position_sizing_report.py — LGR EV Position Sizing V1 レポート
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from archive.lgr.lgr_ev_position_sizing import (
    DEFAULT_BAYES_FEATURE_LOG,
    DEFAULT_EV_REPORT,
    DEFAULT_PURE_FEATURE_LOG,
    EV_FEATURE_WEIGHTS,
    PURE_BASELINE,
    SIZING_PATTERNS,
    TOP_EV_COHORTS,
    build_session_minutes_cross,
    cohen_d,
    enrich_with_ev_scores,
    select_recommended_sizing,
    simulate_bayes_gate,
    simulate_pattern,
    simulate_rank_gate,
    summarize_cohort,
    summarize_group,
    summarize_sized_trades,
)


def _fmt_pf(v: float) -> str:
    return "inf" if math.isinf(v) else f"{v:.3f}"


def _fmt_r(v: float) -> str:
    return f"{v:+.2f}"


def _stats_row(label: str, stats: dict[str, Any]) -> str:
    return (
        f"| {label} | {stats['trades']} | {_fmt_pf(stats['pf'])} | {_fmt_r(stats['total_r'])} | "
        f"{stats['max_dd_pct']:.2f}% | {stats['sharpe']:.3f} | {stats['mar']:.2f} |"
    )


def build_ev_score_design_section(df: pd.DataFrame) -> str:
    weight_rows = [f"| {k} | {v:.2f} |" for k, v in EV_FEATURE_WEIGHTS.items()]
    return "\n".join(
        [
            "各特徴量ビンの **平均 R (base_r)** から quality score (0–1) を算出し、重み付き合成。",
            "勝敗分類器ではなく **期待値ランキング** 専用。",
            "",
            "| Feature | Weight |",
            "| --- | ---: |",
            *weight_rows,
            "",
            f"- `ev_score` range: **{df['ev_score'].min():.4f} – {df['ev_score'].max():.4f}**",
            f"- `ev_rank` = percentile rank of `ev_score` (0=bottom, 1=top)",
            f"- Population: **{len(df)}** trades (all executed, no rejection)",
        ]
    )


def build_ev_rank_distribution(df: pd.DataFrame) -> str:
    rows: list[str] = []
    for start in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        end = start + 0.1
        if end >= 1.0:
            mask = (df["ev_rank"] >= start) & (df["ev_rank"] <= end)
            label = f"{start:.1f}-{end:.1f}"
        else:
            mask = (df["ev_rank"] >= start) & (df["ev_rank"] < end)
            label = f"{start:.1f}-{end:.1f}"
        sub = df[mask]
        if sub.empty:
            continue
        stats = summarize_sized_trades(sub.assign(sized_r=sub["base_r"]))
        rows.append(
            f"| {label} | {stats['trades']} | {stats['wr_pct']:.1f}% | {_fmt_pf(stats['pf'])} | "
            f"{_fmt_r(stats['total_r'])} | {stats['avg_r']:+.3f} |"
        )
    return "\n".join(
        [
            "| EV Rank Bin | Count | WR | PF | TotalR | AvgR |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *rows,
        ]
    )


def build_pattern_comparison(df: pd.DataFrame) -> tuple[str, dict[str, dict[str, Any]]]:
    rows: list[str] = []
    stats_map: dict[str, dict[str, Any]] = {}
    for name in SIZING_PATTERNS:
        sim = simulate_pattern(df, name)
        stats = summarize_sized_trades(sim)
        stats_map[name] = stats
        rows.append(_stats_row(name, stats))
    table = "\n".join(
        [
            "| Pattern | Trades | PF | TotalR | MaxDD | Sharpe | MAR |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *rows,
        ]
    )
    return table, stats_map


def build_top_ev_cohort_table(df: pd.DataFrame) -> str:
    rows: list[str] = []
    prev_avg = None
    monotonic = True
    for n in TOP_EV_COHORTS:
        stats = summarize_cohort(df, n)
        rows.append(
            f"| Top {n} | {stats['trades']} | {stats['wr_pct']:.1f}% | {_fmt_pf(stats['pf'])} | "
            f"{_fmt_r(stats['total_r'])} | {stats['max_dd_pct']:.2f}% | {stats['sharpe']:.3f} |"
        )
        if prev_avg is not None and stats["avg_r"] < prev_avg - 1e-9:
            monotonic = False
        prev_avg = stats["avg_r"]
    note = (
        "AvgR は Top100 → Top1000 で **概ね単調増加**（EV rank と期待値が整合）。"
        if monotonic
        else "AvgR に一部非単調区間あり — 上位コホート内でもノイズ存在。"
    )
    table = "\n".join(
        [
            "| Top N | Count | WR | PF | TotalR | MaxDD | Sharpe |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            note,
        ]
    )
    return table


def build_fine_minutes_table(df: pd.DataFrame) -> str:
    grouped = summarize_group(df, "fine_minutes_bucket")
    grouped = grouped.sort_values("group")
    rows = [
        f"| {row['group']} | {int(row['trades'])} | {row['wr_pct']:.1f}% | "
        f"{_fmt_pf(row['pf'])} | {_fmt_r(row['total_r'])} |"
        for _, row in grouped.iterrows()
    ]
    return "\n".join(
        [
            "| Minutes | Count | WR | PF | TotalR |",
            "| --- | ---: | ---: | ---: | ---: |",
            *rows,
        ]
    )


def build_session_heatmap(df: pd.DataFrame) -> tuple[str, str]:
    session_df = summarize_group(df, "session_norm")
    session_rows = [
        f"| {row['group']} | {int(row['trades'])} | {row['wr_pct']:.1f}% | "
        f"{_fmt_pf(row['pf'])} | {_fmt_r(row['total_r'])} |"
        for _, row in session_df.iterrows()
    ]
    session_table = "\n".join(
        [
            "| Session | Count | WR | PF | TotalR |",
            "| --- | ---: | ---: | ---: | ---: |",
            *session_rows,
        ]
    )

    cross = build_session_minutes_cross(df)
    cross = cross.sort_values(["session", "minutes_bucket"])
    cross_rows = [
        f"| {row['session']} | {row['minutes_bucket']} | {int(row['count'])} | "
        f"{row['wr_pct']:.1f}% | {_fmt_pf(row['pf'])} | {_fmt_r(row['total_r'])} |"
        for _, row in cross.iterrows()
    ]
    cross_table = "\n".join(
        [
            "| Session | Minutes Bucket | Count | WR | PF | TotalR |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
            *cross_rows,
        ]
    )
    return session_table, cross_table


def build_feature_importance_table(df: pd.DataFrame) -> str:
    sorted_df = df.sort_values("ev_rank", ascending=False)
    top500 = sorted_df.head(500)
    bottom500 = sorted_df.tail(500)

    numeric = [
        ("positive_close_ratio", "positive_close_ratio"),
        ("directional_efficiency", "directional_efficiency"),
        ("minutes_from_session_open", "minutes_from_session_open"),
        ("volatility_percentile", "volatility_percentile"),
        ("atr_ratio", "atr_ratio"),
        ("position_in_range", "position_in_range"),
    ]
    rows: list[tuple[float, str]] = []
    for label, col in numeric:
        if col not in df.columns:
            continue
        top_v = pd.to_numeric(top500[col], errors="coerce")
        bot_v = pd.to_numeric(bottom500[col], errors="coerce")
        effect = cohen_d(top_v, bot_v)
        rows.append(
            (
                abs(effect),
                f"| {label} | {effect:+.3f} | {top_v.mean():.4f} | {bot_v.mean():.4f} |",
            )
        )

    top_asia = (top500["session_norm"] == "ASIA").mean() * 100
    bot_asia = (bottom500["session_norm"] == "ASIA").mean() * 100
    rows.append(
        (
            abs(top_asia - bot_asia) / 100,
            f"| session_type (ASIA share) | {top_asia - bot_asia:+.1f}pp | "
            f"ASIA {top_asia:.1f}% | ASIA {bot_asia:.1f}% |",
        )
    )
    rows.sort(key=lambda x: x[0], reverse=True)
    return "\n".join(
        [
            "| Feature | Effect Size | Top500 Mean | Bottom500 Mean |",
            "| --- | ---: | ---: | ---: |",
            *(r for _, r in rows),
        ]
    )


def build_model_comparison(df: pd.DataFrame, bayes_df: pd.DataFrame) -> str:
    pure_stats = summarize_sized_trades(df.assign(sized_r=df["base_r"]))

    bayes_exec = simulate_bayes_gate(df, bayes_df)
    bayes_stats = summarize_sized_trades(bayes_exec)

    rank_exec = simulate_rank_gate(df, bayes_df, allow_top_pct=10.0, caution_top_pct=30.0)
    rank_stats = summarize_sized_trades(rank_exec)

    ev_sim = simulate_pattern(df, "Pattern B")
    ev_stats = summarize_sized_trades(ev_sim)

    def pass_mark(stats: dict[str, Any]) -> str:
        ok = (
            stats["pf"] > PURE_BASELINE["pf"]
            and stats["total_r"] > PURE_BASELINE["total_r"]
            and stats["max_dd_pct"] < PURE_BASELINE["max_dd_pct"]
        )
        return "PASS" if ok else "FAIL"

    rows = [
        _stats_row("Pure (1.0R fixed)", pure_stats),
        _stats_row("Bayes Gate (V1 threshold)", bayes_stats),
        _stats_row("Rank Gate (Top10%/30%)", rank_stats),
        _stats_row("EV Sizing (Pattern B)", ev_stats),
    ]
    return "\n".join(
        [
            "| Model | Trades | PF | TotalR | MaxDD | Sharpe | MAR |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            "**Success criteria** (vs Pure reference PF>1.936, TotalR>+705R, MaxDD<16.62%):",
            f"- Pure: {pass_mark(pure_stats)}",
            f"- Bayes Gate: {pass_mark(bayes_stats)}",
            f"- Rank Gate: {pass_mark(rank_stats)}",
            f"- EV Sizing Pattern B: {pass_mark(ev_stats)}",
            "",
            "注: Bayes/Rank Gate は **拒否=0R** のゲート戦略。EV Sizing は **全件執行** で倍率のみ変更。",
        ]
    )


def build_lgr_ev_position_sizing_report(pure_df: pd.DataFrame, bayes_df: pd.DataFrame) -> str:
    df = enrich_with_ev_scores(pure_df)
    design = build_ev_score_design_section(df)
    rank_dist = build_ev_rank_distribution(df)
    pattern_table, pattern_stats = build_pattern_comparison(df)
    cohort_table = build_top_ev_cohort_table(df)
    fine_minutes = build_fine_minutes_table(df)
    session_table, cross_table = build_session_heatmap(df)
    feature_table = build_feature_importance_table(df)
    model_cmp = build_model_comparison(df, bayes_df)
    recommended = select_recommended_sizing(pattern_stats)

    return "\n".join(
        [
            "# LGR EV Position Sizing V1",
            "",
            "## Philosophy",
            "",
            "LGR は **4050件全執行** を基本とし、拒否ではなく **EV rank に応じたリスク配分** で最適化する。",
            "Reject PF < 1.0 を目標とするゲート思想は不適切 — 母集団自体が PF≈1.3–1.9 の優良セット。",
            "",
            f"Pure reference: **{PURE_BASELINE['trades']}** trades, PF **{PURE_BASELINE['pf']}**, "
            f"TotalR **+{PURE_BASELINE['total_r']}R**, MaxDD **{PURE_BASELINE['max_dd_pct']}%**",
            "",
            "## 1. EV Score Design",
            "",
            design,
            "",
            "## 2. EV Rank Distribution",
            "",
            rank_dist,
            "",
            "## 3. Lot Factor Pattern Comparison (all trades executed)",
            "",
            pattern_table,
            "",
            "| Tier | Pattern B (V1) | Pattern C | Pattern D |",
            "| --- | ---: | ---: | ---: |",
            "| Top 5% | 1.50R | 2.00R | 2.50R |",
            "| Top 20% | 1.25R | 1.25R | 1.50R |",
            "| Top 50% | 1.00R | 0.75R | 1.00R |",
            "| Bottom 50% | 0.50R | 0.25R | 0.50R |",
            "| Pattern E | risk = 0.25 + 1.75×ev_score (0.25–2.0) | | |",
            "",
            "## 4. Top EV Cohort Analysis",
            "",
            cohort_table,
            "",
            "## 5. Feature Importance (Top500 vs Bottom500 by EV rank)",
            "",
            feature_table,
            "",
            "## 6. minutes_bucket Deep Dive",
            "",
            fine_minutes,
            "",
            "## 7. Session Heatmap",
            "",
            "### By Session",
            "",
            session_table,
            "",
            "### Session × minutes_bucket",
            "",
            cross_table,
            "",
            "## 8. Model Comparison (Pure vs Gate vs EV Sizing)",
            "",
            model_cmp,
            "",
            "## 9. Recommended Sizing Setting",
            "",
            f"### **{recommended.label}**",
            "",
            recommended.reason,
            "",
            f"Selected pattern: **{recommended.pattern}**",
            "",
        ]
    )


def write_lgr_ev_position_sizing_report(
    *,
    pure_log: Path | None = None,
    bayes_log: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    pure_path = Path(pure_log or DEFAULT_PURE_FEATURE_LOG)
    bayes_path = Path(bayes_log or DEFAULT_BAYES_FEATURE_LOG)
    out_path = Path(output_path or DEFAULT_EV_REPORT)
    pure_df = pd.read_csv(pure_path)
    bayes_df = pd.read_csv(bayes_path)
    report = build_lgr_ev_position_sizing_report(pure_df, bayes_df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    return out_path


__all__ = ["build_lgr_ev_position_sizing_report", "write_lgr_ev_position_sizing_report"]

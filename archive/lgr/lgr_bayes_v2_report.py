"""
lgr_bayes_v2_report.py — LGR Bayes V2 Rank Gate 評価レポート生成
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from archive.lgr.lgr_bayes_v2 import (
    DEFAULT_V2_FEATURE_LOG,
    DEFAULT_V2_REPORT,
    DEFAULT_V2_TRAIN_CSV,
    RANK_GATE_PRESETS,
    TOP_N_FRONTIER,
    V2_FEATURE_NAMES,
    apply_rank_gate,
    cohen_d,
    derive_atr_ratio,
    derive_position_in_range,
    initialize_lgr_bayes_v2,
    minutes_bucket_label,
    predict_log_likelihood_breakdown,
    prepare_evaluation_frame,
    summarize_outcomes,
)


def _fmt_pf(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.3f}"


def _fmt_r(value: float) -> str:
    return f"{value:+.2f}"


def _fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def build_rank_gate_comparison_table(df: pd.DataFrame) -> str:
    rows: list[str] = []
    for gate_name, cfg in RANK_GATE_PRESETS.items():
        regime = apply_rank_gate(
            df,
            allow_top_pct=cfg["allow_top_pct"],
            caution_top_pct=cfg["caution_top_pct"],
        )
        executed = df[regime.isin(["ALLOW", "CAUTION"])]
        stats = summarize_outcomes(executed)
        rows.append(
            f"| {gate_name} | Top {cfg['allow_top_pct']:.0f}% ALLOW / Top {cfg['caution_top_pct']:.0f}% CAUTION | "
            f"{stats['count']} | {_fmt_pct(stats['wr_pct'])} | {_fmt_pf(stats['pf'])} | "
            f"{_fmt_r(stats['total_r'])} | {stats['max_dd_pct']:.2f}% |"
        )
    return "\n".join(
        [
            "| Gate | Policy | Executed | WR | PF | TotalR | MaxDD |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            *rows,
        ]
    )


def build_top_n_frontier_table(df: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    sorted_df = df.sort_values("bayes_probability_rank", ascending=False)
    rows: list[str] = []
    stats_list: list[dict[str, Any]] = []
    for n in TOP_N_FRONTIER:
        sub = sorted_df.head(min(n, len(sorted_df)))
        stats = summarize_outcomes(sub)
        stats["top_n"] = n
        stats_list.append(stats)
        rows.append(
            f"| Top {n} | {stats['count']} | {_fmt_pct(stats['wr_pct'])} | {_fmt_pf(stats['pf'])} | "
            f"{_fmt_r(stats['total_r'])} | {stats['avg_r']:+.3f} | {stats['max_dd_pct']:.2f}% | "
            f"{stats['sharpe']:.3f} |"
        )

    valid = [s for s in stats_list if s["count"] > 0]
    highlights = {}
    if valid:
        highlights["max_pf"] = max(valid, key=lambda s: s["pf"] if not math.isinf(s["pf"]) else -1)
        highlights["max_total_r"] = max(valid, key=lambda s: s["total_r"])
        highlights["max_sharpe"] = max(valid, key=lambda s: s["sharpe"])

    table = "\n".join(
        [
            "| Top N | Count | WR | PF | TotalR | AvgR | MaxDD | Sharpe |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *rows,
        ]
    )
    return table, highlights


def build_feature_importance_table(df: pd.DataFrame) -> str:
    sorted_df = df.sort_values("bayes_probability_rank", ascending=False)
    top500 = sorted_df.head(500)
    bottom500 = sorted_df.tail(500)

    work = df.copy()
    work["atr_ratio"] = work.apply(derive_atr_ratio, axis=1)
    work["position_in_range"] = work.apply(derive_position_in_range, axis=1)
    sorted_work = work.sort_values("bayes_probability_rank", ascending=False)
    top500w = sorted_work.head(500)
    bottom500w = sorted_work.tail(500)

    numeric_features = [
        ("positive_close_ratio", "positive_close_ratio"),
        ("directional_efficiency", "directional_efficiency"),
        ("minutes_from_session_open", "minutes_from_session_open"),
        ("volatility_percentile", "volatility_percentile"),
        ("atr_ratio", "atr_ratio"),
        ("position_in_range", "position_in_range"),
    ]

    rows: list[tuple[float, str]] = []
    for label, col in numeric_features:
        top_vals = pd.to_numeric(top500w[col], errors="coerce")
        bottom_vals = pd.to_numeric(bottom500w[col], errors="coerce")
        effect = cohen_d(top_vals, bottom_vals)
        rows.append(
            (
                abs(effect),
                f"| {label} | {effect:+.3f} | {top_vals.mean():.4f} | {bottom_vals.mean():.4f} |",
            )
        )

    # session_type: WR difference
    top_wr = (top500["session_type"].value_counts(normalize=True) * 100.0).to_dict()
    bottom_wr = (bottom500["session_type"].value_counts(normalize=True) * 100.0).to_dict()
    session_keys = sorted(set(top_wr) | set(bottom_wr))
    max_gap = max(abs(top_wr.get(k, 0) - bottom_wr.get(k, 0)) for k in session_keys) if session_keys else 0.0
    top_asia = top_wr.get("ASIA", 0.0)
    bottom_asia = bottom_wr.get("ASIA", 0.0)
    rows.append(
        (
            max_gap / 100.0,
            f"| session_type (ASIA share pp) | {top_asia - bottom_asia:+.1f}pp | "
            f"ASIA {top_asia:.1f}% | ASIA {bottom_asia:.1f}% |",
        )
    )

    rows.sort(key=lambda item: item[0], reverse=True)
    return "\n".join(
        [
            "| Feature | Effect Size | Top500 Mean | Bottom500 Mean |",
            "| --- | ---: | ---: | ---: |",
            *(row for _, row in rows),
        ]
    )


def build_log_likelihood_compression_table(df: pd.DataFrame, model: Any) -> str:
    """各特徴量の log-ratio 分布幅 — 圧縮に寄与する特徴量を特定。"""
    feature_stats: dict[str, list[float]] = {name: [] for name in V2_FEATURE_NAMES}
    log_odds_list: list[float] = []

    for row in df.itertuples(index=False):
        breakdown = predict_log_likelihood_breakdown(
            {
                "pair": row.pair,
                "session_type": row.session_type,
                "minutes_from_session_open": row.minutes_from_session_open,
                "positive_close_ratio": row.positive_close_ratio,
                "directional_efficiency": row.directional_efficiency,
            },
            model=model,
        )
        log_odds_list.append(breakdown["log_odds"])
        for feat, payload in breakdown["per_feature"].items():
            feature_stats[feat].append(payload["log_ratio"])

    rows: list[str] = []
    for feat in V2_FEATURE_NAMES:
        ratios = pd.Series(feature_stats[feat])
        rows.append(
            f"| {feat} | {ratios.mean():+.4f} | {ratios.std(ddof=1):.4f} | "
            f"{ratios.min():+.4f} | {ratios.max():+.4f} | {ratios.max() - ratios.min():.4f} |"
        )

    probs = df["bayes_probability_v2"]
    log_odds = pd.Series(log_odds_list)
    summary = "\n".join(
        [
            "| Feature | Mean log-ratio | Std log-ratio | Min | Max | Range |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            f"- Probability range: **{probs.min():.4f} – {probs.max():.4f}** (span {probs.max() - probs.min():.4f})",
            f"- Log-odds range: **{log_odds.min():.4f} – {log_odds.max():.4f}**",
            "",
            "圧縮寄与が大きい特徴量 = log-ratio **Range が小さい**（win/loss をあまり分離しない）",
            "識別寄与が大きい特徴量 = log-ratio **Range / Std が大きい**",
        ]
    )
    return summary


def build_shadow_analysis_table(df: pd.DataFrame, *, gate_name: str, cfg: dict[str, float]) -> tuple[str, dict[str, Any]]:
    regime = apply_rank_gate(
        df,
        allow_top_pct=cfg["allow_top_pct"],
        caution_top_pct=cfg["caution_top_pct"],
    )
    reject = df[regime == "REJECT"]
    stats = summarize_outcomes(reject)
    table = "\n".join(
        [
            f"### {gate_name} — REJECT population",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Count | {stats['count']} |",
            f"| WR | {_fmt_pct(stats['wr_pct'])} |",
            f"| PF | {_fmt_pf(stats['pf'])} |",
            f"| TotalR | {_fmt_r(stats['total_r'])} |",
            f"| AvgR | {stats['avg_r']:+.3f} |",
            f"| Success (PF < 1.0) | {'PASS' if stats['pf'] < 1.0 else 'FAIL'} |",
            "",
        ]
    )
    return table, stats


def select_recommended_gate(
    df: pd.DataFrame,
    *,
    frontier_highlights: dict[str, Any],
) -> tuple[str, dict[str, float], str]:
    """
    推奨 Gate を1つ選定。
    優先: Reject PF < 1.0 → Executed PF/TotalR/Sharpe バランス
    """
    candidates: list[dict[str, Any]] = []
    for gate_name, cfg in RANK_GATE_PRESETS.items():
        regime = apply_rank_gate(
            df,
            allow_top_pct=cfg["allow_top_pct"],
            caution_top_pct=cfg["caution_top_pct"],
        )
        executed = df[regime.isin(["ALLOW", "CAUTION"])]
        reject = df[regime == "REJECT"]
        exec_stats = summarize_outcomes(executed)
        reject_stats = summarize_outcomes(reject)
        candidates.append(
            {
                "name": gate_name,
                "cfg": cfg,
                "exec": exec_stats,
                "reject": reject_stats,
                "reject_pass": reject_stats["pf"] < 1.0,
            }
        )

    passing = [c for c in candidates if c["reject_pass"]]
    pool = passing if passing else candidates

    def score(item: dict[str, Any]) -> float:
        ex = item["exec"]
        rej = item["reject"]
        pf = ex["pf"] if not math.isinf(ex["pf"]) else 5.0
        sharpe = ex["sharpe"]
        if item["reject_pass"]:
            return pf * 0.35 + (ex["total_r"] / 100.0) * 0.35 + sharpe * 0.15 + 2.0
        # Shadow FAIL for all gates: prefer lowest reject PF + strong executed quality
        reject_penalty = max(0.0, rej["pf"] - 1.0) * 2.0
        tightness_bonus = (100.0 - item["cfg"]["caution_top_pct"]) / 100.0 * 0.5
        return pf * 0.40 + sharpe * 0.25 + (ex["total_r"] / 150.0) * 0.20 - reject_penalty + tightness_bonus

    best = max(pool, key=score)
    cfg = best["cfg"]
    ex = best["exec"]
    rej = best["reject"]

    reason_parts = [
        f"Top **{cfg['allow_top_pct']:.0f}% ALLOW / Top {cfg['caution_top_pct']:.0f}% CAUTION** を推奨。",
        f"Executed **{ex['count']}** trades, WR **{ex['wr_pct']:.1f}%**, PF **{_fmt_pf(ex['pf'])}**, "
        f"TotalR **{_fmt_r(ex['total_r'])}**, MaxDD **{ex['max_dd_pct']:.2f}%**.",
    ]
    if best["reject_pass"]:
        reason_parts.append(
            f"Reject 群 PF **{_fmt_pf(rej['pf'])}** < 1.0（Shadow 成功条件 PASS）。"
        )
    else:
        reason_parts.append(
            f"Reject 群 PF **{_fmt_pf(rej['pf'])}** ≥ 1.0 — 3 Gate いずれも Shadow 成功条件未達。"
            "Top-N を絞るか特徴量追加が必要。"
        )

    if frontier_highlights:
        mpf = frontier_highlights.get("max_pf")
        if mpf:
            reason_parts.append(
                f"Top-N Frontier: PF 最大は Top **{mpf['top_n']}** (PF {_fmt_pf(mpf['pf'])}), "
                f"TotalR 最大は Top **{frontier_highlights['max_total_r']['top_n']}**, "
                f"Sharpe 最大は Top **{frontier_highlights['max_sharpe']['top_n']}**。"
            )

    reason_parts.append(
        "V1 絶対閾値 (ALLOW≥0.55) は max P=0.525 のため機能せず。"
        "Percentile rank 方式で順位付け能力を執行判断に反映する。"
    )

    label = (
        f"Top {cfg['allow_top_pct']:.0f}% ALLOW / "
        f"Top {cfg['caution_top_pct']:.0f}% CAUTION / "
        f"Bottom {100 - cfg['caution_top_pct']:.0f}% REJECT"
    )
    return label, cfg, " ".join(reason_parts)


def build_lgr_bayes_v2_rank_gate_report(
    df: pd.DataFrame,
    *,
    model: Any,
) -> str:
    work = prepare_evaluation_frame(df, model=model)

    rank_gate_table = build_rank_gate_comparison_table(work)
    frontier_table, frontier_highlights = build_top_n_frontier_table(work)
    feature_table = build_feature_importance_table(work)
    ll_table = build_log_likelihood_compression_table(work, model)

    shadow_sections: list[str] = []
    shadow_stats: dict[str, dict[str, Any]] = {}
    for gate_name, cfg in RANK_GATE_PRESETS.items():
        section, stats = build_shadow_analysis_table(work, gate_name=gate_name, cfg=cfg)
        shadow_sections.append(section)
        shadow_stats[gate_name] = stats

    recommended_label, recommended_cfg, recommended_reason = select_recommended_gate(
        work,
        frontier_highlights=frontier_highlights,
    )

    prob = work["bayes_probability_v2"]
    rank = work["bayes_probability_rank"]

    sections = [
        "# LGR Bayes Gate V2 — Rank Gate Evaluation",
        "",
        f"Samples: **{len(work)}** | Model: Naive Bayes V2 (`minutes_bucket` + 4 features)",
        "",
        "## V1 → V2 Motivation",
        "",
        "| Finding | Detail |",
        "| --- | --- |",
        "| Calibration | Normal (V1 Mean Cal Error ≈ 0.008) |",
        f"| V2 Probability Range | {prob.min():.4f} – {prob.max():.4f} |",
        f"| V2 Rank Range | {rank.min():.4f} – {rank.max():.4f} |",
        "| Root Cause | Probability Compression — rank gate replaces absolute thresholds |",
        "",
        "## 1. Top-N Frontier",
        "",
        frontier_table,
        "",
    ]

    if frontier_highlights:
        mpf = frontier_highlights["max_pf"]
        mtr = frontier_highlights["max_total_r"]
        msh = frontier_highlights["max_sharpe"]
        sections.extend(
            [
                "**Frontier peaks:**",
                f"- PF max: Top **{mpf['top_n']}** → PF {_fmt_pf(mpf['pf'])}, TotalR {_fmt_r(mpf['total_r'])}",
                f"- TotalR max: Top **{mtr['top_n']}** → TotalR {_fmt_r(mtr['total_r'])}, PF {_fmt_pf(mtr['pf'])}",
                f"- Sharpe max: Top **{msh['top_n']}** → Sharpe {msh['sharpe']:.3f}, PF {_fmt_pf(msh['pf'])}",
                "",
            ]
        )

    sections.extend(
        [
            "## 2. Rank Gate Comparison",
            "",
            rank_gate_table,
            "",
            "## 3. Feature Importance (Top500 vs Bottom500)",
            "",
            feature_table,
            "",
            "### minutes_bucket (V2 Bayes feature)",
            "",
            "| Bucket | Count | WR | PF | TotalR |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )

    work["_minutes_bucket"] = work["minutes_from_session_open"].apply(minutes_bucket_label)
    for bucket, sub in work.groupby("_minutes_bucket", sort=False):
        stats = summarize_outcomes(sub)
        sections.append(
            f"| {bucket} | {stats['count']} | {_fmt_pct(stats['wr_pct'])} | "
            f"{_fmt_pf(stats['pf'])} | {_fmt_r(stats['total_r'])} |"
        )

    sections.extend(
        [
            "",
            "## 4. Probability Compression — Log-Likelihood Breakdown",
            "",
            ll_table,
            "",
            "## 5. Shadow Analysis (Rank Gate REJECT groups)",
            "",
            *shadow_sections,
            "## 6. Recommended Gate Setting",
            "",
            f"### **{recommended_label}**",
            "",
            recommended_reason,
            "",
            "| Parameter | Value |",
            "| --- | ---: |",
            f"| ALLOW | Top {recommended_cfg['allow_top_pct']:.0f}% by `bayes_probability_rank` |",
            f"| CAUTION | Top {recommended_cfg['caution_top_pct']:.0f}% (excluding ALLOW band) |",
            f"| REJECT | Bottom {100 - recommended_cfg['caution_top_pct']:.0f}% |",
            "",
            "### Implementation Notes",
            "",
            "- `bayes_probability_rank` = `rank(probability, pct=True)` over full reference population",
            "- 0.0 = lowest probability, 1.0 = highest",
            "- V2 replaces `minutes_from_session_open` bin with explicit `minutes_bucket` (0-30/30-60/60-120/120-240/240+)",
            "- Absolute thresholds (0.55 / 0.45) are **deprecated** in V2",
            "",
        ]
    )
    return "\n".join(sections)


def write_lgr_bayes_v2_rank_gate_report(
    *,
    feature_log_path: Path | None = None,
    train_csv: Path | None = None,
    output_path: Path | None = None,
    retrain: bool = False,
) -> Path:
    log_path = Path(feature_log_path or DEFAULT_V2_FEATURE_LOG)
    out_path = Path(output_path or DEFAULT_V2_REPORT)
    model = initialize_lgr_bayes_v2(train_csv=train_csv or DEFAULT_V2_TRAIN_CSV, retrain=retrain)
    df = pd.read_csv(log_path)
    report = build_lgr_bayes_v2_rank_gate_report(df, model=model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    return out_path


__all__ = [
    "build_lgr_bayes_v2_rank_gate_report",
    "write_lgr_bayes_v2_rank_gate_report",
]

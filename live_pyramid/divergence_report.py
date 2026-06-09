"""BT vs Live pyramid divergence analysis and markdown report."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from live_pyramid.l6_log import DEFAULT_LIVE_PYRAMID_LOG_PATH, LIVE_PYRAMID_LOG_COLUMNS, live_pyramid_log_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports"
DEFAULT_REPORT_PATH = REPORT_DIR / "live_pyramid_divergence.md"


@dataclass(frozen=True)
class DivergenceMetrics:
    live_sessions: int
    bt_matched_sessions: int
    avg_live_layers: float
    avg_bt_layers: float
    avg_layer_delta: float
    limit_placed_total: int
    limit_filled_total: int
    limit_cancelled_total: int
    limit_fill_rate: float
    time_limit_sessions: int
    decel_exit_sessions: int
    top_rejected_reasons: dict[str, int]


def load_live_pyramid_log(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or live_pyramid_log_path()
    if not csv_path.exists():
        raise FileNotFoundError(f"Live pyramid log not found: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    for col in LIVE_PYRAMID_LOG_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    if "event_time" in df.columns:
        df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    for col in (
        "pyramid_layers",
        "limit_price",
        "lot_size",
        "sl",
        "tp",
        "fill_price",
        "ws_kalman_velocity",
        "limit_placed_count",
        "limit_filled_count",
        "limit_cancelled_count",
        "market_fallback_count",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("ws_decel_exit", "ws_time_limit_exit", "pending_limit"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin({"true", "1", "yes"})
    return df


def summarize_live_sessions(live_df: pd.DataFrame) -> pd.DataFrame:
    """trade_id ごとの最終 SESSION_CLOSE / 最新行スナップショット。"""
    if live_df.empty:
        return pd.DataFrame()

    close_rows = live_df[live_df["event_type"] == "SESSION_CLOSE"].copy()
    if close_rows.empty:
        close_rows = live_df.sort_values("event_time").groupby("trade_id", as_index=False).tail(1)
    else:
        close_rows = close_rows.sort_values("event_time").groupby("trade_id", as_index=False).tail(1)

    tick_stats = (
        live_df.groupby("trade_id", as_index=False)
        .agg(
            limit_placed_total=("action_type", lambda s: int((s == "PYRAMID_LIMIT").sum())),
            limit_cancelled_total=("action_type", lambda s: int((s == "PYRAMID_CANCEL").sum())),
            market_fallback_total=("action_type", lambda s: int((s == "PYRAMID_MARKET_FALLBACK").sum())),
        )
    )
    fill_stats = (
        live_df[live_df["event_type"] == "FILL"]
        .groupby("trade_id", as_index=False)
        .size()
        .rename(columns={"size": "limit_filled_total"})
    )

    summary = close_rows.merge(tick_stats, on="trade_id", how="left").merge(
        fill_stats, on="trade_id", how="left"
    )
    for col in ("limit_placed_total", "limit_cancelled_total", "market_fallback_total", "limit_filled_total"):
        if col in summary.columns:
            summary[col] = summary[col].fillna(0).astype(int)

    summary["limit_fill_rate"] = summary.apply(
        lambda row: (
            float(row["limit_filled_total"]) / float(row["limit_placed_total"])
            if row.get("limit_placed_total", 0) > 0
            else 0.0
        ),
        axis=1,
    )
    return summary


def load_backtest_pyramid_reference(bt_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(bt_csv, encoding="utf-8-sig")
    if "trade_id" not in df.columns:
        raise ValueError("Backtest CSV must include trade_id column")
    if "pyramid_layers" not in df.columns:
        raise ValueError("Backtest CSV must include pyramid_layers column")
    cols = ["trade_id", "pyramid_layers"]
    for optional in ("profit_r", "shadow_profit_r", "setup_type", "pair", "timestamp"):
        if optional in df.columns:
            cols.append(optional)
    out = df[cols].copy()
    out["pyramid_layers"] = pd.to_numeric(out["pyramid_layers"], errors="coerce").fillna(0).astype(int)
    if "profit_r" in out.columns:
        out["profit_r"] = pd.to_numeric(out["profit_r"], errors="coerce").fillna(0.0)
    return out


def compare_bt_live(
    live_summary: pd.DataFrame,
    bt_df: pd.DataFrame,
) -> tuple[pd.DataFrame, DivergenceMetrics]:
    if live_summary.empty:
        metrics = DivergenceMetrics(
            live_sessions=0,
            bt_matched_sessions=0,
            avg_live_layers=0.0,
            avg_bt_layers=0.0,
            avg_layer_delta=0.0,
            limit_placed_total=0,
            limit_filled_total=0,
            limit_cancelled_total=0,
            limit_fill_rate=0.0,
            time_limit_sessions=0,
            decel_exit_sessions=0,
            top_rejected_reasons={},
        )
        return pd.DataFrame(), metrics

    if bt_df.empty:
        merged = live_summary.copy()
        merged["live_pyramid_layers"] = pd.to_numeric(merged.get("pyramid_layers", 0), errors="coerce").fillna(0)
        merged["bt_pyramid_layers"] = 0
        merged["layer_delta"] = merged["live_pyramid_layers"]
        bt_matched = 0
        avg_bt_layers = 0.0
    else:
        merged = live_summary.merge(bt_df, on="trade_id", how="left", suffixes=("_live", "_bt"))
        if "pyramid_layers_live" in merged.columns:
            merged["live_pyramid_layers"] = pd.to_numeric(merged["pyramid_layers_live"], errors="coerce").fillna(0)
        else:
            merged["live_pyramid_layers"] = pd.to_numeric(merged.get("pyramid_layers", 0), errors="coerce").fillna(0)
        if "pyramid_layers_bt" in merged.columns:
            merged["bt_pyramid_layers"] = pd.to_numeric(merged["pyramid_layers_bt"], errors="coerce").fillna(0)
        elif "pyramid_layers" in merged.columns:
            merged["bt_pyramid_layers"] = pd.to_numeric(merged["pyramid_layers"], errors="coerce").fillna(0)
        else:
            merged["bt_pyramid_layers"] = 0
        merged["layer_delta"] = merged["live_pyramid_layers"] - merged["bt_pyramid_layers"]
        bt_matched = int((merged["trade_id"].isin(bt_df["trade_id"])).sum())
        avg_bt_layers = float(merged["bt_pyramid_layers"].mean())

    rejected = live_summary["ws_pyramid_rejected_reason"].fillna("").astype(str)
    rejected = rejected[rejected != ""]
    top_rejected: dict[str, int] = {}
    if not rejected.empty:
        top_rejected = rejected.value_counts().head(10).astype(int).to_dict()

    placed = int(live_summary.get("limit_placed_total", pd.Series(dtype=int)).sum())
    filled = int(live_summary.get("limit_filled_total", pd.Series(dtype=int)).sum())
    cancelled = int(live_summary.get("limit_cancelled_total", pd.Series(dtype=int)).sum())

    metrics = DivergenceMetrics(
        live_sessions=int(len(live_summary)),
        bt_matched_sessions=bt_matched,
        avg_live_layers=float(merged["live_pyramid_layers"].mean()),
        avg_bt_layers=avg_bt_layers,
        avg_layer_delta=float(merged["layer_delta"].mean()),
        limit_placed_total=placed,
        limit_filled_total=filled,
        limit_cancelled_total=cancelled,
        limit_fill_rate=(filled / placed) if placed > 0 else 0.0,
        time_limit_sessions=int(live_summary.get("ws_time_limit_exit", pd.Series(dtype=bool)).sum()),
        decel_exit_sessions=int(live_summary.get("ws_decel_exit", pd.Series(dtype=bool)).sum()),
        top_rejected_reasons=top_rejected,
    )
    return merged, metrics


def render_divergence_markdown(
    metrics: DivergenceMetrics,
    comparison_df: pd.DataFrame,
    *,
    live_log_path: Path,
    bt_csv_path: Path | None,
) -> str:
    lines = [
        "# Live Pyramid BT Divergence Report",
        "",
        f"- Live log: `{live_log_path}`",
        f"- Backtest CSV: `{bt_csv_path}`" if bt_csv_path else "- Backtest CSV: *(not provided)*",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Live sessions | {metrics.live_sessions} |",
        f"| BT matched sessions | {metrics.bt_matched_sessions} |",
        f"| Avg live pyramid_layers | {metrics.avg_live_layers:.3f} |",
        f"| Avg BT pyramid_layers | {metrics.avg_bt_layers:.3f} |",
        f"| Avg layer delta (live − BT) | {metrics.avg_layer_delta:+.3f} |",
        f"| Limit placed (events) | {metrics.limit_placed_total} |",
        f"| Limit filled (events) | {metrics.limit_filled_total} |",
        f"| Limit cancelled (events) | {metrics.limit_cancelled_total} |",
        f"| Limit fill rate | {metrics.limit_fill_rate:.1%} |",
        f"| WR time_limit sessions | {metrics.time_limit_sessions} |",
        f"| WR decel exit sessions | {metrics.decel_exit_sessions} |",
        "",
    ]

    if metrics.top_rejected_reasons:
        lines.extend(["## Top pyramid rejected reasons", ""])
        for reason, count in metrics.top_rejected_reasons.items():
            lines.append(f"- `{reason}`: {count}")
        lines.append("")

    if not comparison_df.empty and "layer_delta" in comparison_df.columns:
        divergent = comparison_df[comparison_df["layer_delta"] != 0].copy()
        lines.extend(
            [
                "## Trades with layer mismatch (live ≠ BT)",
                "",
                f"Count: **{len(divergent)}** / {len(comparison_df)}",
                "",
            ]
        )
        if not divergent.empty:
            show_cols = [
                c
                for c in (
                    "trade_id",
                    "setup_type_live",
                    "setup_type",
                    "live_pyramid_layers",
                    "bt_pyramid_layers",
                    "layer_delta",
                    "limit_fill_rate",
                    "ws_pyramid_rejected_reason",
                )
                if c in divergent.columns
            ]
            preview = divergent[show_cols].head(25)
            lines.append("| " + " | ".join(preview.columns) + " |")
            lines.append("| " + " | ".join(["---"] * len(preview.columns)) + " |")
            for _, row in preview.iterrows():
                cells = [str(row[c]) for c in preview.columns]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- **layer_delta < 0** — Live Limit 未約定により BT より積み増しが少ない（想定内の保守性）。",
            "- **limit_fill_rate** — BT は bar.close 即時約定=100% 相当。Live は Limit TTL 内の約定率。",
            "- **time_limit** — WR（Wyckoff Reversal）ピラミッドのタイムリミット発火数。",
            "",
        ]
    )
    return "\n".join(lines)


def generate_divergence_report(
    *,
    live_log: Path | None = None,
    bt_csv: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    live_path = live_log or live_pyramid_log_path()
    live_df = load_live_pyramid_log(live_path)
    live_summary = summarize_live_sessions(live_df)

    if bt_csv is not None:
        bt_df = load_backtest_pyramid_reference(bt_csv)
        comparison_df, metrics = compare_bt_live(live_summary, bt_df)
    else:
        comparison_df, metrics = compare_bt_live(live_summary, pd.DataFrame(columns=["trade_id", "pyramid_layers"]))

    report = render_divergence_markdown(
        metrics,
        comparison_df,
        live_log_path=live_path,
        bt_csv_path=bt_csv,
    )
    out = output_path or DEFAULT_REPORT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    return out


def metrics_to_dict(metrics: DivergenceMetrics) -> dict[str, Any]:
    return {
        "live_sessions": metrics.live_sessions,
        "bt_matched_sessions": metrics.bt_matched_sessions,
        "avg_live_layers": metrics.avg_live_layers,
        "avg_bt_layers": metrics.avg_bt_layers,
        "avg_layer_delta": metrics.avg_layer_delta,
        "limit_placed_total": metrics.limit_placed_total,
        "limit_filled_total": metrics.limit_filled_total,
        "limit_cancelled_total": metrics.limit_cancelled_total,
        "limit_fill_rate": metrics.limit_fill_rate,
        "time_limit_sessions": metrics.time_limit_sessions,
        "decel_exit_sessions": metrics.decel_exit_sessions,
        "top_rejected_reasons": metrics.top_rejected_reasons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live pyramid BT divergence report")
    parser.add_argument(
        "--live-log",
        type=Path,
        default=DEFAULT_LIVE_PYRAMID_LOG_PATH,
        help="Live pyramid audit CSV",
    )
    parser.add_argument(
        "--bt-csv",
        type=Path,
        default=None,
        help="Backtest CSV with trade_id and pyramid_layers",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Markdown report output path",
    )
    parser.add_argument("--json", action="store_true", help="Print metrics JSON to stdout")
    args = parser.parse_args(argv)

    live_df = load_live_pyramid_log(args.live_log)
    live_summary = summarize_live_sessions(live_df)
    if args.bt_csv is not None:
        bt_df = load_backtest_pyramid_reference(args.bt_csv)
        comparison_df, metrics = compare_bt_live(live_summary, bt_df)
    else:
        comparison_df, metrics = compare_bt_live(live_summary, pd.DataFrame(columns=["trade_id", "pyramid_layers"]))

    out = generate_divergence_report(live_log=args.live_log, bt_csv=args.bt_csv, output_path=args.out)
    print(f"Report written: {out}")
    if args.json:
        print(json.dumps(metrics_to_dict(metrics), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

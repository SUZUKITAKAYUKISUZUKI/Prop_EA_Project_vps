"""
lgr_bayes_gate.py — LGR 専用 Naive Bayes Gate V1

LSFC/CSPA/Wyckoff ベイズ実装とは完全分離。LGR 専用の単純ナイーブベイズのみ。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import pandas as pd

logger = logging.getLogger("lgr_bayes_gate")

LgrBayesRegime = Literal["ALLOW", "CAUTION", "REJECT"]

LGR_BAYES_ALLOW_THRESHOLD = 0.55
LGR_BAYES_CAUTION_THRESHOLD = 0.45
LGR_BAYES_LAPLACE_ALPHA = 1.0
LGR_BAYES_MODEL_VERSION = 1
LGR_BAYES_REJECT_SOURCE = "REJECT_BY_LGR_BAYES"

V1_FEATURE_NAMES: tuple[str, ...] = (
    "pair",
    "session_type",
    "minutes_from_session_open",
    "positive_close_ratio",
    "directional_efficiency",
)

DEFAULT_TRAIN_CSV = (
    Path(__file__).resolve().parent / "backtest_results" / "logs" / "lgr_features.csv"
)
DEFAULT_MODEL_JSON = (
    Path(__file__).resolve().parent / "backtest_results" / "models" / "lgr_bayes_v1.json"
)

_SESSION_MAP = {
    "ASIA": "ASIA",
    "LONDON": "LONDON",
    "NY": "NEWYORK",
    "NEWYORK": "NEWYORK",
    "OVERLAP": "OVERLAP",
}


def is_lgr_bayes_gate_enabled() -> bool:
    from strategies.archive.liquidity_grab_reversal import is_lgr_bayes_gate_mode

    return is_lgr_bayes_gate_mode()


def normalize_session_type(raw: Any) -> str:
    key = str(raw or "").strip().upper()
    mapped = _SESSION_MAP.get(key)
    if mapped:
        return mapped
    if key in {"ASIA", "LONDON", "NEWYORK", "OVERLAP"}:
        return key
    return "OTHER"


def minutes_bin_label(minutes: Any) -> str:
    try:
        value = int(float(minutes))
    except (TypeError, ValueError):
        value = 0
    if value < 30:
        return "M0_30"
    if value < 60:
        return "M30_60"
    if value < 120:
        return "M60_120"
    if value < 240:
        return "M120_240"
    return "M240_PLUS"


def _quantile_edges(series: pd.Series, *, bins: int = 4) -> list[float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return [0.0, 1.0]
    edges = [float(clean.min())]
    for pct in range(1, bins):
        edges.append(float(clean.quantile(pct / bins)))
    edges.append(float(clean.max()))
    uniq: list[float] = []
    for edge in edges:
        if not uniq or edge > uniq[-1]:
            uniq.append(edge)
    if len(uniq) < 2:
        uniq = [float(clean.min()), float(clean.max()) + 1e-9]
    return uniq


def quantile_bin_label(value: Any, edges: Sequence[float], *, prefix: str) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        num = 0.0
    if len(edges) < 2:
        return f"{prefix}Q1"
    for idx in range(len(edges) - 1):
        lower = edges[idx]
        upper = edges[idx + 1]
        is_last = idx == len(edges) - 2
        if num >= lower and (num <= upper if is_last else num < upper):
            return f"{prefix}_Q{idx + 1}"
    return f"{prefix}_Q{len(edges) - 1}"


@dataclass
class LgrBayesModel:
    version: int = LGR_BAYES_MODEL_VERSION
    alpha: float = LGR_BAYES_LAPLACE_ALPHA
    priors: dict[str, float] = field(default_factory=lambda: {"win": 0.5, "loss": 0.5})
    class_counts: dict[str, int] = field(default_factory=lambda: {"win": 0, "loss": 0})
    pcr_edges: list[float] = field(default_factory=lambda: [0.0, 1.0])
    de_edges: list[float] = field(default_factory=lambda: [0.0, 1.0])
    likelihoods: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    vocab: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "alpha": self.alpha,
            "priors": dict(self.priors),
            "class_counts": dict(self.class_counts),
            "pcr_edges": list(self.pcr_edges),
            "de_edges": list(self.de_edges),
            "likelihoods": self.likelihoods,
            "vocab": self.vocab,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LgrBayesModel":
        return cls(
            version=int(payload.get("version", LGR_BAYES_MODEL_VERSION)),
            alpha=float(payload.get("alpha", LGR_BAYES_LAPLACE_ALPHA)),
            priors=dict(payload.get("priors", {})),
            class_counts=dict(payload.get("class_counts", {})),
            pcr_edges=[float(x) for x in payload.get("pcr_edges", [0.0, 1.0])],
            de_edges=[float(x) for x in payload.get("de_edges", [0.0, 1.0])],
            likelihoods={
                str(label): {
                    str(feature): {str(k): int(v) for k, v in values.items()}
                    for feature, values in feature_map.items()
                }
                for label, feature_map in dict(payload.get("likelihoods", {})).items()
            },
            vocab={str(k): [str(v) for v in vals] for k, vals in dict(payload.get("vocab", {})).items()},
        )


_MODEL: LgrBayesModel | None = None


def encode_lgr_bayes_features(
    features: Mapping[str, Any],
    *,
    model: LgrBayesModel,
) -> dict[str, str]:
    pair = str(features.get("pair", "")).upper()
    if pair not in {"EURUSD", "GBPUSD"}:
        pair = "OTHER"
    return {
        "pair": pair,
        "session_type": normalize_session_type(features.get("session_type")),
        "minutes_from_session_open": minutes_bin_label(features.get("minutes_from_session_open")),
        "positive_close_ratio": quantile_bin_label(
            features.get("positive_close_ratio"),
            model.pcr_edges,
            prefix="PCR",
        ),
        "directional_efficiency": quantile_bin_label(
            features.get("directional_efficiency"),
            model.de_edges,
            prefix="DE",
        ),
    }


def build_bayes_reason(encoded: Mapping[str, str]) -> str:
    return f"{encoded['positive_close_ratio']}_{encoded['directional_efficiency']}"


def train_lgr_bayes_v1(
    df: pd.DataFrame,
    *,
    alpha: float = LGR_BAYES_LAPLACE_ALPHA,
) -> LgrBayesModel:
    train = df[df["trade_result"].isin(["WIN", "LOSS"])].copy()
    if train.empty:
        raise ValueError("LGR Bayes V1 training requires WIN/LOSS rows")

    train["label"] = train["trade_result"].str.upper().map({"WIN": "win", "LOSS": "loss"})
    model = LgrBayesModel(alpha=alpha)
    model.pcr_edges = _quantile_edges(train["positive_close_ratio"], bins=4)
    model.de_edges = _quantile_edges(train["directional_efficiency"], bins=4)

    win_count = int((train["label"] == "win").sum())
    loss_count = int((train["label"] == "loss").sum())
    total = win_count + loss_count
    model.class_counts = {"win": win_count, "loss": loss_count}
    model.priors = {
        "win": win_count / total,
        "loss": loss_count / total,
    }

    likelihoods: dict[str, dict[str, dict[str, int]]] = {"win": {}, "loss": {}}
    vocab: dict[str, set[str]] = {name: set() for name in V1_FEATURE_NAMES}

    for _, row in train.iterrows():
        encoded = encode_lgr_bayes_features(row, model=model)
        label = str(row["label"])
        for feature, value in encoded.items():
            likelihoods[label].setdefault(feature, {})
            likelihoods[label][feature][value] = likelihoods[label][feature].get(value, 0) + 1
            vocab[feature].add(value)

    model.likelihoods = likelihoods
    model.vocab = {feature: sorted(values) for feature, values in vocab.items()}
    return model


def _laplace_prob(
    *,
    label: str,
    feature: str,
    value: str,
    model: LgrBayesModel,
) -> float:
    counts = model.likelihoods.get(label, {}).get(feature, {})
    class_total = model.class_counts.get(label, 0)
    vocab_size = max(len(model.vocab.get(feature, [])), 1)
    numerator = float(counts.get(value, 0)) + model.alpha
    denominator = float(class_total) + model.alpha * float(vocab_size)
    return numerator / denominator if denominator > 0 else 1.0 / vocab_size


def predict_win_probability(
    features: Mapping[str, Any],
    *,
    model: LgrBayesModel,
) -> tuple[float, dict[str, str]]:
    encoded = encode_lgr_bayes_features(features, model=model)
    win_log = model.priors["win"]
    loss_log = model.priors["loss"]
    win_score = win_log
    loss_score = loss_log
    for feature in V1_FEATURE_NAMES:
        value = encoded[feature]
        win_score *= _laplace_prob(label="win", feature=feature, value=value, model=model)
        loss_score *= _laplace_prob(label="loss", feature=feature, value=value, model=model)
    denom = win_score + loss_score
    if denom <= 0:
        return 0.5, encoded
    return win_score / denom, encoded


def classify_regime(probability: float) -> LgrBayesRegime:
    if probability >= LGR_BAYES_ALLOW_THRESHOLD:
        return "ALLOW"
    if probability >= LGR_BAYES_CAUTION_THRESHOLD:
        return "CAUTION"
    return "REJECT"


def evaluate_lgr_bayes_gate(features: Mapping[str, Any]) -> dict[str, Any]:
    model = get_lgr_bayes_model()
    probability, encoded = predict_win_probability(features, model=model)
    regime = classify_regime(probability)
    return {
        "bayes_probability": round(float(probability), 4),
        "bayes_regime": regime,
        "bayes_reason": build_bayes_reason(encoded),
    }


def features_from_lgr_setup(setup: Any) -> dict[str, Any]:
    feat = setup.lgr_features
    return {
        "pair": setup.pair,
        "session_type": feat.session_type,
        "minutes_from_session_open": feat.minutes_from_session_open,
        "positive_close_ratio": feat.positive_close_ratio,
        "directional_efficiency": feat.directional_efficiency,
    }


def save_lgr_bayes_model(model: LgrBayesModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_lgr_bayes_model(path: Path) -> LgrBayesModel:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return LgrBayesModel.from_dict(payload)


def reset_lgr_bayes_model(model: LgrBayesModel | None = None) -> LgrBayesModel | None:
    global _MODEL
    _MODEL = model
    return _MODEL


def get_lgr_bayes_model() -> LgrBayesModel:
    global _MODEL
    if _MODEL is None:
        raise RuntimeError("LGR Bayes model is not initialized — call initialize_lgr_bayes_gate() first")
    return _MODEL


def initialize_lgr_bayes_gate(
    *,
    train_csv: Path | None = None,
    model_json: Path | None = None,
    retrain: bool = False,
) -> LgrBayesModel:
    csv_path = Path(train_csv or os.getenv("LGR_BAYES_TRAIN_CSV", DEFAULT_TRAIN_CSV))
    json_path = Path(model_json or os.getenv("LGR_BAYES_MODEL_JSON", DEFAULT_MODEL_JSON))

    if json_path.is_file() and not retrain:
        model = load_lgr_bayes_model(json_path)
        logger.info("LGR Bayes V1 loaded from %s", json_path)
    else:
        if not csv_path.is_file():
            raise FileNotFoundError(f"LGR Bayes training CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        model = train_lgr_bayes_v1(df)
        save_lgr_bayes_model(model, json_path)
        logger.info(
            "LGR Bayes V1 trained from %s (%d WIN/LOSS rows) -> %s",
            csv_path,
            model.class_counts["win"] + model.class_counts["loss"],
            json_path,
        )
    return reset_lgr_bayes_model(model)  # type: ignore[return-value]


def profit_factor(profit_r: pd.Series) -> float:
    wins = profit_r[profit_r > 0].sum()
    losses = abs(profit_r[profit_r < 0].sum())
    if losses <= 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def summarize_trade_block(df: pd.DataFrame, *, label: str) -> dict[str, Any]:
    if df.empty:
        return {
            "label": label,
            "count": 0,
            "wr_pct": 0.0,
            "pf": 0.0,
            "total_r": 0.0,
            "avg_r": 0.0,
            "max_dd_pct": 0.0,
        }
    wins = df[df["profit_r"] > 0]
    wr = len(wins) / len(df) * 100.0
    total_r = float(df["profit_r"].sum())
    return {
        "label": label,
        "count": len(df),
        "wr_pct": wr,
        "pf": profit_factor(df["profit_r"]),
        "total_r": total_r,
        "avg_r": float(df["profit_r"].mean()),
        "max_dd_pct": 0.0,
    }


def compute_max_dd_pct_from_r(profit_r: pd.Series) -> float:
    equity = 100.0
    peak = equity
    max_dd = 0.0
    for r in profit_r:
        equity += float(r)
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak * 100.0
            max_dd = max(max_dd, dd)
    return max_dd


def build_feature_importance_tables(model: LgrBayesModel, df: pd.DataFrame) -> tuple[str, str]:
    train_base = df[df["trade_result"].isin(["WIN", "LOSS"])].copy()
    overall_wr = (train_base["trade_result"] == "WIN").mean() * 100.0 if not train_base.empty else 0.0

    feature_rows: list[str] = []
    bin_rows: list[str] = []

    for feature in V1_FEATURE_NAMES:
        bins: list[str] = []
        for _, row in train_base.iterrows():
            encoded = encode_lgr_bayes_features(row, model=model)
            bins.append(encoded[feature])
        work = train_base.copy()
        work["_bin"] = bins
        grouped = work.groupby("_bin", dropna=False)
        feature_wr_spread = 0.0
        for bin_label, sub in grouped:
            if sub.empty:
                continue
            wr = (sub["trade_result"] == "WIN").mean() * 100.0
            pf = profit_factor(sub["profit_r"])
            feature_wr_spread = max(feature_wr_spread, abs(wr - overall_wr))
            bin_rows.append(
                f"| {feature} | {bin_label} | {len(sub)} | {wr:.1f}% | {pf:.3f} | {sub['profit_r'].sum():+.2f} |"
            )
        feature_rows.append(f"| {feature} | {overall_wr:.1f}% | {feature_wr_spread:.1f}pp |")

    feature_table = "\n".join(
        [
            "| Feature | Overall WR | Max bin WR spread |",
            "| --- | ---: | ---: |",
            *feature_rows,
        ]
    )
    bin_table = "\n".join(
        [
            "| Feature | Bin | Count | WR | PF | Total R |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
            *bin_rows,
        ]
    )
    return feature_table, bin_table


def build_lgr_bayes_gate_report(
    records: list[dict[str, Any]],
    *,
    feature_log_df: pd.DataFrame | None = None,
    model: LgrBayesModel | None = None,
    pure_baseline: dict[str, Any] | None = None,
) -> str:
    df = pd.DataFrame(records)
    if df.empty:
        return "# LGR Bayes Gate V1 Report\n\nNo records.\n"

    executed = df[df["trade_result"].isin(["WIN", "LOSS"])].copy()
    shadow = df[
        (df["trade_result"] == "NOT_EXECUTED")
        & df["shadow_result"].isin(["WIN", "LOSS"])
    ].copy()
    rejected = df[df["decision_source"] == "REJECT_BY_LGR_BAYES"].copy()
    reject_shadow = rejected[
        rejected["shadow_result"].isin(["WIN", "LOSS"])
    ].copy()

    exec_summary = summarize_trade_block(executed, label="Executed")
    exec_summary["max_dd_pct"] = compute_max_dd_pct_from_r(executed["profit_r"])

    shadow_summary = summarize_trade_block(
        shadow.assign(profit_r=shadow["shadow_profit_r"]) if not shadow.empty else shadow,
        label="Shadow",
    )

    reject_summary = summarize_trade_block(
        reject_shadow.assign(profit_r=reject_shadow["shadow_profit_r"])
        if not reject_shadow.empty
        else reject_shadow,
        label="Reject",
    )

    regime_counts = {"ALLOW": 0, "CAUTION": 0, "REJECT": 0}
    if feature_log_df is not None and "bayes_regime" in feature_log_df.columns:
        for regime, count in feature_log_df["bayes_regime"].value_counts().items():
            if regime in regime_counts:
                regime_counts[str(regime)] = int(count)

    feature_table = ""
    bin_table = ""
    if feature_log_df is not None and model is not None:
        feature_table, bin_table = build_feature_importance_tables(model, feature_log_df)

    pure_pf = pure_baseline.get("pf") if pure_baseline else None
    pure_dd = pure_baseline.get("max_dd_pct") if pure_baseline else None
    success_lines = []
    if pure_pf is not None:
        success_lines.append(
            f"- Executed PF > Pure PF: **{exec_summary['pf']:.3f} > {pure_pf:.3f}** "
            f"({'PASS' if exec_summary['pf'] > pure_pf else 'FAIL'})"
        )
    if pure_dd is not None:
        success_lines.append(
            f"- Executed MaxDD < Pure MaxDD: **{exec_summary['max_dd_pct']:.2f}% < {pure_dd:.2f}%** "
            f"({'PASS' if exec_summary['max_dd_pct'] < pure_dd else 'FAIL'})"
        )
    success_lines.extend(
        [
            f"- Shadow PF < 1.0: **{shadow_summary['pf']:.3f}** "
            f"({'PASS' if shadow_summary['pf'] < 1.0 else 'FAIL'})",
            f"- Shadow Total R < 0: **{shadow_summary['total_r']:+.2f}** "
            f"({'PASS' if shadow_summary['total_r'] < 0 else 'FAIL'})",
        ]
    )

    comparison_rows = [
        "| Metric | Executed | Shadow | Diff |",
        "| --- | ---: | ---: | ---: |",
        f"| Count | {exec_summary['count']} | {shadow_summary['count']} | {exec_summary['count'] - shadow_summary['count']:+d} |",
        f"| WR | {exec_summary['wr_pct']:.1f}% | {shadow_summary['wr_pct']:.1f}% | {exec_summary['wr_pct'] - shadow_summary['wr_pct']:+.1f}pp |",
        f"| PF | {exec_summary['pf']:.3f} | {shadow_summary['pf']:.3f} | {exec_summary['pf'] - shadow_summary['pf']:+.3f} |",
        f"| Total R | {exec_summary['total_r']:+.2f} | {shadow_summary['total_r']:+.2f} | {exec_summary['total_r'] - shadow_summary['total_r']:+.2f} |",
        f"| Avg R | {exec_summary['avg_r']:+.3f} | {shadow_summary['avg_r']:+.3f} | {exec_summary['avg_r'] - shadow_summary['avg_r']:+.3f} |",
    ]

    lines = [
        "# LGR Bayes Gate V1 Report",
        "",
        "## Executed",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Count | {exec_summary['count']} |",
        f"| WR | {exec_summary['wr_pct']:.1f}% |",
        f"| PF | {exec_summary['pf']:.3f} |",
        f"| Total R | {exec_summary['total_r']:+.2f} |",
        f"| Max DD | {exec_summary['max_dd_pct']:.2f}% |",
        "",
        "## Shadow",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Count | {shadow_summary['count']} |",
        f"| WR | {shadow_summary['wr_pct']:.1f}% |",
        f"| PF | {shadow_summary['pf']:.3f} |",
        f"| Total R | {shadow_summary['total_r']:+.2f} |",
        f"| Avg R | {shadow_summary['avg_r']:+.3f} |",
        "",
        "## Reject Effectiveness",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Reject Count | {len(rejected)} |",
        f"| Reject WR | {reject_summary['wr_pct']:.1f}% |",
        f"| Reject PF | {reject_summary['pf']:.3f} |",
        f"| Reject Total R | {reject_summary['total_r']:+.2f} |",
        f"| Reject Avg R | {reject_summary['avg_r']:+.3f} |",
        "",
        "## Bayes Regime Distribution",
        "",
        "| Regime | Count |",
        "| --- | ---: |",
        f"| ALLOW | {regime_counts['ALLOW']} |",
        f"| CAUTION | {regime_counts['CAUTION']} |",
        f"| REJECT | {regime_counts['REJECT']} |",
        "",
        "## Shadow Comparison",
        "",
        *comparison_rows,
        "",
        "## Success Criteria",
        "",
        *success_lines,
    ]

    if feature_table:
        lines.extend(["", "## Feature Importance", "", feature_table])
    if bin_table:
        lines.extend(["", "### Bin Breakdown", "", bin_table])

    return "\n".join(lines) + "\n"


__all__ = [
    "LGR_BAYES_ALLOW_THRESHOLD",
    "LGR_BAYES_CAUTION_THRESHOLD",
    "LGR_BAYES_LAPLACE_ALPHA",
    "LgrBayesModel",
    "LGR_BAYES_REJECT_SOURCE",
    "build_lgr_bayes_gate_report",
    "evaluate_lgr_bayes_gate",
    "features_from_lgr_setup",
    "initialize_lgr_bayes_gate",
    "is_lgr_bayes_gate_enabled",
    "reset_lgr_bayes_model",
    "train_lgr_bayes_v1",
]

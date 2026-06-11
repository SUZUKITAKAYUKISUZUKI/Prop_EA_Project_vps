"""
lgr_bayes_v2.py — LGR 専用 Naive Bayes Gate V2 (Rank-based)

V1 診断: Calibration 正常だが Probability Compression あり。
V2: minutes_bucket 強化 + 絶対閾値廃止 → percentile rank で ALLOW/CAUTION/REJECT。
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from archive.lgr.lgr_bayes_gate import (
    LGR_BAYES_LAPLACE_ALPHA,
    LGR_BAYES_REJECT_SOURCE,
    LgrBayesModel,
    _laplace_prob,
    _quantile_edges,
    normalize_session_type,
    profit_factor,
    quantile_bin_label,
)

logger = logging.getLogger("lgr_bayes_v2")

LgrBayesRegime = Literal["ALLOW", "CAUTION", "REJECT"]

LGR_BAYES_V2_MODEL_VERSION = 2
DEFAULT_V2_TRAIN_CSV = (
    Path(__file__).resolve().parent / "backtest_results" / "logs" / "lgr_features.csv"
)
DEFAULT_V2_MODEL_JSON = (
    Path(__file__).resolve().parent / "backtest_results" / "models" / "lgr_bayes_v2.json"
)
DEFAULT_V2_FEATURE_LOG = (
    Path(__file__).resolve().parent / "backtest_results" / "logs" / "lgr_bayes_features_3y.csv"
)
DEFAULT_V2_REPORT = (
    Path(__file__).resolve().parent / "backtest_results" / "LGR_BAYES_V2_RANK_GATE.md"
)

V2_FEATURE_NAMES: tuple[str, ...] = (
    "pair",
    "session_type",
    "minutes_bucket",
    "positive_close_ratio",
    "directional_efficiency",
)

# Rank gate presets (percent of population from top)
RANK_GATE_PRESETS: dict[str, dict[str, float]] = {
    "Gate A": {"allow_top_pct": 5.0, "caution_top_pct": 20.0},
    "Gate B": {"allow_top_pct": 10.0, "caution_top_pct": 30.0},
    "Gate C": {"allow_top_pct": 20.0, "caution_top_pct": 50.0},
}

TOP_N_FRONTIER = (50, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000)


def minutes_bucket_label(minutes: Any) -> str:
    """0-30 / 30-60 / 60-120 / 120-240 / 240+"""
    try:
        value = int(float(minutes))
    except (TypeError, ValueError):
        value = 0
    if value < 30:
        return "MB_0_30"
    if value < 60:
        return "MB_30_60"
    if value < 120:
        return "MB_60_120"
    if value < 240:
        return "MB_120_240"
    return "MB_240_PLUS"


def encode_lgr_bayes_v2_features(
    features: Mapping[str, Any],
    *,
    model: LgrBayesModel,
) -> dict[str, str]:
    pair = str(features.get("pair", "")).upper()
    if pair not in {"EURUSD", "GBPUSD"}:
        pair = "OTHER"
    minutes_raw = features.get("minutes_bucket", features.get("minutes_from_session_open"))
    if isinstance(minutes_raw, str) and minutes_raw.startswith("MB_"):
        minutes_bin = minutes_raw
    else:
        minutes_bin = minutes_bucket_label(minutes_raw)
    return {
        "pair": pair,
        "session_type": normalize_session_type(features.get("session_type")),
        "minutes_bucket": minutes_bin,
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


def train_lgr_bayes_v2(
    df: pd.DataFrame,
    *,
    alpha: float = LGR_BAYES_LAPLACE_ALPHA,
) -> LgrBayesModel:
    train = df[df["trade_result"].isin(["WIN", "LOSS"])].copy()
    if train.empty:
        raise ValueError("LGR Bayes V2 training requires WIN/LOSS rows")

    train["label"] = train["trade_result"].str.upper().map({"WIN": "win", "LOSS": "loss"})
    model = LgrBayesModel(version=LGR_BAYES_V2_MODEL_VERSION, alpha=alpha)
    model.pcr_edges = _quantile_edges(train["positive_close_ratio"], bins=4)
    model.de_edges = _quantile_edges(train["directional_efficiency"], bins=4)

    win_count = int((train["label"] == "win").sum())
    loss_count = int((train["label"] == "loss").sum())
    total = win_count + loss_count
    model.class_counts = {"win": win_count, "loss": loss_count}
    model.priors = {"win": win_count / total, "loss": loss_count / total}

    likelihoods: dict[str, dict[str, dict[str, int]]] = {"win": {}, "loss": {}}
    vocab: dict[str, set[str]] = {name: set() for name in V2_FEATURE_NAMES}

    for _, row in train.iterrows():
        encoded = encode_lgr_bayes_v2_features(row, model=model)
        label = str(row["label"])
        for feature, value in encoded.items():
            likelihoods[label].setdefault(feature, {})
            likelihoods[label][feature][value] = likelihoods[label][feature].get(value, 0) + 1
            vocab[feature].add(value)

    model.likelihoods = likelihoods
    model.vocab = {feature: sorted(values) for feature, values in vocab.items()}
    return model


def predict_win_probability_v2(
    features: Mapping[str, Any],
    *,
    model: LgrBayesModel,
) -> tuple[float, dict[str, str]]:
    encoded = encode_lgr_bayes_v2_features(features, model=model)
    win_score = model.priors["win"]
    loss_score = model.priors["loss"]
    for feature in V2_FEATURE_NAMES:
        value = encoded[feature]
        win_score *= _laplace_prob(label="win", feature=feature, value=value, model=model)
        loss_score *= _laplace_prob(label="loss", feature=feature, value=value, model=model)
    denom = win_score + loss_score
    if denom <= 0:
        return 0.5, encoded
    return win_score / denom, encoded


def predict_log_likelihood_breakdown(
    features: Mapping[str, Any],
    *,
    model: LgrBayesModel,
) -> dict[str, Any]:
    """各特徴量の log-likelihood 寄与（win vs loss log-ratio）。"""
    encoded = encode_lgr_bayes_v2_features(features, model=model)
    per_feature: dict[str, dict[str, float]] = {}
    total_log_win = math.log(max(model.priors["win"], 1e-15))
    total_log_loss = math.log(max(model.priors["loss"], 1e-15))

    for feature in V2_FEATURE_NAMES:
        value = encoded[feature]
        p_w = max(_laplace_prob(label="win", feature=feature, value=value, model=model), 1e-15)
        p_l = max(_laplace_prob(label="loss", feature=feature, value=value, model=model), 1e-15)
        log_w = math.log(p_w)
        log_l = math.log(p_l)
        per_feature[feature] = {
            "value": value,
            "log_p_win": log_w,
            "log_p_loss": log_l,
            "log_ratio": log_w - log_l,
        }
        total_log_win += log_w
        total_log_loss += log_l

    log_odds = total_log_win - total_log_loss
    probability = 1.0 / (1.0 + math.exp(-log_odds))
    return {
        "bayes_probability": probability,
        "log_odds": log_odds,
        "per_feature": per_feature,
        "encoded": encoded,
    }


def compute_bayes_probability_rank(
    probabilities: pd.Series,
    *,
    method: str = "average",
) -> pd.Series:
    """
    Percentile rank in [0, 1].
    0.0 = 最下位, 1.0 = 最上位 (highest probability).
    """
    ranks = probabilities.rank(method=method, pct=True)
    return ranks.astype(float)


def classify_regime_by_rank(
    rank: float,
    *,
    allow_top_pct: float,
    caution_top_pct: float,
) -> LgrBayesRegime:
    """
    Top allow_top_pct% → ALLOW
    Next (caution_top_pct - allow_top_pct)% → CAUTION
    Rest → REJECT
    """
    allow_cut = 1.0 - allow_top_pct / 100.0
    caution_cut = 1.0 - caution_top_pct / 100.0
    if rank >= allow_cut:
        return "ALLOW"
    if rank >= caution_cut:
        return "CAUTION"
    return "REJECT"


def apply_rank_gate(
    df: pd.DataFrame,
    *,
    allow_top_pct: float,
    caution_top_pct: float,
    rank_col: str = "bayes_probability_rank",
) -> pd.Series:
    return df[rank_col].apply(
        classify_regime_by_rank,
        allow_top_pct=allow_top_pct,
        caution_top_pct=caution_top_pct,
    )


def prepare_evaluation_frame(df: pd.DataFrame, *, model: LgrBayesModel) -> pd.DataFrame:
    """Outcome + V2 probability + rank を付与。"""
    work = df.copy()
    executed = work["trade_result"].isin(["WIN", "LOSS"])
    work["outcome"] = np.where(executed, work["trade_result"], work["shadow_trade_result"])
    work["outcome_r"] = np.where(executed, work["profit_r"], work["shadow_profit_r"])
    work = work[work["outcome"].isin(["WIN", "LOSS"])].copy()
    work["is_win"] = work["outcome"] == "WIN"

    probs: list[float] = []
    for row in work.itertuples(index=False):
        feats = {
            "pair": row.pair,
            "session_type": row.session_type,
            "minutes_from_session_open": row.minutes_from_session_open,
            "positive_close_ratio": row.positive_close_ratio,
            "directional_efficiency": row.directional_efficiency,
        }
        prob, _ = predict_win_probability_v2(feats, model=model)
        probs.append(prob)
    work["bayes_probability_v2"] = probs
    work["bayes_probability_rank"] = compute_bayes_probability_rank(work["bayes_probability_v2"])
    return work


def derive_position_in_range(row: pd.Series) -> float:
    high = float(row.get("distance_daily_high", 0.0) or 0.0)
    low = float(row.get("distance_daily_low", 0.0) or 0.0)
    span = high + low
    if span <= 0:
        return 0.5
    return float(low / span)


def derive_atr_ratio(row: pd.Series) -> float:
    if "impulse_atr_ratio" in row.index and pd.notna(row["impulse_atr_ratio"]):
        return float(row["impulse_atr_ratio"])
    atr = float(row.get("current_atr_h1", 0.0) or 0.0)
    sweep = float(row.get("sweep_distance_pips", 0.0) or 0.0)
    if atr <= 0:
        return 0.0
    return sweep / atr


def compute_max_dd_pct(profit_r: pd.Series) -> float:
    equity = 100.0
    peak = equity
    max_dd = 0.0
    for r in profit_r:
        equity += float(r)
        if equity > peak:
            peak = equity
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak * 100.0)
    return max_dd


def compute_sharpe(profit_r: pd.Series) -> float:
    if len(profit_r) < 2:
        return 0.0
    std = float(profit_r.std(ddof=1))
    if std <= 0:
        return 0.0
    return float(profit_r.mean() / std)


def summarize_outcomes(sub: pd.DataFrame) -> dict[str, Any]:
    if sub.empty:
        return {
            "count": 0,
            "wr_pct": 0.0,
            "pf": 0.0,
            "total_r": 0.0,
            "avg_r": 0.0,
            "max_dd_pct": 0.0,
            "sharpe": 0.0,
        }
    r = sub["outcome_r"].astype(float)
    wins = sub[sub["is_win"]]
    return {
        "count": len(sub),
        "wr_pct": len(wins) / len(sub) * 100.0,
        "pf": profit_factor(r),
        "total_r": float(r.sum()),
        "avg_r": float(r.mean()),
        "max_dd_pct": compute_max_dd_pct(r),
        "sharpe": compute_sharpe(r),
    }


def cohen_d(top: pd.Series, bottom: pd.Series) -> float:
    top = pd.to_numeric(top, errors="coerce").dropna()
    bottom = pd.to_numeric(bottom, errors="coerce").dropna()
    if top.empty or bottom.empty:
        return 0.0
    n1, n2 = len(top), len(bottom)
    var1 = top.var(ddof=1)
    var2 = bottom.var(ddof=1)
    pooled = ((n1 - 1) * var1 + (n2 - 1) * var2) / max(n1 + n2 - 2, 1)
    if pooled <= 0:
        return 0.0
    return float((top.mean() - bottom.mean()) / math.sqrt(pooled))


def save_lgr_bayes_v2_model(model: LgrBayesModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = model.to_dict()
    payload["version"] = LGR_BAYES_V2_MODEL_VERSION
    payload["feature_names"] = list(V2_FEATURE_NAMES)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_lgr_bayes_v2_model(path: Path) -> LgrBayesModel:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return LgrBayesModel.from_dict(payload)


def initialize_lgr_bayes_v2(
    *,
    train_csv: Path | None = None,
    model_json: Path | None = None,
    retrain: bool = False,
) -> LgrBayesModel:
    csv_path = Path(train_csv or os.getenv("LGR_BAYES_TRAIN_CSV", DEFAULT_V2_TRAIN_CSV))
    json_path = Path(model_json or os.getenv("LGR_BAYES_V2_MODEL_JSON", DEFAULT_V2_MODEL_JSON))

    if json_path.is_file() and not retrain:
        model = load_lgr_bayes_v2_model(json_path)
        logger.info("LGR Bayes V2 loaded from %s", json_path)
        return model

    if not csv_path.is_file():
        raise FileNotFoundError(f"LGR Bayes V2 training CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    model = train_lgr_bayes_v2(df)
    save_lgr_bayes_v2_model(model, json_path)
    logger.info(
        "LGR Bayes V2 trained from %s (%d WIN/LOSS rows) -> %s",
        csv_path,
        model.class_counts["win"] + model.class_counts["loss"],
        json_path,
    )
    return model


def evaluate_lgr_bayes_v2_gate(
    features: Mapping[str, Any],
    *,
    model: LgrBayesModel,
    rank: float,
    allow_top_pct: float,
    caution_top_pct: float,
) -> dict[str, Any]:
    probability, encoded = predict_win_probability_v2(features, model=model)
    regime = classify_regime_by_rank(
        rank,
        allow_top_pct=allow_top_pct,
        caution_top_pct=caution_top_pct,
    )
    return {
        "bayes_probability": round(float(probability), 4),
        "bayes_probability_rank": round(float(rank), 4),
        "bayes_regime": regime,
        "bayes_reason": f"{encoded['minutes_bucket']}_{encoded['positive_close_ratio']}_{encoded['directional_efficiency']}",
    }


__all__ = [
    "LGR_BAYES_REJECT_SOURCE",
    "LGR_BAYES_V2_MODEL_VERSION",
    "RANK_GATE_PRESETS",
    "TOP_N_FRONTIER",
    "V2_FEATURE_NAMES",
    "apply_rank_gate",
    "classify_regime_by_rank",
    "compute_bayes_probability_rank",
    "encode_lgr_bayes_v2_features",
    "evaluate_lgr_bayes_v2_gate",
    "initialize_lgr_bayes_v2",
    "minutes_bucket_label",
    "predict_log_likelihood_breakdown",
    "predict_win_probability_v2",
    "prepare_evaluation_frame",
    "train_lgr_bayes_v2",
]

"""
strategies/vamr_bayes.py — VAMR Phase 3 Naive Bayes probability layer.

Pattern 5 fixed rules are frozen. No ML optimization / feature selection.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from strategies.vamr import STRATEGY_FULL_NAME
from strategies.vamr_features import add_derived_features, load_poc_cohort, profit_factor
from strategies.vamr_phase2 import (
    PATTERN_SPECS,
    Phase2Thresholds,
    apply_pattern_mask,
    max_drawdown_r,
)

STRATEGY_NAME = STRATEGY_FULL_NAME
VAMR_BAYES_MODEL_VERSION = 1
VAMR_BAYES_LAPLACE_ALPHA = 1.0
VAMR_BAYES_MIN_TRAIN_ROWS = 30

FROZEN_PATTERN5_THRESHOLDS = Phase2Thresholds(
    poc_distance_q1_max=0.696370,
    va_width_q1_max=1.259182,
    rejection_q4_min=0.674130,
)
PATTERN5_SPEC = PATTERN_SPECS[5]

PROBABILITY_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0.0-0.2", 0.0, 0.2),
    ("0.2-0.4", 0.2, 0.4),
    ("0.4-0.6", 0.4, 0.6),
    ("0.6-0.8", 0.6, 0.8),
    ("0.8-1.0", 0.8, 1.000001),
)

GO_TOP_BUCKET_PF = 1.20
GO_TOP_BUCKET_TRADES = 300
WATCH_BUCKET_PF = 1.05

NUMERIC_FEATURES: tuple[str, ...] = (
    "price_vs_poc",
    "value_area_width_atr_ratio",
    "rejection_strength",
    "volume_ratio_20ma",
    "atr_percentile_100b",
)
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "session_type",
    "vp_touch_side",
    "htf_aligned",
    "retest_count_bin",
    "primary_pa_type",
)
VAMR_BAYES_FEATURES: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

DEFAULT_MODEL_JSON = (
    Path(__file__).resolve().parents[1] / "backtest_results" / "models" / "vamr_bayes_v1.json"
)
DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "backtest_results/logs/var_features_pure_10y.csv"
DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "backtest_results/logs/var_features_pure_10y_enriched.csv"


def pf_str(pf: float) -> str:
    if np.isinf(pf):
        return "inf"
    if not np.isfinite(pf):
        return "n/a"
    return f"{pf:.3f}"


def retest_count_bin(value: Any) -> str:
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        return "1"
    if n >= 4:
        return "4+"
    return str(max(n, 1))


def normalize_htf_aligned(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"aligned", "not_aligned"}:
        return text
    return "unknown"


def normalize_pa_type(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"ENGULFING", "INSIDE_BAR", "CLOSE_ONLY"}:
        return text
    return "OTHER"


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
        return f"{prefix}_UNK"
    if len(edges) < 2:
        return f"{prefix}_Q1"
    for idx in range(len(edges) - 1):
        lower = edges[idx]
        upper = edges[idx + 1]
        is_last = idx == len(edges) - 2
        if num >= lower and (num <= upper if is_last else num < upper):
            return f"{prefix}_Q{idx + 1}"
    return f"{prefix}_Q{len(edges) - 1}"


@dataclass
class VamrBayesModel:
    version: int = VAMR_BAYES_MODEL_VERSION
    alpha: float = VAMR_BAYES_LAPLACE_ALPHA
    priors: dict[str, float] = field(default_factory=lambda: {"win": 0.5, "loss": 0.5})
    class_counts: dict[str, int] = field(default_factory=lambda: {"win": 0, "loss": 0})
    numeric_edges: dict[str, list[float]] = field(default_factory=dict)
    likelihoods: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    vocab: dict[str, list[str]] = field(default_factory=dict)
    avg_win_r: float = 1.0
    avg_loss_r: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "alpha": self.alpha,
            "priors": dict(self.priors),
            "class_counts": dict(self.class_counts),
            "numeric_edges": self.numeric_edges,
            "likelihoods": self.likelihoods,
            "vocab": self.vocab,
            "avg_win_r": self.avg_win_r,
            "avg_loss_r": self.avg_loss_r,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VamrBayesModel:
        return cls(
            version=int(payload.get("version", VAMR_BAYES_MODEL_VERSION)),
            alpha=float(payload.get("alpha", VAMR_BAYES_LAPLACE_ALPHA)),
            priors=dict(payload.get("priors", {})),
            class_counts=dict(payload.get("class_counts", {})),
            numeric_edges={str(k): [float(x) for x in v] for k, v in dict(payload.get("numeric_edges", {})).items()},
            likelihoods={
                str(label): {
                    str(feature): {str(k): int(v) for k, v in values.items()}
                    for feature, values in feature_map.items()
                }
                for label, feature_map in dict(payload.get("likelihoods", {})).items()
            },
            vocab={str(k): [str(v) for v in vals] for k, vals in dict(payload.get("vocab", {})).items()},
            avg_win_r=float(payload.get("avg_win_r", 1.0)),
            avg_loss_r=float(payload.get("avg_loss_r", 1.0)),
        )


def prepare_pattern5_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = add_derived_features(df.copy())
    work["outcome_label"] = np.where(pd.to_numeric(work["result_r"], errors="coerce").fillna(0.0) > 0, "WIN", "LOSS")
    work["target_win"] = (work["outcome_label"] == "WIN").astype(int)
    work["retest_count_bin"] = work["retest_count"].map(retest_count_bin)
    work["htf_aligned"] = work["htf_aligned"].map(normalize_htf_aligned)
    mask = apply_pattern_mask(work, PATTERN5_SPEC, FROZEN_PATTERN5_THRESHOLDS)
    work = work.loc[mask].copy()
    work["primary_pa_type"] = work["primary_pa_type"].map(normalize_pa_type)
    return work.sort_values("timestamp").reset_index(drop=True)


def load_pattern5_dataset(
    input_path: Path = DEFAULT_INPUT,
    cache_path: Path = DEFAULT_CACHE,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    cohort = load_poc_cohort(input_path, start=start, end=end, enrich=True, cache_path=cache_path)
    return prepare_pattern5_frame(cohort)


def encode_vamr_bayes_features(row: Mapping[str, Any], *, model: VamrBayesModel) -> dict[str, str]:
    encoded: dict[str, str] = {}
    for name in NUMERIC_FEATURES:
        edges = model.numeric_edges.get(name, [0.0, 1.0])
        encoded[name] = quantile_bin_label(row.get(name), edges, prefix=name.upper())
    encoded["session_type"] = str(row.get("session_type", "ASIA")).upper()
    encoded["vp_touch_side"] = str(row.get("vp_touch_side", "VAH")).upper()
    encoded["htf_aligned"] = normalize_htf_aligned(row.get("htf_aligned"))
    encoded["retest_count_bin"] = retest_count_bin(row.get("retest_count"))
    encoded["primary_pa_type"] = normalize_pa_type(row.get("primary_pa_type"))
    return encoded


def train_vamr_bayes(
    df: pd.DataFrame,
    *,
    alpha: float = VAMR_BAYES_LAPLACE_ALPHA,
    min_rows: int = VAMR_BAYES_MIN_TRAIN_ROWS,
) -> VamrBayesModel:
    train = df[df["outcome_label"].isin(["WIN", "LOSS"])].copy()
    if len(train) < min_rows:
        raise ValueError(f"VAMR Bayes requires at least {min_rows} training rows, got {len(train)}")

    model = VamrBayesModel(alpha=alpha)
    for name in NUMERIC_FEATURES:
        model.numeric_edges[name] = _quantile_edges(train[name], bins=4)

    wins = train[train["outcome_label"] == "WIN"]
    losses = train[train["outcome_label"] == "LOSS"]
    win_count = int(len(wins))
    loss_count = int(len(losses))
    total = win_count + loss_count
    model.class_counts = {"win": win_count, "loss": loss_count}
    model.priors = {"win": win_count / total, "loss": loss_count / total}
    model.avg_win_r = float(pd.to_numeric(wins["result_r"], errors="coerce").mean()) if win_count else 1.0
    model.avg_loss_r = float(abs(pd.to_numeric(losses["result_r"], errors="coerce").mean())) if loss_count else 1.0

    likelihoods: dict[str, dict[str, dict[str, int]]] = {"win": {}, "loss": {}}
    vocab: dict[str, set[str]] = {name: set() for name in VAMR_BAYES_FEATURES}

    for row in train.to_dict(orient="records"):
        encoded = encode_vamr_bayes_features(row, model=model)
        label = "win" if row["outcome_label"] == "WIN" else "loss"
        for feature, value in encoded.items():
            likelihoods[label].setdefault(feature, {})
            likelihoods[label][feature][value] = likelihoods[label][feature].get(value, 0) + 1
            vocab[feature].add(value)

    model.likelihoods = likelihoods
    model.vocab = {feature: sorted(values) for feature, values in vocab.items()}
    return model


def _laplace_prob(*, label: str, feature: str, value: str, model: VamrBayesModel) -> float:
    counts = model.likelihoods.get(label, {}).get(feature, {})
    class_total = model.class_counts.get(label, 0)
    vocab_size = max(len(model.vocab.get(feature, [])), 1)
    numerator = float(counts.get(value, 0)) + model.alpha
    denominator = float(class_total) + model.alpha * float(vocab_size)
    return numerator / denominator if denominator > 0 else 1.0 / vocab_size


def predict_win_probability(row: Mapping[str, Any], *, model: VamrBayesModel) -> tuple[float, dict[str, str]]:
    encoded = encode_vamr_bayes_features(row, model=model)
    win_score = model.priors["win"]
    loss_score = model.priors["loss"]
    for feature in VAMR_BAYES_FEATURES:
        value = encoded[feature]
        win_score *= _laplace_prob(label="win", feature=feature, value=value, model=model)
        loss_score *= _laplace_prob(label="loss", feature=feature, value=value, model=model)
    denom = win_score + loss_score
    if denom <= 0:
        return 0.5, encoded
    return float(win_score / denom), encoded


def probability_bucket(probability: float) -> str:
    p = float(np.clip(probability, 0.0, 1.0))
    for label, low, high in PROBABILITY_BUCKETS:
        if low <= p < high or (label == "0.8-1.0" and p >= 0.8):
            return label
    return "0.8-1.0"


def expected_metrics(probability: float, *, model: VamrBayesModel) -> dict[str, float]:
    p = float(np.clip(probability, 0.0, 1.0))
    ev_r = p * model.avg_win_r - (1.0 - p) * model.avg_loss_r
    win_r = model.avg_win_r
    loss_r = model.avg_loss_r
    pf_exp = (p * win_r) / max((1.0 - p) * loss_r, 1e-9) if p < 1.0 else float("inf")
    return {
        "bayes_probability": round(p, 4),
        "expected_ev_r": round(ev_r, 4),
        "expected_pf": round(pf_exp, 4) if np.isfinite(pf_exp) else pf_exp,
        "expected_win_rate_pct": round(p * 100.0, 2),
    }


def score_dataframe(df: pd.DataFrame, model: VamrBayesModel) -> pd.DataFrame:
    probs: list[float] = []
    buckets: list[str] = []
    evs: list[float] = []
    pfs: list[float] = []
    for row in df.to_dict(orient="records"):
        prob, _ = predict_win_probability(row, model=model)
        exp = expected_metrics(prob, model=model)
        probs.append(prob)
        buckets.append(probability_bucket(prob))
        evs.append(exp["expected_ev_r"])
        pf_val = exp["expected_pf"]
        pfs.append(float(pf_val) if np.isfinite(pf_val) else np.nan)
    out = df.copy()
    out["bayes_probability"] = probs
    out["probability_bucket"] = buckets
    out["expected_ev_r"] = evs
    out["expected_pf"] = pfs
    return out


@dataclass
class BucketMetrics:
    bucket: str
    trades: int
    win_rate: float
    pf: float
    avg_r: float
    total_r: float
    max_dd_r: float


def summarize_bucket(bucket: str, df: pd.DataFrame) -> BucketMetrics:
    r = pd.to_numeric(df["result_r"], errors="coerce").fillna(0.0)
    pf = profit_factor(r)
    return BucketMetrics(
        bucket=bucket,
        trades=int(len(df)),
        win_rate=round(float((r > 0).mean() * 100.0), 2) if len(df) else 0.0,
        pf=round(pf, 4) if np.isfinite(pf) else pf,
        avg_r=round(float(r.mean()), 4) if len(df) else 0.0,
        total_r=round(float(r.sum()), 2) if len(df) else 0.0,
        max_dd_r=max_drawdown_r(r),
    )


def bucket_metrics_table(df: pd.DataFrame) -> list[BucketMetrics]:
    rows: list[BucketMetrics] = []
    for label, _, _ in PROBABILITY_BUCKETS:
        seg = df[df["probability_bucket"] == label]
        rows.append(summarize_bucket(label, seg))
    return rows


def phase3_verdict(bucket_rows: list[BucketMetrics]) -> str:
    top = bucket_rows[-1] if bucket_rows else None
    if top and top.trades >= GO_TOP_BUCKET_TRADES and top.avg_r > 0 and np.isfinite(float(top.pf)) and float(top.pf) >= GO_TOP_BUCKET_PF:
        return "GO"
    best_pf = max((float(r.pf) for r in bucket_rows if np.isfinite(float(r.pf)) and r.trades > 0), default=0.0)
    if best_pf >= WATCH_BUCKET_PF:
        return "WATCH"
    return "KILL"


def save_vamr_bayes_model(model: VamrBayesModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_vamr_bayes_model(path: Path) -> VamrBayesModel:
    return VamrBayesModel.from_dict(json.loads(path.read_text(encoding="utf-8")))


def check_probability_monotonicity(bucket_rows: list[BucketMetrics]) -> bool:
    finite = [float(r.pf) for r in bucket_rows if r.trades >= 20 and np.isfinite(float(r.pf))]
    if len(finite) < 2:
        return True
    return finite[-1] >= finite[0]


__all__ = [
    "FROZEN_PATTERN5_THRESHOLDS",
    "PATTERN5_SPEC",
    "PROBABILITY_BUCKETS",
    "VamrBayesModel",
    "bucket_metrics_table",
    "check_probability_monotonicity",
    "expected_metrics",
    "load_pattern5_dataset",
    "load_vamr_bayes_model",
    "phase3_verdict",
    "predict_win_probability",
    "prepare_pattern5_frame",
    "probability_bucket",
    "save_vamr_bayes_model",
    "score_dataframe",
    "summarize_bucket",
    "train_vamr_bayes",
]

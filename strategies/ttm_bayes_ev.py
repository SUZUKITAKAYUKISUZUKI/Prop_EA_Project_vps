"""
TTMS (TTM Short) — Bayesian EV Layer V1.

Categorical Naive Bayes → bayes_win_prob → raw EV → ev_rank → variable sizing.
Entry rejection is forbidden; all trades execute with tiered lot multipliers.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_VERSION = "ttm_bayes_ev_v1"
TTM_BAYES_LAPLACE_ALPHA = 1.0
TTM_BAYES_MIN_TRAIN_ROWS = 30
TTM_BAYES_QCUT_BINS = 5

TTM_EV_PATTERNS: dict[str, tuple[tuple[float, float], ...]] = {
    "A": ((0.95, 1.50), (0.80, 1.25), (0.50, 1.00), (0.0, 0.50)),
    "B": ((0.95, 1.75), (0.80, 1.25), (0.50, 1.00), (0.0, 0.50)),
    "C": ((0.95, 2.00), (0.80, 1.25), (0.50, 1.00), (0.0, 0.50)),
    "D": ((0.95, 2.00), (0.80, 1.50), (0.50, 1.00), (0.0, 0.50)),
    "E": ((0.95, 2.25), (0.80, 1.50), (0.50, 1.00), (0.0, 0.50)),
    "F": ((0.95, 2.50), (0.80, 1.50), (0.50, 1.00), (0.0, 0.50)),
    "G": ((0.95, 3.00), (0.80, 1.50), (0.50, 1.00), (0.0, 0.50)),
    "H": ((0.95, 4.00), (0.80, 1.75), (0.50, 0.75), (0.0, 0.25)),
    "I": ((0.95, 8.00), (0.80, 1.00), (0.50, 0.75), (0.0, 0.25)),
}
TTM_EV_OPTUNA_TRIAL = 12
TTM_EV_OFFICIAL_PATTERN = "I"
TTM_EV_PATTERN_DEFAULT = TTM_EV_OFFICIAL_PATTERN
TTM_EV_TIERS: tuple[tuple[float, float], ...] = TTM_EV_PATTERNS[TTM_EV_OFFICIAL_PATTERN]

WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
VOL_REGIME_LABELS = ("LOW_VOL", "MID_VOL", "HIGH_VOL", "EXTREME_VOL")

TTM_BAYES_TIER1_CATEGORICAL: tuple[str, ...] = (
    "is_gotobi",
    "weekday",
    "week_of_month",
    "minute_bucket",
)

TTM_BAYES_TIER1_NUMERIC: tuple[str, ...] = (
    "minutes_to_ttm",
    "minutes_after_ttm",
)

TTM_BAYES_TIER2_NUMERIC: tuple[str, ...] = (
    "pre_ttm_return",
    "pre_ttm_velocity",
    "pre_ttm_range",
    "pre_ttm_atr_ratio",
    "asian_range",
    "asian_low_distance",
    "asian_range_pct",
    "low_break_distance",
    "low_break_velocity",
)

TTM_BAYES_TIER3_NUMERIC: tuple[str, ...] = (
    "atr_m5",
    "atr_m15",
    "atr_h1",
    "atr_ratio_m5_h1",
    "atr_ratio_m15_h1",
)

TTM_BAYES_TIER4_CATEGORICAL: tuple[str, ...] = (
    "month",
    "quarter",
    "is_month_end",
    "is_quarter_end",
)

TTM_BAYES_ENGINEERED: tuple[str, ...] = (
    "weekday_gotobi",
    "minute_bucket_gotobi",
    "momentum_sign",
    "volatility_regime",
)

TTM_BAYES_QCUT_NUMERIC: tuple[str, ...] = (
    *TTM_BAYES_TIER1_NUMERIC,
    *TTM_BAYES_TIER2_NUMERIC,
    *TTM_BAYES_TIER3_NUMERIC,
)

TTM_BAYES_CATEGORICAL: tuple[str, ...] = (
    *TTM_BAYES_TIER1_CATEGORICAL,
    *TTM_BAYES_TIER4_CATEGORICAL,
    *TTM_BAYES_ENGINEERED,
)

TTM_BAYES_FEATURES: tuple[str, ...] = TTM_BAYES_CATEGORICAL + tuple(
    f"{name}_q" for name in TTM_BAYES_QCUT_NUMERIC
)

TTM_BAYES_EXCLUDED: frozenset[str] = frozenset(
    {
        "trade_id",
        "timestamp",
        "pair",
        "direction",
        "event_type",
        "event_trigger",
        "pattern_class",
        "profit_r",
        "result_r",
        "sized_result_r",
        "bayes_win_prob",
        "ev_rank",
        "ev_lot_multiplier",
        "raw_ev",
        "target_win",
    }
)

DEFAULT_TTM_BAYES_MODEL_JSON = (
    Path(__file__).resolve().parents[1] / "backtest_results" / "models" / "ttm_bayes_ev_v1.json"
)
DEFAULT_TTM_EV_TRAIN_CSV = (
    Path(__file__).resolve().parents[1] / "backtest_results" / "logs" / "ttm_short_features_5y.csv"
)
DEFAULT_TTM_EV_VALID_CSV = (
    Path(__file__).resolve().parents[1] / "backtest_results" / "logs" / "ttm_short_features_wft_5y.csv"
)

EV_BUCKET_EDGES: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
TTM_EV_BOTTOM20_MAX_RANK = 0.2
TTM_EV_TOP20_MIN_RANK = 0.80
DEFAULT_TTM_EV_BOTTOM20_MULT = 0.05
TTM_EV_PYRAMID_TOP20_ENV = "TTM_EV_PYRAMID_TOP20"


def is_ttm_top20_pyramid_enabled() -> bool:
    return os.getenv(TTM_EV_PYRAMID_TOP20_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


def is_ttm_top20_ev_rank(ev_rank: float) -> bool:
    return float(ev_rank) >= TTM_EV_TOP20_MIN_RANK


def is_ttm_ev_sizing_mode() -> bool:
    return os.getenv("TTM_EV_SIZING", "0").strip().lower() in ("1", "true", "yes", "on")


def _bool_label(value: Any) -> str:
    if isinstance(value, str):
        return "YES" if value.strip().lower() in ("1", "true", "yes", "y") else "NO"
    return "YES" if bool(value) else "NO"


def _weekday_label(value: Any) -> str:
    try:
        idx = int(value)
    except (TypeError, ValueError):
        return "UNK"
    if 0 <= idx < len(WEEKDAY_NAMES):
        return WEEKDAY_NAMES[idx]
    return f"D{idx}"


def _quantile_edges(series: pd.Series, *, bins: int = TTM_BAYES_QCUT_BINS) -> list[float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return [0.0, 1.0]
    edges = np.unique(np.quantile(clean, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        return [float(clean.min()), float(clean.max()) + 1.0e-9]
    return [float(x) for x in edges]


def quantile_bin_label(value: Any, edges: Sequence[float], *, prefix: str) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return f"{prefix}_UNK"
    if not edges or len(edges) < 2:
        return f"{prefix}_UNK"
    idx = int(np.searchsorted(edges, num, side="right") - 1)
    idx = max(0, min(idx, len(edges) - 2))
    return f"{prefix}{idx}"


def volatility_regime_label(value: Any, edges: Sequence[float]) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "UNK_VOL"
    if not edges or len(edges) < 2:
        return "UNK_VOL"
    idx = int(np.searchsorted(edges, num, side="right") - 1)
    idx = max(0, min(idx, len(edges) - 2))
    if len(edges) - 1 <= 4:
        return VOL_REGIME_LABELS[min(idx, len(VOL_REGIME_LABELS) - 1)]
    n_bins = len(edges) - 1
    scaled = int(idx * len(VOL_REGIME_LABELS) / max(n_bins, 1))
    return VOL_REGIME_LABELS[min(scaled, len(VOL_REGIME_LABELS) - 1)]


def momentum_sign_label(pre_ttm_return: Any) -> str:
    try:
        return "POS" if float(pre_ttm_return) >= 0.0 else "NEG"
    except (TypeError, ValueError):
        return "UNK"


def target_win_from_row(row: Mapping[str, Any]) -> int:
    r = row.get("result_r", row.get("profit_r", 0))
    try:
        return 1 if float(r) > 0.0 else 0
    except (TypeError, ValueError):
        return 0


def prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    train = df[df["trade_result"].isin(["WIN", "LOSS"])].copy()
    if "result_r" in train.columns:
        r = pd.to_numeric(train["result_r"], errors="coerce")
    else:
        r = pd.to_numeric(train["profit_r"], errors="coerce")
    train["result_r"] = r
    train["target_win"] = (r > 0.0).astype(int)
    return train


@dataclass
class TtmBayesModel:
    version: str = MODEL_VERSION
    alpha: float = TTM_BAYES_LAPLACE_ALPHA
    numeric_edges: dict[str, list[float]] = field(default_factory=dict)
    atr_h1_regime_edges: list[float] = field(default_factory=list)
    class_counts: dict[str, int] = field(default_factory=dict)
    priors: dict[str, float] = field(default_factory=dict)
    likelihoods: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    vocab: dict[str, list[str]] = field(default_factory=dict)
    avg_win_r: float = 1.0
    avg_loss_r: float = 1.0
    reference_probs: tuple[float, ...] = ()
    reference_raw_evs: tuple[float, ...] = ()
    train_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "alpha": self.alpha,
            "numeric_edges": self.numeric_edges,
            "atr_h1_regime_edges": self.atr_h1_regime_edges,
            "class_counts": self.class_counts,
            "priors": self.priors,
            "likelihoods": self.likelihoods,
            "vocab": self.vocab,
            "avg_win_r": self.avg_win_r,
            "avg_loss_r": self.avg_loss_r,
            "reference_probs": list(self.reference_probs),
            "reference_raw_evs": list(self.reference_raw_evs),
            "train_rows": self.train_rows,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TtmBayesModel:
        ref_evs = payload.get("reference_raw_evs")
        if ref_evs is None:
            ref_evs = payload.get("reference_probs", [])
        return cls(
            version=str(payload.get("version", MODEL_VERSION)),
            alpha=float(payload.get("alpha", TTM_BAYES_LAPLACE_ALPHA)),
            numeric_edges={k: list(v) for k, v in dict(payload.get("numeric_edges", {})).items()},
            atr_h1_regime_edges=list(payload.get("atr_h1_regime_edges", [])),
            class_counts={str(k): int(v) for k, v in dict(payload.get("class_counts", {})).items()},
            priors={str(k): float(v) for k, v in dict(payload.get("priors", {})).items()},
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
            reference_probs=tuple(float(x) for x in payload.get("reference_probs", [])),
            reference_raw_evs=tuple(float(x) for x in ref_evs),
            train_rows=int(payload.get("train_rows", 0)),
        )


_MODEL: TtmBayesModel | None = None


def encode_ttm_bayes_features(
    features: Mapping[str, Any],
    *,
    model: TtmBayesModel,
) -> dict[str, str]:
    gotobi = _bool_label(features.get("is_gotobi"))
    weekday = _weekday_label(features.get("weekday", 0))
    minute_bucket = str(features.get("minute_bucket", "UNK"))
    gotobi_suffix = "Gotobi" if gotobi == "YES" else "NotGotobi"

    encoded: dict[str, str] = {
        "is_gotobi": gotobi,
        "weekday": weekday,
        "week_of_month": f"WOM_{int(features.get('week_of_month', 1))}",
        "minute_bucket": minute_bucket,
        "month": f"M_{int(features.get('month', 0))}",
        "quarter": f"Q_{int(features.get('quarter', 0))}",
        "is_month_end": _bool_label(features.get("is_month_end")),
        "is_quarter_end": _bool_label(features.get("is_quarter_end")),
        "weekday_gotobi": f"{weekday}_{gotobi_suffix}",
        "minute_bucket_gotobi": f"{minute_bucket}_{gotobi_suffix}",
        "momentum_sign": momentum_sign_label(features.get("pre_ttm_return")),
        "volatility_regime": volatility_regime_label(
            features.get("atr_h1"),
            model.atr_h1_regime_edges,
        ),
    }
    for name in TTM_BAYES_QCUT_NUMERIC:
        edges = model.numeric_edges.get(name, [])
        encoded[f"{name}_q"] = quantile_bin_label(features.get(name), edges, prefix=f"{name}_q")
    return encoded


def _laplace_prob(*, label: str, feature: str, value: str, model: TtmBayesModel) -> float:
    counts = model.likelihoods.get(label, {}).get(feature, {})
    class_total = model.class_counts.get(label, 0)
    vocab_size = max(len(model.vocab.get(feature, [])), 1)
    numerator = float(counts.get(value, 0)) + model.alpha
    denominator = float(class_total) + model.alpha * float(vocab_size)
    return numerator / denominator if denominator > 0 else 1.0 / vocab_size


def predict_ttm_win_probability(
    features: Mapping[str, Any],
    *,
    model: TtmBayesModel,
) -> float:
    encoded = encode_ttm_bayes_features(features, model=model)
    win_score = model.priors.get("win", 0.5)
    loss_score = model.priors.get("loss", 0.5)
    for feature in TTM_BAYES_FEATURES:
        value = encoded.get(feature, "UNK")
        win_score *= _laplace_prob(label="win", feature=feature, value=value, model=model)
        loss_score *= _laplace_prob(label="loss", feature=feature, value=value, model=model)
    denom = win_score + loss_score
    if denom <= 0:
        return 0.5
    return float(win_score / denom)


def compute_raw_ev(
    bayes_win_prob: float,
    *,
    avg_win_r: float,
    avg_loss_r: float,
) -> float:
    p = min(1.0, max(0.0, float(bayes_win_prob)))
    return float(p * avg_win_r - (1.0 - p) * avg_loss_r)


def compute_ev_rank(score: float, reference_scores: Sequence[float]) -> float:
    if not reference_scores:
        return 0.5
    arr = np.sort(np.asarray(reference_scores, dtype=np.float64))
    rank = float(np.searchsorted(arr, score, side="right") / len(arr))
    return round(min(1.0, max(0.0, rank)), 6)


def get_ttm_ev_pattern() -> str:
    raw = os.getenv("TTM_EV_PATTERN", TTM_EV_PATTERN_DEFAULT).strip().upper()
    if raw.startswith("PATTERN"):
        raw = raw.replace("PATTERN", "").strip()
    return raw if raw in TTM_EV_PATTERNS else TTM_EV_PATTERN_DEFAULT


def get_ttm_ev_tiers() -> tuple[tuple[float, float], ...]:
    return TTM_EV_PATTERNS[get_ttm_ev_pattern()]


def get_ttm_bottom20_mult() -> float:
    raw = os.getenv("TTM_EV_BOTTOM20_MULT", str(DEFAULT_TTM_EV_BOTTOM20_MULT)).strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_TTM_EV_BOTTOM20_MULT


def is_ttm_bottom20_reject() -> bool:
    return get_ttm_bottom20_mult() <= 0.0


def should_reject_ttm_bottom20(ev_rank: float) -> bool:
    return float(ev_rank) < TTM_EV_BOTTOM20_MAX_RANK and is_ttm_bottom20_reject()


def lot_factor_from_ev_rank(
    ev_rank: float,
    *,
    tiers: Sequence[tuple[float, float]] | None = None,
    bottom20_mult: float | None = None,
) -> float:
    rank = float(ev_rank)
    active_tiers = get_ttm_ev_tiers() if tiers is None else tiers
    b20 = DEFAULT_TTM_EV_BOTTOM20_MULT if bottom20_mult is None else max(0.0, float(bottom20_mult))
    if bottom20_mult is None and not is_ttm_bottom20_reject():
        b20 = get_ttm_bottom20_mult()
    if rank < TTM_EV_BOTTOM20_MAX_RANK:
        return b20
    for cutoff, mult in active_tiers:
        if rank >= cutoff:
            return float(mult)
    return float(active_tiers[-1][1])


def filter_training_rows(df: pd.DataFrame, *, train_end: str | None = None) -> pd.DataFrame:
    train = prepare_training_frame(df)
    if train_end and "timestamp" in train.columns:
        ts = pd.to_datetime(train["timestamp"])
        train = train[ts < pd.Timestamp(train_end)]
    return train


def _compute_avg_r_stats(train: pd.DataFrame) -> tuple[float, float]:
    r = pd.to_numeric(train["result_r"], errors="coerce")
    wins = r[r > 0.0]
    losses = r[r <= 0.0]
    avg_win_r = float(wins.mean()) if len(wins) else 1.0
    avg_loss_r = float(abs(losses.mean())) if len(losses) else 1.0
    return avg_win_r, avg_loss_r


def train_ttm_bayes_model(df: pd.DataFrame, *, alpha: float = TTM_BAYES_LAPLACE_ALPHA) -> TtmBayesModel:
    train = prepare_training_frame(df)
    if train.empty:
        raise ValueError("TTM Bayes training requires WIN/LOSS rows")
    if len(train) < TTM_BAYES_MIN_TRAIN_ROWS:
        raise ValueError(
            f"TTM Bayes training produced {len(train)} rows (< {TTM_BAYES_MIN_TRAIN_ROWS})"
        )

    model = TtmBayesModel(alpha=alpha, version=MODEL_VERSION)
    for name in TTM_BAYES_QCUT_NUMERIC:
        if name in train.columns:
            model.numeric_edges[name] = _quantile_edges(train[name], bins=TTM_BAYES_QCUT_BINS)
    if "atr_h1" in train.columns:
        model.atr_h1_regime_edges = _quantile_edges(train["atr_h1"], bins=4)

    model.avg_win_r, model.avg_loss_r = _compute_avg_r_stats(train)

    train = train.copy()
    train["label"] = train["target_win"].map({1: "win", 0: "loss"})
    win_count = int((train["label"] == "win").sum())
    loss_count = int((train["label"] == "loss").sum())
    total = win_count + loss_count
    model.class_counts = {"win": win_count, "loss": loss_count}
    model.priors = {"win": win_count / total, "loss": loss_count / total}

    likelihoods: dict[str, dict[str, dict[str, int]]] = {"win": {}, "loss": {}}
    vocab: dict[str, set[str]] = {name: set() for name in TTM_BAYES_FEATURES}

    for _, row in train.iterrows():
        encoded = encode_ttm_bayes_features(row, model=model)
        label = str(row["label"])
        for feature, value in encoded.items():
            likelihoods[label].setdefault(feature, {})
            likelihoods[label][feature][value] = likelihoods[label][feature].get(value, 0) + 1
            vocab[feature].add(value)

    model.likelihoods = likelihoods
    model.vocab = {feature: sorted(values) for feature, values in vocab.items()}

    ref_probs: list[float] = []
    ref_evs: list[float] = []
    for _, row in train.iterrows():
        p = predict_ttm_win_probability(row, model=model)
        ref_probs.append(p)
        ref_evs.append(
            compute_raw_ev(p, avg_win_r=model.avg_win_r, avg_loss_r=model.avg_loss_r)
        )
    model.reference_probs = tuple(ref_probs)
    model.reference_raw_evs = tuple(ref_evs)
    model.train_rows = len(train)
    return model


def score_ttm_ev_row(
    row: Mapping[str, Any],
    *,
    model: TtmBayesModel,
) -> dict[str, float]:
    bayes_win_prob = predict_ttm_win_probability(row, model=model)
    raw_ev = compute_raw_ev(
        bayes_win_prob,
        avg_win_r=model.avg_win_r,
        avg_loss_r=model.avg_loss_r,
    )
    ev_rank = compute_ev_rank(raw_ev, model.reference_raw_evs)
    lot_multiplier = lot_factor_from_ev_rank(ev_rank)
    return {
        "bayes_win_prob": round(bayes_win_prob, 6),
        "raw_ev": round(raw_ev, 6),
        "ev_rank": round(ev_rank, 6),
        "ev_lot_multiplier": round(lot_multiplier, 4),
    }


def apply_ttm_ev_scores(df: pd.DataFrame, *, model: TtmBayesModel) -> pd.DataFrame:
    out = df.copy()
    for col in ("bayes_win_prob", "raw_ev", "ev_rank", "ev_lot_multiplier", "sized_result_r"):
        if col in out.columns:
            out = out.drop(columns=[col])
    scores = [score_ttm_ev_row(row, model=model) for _, row in out.iterrows()]
    scored = pd.DataFrame(scores, index=out.index)
    out = pd.concat([out, scored], axis=1)
    if "result_r" in out.columns:
        base_r = pd.to_numeric(out["result_r"], errors="coerce").fillna(0.0)
    else:
        base_r = pd.to_numeric(out["profit_r"], errors="coerce").fillna(0.0)
    out["sized_result_r"] = (base_r * out["ev_lot_multiplier"]).round(4)
    return out


def features_from_ttm_setup(setup: Any) -> dict[str, Any]:
    return setup.ttm_features.as_dict()


def evaluate_ttm_ev_sizing_for_setup(setup: Any, *, model: TtmBayesModel | None = None) -> dict[str, float]:
    active = model or get_ttm_bayes_model()
    features = features_from_ttm_setup(setup)
    scored = score_ttm_ev_row(features, model=active)
    return {
        "bayes_win_prob": scored["bayes_win_prob"],
        "ev_rank": scored["ev_rank"],
        "ev_lot_multiplier": scored["ev_lot_multiplier"],
    }


def save_ttm_bayes_model(model: TtmBayesModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_ttm_bayes_model(path: Path) -> TtmBayesModel:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return TtmBayesModel.from_dict(payload)


def reset_ttm_bayes_model(model: TtmBayesModel | None = None) -> TtmBayesModel | None:
    global _MODEL
    _MODEL = model
    return _MODEL


def get_ttm_bayes_model() -> TtmBayesModel:
    if _MODEL is None:
        raise RuntimeError("TTM Bayes model is not initialized — call initialize_ttm_bayes_ev() first")
    return _MODEL


def initialize_ttm_bayes_ev(
    *,
    csv_path: Path | None = None,
    json_path: Path | None = None,
    train_end: str | None = None,
    retrain: bool = False,
    train_rows: pd.DataFrame | None = None,
) -> TtmBayesModel:
    json_path = json_path or DEFAULT_TTM_BAYES_MODEL_JSON
    if train_rows is not None and not train_rows.empty:
        model = train_ttm_bayes_model(train_rows)
        if train_end is None:
            save_ttm_bayes_model(model, json_path)
        logger.info("TTM Bayes V1 trained from in-memory rows (%d)", model.train_rows)
        return reset_ttm_bayes_model(model)  # type: ignore[return-value]

    if json_path.is_file() and not retrain and train_end is None:
        model = load_ttm_bayes_model(json_path)
        logger.info("TTM Bayes V1 loaded from %s (%d train rows)", json_path, model.train_rows)
        return reset_ttm_bayes_model(model)  # type: ignore[return-value]

    csv_path = csv_path or Path(os.getenv("TTM_EV_TRAIN_CSV", str(DEFAULT_TTM_EV_TRAIN_CSV)))
    if not csv_path.is_file():
        raise FileNotFoundError(f"TTM Bayes training CSV not found: {csv_path}")
    raw_df = pd.read_csv(csv_path)
    train_df = filter_training_rows(raw_df, train_end=train_end or os.getenv("TTM_EV_TRAIN_END"))
    if len(train_df) < TTM_BAYES_MIN_TRAIN_ROWS:
        raise ValueError(
            f"TTM Bayes training produced {len(train_df)} rows (< {TTM_BAYES_MIN_TRAIN_ROWS}) from {csv_path}"
        )
    model = train_ttm_bayes_model(train_df)
    if train_end is None:
        save_ttm_bayes_model(model, json_path)
    logger.info(
        "TTM Bayes V1 trained from %s (%d rows%s, avg_win_r=%.3f avg_loss_r=%.3f)",
        csv_path,
        model.train_rows,
        f", cutoff<{train_end}" if train_end else "",
        model.avg_win_r,
        model.avg_loss_r,
    )
    return reset_ttm_bayes_model(model)  # type: ignore[return-value]


def register_ttm_ev_training_row(row: Mapping[str, Any]) -> None:
    _RUNTIME_BUFFER.append(dict(row))


def reset_ttm_ev_runtime() -> None:
    global _RUNTIME_BUFFER
    _RUNTIME_BUFFER = []


def evaluate_ttm_ev_with_runtime(setup: Any) -> dict[str, float]:
    if len(_RUNTIME_BUFFER) >= TTM_BAYES_MIN_TRAIN_ROWS:
        model = train_ttm_bayes_model(pd.DataFrame(_RUNTIME_BUFFER))
        return evaluate_ttm_ev_sizing_for_setup(setup, model=model)
    return {"bayes_win_prob": 0.5, "ev_rank": 0.5, "ev_lot_multiplier": 1.0}


_RUNTIME_BUFFER: list[dict[str, Any]] = []


def profit_factor_from_r(series: pd.Series) -> float:
    r = pd.to_numeric(series, errors="coerce").fillna(0.0)
    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    return float(gains / losses) if losses > 0 else 0.0


def sharpe_from_r(series: pd.Series) -> float:
    r = pd.to_numeric(series, errors="coerce").dropna()
    if len(r) < 2:
        return 0.0
    std = float(r.std(ddof=1))
    if std <= 1e-12:
        return 0.0
    return float(r.mean() / std * math.sqrt(len(r)))


def max_dd_pct_from_r(series: pd.Series) -> float:
    equity = 100.0
    peak = equity
    max_dd = 0.0
    for val in pd.to_numeric(series, errors="coerce").fillna(0.0):
        equity += float(val)
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak * 100.0)
    return max_dd


def calmar_from_r(total_r: float, max_dd_pct: float, *, years: float) -> float:
    if max_dd_pct <= 1e-12 or years <= 0:
        return 0.0
    return float((total_r / years) / (max_dd_pct / 100.0))


def summarize_ev_cohort(df: pd.DataFrame, *, profit_col: str = "sized_result_r") -> dict[str, Any]:
    sub = df[df["trade_result"].isin(["WIN", "LOSS"])].copy()
    if sub.empty:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "total_r": 0.0, "avg_r": 0.0, "sharpe": 0.0}
    r = pd.to_numeric(sub[profit_col], errors="coerce").fillna(0.0)
    wins = int((sub["trade_result"] == "WIN").sum())
    n = len(sub)
    return {
        "n": n,
        "wr": wins / n * 100.0,
        "pf": profit_factor_from_r(r),
        "total_r": float(r.sum()),
        "avg_r": float(r.mean()),
        "sharpe": sharpe_from_r(r),
    }


def build_ttm_ev_bucket_report(records: Sequence[Mapping[str, Any]]) -> str:
    rows: list[dict[str, Any]] = []
    for rec in records:
        if str(rec.get("trade_result", "")) not in ("WIN", "LOSS"):
            continue
        ev_rank = rec.get("ev_rank", rec.get("ttm_ev_rank"))
        if ev_rank in (None, ""):
            continue
        profit_r = float(rec.get("sized_result_r", rec.get("profit_r", 0)) or 0)
        rows.append(
            {
                "ev_rank": float(ev_rank),
                "profit_r": profit_r,
                "win": 1 if rec.get("trade_result") == "WIN" else 0,
            }
        )
    if not rows:
        return "EV bucket report: no executed trades with ev_rank."

    df = pd.DataFrame(rows)
    lines = [
        "",
        "EV Bucket Summary (sized R)",
        "-" * 84,
        f"{'Bucket':<14} {'Trades':>7} {'WR%':>7} {'PF':>7} {'TotalR':>10} {'AvgR':>8}",
    ]
    for lo, hi in zip(EV_BUCKET_EDGES[:-1], EV_BUCKET_EDGES[1:]):
        if hi < 1.0:
            mask = (df["ev_rank"] >= lo) & (df["ev_rank"] < hi)
            label = f"EV {lo:.1f}-{hi:.1f}"
        else:
            mask = (df["ev_rank"] >= lo) & (df["ev_rank"] <= hi)
            label = f"EV {lo:.1f}-{hi:.1f}"
        sub = df[mask]
        if sub.empty:
            lines.append(f"{label:<14} {0:>7} {'-':>7} {'-':>7} {0:>10.2f} {0:>8.3f}")
            continue
        wr = sub["win"].mean() * 100.0
        pf = profit_factor_from_r(sub["profit_r"])
        total_r = sub["profit_r"].sum()
        avg_r = sub["profit_r"].mean()
        lines.append(
            f"{label:<14} {len(sub):>7} {wr:>6.1f}% {pf:>7.3f} {total_r:>10.2f} {avg_r:>8.3f}"
        )
    lines.append("-" * 84)
    return "\n".join(lines)


__all__ = [
    "MODEL_VERSION",
    "TTM_BAYES_FEATURES",
    "TTM_EV_PATTERNS",
    "TTM_EV_OPTUNA_TRIAL",
    "TTM_EV_OFFICIAL_PATTERN",
    "TTM_EV_PATTERN_DEFAULT",
    "TTM_EV_TIERS",
    "apply_ttm_ev_scores",
    "build_ttm_ev_bucket_report",
    "calmar_from_r",
    "compute_ev_rank",
    "compute_raw_ev",
    "evaluate_ttm_ev_with_runtime",
    "initialize_ttm_bayes_ev",
    "DEFAULT_TTM_EV_BOTTOM20_MULT",
    "TTM_EV_BOTTOM20_MAX_RANK",
    "get_ttm_bottom20_mult",
    "get_ttm_ev_pattern",
    "get_ttm_ev_tiers",
    "TTM_EV_TOP20_MIN_RANK",
    "TTM_EV_PYRAMID_TOP20_ENV",
    "is_ttm_top20_ev_rank",
    "is_ttm_top20_pyramid_enabled",
    "is_ttm_ev_sizing_mode",
    "lot_factor_from_ev_rank",
    "max_dd_pct_from_r",
    "should_reject_ttm_bottom20",
    "predict_ttm_win_probability",
    "prepare_training_frame",
    "register_ttm_ev_training_row",
    "reset_ttm_bayes_model",
    "reset_ttm_ev_runtime",
    "score_ttm_ev_row",
    "sharpe_from_r",
    "summarize_ev_cohort",
    "target_win_from_row",
    "train_ttm_bayes_model",
]

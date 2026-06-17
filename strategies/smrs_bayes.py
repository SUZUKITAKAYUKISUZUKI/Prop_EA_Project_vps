"""SMRS Phase 3 — Naive Bayes probability layer (entry features only, frozen rules)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from strategies.smrs_pure import SMRS_PAIRS, STRATEGY_DISPLAY_NAME, STRATEGY_FULL_NAME

STRATEGY_NAME = STRATEGY_FULL_NAME
SMRS_BAYES_MODEL_VERSION = 1
SMRS_BAYES_LAPLACE_ALPHA = 1.0
SMRS_BAYES_MIN_TRAIN_ROWS = 30

DEFAULT_INPUT = (
    Path(__file__).resolve().parents[1] / "backtest_results" / "logs" / "smrs_features_pure_3y.csv"
)
DEFAULT_MODEL_JSON = (
    Path(__file__).resolve().parents[1] / "backtest_results" / "models" / "smrs_bayes_v1.json"
)

PROBABILITY_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0.0-0.2", 0.0, 0.2),
    ("0.2-0.4", 0.2, 0.4),
    ("0.4-0.6", 0.4, 0.6),
    ("0.6-0.8", 0.6, 0.8),
    ("0.8-1.0", 0.8, 1.000001),
)

HIGH_CONF_THRESHOLDS: tuple[float, ...] = (0.70, 0.80, 0.90)

SMRS_BAYES_FEATURES: tuple[str, ...] = (
    "pair",
    "direction",
    "session",
    "hour_bucket",
    "abs_z_score_bucket",
    "atr_percentile_bucket",
    "pair_direction",
)


def pf_str(pf: float) -> str:
    if np.isinf(pf):
        return "inf"
    if not np.isfinite(pf):
        return "n/a"
    return f"{pf:.3f}"


def profit_factor(result_r: pd.Series | np.ndarray) -> float:
    r = np.asarray(result_r, dtype=np.float64)
    if r.size == 0:
        return 0.0
    gw = r[r > 0].sum()
    gl = abs(r[r < 0].sum())
    if gl <= 0:
        return float("inf") if gw > 0 else 0.0
    return float(gw / gl)


def max_drawdown_r(result_r: pd.Series | np.ndarray) -> float:
    r = np.asarray(result_r, dtype=np.float64)
    if r.size == 0:
        return 0.0
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    return float((eq - peak).min())


def normalize_session(value: Any, hour: int | None = None) -> str:
    text = str(value or "").strip().upper()
    if text in {"LONDON", "NY"}:
        return text
    if hour is not None:
        if 7 <= hour < 13:
            return "LONDON"
        if 13 <= hour < 22:
            return "NY"
    return "NY"


def hour_bucket(hour: int) -> str:
    if 7 <= hour <= 9:
        return "07_09"
    if 10 <= hour <= 12:
        return "10_12"
    if 13 <= hour <= 16:
        return "13_16"
    if 17 <= hour <= 20:
        return "17_20"
    if 21 <= hour <= 23:
        return "21_23"
    return "OTHER"


def abs_z_score_bucket(z_score: Any) -> str:
    try:
        az = abs(float(z_score))
    except (TypeError, ValueError):
        return "2.0_2.5"
    if az < 2.5:
        return "2.0_2.5"
    if az < 3.0:
        return "2.5_3.0"
    if az < 3.5:
        return "3.0_3.5"
    if az < 4.0:
        return "3.5_4.0"
    return "4.0_plus"


def atr_percentile_bucket(value: Any, *, edges: Sequence[float]) -> str:
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return "MID"
    if len(edges) < 2:
        return "MID"
    low_edge, high_edge = float(edges[0]), float(edges[1])
    if ratio < low_edge:
        return "LOW"
    if ratio >= high_edge:
        return "HIGH"
    return "MID"


@dataclass
class SmrsBayesModel:
    version: int = SMRS_BAYES_MODEL_VERSION
    alpha: float = SMRS_BAYES_LAPLACE_ALPHA
    priors: dict[str, float] = field(default_factory=lambda: {"win": 0.5, "loss": 0.5})
    class_counts: dict[str, int] = field(default_factory=lambda: {"win": 0, "loss": 0})
    atr_tertile_edges: list[float] = field(default_factory=lambda: [0.0, 1.0])
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
            "atr_tertile_edges": self.atr_tertile_edges,
            "likelihoods": self.likelihoods,
            "vocab": self.vocab,
            "avg_win_r": self.avg_win_r,
            "avg_loss_r": self.avg_loss_r,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SmrsBayesModel:
        return cls(
            version=int(payload.get("version", SMRS_BAYES_MODEL_VERSION)),
            alpha=float(payload.get("alpha", SMRS_BAYES_LAPLACE_ALPHA)),
            priors=dict(payload.get("priors", {})),
            class_counts=dict(payload.get("class_counts", {})),
            atr_tertile_edges=[float(x) for x in payload.get("atr_tertile_edges", [0.0, 1.0])],
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


def prepare_smrs_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    work = work[work["pair"].isin(SMRS_PAIRS)].copy()
    work["pair"] = work["pair"].astype(str).str.upper()
    work["direction"] = work["direction"].astype(str).str.upper()
    work["hour"] = work["timestamp"].dt.hour.astype(int)
    work["session"] = [
        normalize_session(sess, hour=int(h))
        for sess, h in zip(work["session"], work["hour"])
    ]
    work["hour_bucket"] = work["hour"].map(hour_bucket)
    work["abs_z_score_bucket"] = work["z_score_entry"].map(abs_z_score_bucket)
    work["pair_direction"] = work["pair"] + "_" + work["direction"]
    work["atr_ratio"] = (
        pd.to_numeric(work["atr_entry"], errors="coerce")
        / pd.to_numeric(work["atr_p50_entry"], errors="coerce").replace(0, np.nan)
    )
    work["result_r"] = pd.to_numeric(work["result_r"], errors="coerce").fillna(0.0)
    work["target_win"] = np.where(work["result_r"] > 0, 1, 0)
    work["outcome_label"] = np.where(work["target_win"] == 1, "WIN", "LOSS")
    return work.sort_values("timestamp").reset_index(drop=True)


def load_smrs_dataset(
    input_path: Path = DEFAULT_INPUT,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    from src.services.feature_loader import load_feature_dataframe

    df = load_feature_dataframe(input_path, start=start, end=end)
    if df.empty:
        raise FileNotFoundError(f"SMRS feature log not found in SQLite or CSV: {input_path}")
    work = prepare_smrs_frame(df)
    return work.sort_values("timestamp").reset_index(drop=True)


def encode_smrs_bayes_features(row: Mapping[str, Any], *, model: SmrsBayesModel) -> dict[str, str]:
    return {
        "pair": str(row.get("pair", "")).upper(),
        "direction": str(row.get("direction", "")).upper(),
        "session": normalize_session(row.get("session"), hour=int(row.get("hour", 0))),
        "hour_bucket": str(row.get("hour_bucket", hour_bucket(int(row.get("hour", 0))))),
        "abs_z_score_bucket": str(row.get("abs_z_score_bucket", abs_z_score_bucket(row.get("z_score_entry")))),
        "atr_percentile_bucket": atr_percentile_bucket(
            row.get("atr_ratio"),
            edges=model.atr_tertile_edges,
        ),
        "pair_direction": str(row.get("pair_direction", "")).upper(),
    }


def _compute_atr_tertile_edges(train: pd.DataFrame) -> list[float]:
    ratios = pd.to_numeric(train["atr_ratio"], errors="coerce").dropna()
    if ratios.empty:
        return [0.0, 1.0]
    p33 = float(ratios.quantile(1.0 / 3.0))
    p67 = float(ratios.quantile(2.0 / 3.0))
    if p33 >= p67:
        p67 = p33 + 1e-9
    return [p33, p67]


def train_smrs_bayes(
    df: pd.DataFrame,
    *,
    alpha: float = SMRS_BAYES_LAPLACE_ALPHA,
    min_rows: int = SMRS_BAYES_MIN_TRAIN_ROWS,
) -> SmrsBayesModel:
    train = df[df["outcome_label"].isin(["WIN", "LOSS"])].copy()
    if len(train) < min_rows:
        raise ValueError(f"SMRS Bayes requires at least {min_rows} training rows, got {len(train)}")

    model = SmrsBayesModel(alpha=alpha)
    model.atr_tertile_edges = _compute_atr_tertile_edges(train)

    wins = train[train["outcome_label"] == "WIN"]
    losses = train[train["outcome_label"] == "LOSS"]
    win_count = int(len(wins))
    loss_count = int(len(losses))
    total = win_count + loss_count
    model.class_counts = {"win": win_count, "loss": loss_count}
    model.priors = {"win": win_count / total, "loss": loss_count / total}
    model.avg_win_r = float(wins["result_r"].mean()) if win_count else 1.0
    model.avg_loss_r = float(abs(losses["result_r"].mean())) if loss_count else 1.0

    likelihoods: dict[str, dict[str, dict[str, int]]] = {"win": {}, "loss": {}}
    vocab: dict[str, set[str]] = {name: set() for name in SMRS_BAYES_FEATURES}

    for row in train.to_dict(orient="records"):
        encoded = encode_smrs_bayes_features(row, model=model)
        label = "win" if row["outcome_label"] == "WIN" else "loss"
        for feature, value in encoded.items():
            likelihoods[label].setdefault(feature, {})
            likelihoods[label][feature][value] = likelihoods[label][feature].get(value, 0) + 1
            vocab[feature].add(value)

    model.likelihoods = likelihoods
    model.vocab = {feature: sorted(values) for feature, values in vocab.items()}
    return model


def _laplace_prob(*, label: str, feature: str, value: str, model: SmrsBayesModel) -> float:
    counts = model.likelihoods.get(label, {}).get(feature, {})
    class_total = model.class_counts.get(label, 0)
    vocab_size = max(len(model.vocab.get(feature, [])), 1)
    numerator = float(counts.get(value, 0)) + model.alpha
    denominator = float(class_total) + model.alpha * float(vocab_size)
    return numerator / denominator if denominator > 0 else 1.0 / vocab_size


def predict_win_probability(row: Mapping[str, Any], *, model: SmrsBayesModel) -> tuple[float, dict[str, str]]:
    encoded = encode_smrs_bayes_features(row, model=model)
    win_score = math.log(max(model.priors["win"], 1e-15))
    loss_score = math.log(max(model.priors["loss"], 1e-15))
    for feature in SMRS_BAYES_FEATURES:
        value = encoded[feature]
        win_score += math.log(_laplace_prob(label="win", feature=feature, value=value, model=model))
        loss_score += math.log(_laplace_prob(label="loss", feature=feature, value=value, model=model))
    max_score = max(win_score, loss_score)
    win_exp = math.exp(win_score - max_score)
    loss_exp = math.exp(loss_score - max_score)
    denom = win_exp + loss_exp
    if denom <= 0:
        return 0.5, encoded
    return float(win_exp / denom), encoded


def probability_bucket(probability: float) -> str:
    p = float(np.clip(probability, 0.0, 1.0))
    for label, low, high in PROBABILITY_BUCKETS:
        if low <= p < high or (label == "0.8-1.0" and p >= 0.8):
            return label
    return "0.8-1.0"


def score_dataframe(df: pd.DataFrame, model: SmrsBayesModel) -> pd.DataFrame:
    probs: list[float] = []
    buckets: list[str] = []
    encoded_rows: list[dict[str, str]] = []
    for row in df.to_dict(orient="records"):
        prob, encoded = predict_win_probability(row, model=model)
        probs.append(prob)
        buckets.append(probability_bucket(prob))
        encoded_rows.append(encoded)
    out = df.copy()
    out["bayes_probability"] = probs
    out["probability_bucket"] = buckets
    for feature in SMRS_BAYES_FEATURES:
        out[f"feat_{feature}"] = [enc[feature] for enc in encoded_rows]
    out["atr_percentile_bucket"] = out["feat_atr_percentile_bucket"]
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
    mean_prob: float


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
        mean_prob=round(float(df["bayes_probability"].mean()), 4) if len(df) else 0.0,
    )


def bucket_metrics_table(df: pd.DataFrame) -> list[BucketMetrics]:
    rows: list[BucketMetrics] = []
    for label, _, _ in PROBABILITY_BUCKETS:
        seg = df[df["probability_bucket"] == label]
        rows.append(summarize_bucket(label, seg))
    return rows


def check_probability_monotonicity(bucket_rows: list[BucketMetrics], *, metric: str = "pf") -> bool:
    active = [getattr(r, metric) for r in bucket_rows if r.trades >= 20 and np.isfinite(float(getattr(r, metric)))]
    if len(active) < 2:
        return True
    return active[-1] >= active[0]


def check_win_rate_monotonicity(bucket_rows: list[BucketMetrics]) -> bool:
    active = [r.win_rate for r in bucket_rows if r.trades >= 20]
    if len(active) < 2:
        return True
    return active[-1] >= active[0]


def save_smrs_bayes_model(model: SmrsBayesModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_smrs_bayes_model(path: Path) -> SmrsBayesModel:
    return SmrsBayesModel.from_dict(json.loads(path.read_text(encoding="utf-8")))


def export_phase3_results(scored_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(scored_df["timestamp"]),
            "pair": scored_df["pair"].astype(str).str.upper(),
            "direction": scored_df["direction"].astype(str).str.upper(),
            "bayes_probability": pd.to_numeric(scored_df["bayes_probability"], errors="coerce"),
            "probability_bucket": scored_df["probability_bucket"].astype(str),
            "trade_r": pd.to_numeric(scored_df["result_r"], errors="coerce").fillna(0.0),
            "trade_result": pd.to_numeric(scored_df["target_win"], errors="coerce").fillna(0).astype(int),
            "wft_window_id": pd.to_numeric(scored_df.get("wft_window_id"), errors="coerce"),
        }
    ).dropna(subset=["bayes_probability"]).sort_values("timestamp").reset_index(drop=True)


def phase3_verdict(
    *,
    bucket_rows: list[BucketMetrics],
    window_rows: list[dict],
    brier: float,
    calibration_error: float,
    baseline_pf: float,
) -> tuple[str, list[str]]:
    notes: list[str] = []
    monotonic_pf = check_probability_monotonicity(bucket_rows, metric="pf")
    monotonic_wr = check_win_rate_monotonicity(bucket_rows)
    low = bucket_rows[0] if bucket_rows else None
    high = bucket_rows[-1] if bucket_rows else None

    if not monotonic_pf:
        notes.append("PF does not increase monotonically across probability buckets (min 20 trades/bucket).")
    if not monotonic_wr:
        notes.append("Win rate does not increase monotonically across probability buckets.")

    if low and high and high.trades >= 20 and low.trades >= 20:
        if not (high.pf >= low.pf and high.win_rate >= low.win_rate):
            notes.append("High-probability bucket does not outperform low-probability bucket on PF/WR.")

    pfs = [float(w["pf"]) for w in window_rows if w.get("trades", 0) > 0 and np.isfinite(float(w["pf"]))]
    positive_pct = (sum(1 for pf in pfs if pf >= 1.0) / len(pfs) * 100.0) if pfs else 0.0
    mean_pf = float(np.mean(pfs)) if pfs else 0.0
    if positive_pct < 70.0:
        notes.append(f"Only {positive_pct:.0f}% of OOS windows have PF >= 1.0 (threshold 70%).")
    if mean_pf < baseline_pf * 0.85:
        notes.append(f"Mean OOS window PF ({mean_pf:.3f}) materially below baseline ({baseline_pf:.3f}).")

    if brier > 0.25:
        notes.append(f"Brier score {brier:.4f} suggests weak probability calibration.")
    if calibration_error > 0.12:
        notes.append(f"Mean calibration error {calibration_error:.4f} exceeds 0.12.")

    notes.append("Leakage check: each window trains only on trades with timestamp < OOS start (strict).")

    if not notes or (monotonic_pf and monotonic_wr and positive_pct >= 70.0 and mean_pf >= 1.0):
        return "GO", notes
    if monotonic_wr and mean_pf >= 1.0 and positive_pct >= 60.0:
        return "WATCH", notes
    return "NO GO", notes

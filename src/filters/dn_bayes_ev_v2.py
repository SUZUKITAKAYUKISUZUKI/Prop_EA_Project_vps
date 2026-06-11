"""
DiNapoli (DN) — Bayesian EV Rank V2 (interpretable linear score + BT reference ranks).

Trained on BT feature log only; WFT scored against BT reference distribution.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

MODEL_VERSION = "dn_bayes_ev_v2"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "backtest_results" / "models" / "dn_bayes_ev_v2.json"

_default_model: DnEvV2Model | None = None

DEFAULT_FEATURE_COLS: tuple[str, ...] = (
    "momentum_score",
    "pullback_depth",
    "ema_alignment_score",
    "trend_strength",
    "volatility_regime",
    "atr_m15",
    "atr_h4",
    "rsi_m15",
    "tick_volume",
    "volume_zscore",
    "distance_to_daily_high",
    "distance_to_daily_low",
    "distance_to_asia_high",
    "distance_to_asia_low",
    "distance_to_london_high",
    "distance_to_london_low",
)


@dataclass
class DnEvV2Model:
    feature_cols: tuple[str, ...] = DEFAULT_FEATURE_COLS
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    reference_scores: list[float] = field(default_factory=list)
    model_version: str = MODEL_VERSION

    def raw_score_row(self, row: Mapping[str, Any]) -> float:
        total = 0.0
        for col in self.feature_cols:
            w = float(self.weights.get(col, 0.0))
            if w == 0.0:
                continue
            val = float(row.get(col, 0) or 0)
            mu = float(self.means.get(col, 0.0))
            sd = float(self.stds.get(col, 1.0))
            if sd <= 1e-12:
                z = 0.0
            else:
                z = (val - mu) / sd
            total += w * z
        return float(total)

    def rank_score(self, raw: float) -> float:
        if not self.reference_scores:
            return 0.5
        arr = np.asarray(self.reference_scores, dtype=np.float64)
        return float(np.mean(arr <= raw))

    def score_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        raw = self.raw_score_row(row)
        ev_rank = round(max(0.0, min(1.0, self.rank_score(raw))), 6)
        return {
            "ev_rank_v2": ev_rank,
            "ev_raw_v2": round(raw, 6),
            "model_version": self.model_version,
        }

    def score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        raw = out.apply(lambda r: self.raw_score_row(r), axis=1)
        out["ev_raw_v2"] = raw
        if self.reference_scores:
            ref = np.asarray(self.reference_scores, dtype=np.float64)
            out["ev_rank_v2"] = raw.map(lambda x: float(np.mean(ref <= x)))
        else:
            out["ev_rank_v2"] = 0.5
        out["ev_rank_v2"] = out["ev_rank_v2"].clip(0.0, 1.0).round(6)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "feature_cols": list(self.feature_cols),
            "means": self.means,
            "stds": self.stds,
            "weights": self.weights,
            "reference_scores": self.reference_scores,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DnEvV2Model:
        return cls(
            feature_cols=tuple(payload.get("feature_cols", DEFAULT_FEATURE_COLS)),
            means={str(k): float(v) for k, v in dict(payload.get("means", {})).items()},
            stds={str(k): float(v) for k, v in dict(payload.get("stds", {})).items()},
            weights={str(k): float(v) for k, v in dict(payload.get("weights", {})).items()},
            reference_scores=[float(x) for x in payload.get("reference_scores", [])],
            model_version=str(payload.get("model_version", MODEL_VERSION)),
        )

    def save_json(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path | str) -> DnEvV2Model:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


def fit_dn_ev_v2_model(
    train_df: pd.DataFrame,
    *,
    feature_cols: Sequence[str] = DEFAULT_FEATURE_COLS,
    rank_weights: dict[str, float] | None = None,
) -> DnEvV2Model:
    """
    Fit z-score stats on BT and combine rank-based weights (Step 3) with Ridge on result_r.
    """
    from sklearn.linear_model import Ridge

    cols = [c for c in feature_cols if c in train_df.columns]
    sub = train_df.copy()
    for col in cols:
        sub[col] = pd.to_numeric(sub[col], errors="coerce").fillna(0.0)

    means = {c: float(sub[c].mean()) for c in cols}
    stds = {c: float(sub[c].std(ddof=0)) if sub[c].std(ddof=0) > 1e-12 else 1.0 for c in cols}

    X = sub[cols].to_numpy(dtype=np.float64)
    y = pd.to_numeric(sub["result_r"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    Z = np.zeros_like(X)
    for i, c in enumerate(cols):
        sd = stds[c]
        Z[:, i] = (X[:, i] - means[c]) / sd if sd > 1e-12 else 0.0
    ridge = Ridge(alpha=10.0, random_state=42)
    ridge.fit(Z, y)
    ridge_w = {c: float(w) for c, w in zip(cols, ridge.coef_)}

    if rank_weights:
        fused: dict[str, float] = {}
        for c in cols:
            rw = float(rank_weights.get(c, 0.0))
            sign = 1.0 if ridge_w.get(c, 0.0) >= 0 else -1.0
            fused[c] = abs(rw) * sign + 0.25 * ridge_w.get(c, 0.0)
    else:
        fused = ridge_w

    abs_sum = sum(abs(v) for v in fused.values()) or 1.0
    weights = {c: float(fused.get(c, 0.0) / abs_sum) for c in cols}

    model = DnEvV2Model(
        feature_cols=tuple(cols),
        means=means,
        stds=stds,
        weights=weights,
    )
    model.reference_scores = [model.raw_score_row(r) for _, r in sub.iterrows()]
    return model


def get_default_dn_ev_v2_model() -> DnEvV2Model:
    global _default_model
    if _default_model is None:
        path = Path(os.getenv("DN_EV_V2_MODEL", str(DEFAULT_MODEL_PATH)))
        _default_model = DnEvV2Model.load_json(path)
    return _default_model


def score_dn_ev_v2_row(row: Mapping[str, Any], *, model: DnEvV2Model | None = None) -> dict[str, Any]:
    m = model or get_default_dn_ev_v2_model()
    scored = m.score_row(row)
    return {
        "ev_rank_v2": float(scored["ev_rank_v2"]),
        "ev_raw_v2": float(scored["ev_raw_v2"]),
        "model_version": m.model_version,
    }


def apply_dn_ev_v2_to_row(row: dict[str, Any], *, model: DnEvV2Model | None = None) -> dict[str, Any]:
    scored = score_dn_ev_v2_row(row, model=model)
    row["ev_rank_v2"] = scored["ev_rank_v2"]
    row["ev_rank"] = scored["ev_rank_v2"]
    row["model_version"] = scored["model_version"]
    return row

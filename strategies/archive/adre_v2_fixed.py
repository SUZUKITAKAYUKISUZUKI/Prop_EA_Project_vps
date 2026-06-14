"""
strategies/adre_v2_fixed.py — ADRE V2 fixed-rule standalone strategy.

Entry filter (all required):
  EURUSD | BUY | ADR Remaining D8–D9 | MID session (360–720 min)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategies.archive.adre import (
    ADRE_BAYES_PURE_PROB,
    AdreSetup,
    AdreStrategy,
    SETUP_TYPE,
    build_adre_l6_fields,
)
from strategies.base import StrategyResult

V2_SETUP_TYPE = "ADRE_V2"
V2_PAIR = "EURUSD"
V2_DIRECTION = "BUY"
V2_ADR_BUCKETS = frozenset({"D8", "D9"})
V2_SESSION = "MID"

SESSION_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("EARLY", 0, 360),
    ("MID", 360, 720),
    ("LATE", 720, 1080),
    ("CLOSE", 1080, float("inf")),
)

DEFAULT_EDGES_PATH = Path(__file__).resolve().parents[2] / "archive" / "adre" / "adre_bayes_model.json"


def session_bucket(minutes: float) -> str:
    for name, lo, hi in SESSION_BUCKETS:
        if lo <= minutes < hi:
            return name
    return "CLOSE"


def load_frozen_adr_edges(path: Path = DEFAULT_EDGES_PATH) -> np.ndarray:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("adr_decile_edges", [])
        edges = np.array(
            [np.nan if x is None else float(x) for x in raw],
            dtype=np.float64,
        )
        if len(edges) >= 2:
            edges[0] = -np.inf
            edges[-1] = np.inf
            return np.unique(edges)
    return np.array([-np.inf, 0.0, np.inf], dtype=np.float64)


def assign_adr_bucket(adr_remaining: float, edges: np.ndarray) -> str:
    labels = [f"D{i}" for i in range(1, len(edges))]
    bucket = pd.cut(
        [float(adr_remaining)],
        bins=edges,
        labels=labels[: len(edges) - 1],
        include_lowest=True,
    )[0]
    return str(bucket)


def passes_v2_rules(
    setup: AdreSetup,
    *,
    adr_edges: np.ndarray | None = None,
) -> bool:
    if setup.pair.upper() != V2_PAIR:
        return False
    if setup.direction.upper() != V2_DIRECTION:
        return False
    if session_bucket(float(setup.session_minutes_elapsed)) != V2_SESSION:
        return False
    edges = adr_edges if adr_edges is not None else load_frozen_adr_edges()
    bucket = assign_adr_bucket(float(setup.adr_remaining), edges)
    return bucket in V2_ADR_BUCKETS


class ADREV2FixedStrategy(AdreStrategy):
    """ADR Expansion V2 — fixed EURUSD BUY / ADR D8–D9 / MID filter."""

    def __init__(
        self,
        weights_config: dict[str, int] | None = None,
        mode_h1: bool = False,
        *,
        adr_edges: np.ndarray | None = None,
    ) -> None:
        super().__init__(weights_config=weights_config, mode_h1=mode_h1)
        self._adr_edges = adr_edges if adr_edges is not None else load_frozen_adr_edges()

    @property
    def setup_type(self) -> str:
        return V2_SETUP_TYPE

    def analyze_setup(
        self,
        setup: Any,
        gbp_setup: Any | None,
        eur_setup: Any | None,
        h1_gbp: pd.DataFrame,
        h1_eur: pd.DataFrame,
    ) -> StrategyResult:
        result = super().analyze_setup(
            setup, gbp_setup, eur_setup, h1_gbp, h1_eur
        )
        if not result.is_setup or not isinstance(setup, AdreSetup):
            return result
        if passes_v2_rules(setup, adr_edges=self._adr_edges):
            result.setup_type = V2_SETUP_TYPE
            result.strategy_action = "ALLOW"
            result.raw_features["adre_v2_rule"] = "PASS"
        else:
            result.strategy_action = "REJECT"
            result.raw_features["adre_v2_rule"] = "FAIL"
            result.raw_features["reject_reason"] = "ADRE_V2_RULE"
        return result


def enrich_v2_frame(df: pd.DataFrame, adr_edges: np.ndarray) -> pd.DataFrame:
    """Add bucket columns and return EURUSD/BUY weekday rows."""
    out = df.copy()
    out = out[out["pair"].astype(str).str.upper() == V2_PAIR]
    out = out[out["direction"].astype(str).str.upper() == V2_DIRECTION]
    out = out[out["day_of_week"].between(0, 4, inclusive="both")]
    labels = [f"D{i}" for i in range(1, len(adr_edges))]
    out["adr_bucket"] = pd.cut(
        pd.to_numeric(out["adr_remaining"], errors="coerce"),
        bins=adr_edges,
        labels=labels[: len(adr_edges) - 1],
        include_lowest=True,
    ).astype(str)
    out["session_bucket"] = out["session_minutes_elapsed"].apply(session_bucket)
    return out.dropna(subset=["adr_bucket", "session_bucket"]).reset_index(drop=True)


def v2_mask(df: pd.DataFrame) -> pd.Series:
    return (
        (df["pair"].astype(str).str.upper() == V2_PAIR)
        & (df["direction"].astype(str).str.upper() == V2_DIRECTION)
        & (df["adr_bucket"].isin(V2_ADR_BUCKETS))
        & (df["session_bucket"] == V2_SESSION)
    )


__all__ = [
    "ADREV2FixedStrategy",
    "V2_ADR_BUCKETS",
    "V2_SETUP_TYPE",
    "assign_adr_bucket",
    "enrich_v2_frame",
    "load_frozen_adr_edges",
    "passes_v2_rules",
    "session_bucket",
    "v2_mask",
]

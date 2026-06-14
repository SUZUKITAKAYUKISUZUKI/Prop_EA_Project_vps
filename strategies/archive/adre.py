"""
strategies/adre.py — ADR Expansion (ADRE) Pure Discovery strategy.

H1 breakout of prior-day high/low. No scoring, Bayes, or kill switches in pure mode.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from strategies.base import StrategyResult
from strategies.base_strategy import BaseStrategy
from strategies.archive.adre_detector import (
    ADRE_PAIRS,
    RR_RATIO,
    detect_adre_setups_clipped,
    detect_adre_setups_from_arrays,
)
from strategies.htf_trend_analyzer import clip_as_of

SETUP_TYPE = "ADRE"
STRATEGY_ABBREV = "ADRE"
STRATEGY_FULL_NAME = "ADR Expansion"
ADRE_BAYES_PURE_PROB = 1.0
ADRE_BAR_MINUTES = 60

ADRE_L6_EXTRA_COLUMNS: tuple[str, ...] = (
    "adr20",
    "adr_used",
    "adr_remaining",
    "adr_expansion_today",
    "current_day_range",
    "session_minutes_elapsed",
    "day_of_week",
    "month",
    "smt_divergence",
    "smt_strength",
    "leader_pair",
    "breakout_hour_jst",
)

ADRE_FEATURE_COLUMNS: tuple[str, ...] = (
    "trade_id",
    "timestamp",
    "pair",
    "direction",
    *ADRE_L6_EXTRA_COLUMNS,
    "outcome_label",
    "result_r",
    "mfe",
    "mae",
    "rr_ratio",
    "bayes_probability",
    "decision_source",
    "executed",
    "trade_result",
    "profit_r",
)


def is_adre_pure_mode() -> bool:
    raw = os.getenv("ADRE_PURE_MODE", "1")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def is_adre_enabled() -> bool:
    raw = os.getenv("ADRE_ENABLED", "1")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def is_adre_defense_pure_mode() -> bool:
    return is_adre_pure_mode()


def is_adre_l4_bypass() -> bool:
    return is_adre_pure_mode()


def is_adre_generic_bayes_bypass() -> bool:
    return is_adre_pure_mode()


@dataclass(frozen=True)
class AdreSetup:
    timestamp: pd.Timestamp
    pair: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    adr20: float
    adr_used: float
    adr_remaining: float
    adr_expansion_today: float
    day_high: float
    day_low: float
    current_day_range: float = 0.0
    session_minutes_elapsed: int = 0
    day_of_week: int = 0
    month: int = 1
    smt_divergence: bool = False
    smt_strength: float = 0.0
    leader_pair: str = "NONE"
    breakout_hour_jst: int = 0
    setup_type: str = SETUP_TYPE

    @property
    def rr_ratio(self) -> float:
        risk = abs(self.entry_price - self.stop_loss)
        if risk <= 0.0:
            return 0.0
        reward = abs(self.take_profit - self.entry_price)
        return reward / risk


@dataclass
class AdreFeatures:
    adr20: float
    adr_used: float
    adr_remaining: float
    adr_expansion_today: float
    current_day_range: float
    session_minutes_elapsed: int
    day_of_week: int
    month: int
    pair: str
    smt_divergence: bool = False
    smt_strength: float = 0.0
    leader_pair: str = "NONE"
    breakout_hour_jst: int = 0
    outcome_label: str = ""
    result_r: float = 0.0
    mfe: float = 0.0
    mae: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "adr20": self.adr20,
            "adr_used": self.adr_used,
            "adr_remaining": self.adr_remaining,
            "adr_expansion_today": self.adr_expansion_today,
            "current_day_range": self.current_day_range,
            "session_minutes_elapsed": self.session_minutes_elapsed,
            "day_of_week": self.day_of_week,
            "month": self.month,
            "pair": self.pair,
            "smt_divergence": self.smt_divergence,
            "smt_strength": self.smt_strength,
            "leader_pair": self.leader_pair,
            "breakout_hour_jst": self.breakout_hour_jst,
            "outcome_label": self.outcome_label,
            "result_r": self.result_r,
            "mfe": self.mfe,
            "mae": self.mae,
        }


def _payload_to_setup(payload: dict[str, Any]) -> AdreSetup:
    return AdreSetup(
        timestamp=pd.Timestamp(payload["timestamp"]),
        pair=str(payload["pair"]),
        direction=str(payload["direction"]),
        entry_price=float(payload["entry_price"]),
        stop_loss=float(payload["stop_loss"]),
        take_profit=float(payload["take_profit"]),
        adr20=float(payload["adr20"]),
        adr_used=float(payload["adr_used"]),
        adr_remaining=float(payload["adr_remaining"]),
        adr_expansion_today=float(payload["adr_expansion_today"]),
        day_high=float(payload["day_high"]),
        day_low=float(payload["day_low"]),
        current_day_range=float(payload.get("current_day_range", 0.0)),
        session_minutes_elapsed=int(payload.get("session_minutes_elapsed", 0)),
        day_of_week=int(payload.get("day_of_week", 0)),
        month=int(payload.get("month", 1)),
        smt_divergence=bool(payload.get("smt_divergence", False)),
        smt_strength=float(payload.get("smt_strength", 0.0)),
        leader_pair=str(payload.get("leader_pair", "NONE")),
        breakout_hour_jst=int(payload.get("breakout_hour_jst", 0)),
    )


def _correlated_ohlcv_for_pair(pair: str) -> dict[str, Any] | None:
    from strategies.market_utils import correlated_pair, get_bt_pair_frame_registry
    from strategies.bt_ohlcv import BtOhlcvFrame, as_ohlcv

    corr_pair = correlated_pair(pair)
    registry = get_bt_pair_frame_registry()
    if registry is None:
        return None
    frame = registry.get(corr_pair)
    if frame is None:
        return None
    if isinstance(frame, BtOhlcvFrame):
        arr = frame.arrays
    else:
        arr = as_ohlcv(frame)
    return {
        "high": arr.high,
        "low": arr.low,
        "close": arr.close,
        "timestamps": arr.datetime_ns,
    }


def _detect_payloads_from_source(
    source: Any,
    pair: str,
) -> list[dict[str, Any]]:
    from strategies.bt_ohlcv import BtOhlcvFrame, as_ohlcv

    corr = _correlated_ohlcv_for_pair(pair)
    corr_kwargs = {}
    if corr is not None:
        corr_kwargs = {
            "corr_high": corr["high"],
            "corr_low": corr["low"],
            "corr_close": corr["close"],
            "corr_timestamps": corr["timestamps"],
        }

    if isinstance(source, BtOhlcvFrame):
        arr = source.arrays
        return detect_adre_setups_from_arrays(
            open_=arr.open,
            high=arr.high,
            low=arr.low,
            close=arr.close,
            timestamps=arr.datetime_ns,
            pair=pair,
            **corr_kwargs,
        )

    as_of = pd.Timestamp(source["datetime"].iloc[-1])
    clipped = clip_as_of(source, as_of)
    if clipped is None:
        return []
    arr = as_ohlcv(clipped)
    return detect_adre_setups_from_arrays(
        open_=arr.open,
        high=arr.high,
        low=arr.low,
        close=arr.close,
        timestamps=arr.datetime_ns,
        pair=pair,
        **corr_kwargs,
    )


def detect_adre_setups(
    df: pd.DataFrame,
    h1_df: pd.DataFrame | None,
    pair: str,
    *,
    h4_df: pd.DataFrame | None = None,
) -> list[AdreSetup]:
    """Detect ADRE setups on H1 OHLCV (``df`` or ``h1_df``)."""
    del h4_df  # unused — single-TF discovery
    if not is_adre_enabled() or pair not in ADRE_PAIRS:
        return []
    source = h1_df if h1_df is not None else df
    if source is None or len(source) < 20:
        return []

    payloads = _detect_payloads_from_source(source, pair)
    return [_payload_to_setup(p) for p in payloads]


def build_adre_l6_fields(setup: AdreSetup) -> dict[str, Any]:
    return {
        "adr20": round(float(setup.adr20), 6),
        "adr_used": round(float(setup.adr_used), 6),
        "adr_remaining": round(float(setup.adr_remaining), 6),
        "adr_expansion_today": round(float(setup.adr_expansion_today), 6),
        "current_day_range": round(float(setup.current_day_range), 6),
        "session_minutes_elapsed": int(setup.session_minutes_elapsed),
        "day_of_week": int(setup.day_of_week),
        "month": int(setup.month),
        "smt_divergence": bool(setup.smt_divergence),
        "smt_strength": round(float(setup.smt_strength), 4),
        "leader_pair": str(setup.leader_pair),
        "breakout_hour_jst": int(setup.breakout_hour_jst),
    }


def build_adre_feature_log_row(
    *,
    trade_id: str,
    setup: AdreSetup,
    trade_result: str,
    profit_r: float,
    decision_source: str = "ALLOW",
    executed: bool = True,
    mfe: float = 0.0,
    mae: float = 0.0,
) -> dict[str, Any]:
    row = build_adre_l6_fields(setup)
    row.update(
        {
            "trade_id": trade_id,
            "timestamp": setup.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "pair": setup.pair,
            "direction": setup.direction,
            "outcome_label": trade_result if trade_result in ("WIN", "LOSS") else "",
            "result_r": round(float(profit_r), 4),
            "mfe": round(float(mfe), 4),
            "mae": round(float(mae), 4),
            "rr_ratio": round(float(setup.rr_ratio), 4),
            "bayes_probability": ADRE_BAYES_PURE_PROB,
            "decision_source": decision_source,
            "executed": executed,
            "trade_result": trade_result,
            "profit_r": round(float(profit_r), 4),
        }
    )
    return {k: row.get(k, "") for k in ADRE_FEATURE_COLUMNS}


class AdreStrategy(BaseStrategy):
    """ADR Expansion — prior-day breakout pure discovery (H1)."""

    @property
    def setup_type(self) -> str:
        return SETUP_TYPE

    def detect_setups(
        self,
        df: pd.DataFrame,
        pair_name: str,
        h1_df: pd.DataFrame | None = None,
        h4_df: pd.DataFrame | None = None,
    ) -> list[AdreSetup]:
        return detect_adre_setups(df, h1_df, pair_name, h4_df=h4_df)

    def analyze_setup(
        self,
        setup: Any,
        gbp_setup: Any | None,
        eur_setup: Any | None,
        h1_gbp: pd.DataFrame,
        h1_eur: pd.DataFrame,
    ) -> StrategyResult:
        del gbp_setup, eur_setup, h1_gbp, h1_eur
        if not isinstance(setup, AdreSetup):
            return StrategyResult(
                is_setup=False,
                setup_type=self.setup_type,
                direction="FLAT",
                strategy_action="REJECT",
            )

        raw = build_adre_l6_fields(setup)
        raw.update(
            {
                "bayes_probability": ADRE_BAYES_PURE_PROB,
                "smt_intensity": 0.0,
                "smt_diff": 0.0,
                "smt_leader": "NONE",
                "has_bos": False,
                "both_sweep": False,
                "atr_ratio": 1.0,
                "rr_ratio": setup.rr_ratio,
            }
        )

        return StrategyResult(
            is_setup=True,
            setup_type=self.setup_type,
            direction=setup.direction,
            entry_price=setup.entry_price,
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profit,
            candidate_score=0.0,
            raw_features=raw,
            strategy_action="ALLOW",
            base_risk_pct=None,
        )


__all__ = [
    "ADRE_BAYES_PURE_PROB",
    "ADRE_BAR_MINUTES",
    "ADRE_FEATURE_COLUMNS",
    "ADRE_L6_EXTRA_COLUMNS",
    "ADRE_PAIRS",
    "AdreFeatures",
    "AdreSetup",
    "AdreStrategy",
    "SETUP_TYPE",
    "STRATEGY_ABBREV",
    "STRATEGY_FULL_NAME",
    "build_adre_feature_log_row",
    "build_adre_l6_fields",
    "detect_adre_setups",
    "detect_adre_setups_clipped",
    "is_adre_defense_pure_mode",
    "is_adre_enabled",
    "is_adre_generic_bayes_bypass",
    "is_adre_l4_bypass",
    "is_adre_pure_mode",
]

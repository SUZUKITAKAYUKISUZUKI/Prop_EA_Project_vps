"""SMRS pure discovery — constants, baseline params, feature log builder."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from strategies.smrs_scan_numba import (
    ATR_NONE,
    EXIT_D,
    REASON_ENV_B,
    REASON_MA,
    REASON_SL,
    REASON_TIME,
    REASON_TP,
    REASON_Z0,
    SESSION_LONDON_NY,
)

SETUP_TYPE = "SMRS"
# Formal name + abbreviation (single source of truth for Strategy E)
STRATEGY_FULL_NAME = "Statistical Mean Reversion Scalper"
STRATEGY_ABBREV = "SMRS"
STRATEGY_DISPLAY_NAME = f"{STRATEGY_FULL_NAME} ({STRATEGY_ABBREV})"
STRATEGY_PORTFOLIO_LABEL = f"E {STRATEGY_ABBREV}"

SMRS_GEMINI_AUDIT = False
SMRS_LLM_AUDIT = False

# Production pair universe (Phase S1 tier S/A)
SMRS_PAIRS: tuple[str, ...] = (
    "AUDNZD",
    "EURGBP",
    "NZDUSD",
)

EXIT_REASON_LABEL = {
    REASON_SL: "STOP_LOSS",
    REASON_TP: "TAKE_PROFIT_RR",
    REASON_MA: "MA_TOUCH",
    REASON_ENV_B: "ENVELOPE_50",
    REASON_Z0: "ZSCORE_ZERO",
    REASON_TIME: "MAX_HOLD_TIME",
}

SESSION_LABEL = {
    range(0, 7): "ASIA",
}
for h in range(7, 13):
    SESSION_LABEL[h] = "LONDON"
for h in range(13, 22):
    SESSION_LABEL[h] = "NY"
for h in range(22, 24):
    SESSION_LABEL[h] = "ASIA"


def session_label_from_hour(hour: int) -> str:
    if 7 <= hour < 13:
        return "LONDON"
    if 13 <= hour < 22:
        return "NY"
    return "ASIA"


@dataclass(frozen=True)
class SmrsPureParams:
    envelope_dev: float = 0.20
    z_threshold: float = 2.0
    atr_filter: str = "none"
    exit_logic: str = "D"
    session_filter: str = "london_ny"
    max_hold_hours: int = 4
    session_code: int = SESSION_LONDON_NY
    atr_code: int = ATR_NONE
    exit_code: int = EXIT_D


PURE_BASELINE = SmrsPureParams()

SMRS_FEATURE_COLUMNS: tuple[str, ...] = (
    "trade_id",
    "timestamp",
    "pair",
    "setup_type",
    "direction",
    "entry_price",
    "exit_price",
    "stop_loss",
    "take_profit",
    "envelope_dev",
    "z_threshold",
    "z_score_entry",
    "ma_entry",
    "upper_envelope_entry",
    "lower_envelope_entry",
    "atr_entry",
    "atr_p25_entry",
    "atr_p50_entry",
    "atr_p75_entry",
    "session",
    "atr_filter",
    "exit_logic",
    "session_filter",
    "max_hold_hours",
    "bar_index_entry",
    "bars_held",
    "hold_minutes",
    "exit_reason",
    "result_r",
    "profit_r",
    "trade_result",
    "executed",
    "bayes_probability",
    "decision_source",
    "gemini_audit",
    "llm_audit",
)


def _fmt_ts(ts_ns: int) -> str:
    return datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def build_feature_rows(
    pair: str,
    params: SmrsPureParams,
    log: tuple[np.ndarray, ...],
    *,
    prefix: str = "SMRS",
    start_idx: int = 1,
) -> list[dict]:
    (
        profits,
        holds,
        ts_entry,
        _ts_exit,
        direction,
        entry_px,
        exit_px,
        sl_px,
        tp_px,
        z_entry,
        ma_entry,
        upper_entry,
        lower_entry,
        atr_entry,
        atr_p25,
        atr_p50,
        atr_p75,
        hour_entry,
        bars_held,
        exit_reason,
    ) = log

    rows: list[dict] = []
    for i in range(profits.size):
        ts_ns = int(ts_entry[i])
        ts_text = _fmt_ts(ts_ns)
        trade_num = start_idx + i
        trade_id = f"{prefix}_{pair}_{ts_text.replace(' ', '_').replace(':', '')}_{trade_num:06d}"
        direction_label = "BUY" if int(direction[i]) == 1 else "SELL"
        result_r = float(profits[i])
        trade_result = "WIN" if result_r > 0.0 else "LOSS"
        tp_val = tp_px[i]
        rows.append(
            {
                "trade_id": trade_id,
                "timestamp": ts_text,
                "pair": pair,
                "setup_type": SETUP_TYPE,
                "direction": direction_label,
                "entry_price": round(float(entry_px[i]), 5),
                "exit_price": round(float(exit_px[i]), 5),
                "stop_loss": round(float(sl_px[i]), 5),
                "take_profit": round(float(tp_val), 5) if np.isfinite(tp_val) else "",
                "envelope_dev": params.envelope_dev,
                "z_threshold": params.z_threshold,
                "z_score_entry": round(float(z_entry[i]), 4),
                "ma_entry": round(float(ma_entry[i]), 5),
                "upper_envelope_entry": round(float(upper_entry[i]), 5),
                "lower_envelope_entry": round(float(lower_entry[i]), 5),
                "atr_entry": round(float(atr_entry[i]), 6),
                "atr_p25_entry": round(float(atr_p25[i]), 6),
                "atr_p50_entry": round(float(atr_p50[i]), 6),
                "atr_p75_entry": round(float(atr_p75[i]), 6),
                "session": session_label_from_hour(int(hour_entry[i])),
                "atr_filter": params.atr_filter,
                "exit_logic": params.exit_logic,
                "session_filter": params.session_filter,
                "max_hold_hours": params.max_hold_hours,
                "bar_index_entry": "",
                "bars_held": int(bars_held[i]),
                "hold_minutes": round(float(holds[i]), 1),
                "exit_reason": EXIT_REASON_LABEL.get(int(exit_reason[i]), "UNKNOWN"),
                "result_r": round(result_r, 4),
                "profit_r": round(result_r, 4),
                "trade_result": trade_result,
                "executed": 1,
                "bayes_probability": 1.0,
                "decision_source": "RULE_BASE_ONLY",
                "gemini_audit": "OFF",
                "llm_audit": "OFF",
            }
        )
    return rows

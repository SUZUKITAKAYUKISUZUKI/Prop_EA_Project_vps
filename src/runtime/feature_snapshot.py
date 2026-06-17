"""Build JSON-serializable live feature snapshots from pipeline evaluation."""
from __future__ import annotations

from typing import Any

from src.database.data_source import FEATURE_LOG_SCHEMA_VERSION

SESSION_BY_HOUR = (
    (0, 7, "ASIA"),
    (7, 12, "LONDON"),
    (12, 17, "NEW_YORK"),
    (17, 22, "NY_LATE"),
    (22, 24, "ASIA"),
)


def infer_session(hour: int) -> str:
    for start, end, name in SESSION_BY_HOUR:
        if start <= hour < end:
            return name
    return "UNKNOWN"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat(sep=" ", timespec="seconds")
        except TypeError:
            return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


def _setup_fields(setup: Any) -> dict[str, Any]:
    if setup is None:
        return {}
    keys = (
        "pair",
        "direction",
        "entry_price",
        "stop_loss",
        "take_profit",
        "wick_ratio_pct",
        "atr",
        "session_tag",
        "structure_type",
        "fvg_size",
        "zone_type",
    )
    out: dict[str, Any] = {}
    for key in keys:
        if hasattr(setup, key):
            out[key] = _json_safe(getattr(setup, key))
    if hasattr(setup, "timestamp"):
        ts = getattr(setup, "timestamp")
        out["setup_timestamp"] = _json_safe(ts)
        try:
            hour = int(getattr(ts, "hour"))
            out["hour"] = hour
            out["session"] = infer_session(hour)
        except Exception:
            pass
    return out


def build_feature_snapshot(
    pending: Any,
    payload: dict[str, Any] | None = None,
    signal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Canonical live feature vector aligned with BT CSV / features.feature_json."""
    setup = getattr(pending, "setup", None)
    setup_data = _setup_fields(setup)
    symbol = setup_data.get("pair")
    if not symbol and payload:
        symbol = payload.get("market", {}).get("pair")

    ts_text = setup_data.get("setup_timestamp")
    hour = setup_data.get("hour")
    session = setup_data.get("session")
    if hour is None and payload:
        bar_time = payload.get("bar_time") or payload.get("server_time")
        if bar_time:
            try:
                import pandas as pd

                ts = pd.Timestamp(bar_time)
                hour = int(ts.hour)
                session = infer_session(hour)
                ts_text = ts.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass

    features: dict[str, Any] = {
        "trade_id": getattr(pending, "trade_id", None),
        "setup_type": getattr(pending, "setup_type", None),
        "symbol": symbol,
        "timestamp": ts_text,
        "hour": hour,
        "session": session,
        "candidate_score": _json_safe(getattr(pending, "candidate_score", None)),
        "bayes_probability": _json_safe(getattr(pending, "bayes_probability", None)),
        "atr_ratio": _json_safe(getattr(pending, "atr_ratio", None)),
        "smt_intensity": _json_safe(getattr(pending, "smt", None)),
        "smt_diff": _json_safe(getattr(pending, "smt_diff", None)),
        "smt_leader": _json_safe(getattr(pending, "smt_leader", None)),
        "has_bos": _json_safe(getattr(pending, "has_bos", None)),
        "vp_zone": _json_safe(getattr(pending, "vp_zone", None)),
        "l2_regime": _json_safe(getattr(pending, "l2_regime", None)),
        "l2_base_lot_factor": _json_safe(getattr(pending, "l2_base_lot_factor", None)),
        "htf_trend": _json_safe(getattr(pending, "htf_trend", None)),
        "htf_trend_direction": _json_safe(getattr(pending, "htf_trend_direction", None)),
        "divergence_direction": _json_safe(getattr(pending, "divergence_direction", None)),
        "l4_multiplier": _json_safe(getattr(pending, "l4_multiplier", None)),
        "l4_smt_interpretation": _json_safe(getattr(pending, "l4_smt_interpretation", None)),
        "htf_counter_trend": _json_safe(getattr(pending, "htf_counter_trend", None)),
        "htf_lot_multiplier": _json_safe(getattr(pending, "htf_lot_multiplier", None)),
        "lot_factor": _json_safe(getattr(pending, "lot_factor", None)),
        "final_lot_size": _json_safe(getattr(pending, "final_lot_size", None)),
        "risk_score": _json_safe(getattr(pending, "risk_score", None)),
        "decision_source": _json_safe(getattr(pending, "decision_source", None)),
        "is_reject": _json_safe(getattr(pending, "is_reject", None)),
        "tags": _json_safe(getattr(pending, "tags", None)),
        "daily_dd_remaining_percent": _json_safe(getattr(pending, "daily_rem", None)),
        "monthly_dd_remaining_percent": _json_safe(getattr(pending, "monthly_rem", None)),
        "fvg_final_lot_factor": _json_safe(getattr(pending, "fvg_final_lot_factor", None)),
        "ttm_bayes_win_prob": _json_safe(getattr(pending, "ttm_bayes_win_prob", None)),
        "ttm_ev_rank": _json_safe(getattr(pending, "ttm_ev_rank", None)),
        "ttm_ev_lot_multiplier": _json_safe(getattr(pending, "ttm_ev_lot_multiplier", None)),
        "dn_ev_rank": _json_safe(getattr(pending, "dn_ev_rank", None)),
        "dn_ev_bucket": _json_safe(getattr(pending, "dn_ev_bucket", None)),
        "dn_ev_rank_v2": _json_safe(getattr(pending, "dn_ev_rank_v2", None)),
        "dn_prop_gate_tier": _json_safe(getattr(pending, "dn_prop_gate_tier", None)),
        "dn_prop_gate_lot_multiplier": _json_safe(getattr(pending, "dn_prop_gate_lot_multiplier", None)),
        "lgr_bayes_regime": _json_safe(getattr(pending, "lgr_bayes_regime", None)),
        "lgr_ev_rank": _json_safe(getattr(pending, "lgr_ev_rank", None)),
        "cspa_gate_reason": _json_safe(getattr(pending, "cspa_gate_reason", None)),
        "bayes_inputs": {
            "bayes_probability": _json_safe(getattr(pending, "bayes_probability", None)),
            "candidate_score": _json_safe(getattr(pending, "candidate_score", None)),
            "atr_ratio": _json_safe(getattr(pending, "atr_ratio", None)),
            "smt_intensity": _json_safe(getattr(pending, "smt", None)),
            "decision_source": _json_safe(getattr(pending, "decision_source", None)),
        },
        "setup": setup_data,
    }

    if payload:
        account = payload.get("account") or {}
        calendar = payload.get("calendar") or {}
        features["live_context"] = {
            "equity": _json_safe(account.get("equity")),
            "balance": _json_safe(account.get("balance")),
            "spread_points": _json_safe(payload.get("spread_points")),
            "minutes_to_news": _json_safe(calendar.get("minutes_to_next_news")),
            "news_impact_level": _json_safe(calendar.get("news_impact_level")),
            "server_time": _json_safe(payload.get("server_time")),
        }

    if signal:
        features["signal"] = {
            "action": signal.get("action"),
            "lot_size": signal.get("lot_size"),
            "lot_factor": signal.get("lot_factor"),
            "multipliers": _json_safe(signal.get("multipliers")),
            "strategy_letter": signal.get("strategy_letter"),
        }

    features["schema_version"] = FEATURE_LOG_SCHEMA_VERSION
    return {k: v for k, v in features.items() if v is not None}

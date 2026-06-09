"""
audit/gemini_tref_auditor.py — TREF (Tokyo Range Expansion Failure) 専用 Gemini L4 監査

構造化 JSON のみ送信。Idempotency キャッシュ。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TREF_CACHE_PATH = PROJECT_ROOT / "storage" / "tref_llm_cache.json"
PRODUCTION_TREF_CACHE_PATH = PROJECT_ROOT / "storage" / "tref_llm_cache_live_3y.json"
TREF_CACHE_HIT_TAG = "TREF_CACHE_HIT"
TREF_CACHE_MISS_TAG = "TREF_CACHE_MISS"
TREF_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "confidence_score": {"type": "integer"},
        "reason_summary": {"type": "string"},
        "rebalance_probability": {"type": "string"},
    },
    "required": ["confidence_score", "reason_summary"],
}

TREF_PROMPT_VERSION = "v2_liquidity_hack_mean_reversion"

TREF_SYSTEM_PROMPT = """You are a liquidity-hack / mean-reversion auditor for the Tokyo Range Expansion
Failure (TREF) strategy on JPY crosses (AUDJPY, USDJPY). You are NOT a trend-following auditor.

Mindset: Tokyo lunch fake-outs are the DEFAULT. Real trend continuation is the EXCEPTION.
Each setup ALREADY passed deterministic detection (M15 range 09:00–11:15 JST, M5 expansion failure,
trigger in 11:30–15:00 JST). You receive structured JSON ONLY — no raw OHLC. Do NOT invent data.

=== INPUT FIELD MAP ===
- direction: BUY (failure below range) or SELL (failure above range)
- candidate_score: upstream deterministic rank 0–100 (context only)
- htf_trend_direction: BULL | BEAR | NEUTRAL (secondary context — do NOT trend-follow)
- minutes_to_next_news: minutes until next major calendar event (European/US news proxy)
- market_context.range_metrics: tokyo_range_width_pips, h1_atr_20_pips, ratio_range_to_htf_atr
- market_context.expansion_metrics: expansion_depth_pips, m15_atr_20_pips, ratio_depth_to_anchor_atr, bars_stayed_outside_m5
- market_context.execution_metrics: trigger_time_jst (HH:MM JST), trigger_bar_wick_ratio_pct, re_entry_depth_pct
- entry_price, stop_loss, take_profit, range_high, range_low: risk geometry

=== COGNITIVE AXES (MANDATORY — evaluate in this order) ===

【AXIS 1 — MEAN REVERSION VALIDITY / HIGH-SCORE LOGIC】
Tokyo lunch (11:30–14:30 JST) range breaks are, by default, stop-hunts without genuine flow.
When ALL of the following hold, this is a textbook liquidity fake-out — assign 85–92 WITHOUT hesitation:
  • trigger_time_jst between 11:30 and 14:30 (inclusive)
  • ratio_depth_to_anchor_atr < 1.5 (shallow expansion, not a real run)
  • trigger_bar_wick_ratio_pct >= 35 OR re_entry_depth_pct >= 50 (instant re-entry / high wick rejection)
  • bars_stayed_outside_m5 <= 3 (quick failure, not grind-outside)
If 3 of 4 hold (missing only bars count), still floor at 80.
Outside 11:30–14:30 but still shallow + wicky failure: baseline 75–82, not REJECT.

【AXIS 2 — REAL TREND CONTINUATION RISK / HARD CUT ONLY】
Apply REJECT-tier (0–44) ONLY when at least ONE real-break condition is met:
  (A) minutes_to_next_news <= 45 — European/high-impact news window overlaps the setup
  (B) bars_stayed_outside_m5 >= 4 AND ratio_depth_to_anchor_atr >= 1.0 — grind outside range with sustained pressure (gradual HL/LH updates)
If NEITHER (A) nor (B): treat as fake break. Do NOT penalize HTF trend conflict alone.
Everything else is "ダマシ" (fake break) → default score band 78–90, never REJECT for trend reasons alone.

=== MECHANICAL SCORING (show arithmetic in reason_summary) ===
1. Start at 80 (fake-break prior).
2. AXIS 1 full match (4/4): set score to 88 (add +8). 3/4 match: floor 82.
3. AXIS 1 partial (shallow ratio_depth < 1.2 AND wick >= 30): ADD +5.
4. AXIS 2 condition (A): cap score at 35 (REJECT).
5. AXIS 2 condition (B): cap score at 40 (REJECT).
6. Both A and B: cap at 25.
7. Weak trigger only (wick < 15 AND re_entry < 30) with no AXIS 2 hit: SUBTRACT 10 (floor 68, still ALLOW-tier).
Clamp 0–100. Bands: 85–100 STRONG | 60–84 ALLOW | 45–59 CAUTION | 0–44 REJECT.

=== OUTPUT (JSON ONLY) ===
{
  "confidence_score": <integer 0-100>,
  "reason_summary": "<max 120 words: AXIS1/AXIS2 check, arithmetic, final score>",
  "rebalance_probability": "HIGH" | "MEDIUM" | "LOW"
}
Map: HIGH ↔ 75-100, MEDIUM ↔ 45-74, LOW ↔ 0-44. Keep consistent with confidence_score.
"""

_tref_cache: dict[str, dict[str, Any]] | None = None
_tref_cache_path: Path | None = None
_tref_cache_dirty = False
_tref_cache_readonly = False
_tref_cache_stats = {"lookups": 0, "hits": 0, "misses": 0, "api_calls": 0}


def resolve_tref_cache_path() -> Path:
    raw = os.getenv("TREF_LLM_CACHE_PATH", "").strip()
    return Path(raw) if raw else DEFAULT_TREF_CACHE_PATH


def make_tref_hash(
    pair: str,
    timestamp: str,
    range_high: float,
    range_low: float,
    direction: str,
) -> str:
    payload = (
        f"{TREF_PROMPT_VERSION}|{pair}|{timestamp}|{range_high:.6f}|"
        f"{range_low:.6f}|{direction}"
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _load_tref_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        logger.warning("TREF cache load failed: %s", path, exc_info=True)
    return {}


def merge_production_tref_cache(
    target: Path | None = None,
    source: Path | None = None,
) -> int:
    target_path = target or DEFAULT_TREF_CACHE_PATH
    source_path = source or PRODUCTION_TREF_CACHE_PATH
    if not source_path.exists():
        return 0

    target_path.parent.mkdir(parents=True, exist_ok=True)
    merged = _load_tref_cache(target_path)
    source_data = _load_tref_cache(source_path)
    added = 0
    for key, record in source_data.items():
        if key not in merged or not str(merged[key].get("model_version", "")).startswith("gemini"):
            merged[key] = record
            added += 1
    if added:
        target_path.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Merged TREF production cache | source=%s target=%s added=%d total=%d",
            source_path,
            target_path,
            added,
            len(merged),
        )
    return added


def _tref_force_reaudit_enabled() -> bool:
    return os.getenv("TREF_FORCE_REAUDIT", "").strip().lower() in ("1", "true", "yes", "on")


def init_tref_cache(path: Path | None = None, *, readonly: bool = False) -> None:
    global _tref_cache, _tref_cache_path, _tref_cache_dirty, _tref_cache_readonly
    if not _tref_force_reaudit_enabled():
        merge_production_tref_cache()
    _tref_cache_path = path or resolve_tref_cache_path()
    _tref_cache_path.parent.mkdir(parents=True, exist_ok=True)
    _tref_cache = _load_tref_cache(_tref_cache_path)
    _tref_cache_dirty = False
    _tref_cache_readonly = readonly
    if readonly:
        logger.info(
            "TREF LLM cache readonly | path=%s entries=%d (API disabled on miss)",
            _tref_cache_path,
            len(_tref_cache or {}),
        )
    elif _tref_force_reaudit_enabled():
        logger.warning(
            "TREF_FORCE_REAUDIT=1 | path=%s entries=%d (Gemini cache hits bypassed)",
            _tref_cache_path,
            len(_tref_cache or {}),
        )


def flush_tref_cache() -> None:
    global _tref_cache_dirty
    if not _tref_cache_dirty or _tref_cache is None or _tref_cache_path is None:
        return
    _tref_cache_path.parent.mkdir(parents=True, exist_ok=True)
    _tref_cache_path.write_text(
        json.dumps(_tref_cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _tref_cache_dirty = False


def _cache_get(tref_hash: str) -> dict[str, Any] | None:
    if _tref_cache is None:
        init_tref_cache()
    assert _tref_cache is not None
    hit = _tref_cache.get(tref_hash)
    return dict(hit) if hit else None


def _cache_put(tref_hash: str, record: dict[str, Any]) -> None:
    global _tref_cache_dirty
    if _tref_cache_readonly:
        return
    if _tref_cache is None:
        init_tref_cache()
    assert _tref_cache is not None
    _tref_cache[tref_hash] = record
    _tref_cache_dirty = True


def tref_cache_coverage_stats() -> dict[str, float | int]:
    lookups = _tref_cache_stats["lookups"]
    hits = _tref_cache_stats["hits"]
    return {
        "lookups": lookups,
        "hits": hits,
        "misses": _tref_cache_stats["misses"],
        "api_calls": _tref_cache_stats["api_calls"],
        "coverage_pct": round(hits / lookups * 100.0, 2) if lookups else 100.0,
    }


def log_tref_cache_coverage_report() -> None:
    stats = tref_cache_coverage_stats()
    logger.info(
        "TREF LLM cache coverage | lookups=%d hits=%d misses=%d api_calls=%d coverage=%.1f%%",
        stats["lookups"],
        stats["hits"],
        stats["misses"],
        stats["api_calls"],
        stats["coverage_pct"],
    )


def _cache_miss_readonly_result(tref_hash: str) -> dict[str, Any]:
    from audit import risk_manager as audit_rm

    return {
        "confidence_score": 0,
        "reason_summary": "TREF LLM cache miss in readonly backtest (no API call).",
        "rebalance_probability": "LOW",
        "llm_latency_ms": 0,
        "model_version": "TREF_CACHE_MISS",
        "reason_codes": [TREF_CACHE_MISS_TAG],
        "llm_decision": audit_rm.confidence_to_llm_decision(0),
        "risk_score": 100,
        "tref_hash": tref_hash,
    }


def pd_timestamp_str(ts: Any) -> str:
    import pandas as pd

    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _is_backtest_mode() -> bool:
    try:
        from llm_auditor import is_backtest_mode

        return bool(is_backtest_mode())
    except ImportError:
        return os.getenv("BACKTEST_MODE", "").strip().lower() in ("1", "true", "yes", "on")


def _tref_live_gemini_enabled(force_live: bool = False) -> bool:
    if force_live:
        return True
    if os.environ.get("LLM_FORCE_LIVE", "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    try:
        from llm_auditor import is_backtest_mode

        if is_backtest_mode():
            return False
    except ImportError:
        if _is_backtest_mode():
            return False
    try:
        from main_platform import USE_LLM_AUDITOR

        if USE_LLM_AUDITOR:
            return True
    except ImportError:
        pass
    return os.getenv("USE_LLM_AUDITOR", "").strip().lower() in ("1", "true", "yes", "on")


def _mock_tref_audit() -> dict[str, Any]:
    return {
        "confidence_score": 75,
        "reason_summary": "BACKTEST_MODE: TREF expansion-failure mock (moderate confidence).",
        "rebalance_probability": "HIGH",
        "llm_latency_ms": 0,
        "model_version": "TREF_MOCK_BT",
        "reason_codes": ["TREF_MOCK_AUDIT"],
        "llm_decision": "ALLOW",
        "risk_score": 25,
    }


def build_tref_audit_payload(setup: Any, raw: dict[str, Any]) -> dict[str, Any]:
    """Gemini へ送る構造化メタデータ（生 OHLC 禁止）。"""
    payload = raw.get("payload") if isinstance(raw.get("payload"), dict) else {}
    if not payload and isinstance(getattr(setup, "payload", None), dict):
        payload = setup.payload

    market_context = payload.get("market_context", {}) if isinstance(payload, dict) else {}
    return {
        "pair": setup.pair,
        "direction": setup.direction,
        "timestamp": pd_timestamp_str(setup.timestamp),
        "setup_type": "TOKYO_RANGE_EXPANSION_FAILURE",
        "candidate_score": int(raw.get("candidate_score", setup.candidate_score)),
        "htf_trend_direction": str(raw.get("htf_trend_direction", "NEUTRAL")),
        "range_high": round(float(setup.range_high), 6),
        "range_low": round(float(setup.range_low), 6),
        "range_width_pips": round(float(raw.get("range_width_pips", setup.range_width_pips)), 2),
        "expansion_depth_pips": round(float(raw.get("expansion_depth_pips", setup.expansion_depth_pips)), 2),
        "bars_stayed_outside_m5": int(raw.get("bars_stayed_outside_m5", setup.bars_stayed_outside_m5)),
        "entry_price": round(float(setup.entry_price), 6),
        "stop_loss": round(float(setup.stop_loss), 6),
        "take_profit": round(float(setup.take_profit), 6),
        "market_context": market_context,
        "score_breakdown": raw.get("score_breakdown", getattr(setup, "score_breakdown", {})),
        "minutes_to_next_news": int(raw.get("minutes_to_news", raw.get("minutes_to_next_news", 999))),
    }


def _call_gemini_api(payload: dict[str, Any], tref_hash: str) -> dict[str, Any]:
    from llm_auditor import resolve_gemini_api_key, resolve_gemini_model

    logger.warning(
        "🚀 [Gemini API Call] TREF高解像度監査を実行します（ハッシュ: %s）",
        tref_hash,
    )
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=resolve_gemini_api_key())
    started = time.perf_counter()
    response = client.models.generate_content(
        model=resolve_gemini_model(),
        contents=json.dumps(payload, ensure_ascii=False),
        config=types.GenerateContentConfig(
            system_instruction=TREF_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=TREF_RESPONSE_SCHEMA,
            temperature=0.1,
        ),
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    text = getattr(response, "text", None) or "{}"
    parsed = json.loads(text)
    confidence = max(0, min(100, int(parsed.get("confidence_score", 0))))
    from audit import risk_manager as audit_rm

    decision = audit_rm.confidence_to_llm_decision(confidence)
    return {
        "confidence_score": confidence,
        "reason_summary": str(parsed.get("reason_summary", "")),
        "rebalance_probability": str(parsed.get("rebalance_probability", "MEDIUM")),
        "llm_latency_ms": latency_ms,
        "model_version": resolve_gemini_model(),
        "reason_codes": ["TREF_GEMINI_AUDIT"],
        "llm_decision": decision,
        "risk_score": max(0, 100 - confidence),
    }


def audit_tref_setup(
    setup: Any,
    raw: dict[str, Any],
    *,
    force_live: bool = False,
) -> dict[str, Any]:
    global _tref_cache_stats
    payload = build_tref_audit_payload(setup, raw)
    tref_hash = make_tref_hash(
        setup.pair,
        payload["timestamp"],
        float(setup.range_high),
        float(setup.range_low),
        str(setup.direction),
    )

    _tref_cache_stats["lookups"] += 1
    cached = _cache_get(tref_hash)
    skip_cache = (
        cached is not None
        and not _tref_cache_readonly
        and _tref_force_reaudit_enabled()
        and str(cached.get("model_version", "")).startswith("gemini")
    )
    if cached is not None and not skip_cache:
        _tref_cache_stats["hits"] += 1
        hit = dict(cached)
        codes = list(hit.get("reason_codes") or [])
        if TREF_CACHE_HIT_TAG not in codes:
            codes.insert(0, TREF_CACHE_HIT_TAG)
        hit["reason_codes"] = codes
        logger.info(
            "TREF cache hit | hash=%s model=%s confidence=%s",
            tref_hash,
            hit.get("model_version"),
            hit.get("confidence_score"),
        )
        return hit

    _tref_cache_stats["misses"] += 1
    if _tref_cache_readonly or (_is_backtest_mode() and not force_live and not _tref_live_gemini_enabled()):
        logger.warning(
            "TREF cache miss (readonly) | hash=%s pair=%s ts=%s — API skipped",
            tref_hash,
            setup.pair,
            payload["timestamp"],
        )
        return _cache_miss_readonly_result(tref_hash)

    if _tref_live_gemini_enabled(force_live):
        _tref_cache_stats["api_calls"] += 1
        try:
            from llm_auditor import _assert_gemini_api_allowed

            _assert_gemini_api_allowed("TREF Gemini audit")
            result = _call_gemini_api(payload, tref_hash)
        except Exception:
            logger.exception("TREF Gemini audit failed | hash=%s", tref_hash)
            result = {
                "confidence_score": 50,
                "reason_summary": "TREF audit fallback after API error.",
                "rebalance_probability": "MEDIUM",
                "llm_latency_ms": 0,
                "model_version": "TREF_ERROR_FALLBACK",
                "reason_codes": ["TREF_AUDIT_ERROR"],
                "llm_decision": "CAUTION",
                "risk_score": 50,
            }
    else:
        result = _mock_tref_audit()

    record = {
        **result,
        "tref_hash": tref_hash,
        "cached_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _cache_put(tref_hash, record)
    flush_tref_cache()
    return record

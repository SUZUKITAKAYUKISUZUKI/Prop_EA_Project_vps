"""
audit/gemini_fvg_auditor.py — FVG Fill 専用 Gemini L4 監査

構造化 JSON のみ送信。Idempotency キャッシュ + Bar-Lock は戦略層と連携。
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
DEFAULT_FVG_CACHE_PATH = PROJECT_ROOT / "storage" / "fvg_llm_cache.json"
PRODUCTION_FVG_CACHE_PATH = PROJECT_ROOT / "storage" / "fvg_llm_cache_live_3y.json"
FVG_CACHE_HIT_TAG = "FVG_CACHE_HIT"
FVG_CACHE_MISS_TAG = "FVG_CACHE_MISS"
FVG_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "confidence_score": {"type": "integer"},
        "reason_summary": {"type": "string"},
        "rebalance_probability": {"type": "string"},
    },
    "required": ["confidence_score", "reason_summary"],
}

FVG_SYSTEM_PROMPT = """You are a senior quant FX auditor specializing in Fair Value Gap (FVG) /
Liquidity Void mean-reversion (gap fill / rebalance toward imbalance origin).

Each setup has ALREADY passed deterministic FVG detection (valid gap geometry, entry, direction-
appropriate SL/TP at 2R) and HTF trend soft-filter. direction is BUY (FVG_LONG / bullish gap fill)
or SELL (FVG_SHORT / bearish gap fill). You receive structured JSON metadata ONLY (no raw OHLC).
Do NOT invent prices, RSI, or indicators not present in the payload. Compute confidence_score ONLY
via the mechanical algorithm below.

=== INPUT FIELD MAP ===
- direction: BUY | SELL
- fvg_size_pips: gap width in pips (compare to volatility_20d as daily-range context)
- nearby_order_block.present / distance_pips:
  * BUY: demand wall BELOW the FVG (LONG support)
  * SELL: supply wall ABOVE the FVG (SHORT resistance)
- volatility_20d: 20-day realized vol proxy (use with fvg_size_pips to judge overextension)
- current_session: ASIA | LONDON | NY | OFF_SESSION (UTC hour bucket from timestamp)
- htf_trend_direction: BULL | BEAR | NEUTRAL (H1 structure bias)
- candidate_score: upstream deterministic rank (context only, not your output)
- entry_price, stop_loss, take_profit, fvg_top, fvg_bottom: geometry for risk context

=== FVG SCORING ALGORITHM (MANDATORY MECHANICAL EVALUATION) ===
You must calculate the confidence_score using a strict additive/subtractive baseline method.
Do NOT rely on subjective impressions. Prevent mid-range clustering by following this exact logic:

1. BASELINE SCORE START (trend-aligned by direction)
   - BUY + htf_trend_direction == BULL: baseline 75 (ALLOW-tier)
   - SELL + htf_trend_direction == BEAR: baseline 75 (ALLOW-tier)
   - htf_trend_direction == NEUTRAL: baseline 60 (ALLOW-tier)
   - BUY + htf_trend_direction == BEAR: baseline 30 (REJECT-tier)
   - SELL + htf_trend_direction == BULL: baseline 30 (REJECT-tier)

2. MECHANICAL MODIFIERS (Apply strictly based on payload fields)
   - Order Block Confluence:
     * If nearby_order_block.present == true AND distance_pips ≤ 15: ADD 15 points (Cap the final score at 98).
     * If nearby_order_block.present == false OR distance_pips > 20: SUBTRACT 5 points only.
       (Do NOT drop a trend-aligned setup below 65 based on this factor alone).
   - Session Energy:
     * If current_session is LONDON or NY: No change (0 points).
     * If current_session is ASIA or OFF_SESSION: SUBTRACT 10 points.
   - Volatility & Gap Size Context:
     * If fvg_size_pips is large relative to volatility_20d (implied strong momentum vacuum): ADD 5 points.
     * If fvg_size_pips is minimal with low volatility_20d: SUBTRACT 5 points.

3. FINAL SCORE RANGE ENFORCEMENT
   - 85-100 (STRONG): Reserved for HTF-aligned + close Order Block setups.
   - 60-84 (ALLOW): Default zone for standard trend-aligned trades.
     Even if the Order Block is missing or distant, as long as the session is active (LONDON/NY),
     the score MUST remain in the 65-75 range for aligned setups. Do NOT downgrade these to CAUTION.
   - 45-59 (CAUTION): Use ONLY for highly conflicting setups
     (e.g., HTF NEUTRAL combined with ASIA/OFF_SESSION timing).
   - 0-44 (REJECT): For structurally flawed setups (HTF counter-trend or toxic combinations).

In reason_summary, show the arithmetic: baseline → each modifier (+/−) → final score.
Do NOT invent modifiers outside the list above.

=== OUTPUT (JSON ONLY — no markdown, no prose outside JSON) ===
{
  "confidence_score": <integer 0-100>,
  "reason_summary": "<max 120 words: baseline, each modifier applied, final arithmetic>",
  "rebalance_probability": "HIGH" | "MEDIUM" | "LOW"
}
Map: HIGH ↔ 75-100, MEDIUM ↔ 45-74, LOW ↔ 0-44. Keep all three fields consistent.
"""

_fvg_cache: dict[str, dict[str, Any]] | None = None
_fvg_cache_path: Path | None = None
_fvg_cache_dirty = False
_fvg_cache_readonly = False
_fvg_cache_stats = {"lookups": 0, "hits": 0, "misses": 0, "api_calls": 0}


def resolve_fvg_cache_path() -> Path:
    raw = os.getenv("FVG_LLM_CACHE_PATH", "").strip()
    return Path(raw) if raw else DEFAULT_FVG_CACHE_PATH


def make_fvg_hash(
    pair: str,
    timestamp: str,
    fvg_top: float,
    fvg_bottom: float,
) -> str:
    """[pair, timestamp, fvg_top, fvg_bottom] から MD5 ハッシュ。"""
    payload = f"{pair}|{timestamp}|{fvg_top:.6f}|{fvg_bottom:.6f}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _load_fvg_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        logger.warning("FVG cache load failed: %s", path, exc_info=True)
    return {}


def merge_production_fvg_cache(
    target: Path | None = None,
    source: Path | None = None,
) -> int:
    """
    本番 Gemini 監査キャッシュ (live_3y) を canonical キャッシュへマージ。
    同一ハッシュは source（本番）を優先。
    """
    target_path = target or DEFAULT_FVG_CACHE_PATH
    source_path = source or PRODUCTION_FVG_CACHE_PATH
    if not source_path.exists():
        return 0

    target_path.parent.mkdir(parents=True, exist_ok=True)
    merged = _load_fvg_cache(target_path)
    source_data = _load_fvg_cache(source_path)
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
            "Merged FVG production cache | source=%s target=%s added=%d total=%d",
            source_path,
            target_path,
            added,
            len(merged),
        )
    return added


def _fvg_force_reaudit_enabled() -> bool:
    return os.getenv("FVG_FORCE_REAUDIT", "").strip().lower() in ("1", "true", "yes", "on")


def init_fvg_cache(path: Path | None = None, *, readonly: bool = False) -> None:
    global _fvg_cache, _fvg_cache_path, _fvg_cache_dirty, _fvg_cache_readonly
    if not _fvg_force_reaudit_enabled():
        merge_production_fvg_cache()
    _fvg_cache_path = path or resolve_fvg_cache_path()
    _fvg_cache_path.parent.mkdir(parents=True, exist_ok=True)
    _fvg_cache = _load_fvg_cache(_fvg_cache_path)
    _fvg_cache_dirty = False
    _fvg_cache_readonly = readonly
    if readonly:
        logger.info(
            "FVG LLM cache readonly | path=%s entries=%d (API disabled on miss)",
            _fvg_cache_path,
            len(_fvg_cache or {}),
        )
    elif _fvg_force_reaudit_enabled():
        logger.warning(
            "FVG_FORCE_REAUDIT=1 | path=%s entries=%d (Gemini cache hits bypassed)",
            _fvg_cache_path,
            len(_fvg_cache or {}),
        )


def flush_fvg_cache() -> None:
    global _fvg_cache_dirty
    if not _fvg_cache_dirty or _fvg_cache is None or _fvg_cache_path is None:
        return
    _fvg_cache_path.parent.mkdir(parents=True, exist_ok=True)
    _fvg_cache_path.write_text(
        json.dumps(_fvg_cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _fvg_cache_dirty = False


def _cache_get(fvg_hash: str) -> dict[str, Any] | None:
    if _fvg_cache is None:
        init_fvg_cache()
    assert _fvg_cache is not None
    hit = _fvg_cache.get(fvg_hash)
    return dict(hit) if hit else None


def _cache_put(fvg_hash: str, record: dict[str, Any]) -> None:
    global _fvg_cache_dirty
    if _fvg_cache_readonly:
        return
    if _fvg_cache is None:
        init_fvg_cache()
    assert _fvg_cache is not None
    _fvg_cache[fvg_hash] = record
    _fvg_cache_dirty = True


def fvg_cache_coverage_stats() -> dict[str, float | int]:
    lookups = _fvg_cache_stats["lookups"]
    hits = _fvg_cache_stats["hits"]
    return {
        "lookups": lookups,
        "hits": hits,
        "misses": _fvg_cache_stats["misses"],
        "api_calls": _fvg_cache_stats["api_calls"],
        "coverage_pct": round(hits / lookups * 100.0, 2) if lookups else 100.0,
    }


def log_fvg_cache_coverage_report() -> None:
    stats = fvg_cache_coverage_stats()
    logger.info(
        "FVG LLM cache coverage | lookups=%d hits=%d misses=%d api_calls=%d coverage=%.1f%%",
        stats["lookups"],
        stats["hits"],
        stats["misses"],
        stats["api_calls"],
        stats["coverage_pct"],
    )


def _cache_miss_readonly_result(fvg_hash: str) -> dict[str, Any]:
    """readonly BT: キャッシュ未命中時は API を叩かず拒否扱い。"""
    from audit import risk_manager as audit_rm

    return {
        "confidence_score": 0,
        "reason_summary": "FVG LLM cache miss in readonly backtest (no API call).",
        "rebalance_probability": "LOW",
        "llm_latency_ms": 0,
        "model_version": "FVG_CACHE_MISS",
        "reason_codes": [FVG_CACHE_MISS_TAG],
        "llm_decision": audit_rm.confidence_to_llm_decision(0),
        "risk_score": 100,
        "fvg_hash": fvg_hash,
    }


def build_fvg_audit_payload(setup: Any, raw: dict[str, Any]) -> dict[str, Any]:
    """Gemini へ送る構造化メタデータ（生 OHLC 禁止）。"""
    return {
        "pair": setup.pair,
        "direction": setup.direction,
        "timestamp": pd_timestamp_str(setup.timestamp),
        "setup_type": "FVG_FILL",
        "fvg_top": round(float(setup.fvg_top), 6),
        "fvg_bottom": round(float(setup.fvg_bottom), 6),
        "fvg_size_pips": round(float(raw.get("fvg_size_pips", setup.fvg_size_pips)), 2),
        "entry_price": round(float(setup.entry_price), 6),
        "stop_loss": round(float(setup.stop_loss), 6),
        "take_profit": round(float(setup.take_profit), 6),
        "nearby_order_block": {
            "present": bool(raw.get("nearby_order_block_present", setup.nearby_order_block_present)),
            "distance_pips": round(
                float(raw.get("nearby_order_block_distance_pips", setup.nearby_order_block_distance_pips)),
                2,
            ),
        },
        "volatility_20d": round(float(raw.get("volatility_20d", setup.volatility_20d)), 4),
        "current_session": str(raw.get("current_session", setup.current_session)),
        "candidate_score": round(float(raw.get("candidate_score", 0.0)), 2),
        "htf_trend_direction": str(raw.get("htf_trend_direction", "NEUTRAL")),
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


def _fvg_live_gemini_enabled(force_live: bool = False) -> bool:
    """BACKTEST_MODE モック vs 本番 Gemini（--use-llm / LLM_FORCE_LIVE）。"""
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


def _mock_fvg_audit() -> dict[str, Any]:
    return {
        "confidence_score": 90,
        "reason_summary": "BACKTEST_MODE: FVG rebalance mock (high fill probability).",
        "rebalance_probability": "HIGH",
        "llm_latency_ms": 0,
        "model_version": "FVG_MOCK_BT",
        "reason_codes": ["FVG_MOCK_AUDIT"],
        "llm_decision": "ALLOW",
        "risk_score": 10,
    }


def _call_gemini_api(payload: dict[str, Any], fvg_hash: str) -> dict[str, Any]:
    from llm_auditor import resolve_gemini_api_key, resolve_gemini_model

    logger.warning(
        "🚀 [Gemini API Call] FVG高解像度監査を実行します（ハッシュ: %s）",
        fvg_hash,
    )
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=resolve_gemini_api_key())
    started = time.perf_counter()
    response = client.models.generate_content(
        model=resolve_gemini_model(),
        contents=json.dumps(payload, ensure_ascii=False),
        config=types.GenerateContentConfig(
            system_instruction=FVG_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=FVG_RESPONSE_SCHEMA,
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
        "reason_codes": ["FVG_GEMINI_AUDIT"],
        "llm_decision": decision,
        "risk_score": max(0, 100 - confidence),
    }


def audit_fvg_setup(
    setup: Any,
    raw: dict[str, Any],
    *,
    force_live: bool = False,
) -> dict[str, Any]:
    """
    Idempotency Guard: キャッシュヒット時は API ゼロ。
    readonly モード（通常 BT）: キャッシュのみ参照、未命中は API 禁止。
    live モード（--use-llm）: 未命中時のみ Gemini API → キャッシュ追記。
    """
    global _fvg_cache_stats
    payload = build_fvg_audit_payload(setup, raw)
    fvg_hash = make_fvg_hash(
        setup.pair,
        payload["timestamp"],
        float(setup.fvg_top),
        float(setup.fvg_bottom),
    )

    _fvg_cache_stats["lookups"] += 1
    cached = _cache_get(fvg_hash)
    skip_cache = (
        cached is not None
        and not _fvg_cache_readonly
        and _fvg_force_reaudit_enabled()
        and str(cached.get("model_version", "")).startswith("gemini")
    )
    if cached is not None and not skip_cache:
        _fvg_cache_stats["hits"] += 1
        hit = dict(cached)
        codes = list(hit.get("reason_codes") or [])
        if FVG_CACHE_HIT_TAG not in codes:
            codes.insert(0, FVG_CACHE_HIT_TAG)
        hit["reason_codes"] = codes
        logger.info(
            "FVG cache hit | hash=%s model=%s confidence=%s",
            fvg_hash,
            hit.get("model_version"),
            hit.get("confidence_score"),
        )
        return hit

    _fvg_cache_stats["misses"] += 1
    if _fvg_cache_readonly or (_is_backtest_mode() and not force_live and not _fvg_live_gemini_enabled()):
        logger.warning(
            "FVG cache miss (readonly) | hash=%s pair=%s ts=%s — API skipped",
            fvg_hash,
            setup.pair,
            payload["timestamp"],
        )
        return _cache_miss_readonly_result(fvg_hash)

    if _fvg_live_gemini_enabled(force_live):
        _fvg_cache_stats["api_calls"] += 1
        try:
            from llm_auditor import _assert_gemini_api_allowed

            _assert_gemini_api_allowed("FVG Gemini audit")
            result = _call_gemini_api(payload, fvg_hash)
        except Exception:
            logger.exception("FVG Gemini audit failed | hash=%s", fvg_hash)
            result = {
                "confidence_score": 50,
                "reason_summary": "FVG audit fallback after API error.",
                "rebalance_probability": "MEDIUM",
                "llm_latency_ms": 0,
                "model_version": "FVG_ERROR_FALLBACK",
                "reason_codes": ["FVG_AUDIT_ERROR"],
                "llm_decision": "CAUTION",
                "risk_score": 50,
            }
    else:
        result = _mock_fvg_audit()

    record = {
        **result,
        "fvg_hash": fvg_hash,
        "cached_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _cache_put(fvg_hash, record)
    flush_fvg_cache()
    return record

"""
audit/gemini_dtpa_auditor.py — DTPA 専用 Gemini L4 監査

Dow Theory H4 BOS → H1 pullback HL/LH → PA trigger。構造化 JSON のみ送信。
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
DEFAULT_DTPA_CACHE_PATH = PROJECT_ROOT / "storage" / "dtpa_llm_cache.json"
PRODUCTION_DTPA_CACHE_PATH = PROJECT_ROOT / "storage" / "dtpa_llm_cache_live_3y.json"
DTPA_CACHE_HIT_TAG = "DTPA_CACHE_HIT"
DTPA_CACHE_MISS_TAG = "DTPA_CACHE_MISS"
DTPA_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "confidence_score": {"type": "integer"},
        "reason_summary": {"type": "string"},
        "continuation_probability": {"type": "string"},
    },
    "required": ["confidence_score", "reason_summary"],
}

DTPA_PROMPT_VERSION = "v1_dow_bos_pullback_pa"

DTPA_SYSTEM_PROMPT = """You are a senior FX auditor for the DTPA strategy:
H4 Break-of-Structure (Dow Theory) → H1 wave-2 pullback (HL/LH in zone) → H1 PA trigger (3rd wave entry).

Each setup ALREADY passed deterministic rules (BOS, pullback READY, PA at trigger close, HTF alignment).
You receive structured JSON ONLY — no raw OHLC. Do NOT invent indicators or prices.

=== INPUT FIELD MAP ===
- bos_direction: LONG | SHORT (H4 structure shift)
- direction: BUY | SELL (trade direction, mirrors bos)
- consecutive_structure_count: consecutive LH/LL (long) or HH/HL (short) before BOS
- broken_level: H4 level broken on BOS close
- structure_invalidation_level: invalidation if structure fails
- bars_since_bos: H1 bars from BOS to PA trigger
- pullback_depth_atr_ratio: pullback depth vs H1 ATR
- pa_trigger_type: ENGULFING | PIN_BAR | INSIDE_BAR_BREAK
- anchor_price: H1 HL (long) or LH (short) anchor in pullback zone
- htf_aligned: true if H1 HTF trend supports trade direction
- htf_trend_direction: BULL | BEAR | NEUTRAL
- entry_price, stop_loss, take_profit: risk geometry (2R TP)
- candidate_score: upstream L2 score (context only)

=== SCORING (MANDATORY MECHANICAL) ===
1. BASELINE
   - htf_aligned == true AND htf_trend_direction matches direction: start 78
   - htf_aligned == true but NEUTRAL HTF: start 72
   - htf_aligned == false: cap at 35 (REJECT) — should rarely arrive

2. STRUCTURE QUALITY
   - consecutive_structure_count >= 2: +5
   - bars_since_bos between 4 and 24 (timely 3rd wave): +5
   - bars_since_bos > 36: -10
   - pullback_depth_atr_ratio between 0.3 and 1.2 (healthy retrace): +5
   - pullback_depth_atr_ratio > 2.0 (deep/choppy): -8

3. PA TRIGGER
   - ENGULFING: +8
   - PIN_BAR: +6
   - INSIDE_BAR_BREAK: +4

4. EXECUTION TARGET
   - Final score >= 85: STRONG (ALLOW tier, full lot)
   - 65-84: CAUTION tier (half lot downstream)
   - 0-64: REJECT (no entry)

Show arithmetic in reason_summary: baseline → modifiers → final.
Clamp 0-100.

=== OUTPUT (JSON ONLY) ===
{
  "confidence_score": <integer 0-100>,
  "reason_summary": "<max 120 words>",
  "continuation_probability": "HIGH" | "MEDIUM" | "LOW"
}
Map: HIGH ↔ 75-100, MEDIUM ↔ 45-74, LOW ↔ 0-44.
"""

_dtpa_cache: dict[str, dict[str, Any]] | None = None
_dtpa_cache_path: Path | None = None
_dtpa_cache_dirty = False
_dtpa_cache_readonly = False
_dtpa_cache_stats = {"lookups": 0, "hits": 0, "misses": 0, "api_calls": 0}


def resolve_dtpa_cache_path() -> Path:
    raw = os.getenv("DTPA_LLM_CACHE_PATH", "").strip()
    return Path(raw) if raw else DEFAULT_DTPA_CACHE_PATH


def make_dtpa_hash(
    pair: str,
    timestamp: str,
    direction: str,
    broken_level: float | None,
    pa_trigger_type: str,
    anchor_price: float | None,
) -> str:
    broken = f"{broken_level:.6f}" if broken_level is not None else "none"
    anchor = f"{anchor_price:.6f}" if anchor_price is not None else "none"
    payload = (
        f"{DTPA_PROMPT_VERSION}|{pair}|{timestamp}|{direction}|"
        f"{broken}|{pa_trigger_type}|{anchor}"
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _load_dtpa_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        logger.warning("DTPA cache load failed: %s", path, exc_info=True)
    return {}


def merge_production_dtpa_cache(
    target: Path | None = None,
    source: Path | None = None,
) -> int:
    target_path = target or DEFAULT_DTPA_CACHE_PATH
    source_path = source or PRODUCTION_DTPA_CACHE_PATH
    if not source_path.exists():
        return 0
    target_path.parent.mkdir(parents=True, exist_ok=True)
    merged = _load_dtpa_cache(target_path)
    source_data = _load_dtpa_cache(source_path)
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
    return added


def init_dtpa_cache(path: Path | None = None, *, readonly: bool = False) -> None:
    global _dtpa_cache, _dtpa_cache_path, _dtpa_cache_dirty, _dtpa_cache_readonly
    merge_production_dtpa_cache()
    _dtpa_cache_path = path or resolve_dtpa_cache_path()
    _dtpa_cache_path.parent.mkdir(parents=True, exist_ok=True)
    _dtpa_cache = _load_dtpa_cache(_dtpa_cache_path)
    _dtpa_cache_dirty = False
    _dtpa_cache_readonly = readonly


def flush_dtpa_cache() -> None:
    global _dtpa_cache_dirty
    if not _dtpa_cache_dirty or _dtpa_cache is None or _dtpa_cache_path is None:
        return
    _dtpa_cache_path.parent.mkdir(parents=True, exist_ok=True)
    _dtpa_cache_path.write_text(
        json.dumps(_dtpa_cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _dtpa_cache_dirty = False


def dtpa_cache_coverage_stats() -> dict[str, float | int]:
    lookups = _dtpa_cache_stats["lookups"]
    hits = _dtpa_cache_stats["hits"]
    return {
        "lookups": lookups,
        "hits": hits,
        "misses": _dtpa_cache_stats["misses"],
        "api_calls": _dtpa_cache_stats["api_calls"],
        "coverage_pct": round(hits / lookups * 100.0, 2) if lookups else 100.0,
    }


def _cache_get(dtpa_hash: str) -> dict[str, Any] | None:
    if _dtpa_cache is None:
        init_dtpa_cache()
    assert _dtpa_cache is not None
    hit = _dtpa_cache.get(dtpa_hash)
    return dict(hit) if hit else None


def _cache_put(dtpa_hash: str, record: dict[str, Any]) -> None:
    global _dtpa_cache_dirty
    if _dtpa_cache_readonly:
        return
    if _dtpa_cache is None:
        init_dtpa_cache()
    assert _dtpa_cache is not None
    _dtpa_cache[dtpa_hash] = record
    _dtpa_cache_dirty = True


def pd_timestamp_str(ts: Any) -> str:
    import pandas as pd

    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _is_backtest_mode() -> bool:
    try:
        from llm_auditor import is_backtest_mode

        return bool(is_backtest_mode())
    except ImportError:
        return os.getenv("BACKTEST_MODE", "").strip().lower() in ("1", "true", "yes", "on")


def _dtpa_live_gemini_enabled(force_live: bool = False) -> bool:
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


def _mock_dtpa_audit() -> dict[str, Any]:
    return {
        "confidence_score": 85,
        "reason_summary": "BACKTEST_MODE: DTPA 3rd-wave mock (textbook continuation).",
        "continuation_probability": "HIGH",
        "llm_latency_ms": 0,
        "model_version": "DTPA_MOCK_BT",
        "reason_codes": ["DTPA_MOCK_AUDIT"],
        "llm_decision": "ALLOW",
        "risk_score": 15,
    }


def _cache_miss_readonly_result(dtpa_hash: str) -> dict[str, Any]:
    from audit import risk_manager as audit_rm

    return {
        "confidence_score": 0,
        "reason_summary": "DTPA LLM cache miss in readonly backtest (no API call).",
        "continuation_probability": "LOW",
        "llm_latency_ms": 0,
        "model_version": "DTPA_CACHE_MISS",
        "reason_codes": [DTPA_CACHE_MISS_TAG],
        "llm_decision": audit_rm.confidence_to_llm_decision(0),
        "risk_score": 100,
        "dtpa_hash": dtpa_hash,
    }


def build_dtpa_audit_payload(setup: Any, raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair": setup.pair,
        "direction": setup.direction,
        "timestamp": pd_timestamp_str(setup.timestamp),
        "setup_type": "DTPA",
        "candidate_score": int(raw.get("candidate_score", setup.candidate_score)),
        "bos_direction": str(raw.get("bos_direction", setup.bos.direction)),
        "consecutive_structure_count": int(
            raw.get("consecutive_structure_count", setup.bos.consecutive_structure_count)
        ),
        "broken_level": raw.get("broken_level", setup.bos.broken_level),
        "structure_invalidation_level": raw.get(
            "structure_invalidation_level", setup.bos.structure_invalidation_level
        ),
        "bars_since_bos": int(raw.get("bars_since_bos", setup.pullback.bars_since_bos)),
        "pullback_depth_atr_ratio": round(
            float(raw.get("pullback_depth_atr_ratio", setup.pullback.pullback_depth_atr_ratio)), 4
        ),
        "pa_trigger_type": str(raw.get("pa_trigger_type", setup.pa_trigger.trigger_type)),
        "anchor_price": raw.get("anchor_price"),
        "htf_aligned": bool(raw.get("htf_aligned", setup.htf_aligned)),
        "htf_trend_direction": str(raw.get("htf_trend_direction", setup.h1_trend)),
        "entry_price": round(float(setup.entry_price), 6),
        "stop_loss": round(float(setup.stop_loss), 6),
        "take_profit": round(float(setup.take_profit), 6),
        "minutes_to_next_news": int(raw.get("minutes_to_news", raw.get("minutes_to_next_news", 999))),
    }


def _call_gemini_api(payload: dict[str, Any], dtpa_hash: str) -> dict[str, Any]:
    from llm_auditor import resolve_gemini_api_key, resolve_gemini_model

    logger.warning(
        "🚀 [Gemini API Call] DTPA 3rd-wave audit (hash: %s)",
        dtpa_hash,
    )
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=resolve_gemini_api_key())
    started = time.perf_counter()
    response = client.models.generate_content(
        model=resolve_gemini_model(),
        contents=json.dumps(payload, ensure_ascii=False),
        config=types.GenerateContentConfig(
            system_instruction=DTPA_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=DTPA_RESPONSE_SCHEMA,
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
        "continuation_probability": str(parsed.get("continuation_probability", "MEDIUM")),
        "llm_latency_ms": latency_ms,
        "model_version": resolve_gemini_model(),
        "reason_codes": ["DTPA_GEMINI_AUDIT"],
        "llm_decision": decision,
        "risk_score": max(0, 100 - confidence),
    }


def audit_dtpa_setup(
    setup: Any,
    raw: dict[str, Any],
    *,
    force_live: bool = False,
) -> dict[str, Any]:
    global _dtpa_cache_stats
    payload = build_dtpa_audit_payload(setup, raw)
    broken = payload.get("broken_level")
    dtpa_hash = make_dtpa_hash(
        setup.pair,
        payload["timestamp"],
        str(setup.direction),
        float(broken) if broken is not None else None,
        str(payload["pa_trigger_type"]),
        float(payload["anchor_price"]) if payload.get("anchor_price") is not None else None,
    )

    _dtpa_cache_stats["lookups"] += 1
    cached = _cache_get(dtpa_hash)
    if cached is not None:
        _dtpa_cache_stats["hits"] += 1
        hit = dict(cached)
        codes = list(hit.get("reason_codes") or [])
        if DTPA_CACHE_HIT_TAG not in codes:
            codes.insert(0, DTPA_CACHE_HIT_TAG)
        hit["reason_codes"] = codes
        return hit

    _dtpa_cache_stats["misses"] += 1
    if _dtpa_cache_readonly or (_is_backtest_mode() and not force_live and not _dtpa_live_gemini_enabled()):
        return _cache_miss_readonly_result(dtpa_hash)

    if _dtpa_live_gemini_enabled(force_live):
        _dtpa_cache_stats["api_calls"] += 1
        try:
            from llm_auditor import _assert_gemini_api_allowed

            _assert_gemini_api_allowed("DTPA Gemini audit")
            result = _call_gemini_api(payload, dtpa_hash)
        except Exception:
            logger.exception("DTPA Gemini audit failed | hash=%s", dtpa_hash)
            result = {
                "confidence_score": 50,
                "reason_summary": "DTPA audit fallback after API error.",
                "continuation_probability": "MEDIUM",
                "llm_latency_ms": 0,
                "model_version": "DTPA_ERROR_FALLBACK",
                "reason_codes": ["DTPA_AUDIT_ERROR"],
                "llm_decision": "CAUTION",
                "risk_score": 50,
            }
    else:
        result = _mock_dtpa_audit()

    record = {
        **result,
        "dtpa_hash": dtpa_hash,
        "cached_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _cache_put(dtpa_hash, record)
    flush_dtpa_cache()
    return record

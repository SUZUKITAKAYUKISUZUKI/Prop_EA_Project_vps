"""
llm_auditor.py — L4 LLM リスク監査

- 本番 / 最終 BT 検証: Google Gemini API（JSON モード）
- Optuna 走査: ローカル Ollama（既定 gemma4:e4b）— Gemini API は遮断
- 通常 BT: BACKTEST_MODE モック（strategy_edge）
- 3 秒タイムアウト時は LLM_TIMEOUT_FALLBACK（confidence=50）で即時フォールバック
- v3.4: LLM 監査結果 CSV キャッシュ
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger("llm_auditor")

# =============================================================================
# タグ体系の二重管理（開発者向け — 非対称性に注意）
# =============================================================================
# - GEMINI_ASSIGNABLE_TAGS … ライブ L4（Gemini JSON 監査）専用
# - RISK_TAG_WEIGHTS (main_platform) … レガシーシミュレータ + Optuna 重み探索用
# 両者は名前・重み・評価経路が一致しない。バックテスト BT 結果と Live 結果を
# 直接比較する際はタグ非対称性（特に AGAINST_HTF_TREND）に留意すること。
# =============================================================================

DEFAULT_MODEL = "gemini-3.1-flash-lite"  # L4 監査デフォルト。GEMINI_MODEL で上書き可
DEFAULT_OLLAMA_MODEL = "gemma4:e4b"  # Optuna L4 デフォルト。OLLAMA_MODEL で上書き可
LIVE_INFERENCE_TIMEOUT_SEC = 3.0
DEFAULT_TIMEOUT_SEC = LIVE_INFERENCE_TIMEOUT_SEC
WARMUP_TIMEOUT_SEC = 30.0
TIMEOUT_FALLBACK_RISK_SCORE = 50
DEFAULT_TEMPERATURE = 0.1

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LLM_CACHE_PATH = PROJECT_ROOT / "cache" / "llm_audit_cache.csv"
DEFAULT_LLM_CANDIDATES_PATH = PROJECT_ROOT / "cache" / "llm_candidates.csv"

MockMode = Literal["strategy_edge", "deterministic"]

CACHE_COLUMNS: tuple[str, ...] = (
    "cache_key",
    "timestamp",
    "pair",
    "direction",
    "setup_type",
    "candidate_score",
    "bayes_probability",
    "confidence_score",
    "reason_summary",
    "risk_score",
    "reason_codes",
    "llm_decision",
    "thinking",
    "llm_latency_ms",
    "model_version",
    "cached_at",
)

CANDIDATE_COLUMNS: tuple[str, ...] = (
    "cache_key",
    "timestamp",
    "pair",
    "direction",
    "setup_type",
    "candidate_score",
    "bayes_probability",
    "smt_intensity",
    "smt_leader",
    "smt_diff",
    "session_type",
    "adr_remaining",
    "htf_trend_direction",
    "minutes_to_next_news",
    "recent_losses",
    "has_bos",
    "both_sweep",
    "atr_ratio",
    "wick_ratio_pct",
    "entry_price",
    "trigger_price",
    "vp_vah",
    "vp_val",
    "vp_poc",
    "vp_is_allowed",
    "vp_location_score",
    "volume_profile_context_json",
)

_BACKTEST_MODE: bool | None = None
_MOCK_MODE: MockMode = "strategy_edge"
_OPTUNA_OLLAMA_ENABLED: bool = False
_OPTUNA_OLLAMA_MODEL: str = DEFAULT_OLLAMA_MODEL
_llm_cache: dict[str, dict[str, Any]] | None = None
_llm_cache_path: Path | None = None
_llm_cache_readonly: bool = True
DEFAULT_SMT_CONTEXT_CACHE_PATH = Path(__file__).resolve().parent / "cache" / "smt_context_cache_lsfc.json"
_smt_context_cache: dict[str, dict[str, Any]] | None = None
_smt_context_cache_path: Path | None = None
_smt_context_cache_writable: bool = False


@dataclass
class LLMCacheCoverageStats:
    """--llm-cache モードでの L4 監査カバレッジ集計。"""

    audits_total: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def coverage_pct(self) -> float:
        if self.audits_total <= 0:
            return 100.0
        return self.cache_hits / self.audits_total * 100.0


_cache_coverage_stats = LLMCacheCoverageStats()
_track_cache_coverage = False
_llm_cache_miss_sink: list[dict[str, Any]] | None = None

GEMINI_ASSIGNABLE_TAGS: tuple[str, ...] = (
    "NO_BOS",
    "SMT_DIVERGENCE_MISSING",
    "HIGH_ATR_VOLATILITY",
    # AGAINST_HTF_TREND (重み25): Gemini ライブ監査のみ。Optuna / RISK_TAG_WEIGHTS 非対称。
    "AGAINST_HTF_TREND",
    "POOR_SWEEP_REJECTION",
)

SYSTEM_REASON_CODES: frozenset[str] = frozenset(
    {
        "LLM_TIMEOUT",
        "LLM_TIMEOUT_FALLBACK",
        "LLM_PARSE_ERROR",
    }
)

ALLOWED_REASON_CODES: frozenset[str] = frozenset(GEMINI_ASSIGNABLE_TAGS) | SYSTEM_REASON_CODES

RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "confidence_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "vp_interpretation": {
            "type": "string",
            "enum": [
                "MEAN_REVERSION_SYNC",
                "EXPANSION_TRAP",
                "STANDARD_REJECTION",
                "POOR_LOCATION",
            ],
        },
        "reason_summary": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["confidence_score", "vp_interpretation", "reason_summary"],
}

SYSTEM_PROMPT = """# System Prompt: Setup Confidence Scorer (L4 Layer — Offense-Type LLM)

## Role
You are an elite institutional Quantitative Analyst for a Fintokei prop-firm desk.
Your task is NOT a binary allow/reject gate. You must score **how likely this setup is a genuine trend continuation/reversal rather than a fakeout (liquidity trap)**.

Higher scores mean the technical + liquidity structure supports real displacement; lower scores mean elevated fakeout / stop-hunt risk.

---

## Scoring Task
From a **technical and liquidity perspective**, rate confidence that this setup avoids fakeouts and can develop into a real trend move.

Output an integer **confidence_score from 0 to 100**:
- **85–100**: High conviction — strong HTF structure, clean BOS/MSS, liquidity sweep well absorbed, momentum aligned.
- **60–84**: Normal — tradable but not a "A+ spot"; some minor structural friction.
- **40–59**: Low — meaningful fakeout risk; survival sizing only if taken at all downstream.
- **0–39**: High fakeout risk — structurally unreliable (missing BOS, no SMT, poor wick rejection, news volatility, against HTF trend, etc.).

Evaluate at least:
1. Break of Structure / MSS clarity after the sweep or continuation trigger.
2. Correlated-pair SMT divergence at the liquidity event.
3. ATR expansion vs average (slippage / news risk).
4. Alignment or conflict with HTF (H1/H4) institutional trend.
5. Wick rejection quality at London high/low sweep.

When **`smt_context`** is present in the input JSON, treat it as a primary driver of `confidence_score`:
- Compare the traded **`symbol`** with **`smt_leader`** and **`divergence_direction`** (`{PAIR}_LEADING`).
- **Leader continuation**: if the setup is on the **leader** pair, reward confidence when price action / strategy intent align with the leader **continuing** the displacement (momentum persistence, not mean-reversion fade).
- **Lagging reversion**: if the setup is on the **lagging** pair, reward confidence when the setup expresses **catch-up / reversion** toward the leader (closing the cross-pair gap), not blind chase of the leader move.
- **Penalize** structural mismatch: leader pair + fade/chop setup, or lagging pair + chasing leader extension without reversion evidence.
- Weight by **`smt_intensity`** / **`smt_diff_pips`**: stronger divergence → stronger continuation vs reversion inference.
- Modulate with **`session_type`** and **`adr_remaining`**: late session / low ADR headroom reduces continuation follow-through; reversion setups may score better when ADR is nearly exhausted.

---

## Input Format
You receive JSON with timestamp, symbol, strategy, candidate_score, bayes_probability, market_data, technical_checks, and optionally **`smt_context`**.

Example `smt_context` block:
```json
{
  "smt_context": {
    "smt_intensity": 0.82,
    "smt_diff_pips": 8.3,
    "smt_leader": "GBPUSD",
    "divergence_direction": "GBPUSD_LEADING",
    "session_type": "LONDON",
    "adr_remaining": 0.34
  }
}
```

When **`volume_profile_context`** is present, cross-check it with `smt_context` (SMT × VP matrix):

Analyze `smt_context` and `volume_profile_context` together to evaluate setup validity.

**VP Base Score**
- `location_score` == 30: Strong reversal edge at VA sweep zone. (Add score)
- `location_score` == 10: Favorable value-area side. (Small add)
- `location_score` == 0: Neutral / POC vicinity. (No VP edge)
- `location_score` == -20: High trend-chasing / fade risk. (Deduct score)

**SMT × VP Matrix (CRITICAL)**
Interpretation depends on **strategy type** (continuation vs mean-reversion), not a fixed "continuation = high risk" rule:
- **MEAN_REVERSION_SYNC**: `location_score` == 30 AND SMT/leader context supports **catch-up or liquidity fade** (lagging pair, exhausted ADR, reversion intent). Optimal trap setup → `confidence_score` 90–95.
- **EXPANSION_TRAP**: `location_score` == 30 BUT SMT implies **leader continuation / displacement** while price sits at a reversal-style VA edge **against** that continuation (e.g., fade at sweep zone during leader extension). Fake reversal risk → deduct heavily (`confidence_score` < 40).
- **CONTINUATION_SUPPORT**: `location_score` in {10, 30} AND strategy + SMT align with **trend continuation** (leader pair, HTF aligned, ADR headroom). Do **not** auto-penalize; score 60–90 unless other checks fail.
- **STANDARD_REJECTION**: VP edge present without clear SMT sync/trap conflict.
- **POOR_LOCATION**: `location_score` == -20 or `is_allowed` == false.

Use `position_regime`, `price_to_poc_pips`, and VA levels (`vah` / `val` / `poc`) to justify the matrix label.

## Output JSON (strict — no markdown outside JSON)
```json
{
  "confidence_score": 92,
  "vp_interpretation": "MEAN_REVERSION_SYNC",
  "reason_summary": "VAL sweep sync with lagging SMT reversion."
}
```

- confidence_score: integer 0–100
- vp_interpretation: one of MEAN_REVERSION_SYNC | EXPANSION_TRAP | STANDARD_REJECTION | POOR_LOCATION
- reason_summary: concise rationale (max 10 words when VP context present; otherwise max 2 short sentences)
"""


def resolve_system_prompt(setup_type: str | None) -> str:
    """L4 システムプロンプト（全戦略共通）。"""
    return SYSTEM_PROMPT


def _load_dotenv() -> None:
    """プロジェクトルートの .env を os.environ へ読み込む（未設定キーのみ）。"""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_gemini_model() -> str:
    _load_dotenv()
    return os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)


def resolve_ollama_model() -> str:
    _load_dotenv()
    return os.environ.get("OLLAMA_MODEL", _OPTUNA_OLLAMA_MODEL or DEFAULT_OLLAMA_MODEL)


def resolve_ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")


def configure_optuna_ollama(enabled: bool, model: str | None = None) -> None:
    """Optuna 走査中の L4 を Ollama 経由に切り替える（Gemini は別途遮断）。"""
    global _OPTUNA_OLLAMA_ENABLED, _OPTUNA_OLLAMA_MODEL
    _OPTUNA_OLLAMA_ENABLED = enabled
    if model:
        _OPTUNA_OLLAMA_MODEL = model


def is_optuna_ollama_mode() -> bool:
    """Optuna 中かつ Ollama L4 が有効。"""
    try:
        from optuna_runtime import is_optuna_ollama_enabled
    except ImportError:
        return _OPTUNA_OLLAMA_ENABLED
    return is_optuna_ollama_enabled() or _OPTUNA_OLLAMA_ENABLED


def resolve_gemini_api_key() -> str:
    _load_dotenv()
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to .env or set the environment variable."
        )
    return key


def migrate_legacy_confidence(row: dict[str, Any]) -> int:
    """
    レガシー risk_score / llm_decision キャッシュ行を v3.4 confidence_score へ昇格。

    旧スキーマでは risk_score=危険度（高いほど悪い）だったため、
    confidence ≈ 100 - risk_score の逆写像 + 閾値帯補正で段階的に移行する。
    """
    raw = row.get("confidence_score")
    if raw not in (None, ""):
        return max(0, min(100, int(float(raw))))

    decision = str(row.get("llm_decision", row.get("decision_source", "ALLOW"))).upper()
    risk = int(float(row.get("risk_score", 50)))

    if decision in ("REJECT_BY_L4", "REJECT", "REJECT_BY_LLM") or risk >= 41:
        return max(0, min(39, 100 - risk))
    if risk >= 21:
        return max(40, min(59, 100 - risk))
    if risk <= 10:
        return max(85, min(100, 100 - risk))
    return max(60, min(84, 100 - risk))


def confidence_to_llm_decision(confidence_score: int) -> str:
    """確信度スコア → L0/L2 意思決定ラベル（lot_multiplier 帯と同期）。"""
    score = max(0, min(100, int(confidence_score)))
    if score < 40:
        return "REJECT_BY_LLM"
    if score < 60:
        return "CAUTION"
    return "ALLOW"


def parse_confidence_from_payload(parsed: dict[str, Any]) -> tuple[int, str]:
    """Gemini JSON またはレガシー risk_score ペイロードから (confidence, reason_summary) を抽出。"""
    if "confidence_score" in parsed:
        confidence = max(0, min(100, int(parsed["confidence_score"])))
        reason = str(
            parsed.get("reason_summary", parsed.get("reason", parsed.get("thinking", "")))
        )
        return confidence, reason

    legacy_row = {
        "risk_score": parsed.get("risk_score", 50),
        "llm_decision": parsed.get("action", parsed.get("llm_decision", "ALLOW")),
    }
    if str(parsed.get("action", "")).upper() == "REJECT":
        legacy_row["llm_decision"] = "REJECT_BY_L4"
    confidence = migrate_legacy_confidence(legacy_row)
    reason = str(parsed.get("reason_summary", parsed.get("reason", parsed.get("thinking", ""))))
    return confidence, reason


def parse_vp_interpretation(parsed: dict[str, Any]) -> str:
    raw = parsed.get("vp_interpretation")
    if raw not in (None, ""):
        return str(raw)
    return "STANDARD_REJECTION"


def is_backtest_mode() -> bool:
    """BACKTEST_MODE=1 のとき True（Optuna / 戦略層 BT 用 API 完全遮断）。"""
    if _BACKTEST_MODE is not None:
        return _BACKTEST_MODE
    return _backtest_mode_env_active()


def _backtest_mode_env_active() -> bool:
    return os.environ.get("BACKTEST_MODE", "").strip().lower() in ("1", "true", "yes", "on")


def _live_api_explicitly_allowed() -> bool:
    """batch_llm_audit.py / MT5 ライブのみ True。backtest_runner BT では常に False。"""
    return os.environ.get("LLM_FORCE_LIVE", "").strip().lower() in ("1", "true", "yes", "on")


def _assert_gemini_api_allowed(caller: str = "Gemini API") -> None:
    try:
        from optuna_runtime import is_optuna_runtime
    except ImportError:
        is_optuna_runtime = lambda: False  # noqa: E731

    if is_optuna_runtime():
        raise RuntimeError(
            f"{caller} blocked during Optuna (OPTUNA_MODE=1). "
            "Run backtest_runner.py --use-llm separately for final L4 validation."
        )
    if _backtest_mode_env_active() and not _live_api_explicitly_allowed():
        raise RuntimeError(
            f"{caller} blocked while BACKTEST_MODE=1. "
            "Optuna/BT must use mock, --llm-cache, or --production-llm (batch audit)."
        )


def configure_backtest_mode(enabled: bool, mock_mode: MockMode = "strategy_edge") -> None:
    """バックテスト/最適化フェーズで Gemini API を呼ばない。"""
    global _BACKTEST_MODE, _MOCK_MODE
    try:
        from optuna_runtime import is_optuna_runtime
    except ImportError:
        is_optuna_runtime = lambda: False  # noqa: E731
    if is_optuna_runtime() and not enabled:
        logger.warning("Ignoring configure_backtest_mode(False) during Optuna runtime")
        enabled = True
    _BACKTEST_MODE = enabled
    _MOCK_MODE = mock_mode


def make_cache_key(trade_context: dict[str, Any]) -> str:
    ts = str(trade_context.get("timestamp", ""))
    pair = str(trade_context.get("pair") or trade_context.get("symbol") or "")
    direction = str(trade_context.get("direction", ""))
    setup_type = str(trade_context.get("setup_type", "LONDON_SWEEP_FAILURE_CONTINUATION"))
    return f"{ts}|{pair}|{direction}|{setup_type}"


def _csv_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:  # NaN
        return None
    return parsed


def _csv_optional_int(value: Any) -> int | None:
    parsed = _csv_optional_float(value)
    if parsed is None:
        return None
    return int(parsed)


def trade_context_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """候補 CSV / キャッシュ行 → audit_trade 用 context。"""
    adr = _csv_optional_float(row.get("adr_remaining"))
    ctx: dict[str, Any] = {
        "timestamp": row.get("timestamp"),
        "pair": row.get("pair"),
        "setup_type": row.get("setup_type", "LONDON_SWEEP_FAILURE_CONTINUATION"),
        "direction": row.get("direction"),
        "smt_intensity": float(row.get("smt_intensity", 0) or 0),
        "smt_leader": row.get("smt_leader", "NONE"),
        "smt_diff": float(row.get("smt_diff", 0) or 0),
        "session_type": row.get("session_type"),
        "adr_remaining": adr,
        "minutes_to_next_news": int(float(row.get("minutes_to_next_news", 999) or 999)),
        "recent_losses": int(float(row.get("recent_losses", 0) or 0)),
        "has_bos": str(row.get("has_bos", "True")).lower() in ("1", "true", "yes"),
        "both_sweep": str(row.get("both_sweep", "True")).lower() in ("1", "true", "yes"),
        "atr_ratio": float(row.get("atr_ratio", 1.0) or 1.0),
        "wick_ratio_pct": float(row.get("wick_ratio_pct", 0) or 0),
        "candidate_score": float(row.get("candidate_score", 0) or 0),
        "bayes_probability": float(row.get("bayes_probability", 0) or 0),
    }
    htf = row.get("htf_trend_direction")
    if htf not in (None, ""):
        ctx["htf_trend_direction"] = str(htf)
    for price_key in ("entry_price", "trigger_price"):
        parsed = _csv_optional_float(row.get(price_key))
        if parsed is not None:
            ctx[price_key] = parsed
    for vp_key in ("vp_vah", "vp_val", "vp_poc"):
        parsed = _csv_optional_float(row.get(vp_key))
        if parsed is not None:
            ctx[vp_key] = parsed
    vp_allowed = row.get("vp_is_allowed")
    if vp_allowed not in (None, ""):
        ctx["vp_is_allowed"] = str(vp_allowed).lower() in ("1", "true", "yes")
    vp_loc = _csv_optional_int(row.get("vp_location_score"))
    if vp_loc is not None:
        ctx["vp_location_score"] = vp_loc
    vp_json = row.get("volume_profile_context_json")
    if vp_json not in (None, "", "nan"):
        try:
            embedded = json.loads(str(vp_json))
            if isinstance(embedded, dict) and embedded:
                ctx["volume_profile_context"] = embedded
        except json.JSONDecodeError:
            pass
    return ctx


def candidate_row_from_context(trade_context: dict[str, Any]) -> dict[str, Any]:
    key = make_cache_key(trade_context)
    row: dict[str, Any] = {
        "cache_key": key,
        "timestamp": trade_context.get("timestamp"),
        "pair": trade_context.get("pair"),
        "direction": trade_context.get("direction"),
        "setup_type": trade_context.get("setup_type", "LONDON_SWEEP_FAILURE_CONTINUATION"),
        "candidate_score": trade_context.get("candidate_score"),
        "bayes_probability": trade_context.get("bayes_probability"),
        "smt_intensity": trade_context.get("smt_intensity"),
        "smt_leader": trade_context.get("smt_leader", "NONE"),
        "smt_diff": trade_context.get("smt_diff", 0),
        "session_type": trade_context.get("session_type"),
        "adr_remaining": trade_context.get("adr_remaining"),
        "htf_trend_direction": trade_context.get("htf_trend_direction"),
        "minutes_to_next_news": trade_context.get("minutes_to_next_news"),
        "recent_losses": trade_context.get("recent_losses"),
        "has_bos": trade_context.get("has_bos"),
        "both_sweep": trade_context.get("both_sweep"),
        "atr_ratio": trade_context.get("atr_ratio"),
        "wick_ratio_pct": trade_context.get("wick_ratio_pct"),
        "entry_price": trade_context.get("entry_price"),
        "trigger_price": trade_context.get("trigger_price"),
    }
    vp = trade_context.get("volume_profile_context")
    if isinstance(vp, dict) and vp:
        row["volume_profile_context_json"] = json.dumps(vp, ensure_ascii=False)
        row["vp_vah"] = vp.get("vah")
        row["vp_val"] = vp.get("val")
        row["vp_poc"] = vp.get("poc")
        row["vp_is_allowed"] = vp.get("is_allowed")
        row["vp_location_score"] = vp.get("location_score")
    return row


def write_candidates_csv(candidates: list[dict[str, Any]], output_path: Path) -> int:
    """L4 候補 CSV を canonical 列順で書き出す。"""
    import pandas as pd

    if not candidates:
        return 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(candidates).drop_duplicates(subset=["cache_key"])
    for col in CANDIDATE_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df[list(CANDIDATE_COLUMNS)].to_csv(output_path, index=False, encoding="utf-8-sig")
    return len(df)


def load_llm_audit_cache(path: Path | None = None) -> dict[str, dict[str, Any]]:
    cache_path = path or _llm_cache_path or DEFAULT_LLM_CACHE_PATH
    if not cache_path.is_file():
        return {}
    loaded: dict[str, dict[str, Any]] = {}
    with cache_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = row.get("cache_key", "")
            if not key:
                continue
            tags_raw = row.get("reason_codes", "[]")
            try:
                tags = json.loads(tags_raw) if tags_raw.startswith("[") else []
            except json.JSONDecodeError:
                tags = []
            confidence = migrate_legacy_confidence(row)
            reason_summary = str(
                row.get("reason_summary") or row.get("thinking") or ""
            )
            llm_decision = confidence_to_llm_decision(confidence)
            loaded[key] = {
                "confidence_score": confidence,
                "vp_interpretation": str(row.get("vp_interpretation", "STANDARD_REJECTION")),
                "reason_summary": reason_summary,
                "reason_codes": tags,
                "risk_score": max(0, 100 - confidence),
                "llm_risk_score": max(0, 100 - confidence),
                "llm_latency_ms": int(float(row.get("llm_latency_ms", 0))),
                "decision_source": llm_decision,
                "llm_decision": llm_decision,
                "model_version": row.get("model_version", DEFAULT_MODEL),
                "llm_status": "cache_hit",
                "thinking": reason_summary,
            }
    return loaded


def configure_llm_cache(path: Path | None, readonly: bool = True) -> None:
    """LLM 監査キャッシュを有効化。readonly=True なら API はキャッシュミス時のみ警告。"""
    global _llm_cache, _llm_cache_path, _llm_cache_readonly
    if path is None:
        _llm_cache = None
        _llm_cache_path = None
        _llm_cache_readonly = True
        enable_cache_coverage_tracking(False)
        return
    _llm_cache_path = path
    _llm_cache_readonly = readonly
    _llm_cache = load_llm_audit_cache(path)
    enable_cache_coverage_tracking(readonly)
    logger.info("LLM audit cache loaded: %d entries from %s", len(_llm_cache), path)


def enable_cache_coverage_tracking(enabled: bool = True) -> None:
    """backtest_runner --llm-cache 時にカバレッジ計測を ON/OFF。"""
    global _track_cache_coverage
    _track_cache_coverage = enabled
    reset_cache_coverage_stats()


def reset_cache_coverage_stats() -> None:
    global _cache_coverage_stats
    _cache_coverage_stats = LLMCacheCoverageStats()


def enable_cache_miss_collection(sink: list[dict[str, Any]] | None) -> None:
    """readonly キャッシュミス時に candidate_row を sink へ追記（増分 batch audit 用）。"""
    global _llm_cache_miss_sink
    _llm_cache_miss_sink = sink


def get_cache_coverage_stats() -> LLMCacheCoverageStats:
    return _cache_coverage_stats


def _record_cache_coverage(llm_status: str) -> None:
    if not _track_cache_coverage:
        return
    global _cache_coverage_stats
    _cache_coverage_stats.audits_total += 1
    if llm_status == "cache_hit":
        _cache_coverage_stats.cache_hits += 1
    elif llm_status == "cache_miss_fallback":
        _cache_coverage_stats.cache_misses += 1


LLM_CACHE_COVERAGE_WARN_THRESHOLD_PCT = 95.0
LLM_CACHE_COVERAGE_WARNING = (
    "⚠️ 警告: キャッシュミス率が高いため、LLM監査をバイパスした不正確な判定が含まれています。"
    "batch_llm_audit.py を再実行してキャッシュを更新してください。"
)


def get_cached_audit(trade_context: dict[str, Any]) -> dict[str, Any] | None:
    if _llm_cache is None:
        return None
    return _llm_cache.get(make_cache_key(trade_context))


def append_cache_entry(
    trade_context: dict[str, Any],
    result: dict[str, Any],
    path: Path | None = None,
) -> None:
    """バッチ監査結果をキャッシュ CSV へ追記（同一 cache_key は上書き）。"""
    global _llm_cache, _llm_cache_path

    cache_path = path or _llm_cache_path or DEFAULT_LLM_CACHE_PATH
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    rows_by_key: dict[str, dict[str, Any]] = {}
    if cache_path.is_file():
        with cache_path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                key = row.get("cache_key", "")
                if key:
                    rows_by_key[key] = row

    key = make_cache_key(trade_context)
    rows_by_key[key] = {
        "cache_key": key,
        "timestamp": trade_context.get("timestamp"),
        "pair": trade_context.get("pair"),
        "direction": trade_context.get("direction"),
        "setup_type": trade_context.get("setup_type", "LONDON_SWEEP_FAILURE_CONTINUATION"),
        "candidate_score": trade_context.get("candidate_score"),
        "bayes_probability": trade_context.get("bayes_probability"),
        "confidence_score": result.get("confidence_score"),
        "reason_summary": result.get("reason_summary", result.get("thinking", "")),
        "risk_score": result.get("risk_score"),
        "reason_codes": json.dumps(result.get("reason_codes", []), ensure_ascii=False),
        "llm_decision": result.get("llm_decision"),
        "thinking": result.get("reason_summary", result.get("thinking", "")),
        "llm_latency_ms": result.get("llm_latency_ms", 0),
        "model_version": result.get("model_version", resolve_gemini_model()),
        "cached_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    with cache_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CACHE_COLUMNS))
        writer.writeheader()
        for row in rows_by_key.values():
            writer.writerow(row)

    _llm_cache_path = cache_path
    _llm_cache = load_llm_audit_cache(cache_path)


def llm_cache_coverage_stats(
    candidates_path: Path,
    cache_path: Path | None = None,
) -> dict[str, int]:
    """候補 CSV と監査キャッシュのキー一致数（Step 2 進捗確認用）。"""
    import pandas as pd

    df = pd.read_csv(candidates_path)
    total = len(df)
    cached_keys: set[str] = set()
    if cache_path is not None and cache_path.is_file():
        cache_df = pd.read_csv(cache_path)
        if "cache_key" in cache_df.columns:
            cached_keys = set(cache_df["cache_key"].dropna().astype(str))

    missing = 0
    for rec in df.to_dict(orient="records"):
        ctx = trade_context_from_row(rec)
        if make_cache_key(ctx) not in cached_keys:
            missing += 1
    cached = total - missing
    return {"total": total, "cached": cached, "missing": missing}


def batch_audit_candidates(
    candidates_path: Path,
    cache_path: Path | None = None,
    *,
    skip_cached: bool = True,
) -> dict[str, int]:
    """
    候補 CSV に対して Gemini 一括監査し、結果をキャッシュ CSV へ保存。

    Returns: {"total", "audited", "skipped", "failed"}
    """
    import pandas as pd

    out_path = cache_path or DEFAULT_LLM_CACHE_PATH
    os.environ["LLM_FORCE_LIVE"] = "1"
    os.environ.pop("BACKTEST_MODE", None)
    configure_backtest_mode(False)
    configure_llm_cache(out_path, readonly=False)
    df = pd.read_csv(candidates_path)
    auditor = AuditAuditor(use_mock=False)
    stats = {"total": len(df), "audited": 0, "skipped": 0, "failed": 0}
    logger.info(
        "Batch LLM audit started (Step 2 only — BT simulation does not call Gemini): %d candidates",
        stats["total"],
    )

    for rec in df.to_dict(orient="records"):
        ctx = trade_context_from_row(rec)
        key = make_cache_key(ctx)
        if skip_cached and key in (_llm_cache or {}):
            stats["skipped"] += 1
            continue
        try:
            result = auditor.audit_trade(ctx)
            append_cache_entry(ctx, result, out_path)
            stats["audited"] += 1
            done = stats["audited"] + stats["skipped"] + stats["failed"]
            if done % 25 == 0 or done == stats["total"]:
                logger.info(
                    "Batch audit progress: %d/%d (audited=%d skipped=%d failed=%d)",
                    done,
                    stats["total"],
                    stats["audited"],
                    stats["skipped"],
                    stats["failed"],
                )
        except Exception as exc:
            logger.exception("Batch audit failed for %s: %s", key, exc)
            stats["failed"] += 1

    auditor.close()
    configure_llm_cache(out_path, readonly=True)
    return stats


def _import_pipeline_rules() -> tuple[dict[str, int], Any, str]:
    try:
        from feature_engineering import MODEL_VERSION, RISK_TAG_WEIGHTS, risk_to_decision

        return RISK_TAG_WEIGHTS, risk_to_decision, MODEL_VERSION
    except ImportError:  # pragma: no cover
        def risk_to_decision(score: int) -> str:
            if score >= 41:
                return "REJECT_BY_L4"
            if score >= 21:
                return "CAUTION"
            return "ALLOW"

        return {}, risk_to_decision, DEFAULT_MODEL


RISK_TAG_WEIGHTS, risk_to_decision, DEFAULT_MODEL_VERSION = _import_pipeline_rules()


def _infer_session_label(timestamp: str | None) -> str:
    if not timestamp:
        return "Unknown"
    try:
        hour = int(str(timestamp)[11:13])
    except (ValueError, IndexError):
        return "Unknown"
    if 15 <= hour <= 20:
        return "London_Session"
    if hour == 21:
        return "NY_Open"
    return "Off_Session"


def _infer_htf_trend(direction: str | None, candidate_score: float | None) -> str:
    """H1 トレンド未提供時の保守的プレースホルダー。"""
    _ = direction
    if candidate_score is not None and float(candidate_score) >= 80:
        return "Neutral"
    return "Unknown"


_LEADER_PAIR_MAP: dict[str, str] = {
    "GBP": "GBPUSD",
    "EUR": "EURUSD",
    "GBPUSD": "GBPUSD",
    "EURUSD": "EURUSD",
}


def normalize_smt_leader_pair(leader: str | None) -> str:
    key = str(leader or "NONE").strip().upper()
    if key in ("NONE", "", "UNK", "UNKNOWN"):
        return "NONE"
    if key in _LEADER_PAIR_MAP:
        return _LEADER_PAIR_MAP[key]
    if key.startswith("GBP"):
        return "GBPUSD"
    if key.startswith("EUR"):
        return "EURUSD"
    return "NONE"


def build_divergence_direction(leader_pair: str) -> str:
    if leader_pair == "NONE":
        return "NONE"
    return f"{leader_pair}_LEADING"


def _resolve_session_type_label(trade_context: dict[str, Any]) -> str:
    explicit = trade_context.get("session_type")
    if explicit not in (None, ""):
        return str(explicit).upper()
    session = trade_context.get("session")
    if session not in (None, ""):
        return str(session).upper()
    return _infer_session_label(str(trade_context.get("timestamp") or "")).upper()


def has_smt_context(trade_context: dict[str, Any]) -> bool:
    if isinstance(trade_context.get("smt_context"), dict):
        return True
    both_sweep = bool(trade_context.get("both_sweep", trade_context.get("smt_divergence_confirmed", False)))
    intensity = float(trade_context.get("smt_intensity", 0) or 0)
    leader = normalize_smt_leader_pair(str(trade_context.get("smt_leader", "NONE")))
    return both_sweep or intensity > 0.0 or leader != "NONE"


def build_smt_context(trade_context: dict[str, Any]) -> dict[str, Any] | None:
    """Gemini 入力用 smt_context。乖離の leader / lagging 解釈材料を含む。"""
    embedded = trade_context.get("smt_context")
    if isinstance(embedded, dict):
        return dict(embedded)

    if not has_smt_context(trade_context):
        return None

    diff_pips = float(trade_context.get("smt_diff", trade_context.get("smt_diff_pips", 0)) or 0)
    intensity = float(trade_context.get("smt_intensity", abs(diff_pips)) or 0)
    leader_pair = normalize_smt_leader_pair(str(trade_context.get("smt_leader", "NONE")))
    adr_raw = trade_context.get("adr_remaining")
    adr_remaining = round(float(adr_raw), 4) if adr_raw is not None else None

    ctx: dict[str, Any] = {
        "smt_intensity": round(intensity, 4),
        "smt_diff_pips": round(diff_pips, 4),
        "smt_leader": leader_pair,
        "divergence_direction": build_divergence_direction(leader_pair),
        "session_type": _resolve_session_type_label(trade_context),
    }
    if adr_remaining is not None:
        ctx["adr_remaining"] = adr_remaining
    return ctx


def smt_context_confidence_adjustment(trade_context: dict[str, Any], smt_ctx: dict[str, Any]) -> int:
    """
    smt_context に基づく confidence 調整（mock / 決定論的フォールバック用）。

    リーダー通貨ペア + 継続方向整合 → 加点、遅行ペア + リーダー追随（逆方向） → 減点。
    """
    symbol = str(trade_context.get("pair") or trade_context.get("symbol") or "").upper()
    leader = str(smt_ctx.get("smt_leader", "NONE")).upper()
    intensity = float(smt_ctx.get("smt_intensity", 0) or 0)
    if leader == "NONE" or intensity <= 0.0 or not symbol:
        return 0

    traded_is_leader = symbol == leader or symbol.startswith(leader[:3])
    diff_pips = float(smt_ctx.get("smt_diff_pips", 0) or 0)
    direction = str(trade_context.get("direction", "BUY")).upper()
    if leader == "GBPUSD":
        leader_bullish = diff_pips > 0
    elif leader == "EURUSD":
        leader_bullish = diff_pips < 0
    else:
        return 0

    trade_bullish = direction == "BUY"
    aligned = trade_bullish == leader_bullish

    magnitude = min(15, int(intensity // 2))
    if traded_is_leader:
        return magnitude if aligned else -magnitude
    return magnitude if not aligned else -max(5, magnitude // 2)


def map_position_regime(direction: str, location_score: int) -> str:
    """location_score → LLM 向け position_regime ラベル。"""
    from volume_profile_analyzer import normalize_trade_direction

    side = normalize_trade_direction(direction)
    score = int(location_score)
    if score == 30:
        return "BELOW_VAL_SWEEP_ZONE" if side == "BUY" else "ABOVE_VAH_SWEEP_ZONE"
    if score == 10:
        return "割安エリア（VAL～POC）" if side == "BUY" else "割高エリア（VAH～POC）"
    if score == -20:
        return "ペナルティ・エリア（逆張りの踏み上げリスク高）"
    return "適正価格付近（POC周辺）"


def _price_to_poc_pips(trigger_price: float, poc: float, pip_size: float) -> float:
    import math

    if pip_size <= 0.0 or math.isnan(poc):
        return 0.0
    return round((float(trigger_price) - float(poc)) / pip_size, 4)


def build_volume_profile_context_from_levels(
    *,
    levels: dict[str, float],
    direction: str,
    trigger_price: float,
    is_allowed: bool,
    location_score: int,
    pip_size: float,
) -> dict[str, Any]:
    """SessionVolumeProfile 結果から Gemini 用 volume_profile_context を構築。"""
    import math

    vah = float(levels.get("vah", float("nan")))
    val = float(levels.get("val", float("nan")))
    poc = float(levels.get("poc", float("nan")))
    if any(math.isnan(x) for x in (vah, val, poc)):
        return {}

    return {
        "vah": round(vah, 5),
        "val": round(val, 5),
        "poc": round(poc, 5),
        "trigger_price": round(float(trigger_price), 5),
        "is_allowed": bool(is_allowed),
        "location_score": int(location_score),
        "price_to_poc_pips": _price_to_poc_pips(trigger_price, poc, pip_size),
        "position_regime": map_position_regime(direction, int(location_score)),
    }


def build_volume_profile_context(trade_context: dict[str, Any]) -> dict[str, Any] | None:
    """trade_context から volume_profile_context を解決（事前計算済み or スカラー再構成）。"""
    embedded = trade_context.get("volume_profile_context")
    if isinstance(embedded, dict) and embedded:
        return dict(embedded)

    vah = trade_context.get("vp_vah")
    val = trade_context.get("vp_val")
    poc = trade_context.get("vp_poc")
    if vah is None or val is None or poc is None:
        return None

    direction = str(trade_context.get("direction", "BUY"))
    from volume_profile_analyzer import normalize_trade_direction

    direction = normalize_trade_direction(direction)
    trigger_price = float(
        trade_context.get("trigger_price", trade_context.get("entry_price", 0.0)) or 0.0
    )
    pip_size = float(trade_context.get("pip_size", 0.0001) or 0.0001)
    return build_volume_profile_context_from_levels(
        levels={"vah": float(vah), "val": float(val), "poc": float(poc)},
        direction=direction,
        trigger_price=trigger_price,
        is_allowed=bool(trade_context.get("vp_is_allowed", True)),
        location_score=int(trade_context.get("vp_location_score", 0)),
        pip_size=pip_size,
    )


def infer_smt_regime_hint(
    trade_context: dict[str, Any],
    smt_ctx: dict[str, Any] | None,
) -> str:
    """SMT × VP マトリクス用: CONTINUATION / MEAN_REVERSION / NEUTRAL。"""
    if smt_ctx is None:
        return "NEUTRAL"

    symbol = str(trade_context.get("pair") or trade_context.get("symbol") or "").upper()
    leader = str(smt_ctx.get("smt_leader", "NONE")).upper()
    adr = float(smt_ctx.get("adr_remaining", 0.5) if smt_ctx.get("adr_remaining") is not None else 0.5)
    intensity = float(smt_ctx.get("smt_intensity", 0) or 0)

    traded_is_leader = bool(symbol) and (symbol == leader or symbol.startswith(leader[:3]))
    if traded_is_leader and adr >= 0.55 and intensity >= 4.0:
        return "CONTINUATION"
    if (not traded_is_leader and leader != "NONE") or adr <= 0.35:
        return "MEAN_REVERSION"
    return "NEUTRAL"


def infer_vp_interpretation(
    trade_context: dict[str, Any],
    smt_ctx: dict[str, Any] | None,
    vp_ctx: dict[str, Any] | None,
) -> str:
    if vp_ctx is None:
        return "STANDARD_REJECTION"

    location_score = int(vp_ctx.get("location_score", 0))
    if location_score == -20 or not bool(vp_ctx.get("is_allowed", True)):
        return "POOR_LOCATION"
    if location_score != 30:
        return "STANDARD_REJECTION"

    smt_hint = infer_smt_regime_hint(trade_context, smt_ctx)
    if smt_hint == "MEAN_REVERSION":
        return "MEAN_REVERSION_SYNC"
    if smt_hint == "CONTINUATION":
        return "EXPANSION_TRAP"
    return "STANDARD_REJECTION"


def vp_matrix_confidence(
    base_confidence: int,
    trade_context: dict[str, Any],
    smt_ctx: dict[str, Any] | None,
    vp_ctx: dict[str, Any] | None,
) -> tuple[int, str]:
    """VP × SMT マトリクスに基づく confidence と vp_interpretation（mock 用）。"""
    interpretation = infer_vp_interpretation(trade_context, smt_ctx, vp_ctx)
    confidence = int(base_confidence)

    if vp_ctx is not None:
        loc = int(vp_ctx.get("location_score", 0))
        if loc == 30:
            confidence += 8
        elif loc == 10:
            confidence += 3
        elif loc == -20:
            confidence -= 15

    if interpretation == "MEAN_REVERSION_SYNC":
        confidence = max(confidence, 90)
    elif interpretation == "EXPANSION_TRAP":
        confidence = min(confidence, 39)
    elif interpretation == "POOR_LOCATION":
        confidence = min(confidence, 45)

    return max(0, min(100, confidence)), interpretation


def build_gemini_signal_payload(trade_context: dict[str, Any]) -> dict[str, Any]:
    """Gemini 入力 JSON（Market_Context + Strategy_Signal）を構築。"""
    pair = str(trade_context.get("pair") or trade_context.get("symbol") or "UNKNOWN")
    direction = str(trade_context.get("direction") or trade_context.get("signal_direction") or "BUY")
    timestamp = trade_context.get("timestamp")
    candidate_score = trade_context.get("candidate_score")
    bayes_probability = trade_context.get("bayes_probability", 0.0)
    atr_ratio = float(trade_context.get("atr_ratio", trade_context.get("atr_ratio_to_avg", 1.0)))
    wick_pct = float(
        trade_context.get("wick_ratio_pct", trade_context.get("wick_rejection_percent", 0.0))
    )
    both_sweep = bool(trade_context.get("both_sweep", trade_context.get("smt_divergence_confirmed", True)))
    has_bos = bool(trade_context.get("has_bos", trade_context.get("bos_mss_confirmed", True)))
    htf_trend = str(
        trade_context.get("htf_trend_direction")
        or trade_context.get("htf_trend_h1")
        or _infer_htf_trend(direction, float(candidate_score) if candidate_score is not None else None)
    ).upper()

    payload: dict[str, Any] = {
        "timestamp": timestamp,
        "symbol": pair,
        "trade_direction": direction,
        "htf_trend": normalize_htf_trend_label(htf_trend),
        "strategy": trade_context.get("setup_type", "London_Sweep_Reversal"),
        "candidate_score": candidate_score,
        "bayes_probability": bayes_probability,
        "market_data": {
            "session": trade_context.get("session") or _infer_session_label(str(timestamp or "")),
            "atr_ratio_to_avg": round(atr_ratio, 4),
            "htf_trend_h1": htf_trend,
            "htf_trend_direction": htf_trend,
            "signal_direction": direction,
            "minutes_to_next_news": trade_context.get("minutes_to_next_news"),
            "recent_losses": trade_context.get("recent_losses"),
            "smt_intensity": trade_context.get("smt_intensity"),
            "smt_leader": trade_context.get("smt_leader", "NONE"),
            "smt_diff": trade_context.get("smt_diff", 0.0),
        },
        "technical_checks": {
            "smt_divergence_confirmed": both_sweep,
            "bos_mss_confirmed": has_bos,
            "wick_rejection_percent": round(wick_pct, 2),
        },
    }
    smt_ctx = build_smt_context(trade_context)
    if smt_ctx is not None:
        payload["smt_context"] = smt_ctx
    vp_ctx = build_volume_profile_context(trade_context)
    if vp_ctx:
        payload["volume_profile_context"] = vp_ctx
    return payload


def normalize_htf_trend_label(htf: str | None) -> str:
    label = str(htf or "NEUTRAL").upper()
    if label in ("BULL", "BULLISH", "UP"):
        return "BULL"
    if label in ("BEAR", "BEARISH", "DOWN"):
        return "BEAR"
    return "NEUTRAL"


SMT_CONTEXT_SYSTEM_PROMPT = """ROLE:
Qualitative SMT Context Auditor

TASK:
Determine if the SMT correlation anomaly indicates
Continuation or Mean Reversion.

INPUT:
- strategy_type: string (e.g. LONDON_SWEEP_FAILURE_CONTINUATION, FVG_FILL)
- session_type: string
- trade_direction: string ("BUY" | "SELL")
- smt_leader: string
- divergence_direction: string
- htf_trend: string ("BULL" | "BEAR" | "NEUTRAL")

RULES:
- Use strategy_type to judge which interpretation is **favorable**, not uniformly risky:
  - Continuation strategies (LONDON_SWEEP_FAILURE_CONTINUATION): favor CONTINUATION when HTF aligns with trade_direction; penalize MEAN_REVERSION only when HTF/session imply exhaustion trap.
  - Mean-reversion strategies (FVG_FILL): favor MEAN_REVERSION near session liquidity limits; penalize CONTINUATION when HTF opposes trade_direction.
- CONTINUATION: institution-led displacement aligned with HTF trend and trade_direction
- MEAN_REVERSION: exhausted liquidity trap / catch-up fade near session limits
- NEUTRAL: insufficient context to determine
- multiplier is **downward-only** adjustment to base lot (never amplify):
  - Strong alignment (interpretation matches strategy + HTF): 0.95–1.0
  - Mild conflict or NEUTRAL: 0.75–0.90
  - Clear opposed HTF / wrong regime for strategy: 0.50–0.70

OUTPUT JSON strictly:
{
  "smt_interpretation": "CONTINUATION" | "MEAN_REVERSION" | "NEUTRAL",
  "multiplier": <float 0.5 to 1.0>,
  "reason": "<max 5 words>"
}

NOTE: multiplier adjusts base_lot_factor downward only.
1.0 = no change, 0.5 = halve the position."""


SMT_CONTEXT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "smt_interpretation": {
            "type": "string",
            "enum": ["CONTINUATION", "MEAN_REVERSION", "NEUTRAL"],
        },
        "multiplier": {"type": "number", "minimum": 0.5, "maximum": 1.0},
        "reason": {"type": "string"},
    },
    "required": ["smt_interpretation", "multiplier", "reason"],
}


def build_smt_context_audit_payload(
    *,
    session_type: str,
    trade_direction: str,
    smt_leader: str,
    divergence_direction: str,
    htf_trend: str,
    strategy_type: str = "LONDON_SWEEP_FAILURE_CONTINUATION",
) -> dict[str, Any]:
    from volume_profile_analyzer import normalize_trade_direction

    return {
        "strategy_type": str(strategy_type or "LONDON_SWEEP_FAILURE_CONTINUATION").upper(),
        "session_type": str(session_type or "UNKNOWN").upper(),
        "trade_direction": normalize_trade_direction(trade_direction),
        "smt_leader": normalize_smt_leader_pair(smt_leader),
        "divergence_direction": str(divergence_direction or "NONE").upper(),
        "htf_trend": normalize_htf_trend_label(htf_trend),
    }


def clamp_smt_multiplier(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 1.0
    return max(0.5, min(1.0, parsed))


def smt_context_cache_key(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def configure_smt_context_cache(
    path: Path | None,
    *,
    readonly: bool = True,
    clear: bool = False,
) -> None:
    """LSFC L4 SMT 文脈監査 JSON キャッシュ（--production-llm / BT 再開用）。"""
    global _smt_context_cache, _smt_context_cache_path, _smt_context_cache_writable
    if path is None:
        _smt_context_cache = None
        _smt_context_cache_path = None
        _smt_context_cache_writable = False
        return
    if clear and path.is_file():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    _smt_context_cache_path = path
    _smt_context_cache_writable = not readonly
    if path.is_file():
        _smt_context_cache = json.loads(path.read_text(encoding="utf-8"))
    else:
        _smt_context_cache = {}
    logger.info(
        "SMT context cache loaded: %d entries from %s (%s)",
        len(_smt_context_cache),
        path,
        "writable" if _smt_context_cache_writable else "readonly",
    )


def _persist_smt_context_cache_entry(key: str, result: dict[str, Any]) -> None:
    global _smt_context_cache
    if _smt_context_cache is None:
        _smt_context_cache = {}
    stored = {
        "smt_interpretation": str(result.get("smt_interpretation", "NEUTRAL")),
        "multiplier": clamp_smt_multiplier(result.get("multiplier", 1.0)),
        "reason": str(result.get("reason", ""))[:80],
        "llm_latency_ms": int(result.get("llm_latency_ms", 0)),
        "model_version": str(result.get("model_version", resolve_gemini_model())),
    }
    _smt_context_cache[key] = stored
    cache_path = _smt_context_cache_path or DEFAULT_SMT_CONTEXT_CACHE_PATH
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(_smt_context_cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def mock_smt_context_response(
    *,
    session_type: str,
    trade_direction: str,
    smt_leader: str,
    divergence_direction: str,
    htf_trend: str,
    strategy_type: str = "LONDON_SWEEP_FAILURE_CONTINUATION",
) -> dict[str, Any]:
    """決定論的 SMT 文脈モック（trade_direction / htf_trend / strategy_type 同期）。"""
    from volume_profile_analyzer import normalize_trade_direction

    _ = session_type
    direction = normalize_trade_direction(trade_direction)
    trend = normalize_htf_trend_label(htf_trend)
    leader = normalize_smt_leader_pair(smt_leader)
    div = str(divergence_direction or "NONE").upper()
    strat = str(strategy_type or "LONDON_SWEEP_FAILURE_CONTINUATION").upper()
    is_continuation_strat = "CONTINUATION" in strat or strat in (
        "LONDON_SWEEP_FAILURE_CONTINUATION",
        "LSFC",
        "MAIN",
        "ALL",
    )
    is_reversion_strat = "FVG" in strat or "FILL" in strat

    if leader == "NONE" or div == "NONE":
        return {
            "smt_interpretation": "NEUTRAL",
            "multiplier": 0.85,
            "reason": "weak smt context",
            "llm_latency_ms": 0,
            "model_version": "SMT_MOCK_NEUTRAL",
        }

    bullish_trade = direction == "BUY"
    bullish_htf = trend == "BULL"
    bearish_htf = trend == "BEAR"
    aligned = (bullish_trade and bullish_htf) or (not bullish_trade and bearish_htf)
    opposed = (bullish_trade and bearish_htf) or (not bullish_trade and bullish_htf)

    if aligned:
        mult = 1.0 if is_continuation_strat else 0.85
        return {
            "smt_interpretation": "CONTINUATION",
            "multiplier": mult,
            "reason": "htf aligned continuation",
            "llm_latency_ms": 0,
            "model_version": "SMT_MOCK_CONTINUATION",
        }
    if opposed:
        mult = 0.75 if is_reversion_strat else 0.65
        return {
            "smt_interpretation": "MEAN_REVERSION",
            "multiplier": mult,
            "reason": "htf fade trap risk",
            "llm_latency_ms": 0,
            "model_version": "SMT_MOCK_REVERSION",
        }
    return {
        "smt_interpretation": "NEUTRAL",
        "multiplier": 0.85,
        "reason": "neutral htf context",
        "llm_latency_ms": 0,
        "model_version": "SMT_MOCK_NEUTRAL",
    }


def _call_gemini_smt_context(payload: dict[str, Any]) -> dict[str, Any]:
    _assert_gemini_api_allowed("SMT context Gemini audit")
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=resolve_gemini_api_key())
    started = time.perf_counter()
    response = client.models.generate_content(
        model=resolve_gemini_model(),
        contents=json.dumps(payload, ensure_ascii=False),
        config=types.GenerateContentConfig(
            system_instruction=SMT_CONTEXT_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=SMT_CONTEXT_RESPONSE_SCHEMA,
            temperature=0.1,
        ),
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    text = getattr(response, "text", None) or "{}"
    parsed = json.loads(text)
    return {
        "smt_interpretation": str(parsed.get("smt_interpretation", "NEUTRAL")).upper(),
        "multiplier": clamp_smt_multiplier(parsed.get("multiplier", 1.0)),
        "reason": str(parsed.get("reason", ""))[:80],
        "llm_latency_ms": latency_ms,
        "model_version": resolve_gemini_model(),
    }


def audit_smt_context(
    *,
    session_type: str,
    trade_direction: str,
    smt_leader: str,
    divergence_direction: str,
    htf_trend: str,
    strategy_type: str = "LONDON_SWEEP_FAILURE_CONTINUATION",
) -> dict[str, Any]:
    """
    L4 定性 SMT 文脈監査 — base_lot_factor への減衰 multiplier (0.5〜1.0) のみ返す。
    """
    payload = build_smt_context_audit_payload(
        session_type=session_type,
        trade_direction=trade_direction,
        smt_leader=smt_leader,
        divergence_direction=divergence_direction,
        htf_trend=htf_trend,
        strategy_type=strategy_type,
    )
    cache_key = smt_context_cache_key(payload)
    if _smt_context_cache is not None and cache_key in _smt_context_cache:
        hit = dict(_smt_context_cache[cache_key])
        hit.setdefault("llm_latency_ms", 0)
        return hit
    if is_backtest_mode() and not _live_api_explicitly_allowed():
        return mock_smt_context_response(
            session_type=payload["session_type"],
            trade_direction=payload["trade_direction"],
            smt_leader=payload["smt_leader"],
            divergence_direction=payload["divergence_direction"],
            htf_trend=payload["htf_trend"],
            strategy_type=payload["strategy_type"],
        )
    try:
        result = _call_gemini_smt_context(payload)
    except Exception as exc:
        logger.exception("SMT context Gemini audit failed: %s", exc)
        fallback = mock_smt_context_response(
            session_type=payload["session_type"],
            trade_direction=payload["trade_direction"],
            smt_leader=payload["smt_leader"],
            divergence_direction=payload["divergence_direction"],
            htf_trend=payload["htf_trend"],
            strategy_type=payload["strategy_type"],
        )
        fallback["model_version"] = "SMT_ERROR_FALLBACK"
        fallback["reason"] = "api error fallback"
        result = fallback
    if _smt_context_cache_writable:
        _persist_smt_context_cache_entry(cache_key, result)
    return result


@dataclass
class AuditAuditor:
    """
    Google Gemini 1.5 Flash 経由の L4 定量リスク監査。

    Ollama 時代のグローバル直列ロックは廃止 — 並列リクエスト可。
    """

    model: str = field(default_factory=resolve_gemini_model)
    timeout_sec: float = DEFAULT_TIMEOUT_SEC
    temperature: float = DEFAULT_TEMPERATURE
    use_mock: bool = False
    mock_mode: MockMode = "strategy_edge"
    use_ollama: bool = False
    ollama_model: str = field(default_factory=resolve_ollama_model)
    api_key: str | None = None
    _client: Any = field(default=None, init=False, repr=False)
    _executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=8, thread_name_prefix="gemini-audit"),
        init=False,
        repr=False,
    )
    _warmed_up: bool = field(default=False, init=False, repr=False)

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        _assert_gemini_api_allowed("Gemini client init")
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "google-genai package is required. Install with: pip install google-genai"
            ) from exc

        key = self.api_key or resolve_gemini_api_key()
        self._client = genai.Client(api_key=key)

    def _should_use_ollama(self) -> bool:
        return self.use_ollama or is_optuna_ollama_mode()

    def _resolve_model_version(self) -> str:
        if self._should_use_ollama():
            return f"ollama:{self.ollama_model}"
        if self.use_mock or (is_backtest_mode() and not _live_api_explicitly_allowed()):
            return f"mock:{self.mock_mode}"
        return self.model or DEFAULT_MODEL_VERSION

    def _call_ollama(self, trade_context: dict[str, Any]) -> tuple[str, int]:
        """Ollama /api/chat 呼び出し。戻り値: (content, latency_ms)。"""
        import urllib.error
        import urllib.request

        url = f"{resolve_ollama_host()}/api/chat"
        payload = build_gemini_signal_payload(trade_context)
        setup_type = trade_context.get("setup_type")
        body = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": resolve_system_prompt(setup_type)},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "stream": False,
            "format": "json",
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "PropEA-LLMAuditor/1.0",
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec + 5) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama request failed ({url}): {exc}") from exc
        latency_ms = int((time.perf_counter() - started) * 1000)
        parsed_response = json.loads(raw)
        message = parsed_response.get("message") or {}
        content = message.get("content") or parsed_response.get("response") or ""
        return str(content), latency_ms

    def _run_ollama_inference(
        self,
        trade_context: dict[str, Any],
        timeout_sec: float,
    ) -> tuple[str, int]:
        future = self._executor.submit(self._call_ollama, trade_context)
        return future.result(timeout=timeout_sec)

    def _call_gemini(self, trade_context: dict[str, Any]) -> tuple[str, int]:
        """Gemini generate_content 呼び出し。戻り値: (content, latency_ms)。"""
        from google.genai import types

        _assert_gemini_api_allowed("Gemini generate_content")
        self._ensure_client()
        payload = build_gemini_signal_payload(trade_context)
        user_content = json.dumps(payload, ensure_ascii=False)
        setup_type = trade_context.get("setup_type")

        started = time.perf_counter()
        response = self._client.models.generate_content(
            model=self.model,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=resolve_system_prompt(setup_type),
                response_mime_type="application/json",
                response_schema=RESPONSE_JSON_SCHEMA,
                temperature=self.temperature,
            ),
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        content = getattr(response, "text", None) or ""
        return content, latency_ms

    def _run_gemini_inference(
        self,
        trade_context: dict[str, Any],
        timeout_sec: float,
    ) -> tuple[str, int]:
        """Gemini 推論（直列ロックなし — 並列実行可）。"""
        future = self._executor.submit(self._call_gemini, trade_context)
        return future.result(timeout=timeout_sec)

    def _strategy_edge_mock(self, trade_context: dict[str, Any]) -> tuple[str, int]:
        """
        アプローチ①: 戦略層最適化用 — confidence=90 / ALLOW 帯（API 完全遮断）。

        高確信度帯のモック値により L4.5 以降のロジックを通しつつ、
        Gemini コストをゼロに保つ（confidence 1.4x は L0 で後段適用）。
        """
        _ = trade_context
        body = json.dumps(
            {
                "confidence_score": 90,
                "vp_interpretation": "STANDARD_REJECTION",
                "reason_summary": "BACKTEST_MODE: strategy-layer optimization (L4 API bypass).",
            }
        )
        return body, 0

    def _mock_llm_response(self, trade_context: dict[str, Any]) -> tuple[str, int]:
        """API 未接続時の決定論的モック（v3.4 confidence スキーマ準拠）。"""
        tags: list[str] = []
        if float(trade_context.get("atr_ratio", 0)) > 1.5:
            tags.append("HIGH_ATR_VOLATILITY")
        if not trade_context.get("has_bos", True):
            tags.append("NO_BOS")
        if not trade_context.get("both_sweep", True):
            tags.append("SMT_DIVERGENCE_MISSING")
        if float(trade_context.get("wick_ratio_pct", 100)) < 25.0:
            tags.append("POOR_SWEEP_REJECTION")

        weights = {
            "NO_BOS": 10,
            "SMT_DIVERGENCE_MISSING": 20,
            "HIGH_ATR_VOLATILITY": 15,
            "AGAINST_HTF_TREND": 25,
            "POOR_SWEEP_REJECTION": 10,
        }
        risk_score = sum(weights.get(t, 10) for t in tags)
        confidence = max(0, min(100, 100 - risk_score))
        smt_ctx = build_smt_context(trade_context)
        if smt_ctx is not None:
            confidence = max(0, min(100, confidence + smt_context_confidence_adjustment(trade_context, smt_ctx)))
        vp_ctx = build_volume_profile_context(trade_context)
        confidence, vp_interpretation = vp_matrix_confidence(confidence, trade_context, smt_ctx, vp_ctx)
        summary = (
            f"Mock confidence={confidence} vp={vp_interpretation}: tags={tags or 'none'}."
        )
        body = json.dumps(
            {
                "confidence_score": confidence,
                "vp_interpretation": vp_interpretation,
                "reason_summary": summary,
            }
        )
        return body, 120

    def _invoke_llm_with_timeout(self, trade_context: dict[str, Any]) -> tuple[str, int, bool]:
        if self._should_use_ollama():
            try:
                content, latency_ms = self._run_ollama_inference(trade_context, self.timeout_sec)
                return content, latency_ms, False
            except FuturesTimeoutError:
                return "", int(self.timeout_sec * 1000), True
            except Exception as exc:
                logger.warning(
                    "Ollama inference failed during Optuna; falling back to strategy_edge mock: %s",
                    exc,
                )
                content, latency_ms = self._strategy_edge_mock(trade_context)
                return content, latency_ms, False

        if self.use_mock or (is_backtest_mode() and not _live_api_explicitly_allowed()):
            if self.mock_mode == "strategy_edge":
                content, latency_ms = self._strategy_edge_mock(trade_context)
            else:
                content, latency_ms = self._mock_llm_response(trade_context)
            return content, latency_ms, False

        try:
            content, latency_ms = self._run_gemini_inference(trade_context, self.timeout_sec)
            return content, latency_ms, False
        except FuturesTimeoutError:
            return "", int(self.timeout_sec * 1000), True

    def _build_ok_result(
        self,
        trade_context: dict[str, Any],
        parsed: dict[str, Any],
        latency_ms: int,
        llm_status: str = "ok",
    ) -> dict[str, Any]:
        confidence_score, reason_summary = parse_confidence_from_payload(parsed)
        vp_interpretation = parse_vp_interpretation(parsed)
        reason_codes = self._sanitize_tags(parsed.get("reason_codes"))
        llm_decision = confidence_to_llm_decision(confidence_score)
        # レガシー CSV 列互換: risk_score は危険度の逆数写像
        llm_risk_score = max(0, 100 - confidence_score)
        return {
            "confidence_score": confidence_score,
            "vp_interpretation": vp_interpretation,
            "reason_summary": reason_summary,
            "reason_codes": reason_codes,
            "risk_score": llm_risk_score,
            "llm_risk_score": llm_risk_score,
            "llm_latency_ms": latency_ms,
            "decision_source": llm_decision,
            "llm_decision": llm_decision,
            "model_version": self._resolve_model_version(),
            "llm_status": llm_status,
            "thinking": reason_summary,
        }

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        text = text.strip()
        if not text:
            raise ValueError("empty LLM response")
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        raise ValueError("JSON object not found in LLM response")

    @staticmethod
    def _sanitize_tags(raw_tags: Any) -> list[str]:
        if not isinstance(raw_tags, list):
            return []
        cleaned: list[str] = []
        for tag in raw_tags:
            token = str(tag).strip().upper()
            if token in GEMINI_ASSIGNABLE_TAGS and token not in cleaned:
                cleaned.append(token)
        return cleaned

    @staticmethod
    def _decision_from_gemini(risk_score: int, action: str) -> str:
        """レガシー risk_score/action ペイロード互換（非推奨 — confidence 経路を優先）。"""
        confidence = migrate_legacy_confidence(
            {"risk_score": risk_score, "llm_decision": "REJECT_BY_L4" if action == "REJECT" else "ALLOW"}
        )
        return confidence_to_llm_decision(confidence)

    def _timeout_fallback_result(self, latency_ms: int) -> dict[str, Any]:
        # タイムアウト時は Normal 帯中央（60–84）へフォールバック — 攻めすぎず完全停止も避ける
        confidence = 50
        tags = ["LLM_TIMEOUT_FALLBACK"]
        decision = confidence_to_llm_decision(confidence)
        logger.warning(
            "LLM audit timeout fallback: llm_latency_ms=%d confidence_score=%d tags=%s",
            latency_ms,
            confidence,
            tags,
        )
        return {
            "confidence_score": confidence,
            "vp_interpretation": "STANDARD_REJECTION",
            "reason_summary": "LLM timeout - defaulting to neutral confidence (50).",
            "reason_codes": tags,
            "risk_score": max(0, 100 - confidence),
            "llm_risk_score": None,
            "llm_latency_ms": latency_ms,
            "decision_source": decision,
            "llm_decision": decision,
            "model_version": self._resolve_model_version(),
            "llm_status": "timeout_fallback",
            "thinking": "",
        }

    def _parse_error_result(self, latency_ms: int, detail: str) -> dict[str, Any]:
        # パース不能 = 構造不明 → High Risk 帯（<40）で強制 REJECT_BY_LLM
        confidence = 35
        tags = ["LLM_PARSE_ERROR"]
        decision = confidence_to_llm_decision(confidence)
        logger.warning("LLM parse error: llm_latency_ms=%d detail=%s", latency_ms, detail)
        return {
            "confidence_score": confidence,
            "vp_interpretation": "POOR_LOCATION",
            "reason_summary": f"Parse error: {detail}",
            "reason_codes": tags,
            "risk_score": max(0, 100 - confidence),
            "llm_risk_score": None,
            "llm_latency_ms": latency_ms,
            "decision_source": decision,
            "llm_decision": decision,
            "model_version": self._resolve_model_version(),
            "llm_status": "parse_error",
            "llm_error": detail,
            "thinking": "",
        }

    def audit_trade(self, trade_context: dict[str, Any]) -> dict[str, Any]:
        """
        同期版 L4 監査（Gemini 1.5 Flash）。

        優先順: キャッシュヒット → BACKTEST_MODE モック →  live API
        """
        cached = get_cached_audit(trade_context)
        if cached is not None:
            _record_cache_coverage(str(cached.get("llm_status", "cache_hit")))
            return cached

        if _llm_cache is not None and _llm_cache_readonly:
            logger.warning(
                "LLM cache miss for %s — strategy_edge fallback (run batch_llm_audit.py first)",
                make_cache_key(trade_context),
            )
            if _llm_cache_miss_sink is not None:
                _llm_cache_miss_sink.append(candidate_row_from_context(trade_context))
            parsed = self._extract_json_object(self._strategy_edge_mock(trade_context)[0])
            result = self._build_ok_result(trade_context, parsed, 0, llm_status="cache_miss_fallback")
            _record_cache_coverage("cache_miss_fallback")
            return result

        try:
            content, latency_ms, timed_out = self._invoke_llm_with_timeout(trade_context)
        except Exception as exc:  # noqa: BLE001
            logger.exception("LLM call failed: %s", exc)
            return self._parse_error_result(int(self.timeout_sec * 1000), str(exc))

        if timed_out:
            return self._timeout_fallback_result(latency_ms)

        try:
            parsed = self._extract_json_object(content)
        except (ValueError, json.JSONDecodeError) as exc:
            return self._parse_error_result(latency_ms, str(exc))

        result = self._build_ok_result(trade_context, parsed, latency_ms)

        logger.info(
            "LLM audit ok: pair=%s llm_latency_ms=%d confidence=%d decision=%s",
            trade_context.get("pair"),
            latency_ms,
            result["confidence_score"],
            result["llm_decision"],
        )
        return result

    async def audit_trade_async(self, trade_context: dict[str, Any]) -> dict[str, Any]:
        """非同期版 — 執行スレッドをブロックしない（並列 executor 使用）。"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.audit_trade, trade_context)

    def warmup(self) -> dict[str, Any]:
        """ライブ運用前ウォームアップ: ダミーコンテキストで API 接続を確認。"""
        if self.use_mock or (is_backtest_mode() and not self._should_use_ollama()):
            self._warmed_up = True
            return {"llm_status": "mock", "llm_latency_ms": 0, "warmed_up": True}

        if self._should_use_ollama():
            ctx = warmup_trade_context()
            started = time.perf_counter()
            try:
                content, inference_ms = self._run_ollama_inference(ctx, WARMUP_TIMEOUT_SEC)
            except FuturesTimeoutError as exc:
                raise RuntimeError(
                    f"Ollama warmup exceeded {int(WARMUP_TIMEOUT_SEC)}s — check Ollama at {resolve_ollama_host()}"
                ) from exc
            total_ms = int((time.perf_counter() - started) * 1000)
            try:
                parsed = self._extract_json_object(content)
                confidence, _ = parse_confidence_from_payload(parsed)
                tags = self._sanitize_tags(parsed.get("reason_codes"))
            except (ValueError, json.JSONDecodeError):
                tags = []
                confidence = 0
            self._warmed_up = True
            logger.info(
                "Ollama warmup complete: model=%s warmup_latency_ms=%d inference_ms=%d tags=%s",
                self.ollama_model,
                total_ms,
                inference_ms,
                tags,
            )
            return {
                "llm_status": "warm",
                "llm_latency_ms": total_ms,
                "inference_ms": inference_ms,
                "warmed_up": True,
                "warmup_tags": tags,
                "model": self.ollama_model,
                "backend": "ollama",
            }

        _assert_gemini_api_allowed("Gemini warmup")
        ctx = warmup_trade_context()
        started = time.perf_counter()
        try:
            content, inference_ms = self._run_gemini_inference(ctx, WARMUP_TIMEOUT_SEC)
        except FuturesTimeoutError as exc:
            raise RuntimeError(
                f"LLM warmup exceeded {int(WARMUP_TIMEOUT_SEC)}s — check Gemini API key/network"
            ) from exc

        total_ms = int((time.perf_counter() - started) * 1000)
        try:
            parsed = self._extract_json_object(content)
            confidence, _ = parse_confidence_from_payload(parsed)
            tags = self._sanitize_tags(parsed.get("reason_codes"))
        except (ValueError, json.JSONDecodeError):
            tags = []
            confidence = 0

        self._warmed_up = True
        logger.info(
            "LLM warmup complete: model=%s warmup_latency_ms=%d inference_ms=%d tags=%s",
            self.model,
            total_ms,
            inference_ms,
            tags,
        )
        return {
            "llm_status": "warm",
            "llm_latency_ms": total_ms,
            "inference_ms": inference_ms,
            "warmed_up": True,
            "warmup_tags": tags,
            "model": self.model,
        }

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


_default_auditor: AuditAuditor | None = None


def get_auditor(
    model: str | None = None,
    use_mock: bool = False,
    mock_mode: MockMode | None = None,
    api_key: str | None = None,
    use_ollama: bool = False,
    ollama_model: str | None = None,
) -> AuditAuditor:
    """プロセス内シングルトン Auditor。"""
    global _default_auditor
    if _default_auditor is None:
        try:
            from optuna_runtime import is_optuna_runtime
        except ImportError:
            is_optuna_runtime = lambda: False  # noqa: E731

        resolved_mock = mock_mode or _MOCK_MODE
        resolved_ollama_model = ollama_model or resolve_ollama_model()

        if is_optuna_runtime():
            use_mock = False
            use_ollama = True
        elif use_mock or (is_backtest_mode() and not _live_api_explicitly_allowed()):
            use_mock = True
            if mock_mode is None and is_backtest_mode():
                resolved_mock = "strategy_edge"

        _default_auditor = AuditAuditor(
            model=model or resolve_gemini_model(),
            use_mock=use_mock,
            mock_mode=resolved_mock,
            use_ollama=use_ollama,
            ollama_model=resolved_ollama_model,
            api_key=api_key,
        )
    return _default_auditor


def is_gemini_configured() -> bool:
    try:
        resolve_gemini_api_key()
        return True
    except RuntimeError:
        return False


def warmup_trade_context() -> dict[str, Any]:
    return {
        "pair": "EURUSD",
        "setup_type": "London_Sweep_Reversal",
        "direction": "BUY",
        "timestamp": "2026-06-03T21:00:00Z",
        "smt_intensity": 5.0,
        "minutes_to_next_news": 999,
        "recent_losses": 0,
        "has_bos": True,
        "both_sweep": True,
        "atr_ratio": 1.0,
        "wick_ratio_pct": 50.0,
        "candidate_score": 75.0,
        "bayes_probability": 0.55,
    }


def sample_trade_context() -> dict[str, Any]:
    return {
        "pair": "EURUSD",
        "setup_type": "London_Sweep_Reversal",
        "direction": "BUY",
        "timestamp": "2026-06-03T21:00:00Z",
        "smt_intensity": 0.6,
        "minutes_to_next_news": 18,
        "recent_losses": 2,
        "has_bos": False,
        "both_sweep": False,
        "atr_ratio": 1.8,
        "wick_ratio_pct": 22.0,
        "candidate_score": 26.6,
        "bayes_probability": 0.42,
    }


def _print_audit_result(label: str, result: dict[str, Any]) -> None:
    print(f"\n=== {label} ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_self_test(live: bool = False) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    mock_auditor = AuditAuditor(use_mock=True)
    mock_result = mock_auditor.audit_trade(sample_trade_context())
    _print_audit_result("MOCK audit_trade", mock_result)

    async def _async_test() -> dict[str, Any]:
        auditor = AuditAuditor(use_mock=True)
        return await auditor.audit_trade_async(sample_trade_context())

    async_result = asyncio.run(_async_test())
    _print_audit_result("MOCK audit_trade_async", async_result)

    timeout_auditor = AuditAuditor(use_mock=False, timeout_sec=0.001, model=DEFAULT_MODEL)

    class _SlowClient:
        def models(self) -> Any:
            return self

        def generate_content(self, **kwargs: Any) -> Any:
            time.sleep(1.0)
            return type(
                "Resp",
                (),
                {
                    "text": (
                        '{"confidence_score":75,"vp_interpretation":"STANDARD_REJECTION",'
                        '"reason_summary":"slow mock"}'
                    )
                },
            )()

    timeout_auditor._client = _SlowClient()
    timeout_result = timeout_auditor.audit_trade(sample_trade_context())
    _print_audit_result("TIMEOUT fail-safe", timeout_result)
    assert timeout_result["reason_codes"] == ["LLM_TIMEOUT_FALLBACK"]
    assert timeout_result["risk_score"] == TIMEOUT_FALLBACK_RISK_SCORE

    if live:
        live_auditor = AuditAuditor(use_mock=False, model=DEFAULT_MODEL)
        try:
            warm = live_auditor.warmup()
            print(f"\n[LIVE] Warmup: {json.dumps(warm, ensure_ascii=False)}")
            live_result = live_auditor.audit_trade(sample_trade_context())
            _print_audit_result("LIVE Gemini audit_trade", live_result)
        except Exception as exc:
            print(f"\n[LIVE] Skipped or failed: {exc}")
        finally:
            live_auditor.close()
    else:
        print("\n[INFO] Live Gemini test skipped. Run: python llm_auditor.py --live")

    mock_auditor.close()
    print("\n[OK] llm_auditor self-test completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM risk auditor self-test (Gemini 1.5 Flash)")
    parser.add_argument("--live", action="store_true", help="Run live Gemini API test")
    args = parser.parse_args()
    run_self_test(live=args.live)

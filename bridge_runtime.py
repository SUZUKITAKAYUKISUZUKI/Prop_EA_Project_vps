"""
bridge_runtime.py — MT5 Bridge 統合ランタイム（Gemini / カレンダー / LLM監査）
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("bridge_runtime")

_calendar_service: Any = None
_llm_auditor_enabled = False
_runtime_started = False
_live_pyramid_registry: Any = None


def get_live_pyramid_registry():
    """アクティブな Live Pyramid セッション registry（lazy init）。"""
    global _live_pyramid_registry
    if _live_pyramid_registry is None:
        from live_pyramid.registry import LivePyramidRegistry

        _live_pyramid_registry = LivePyramidRegistry()
    return _live_pyramid_registry


def startup_bridge_runtime() -> dict[str, Any]:
    """Bridge 起動時: Gemini API + カレンダーキャッシュ + LLM監査を有効化。"""
    global _calendar_service, _llm_auditor_enabled, _runtime_started

    from calendar_service import CalendarBackgroundService, get_calendar_status
    from feature_engineering import configure_live_runtime
    from llm_auditor import LIVE_INFERENCE_TIMEOUT_SEC, get_auditor, is_gemini_configured, resolve_gemini_model

    summary: dict[str, Any] = {
        "gemini": "unconfigured",
        "calendar": "unavailable",
        "llm_auditor": "disabled",
    }

    # 1) 経済カレンダー（L1 環境認識）
    try:
        _calendar_service = CalendarBackgroundService()
        _calendar_service.refresh_once()
        _calendar_service.start()
        cal = get_calendar_status()
        summary["calendar"] = cal.get("calendar", "ready")
        summary["calendar_detail"] = cal.get("detail", "")
        print(f"[Prop EA] Calendar: {summary['calendar']}")
        logger.info("Calendar service started")
    except Exception as exc:
        logger.warning("Calendar service failed to start: %s", exc)
        summary["calendar"] = "unavailable"
        summary["calendar_detail"] = str(exc)
        print(f"[Prop EA] Calendar: unavailable ({exc})")

    # 2) Gemini API
    gemini_ok = is_gemini_configured()
    summary["gemini"] = "ready" if gemini_ok else "unconfigured"

    from strategies.dinapoli import configure_dinapoli_defense_env
    from strategies.dbbs_common import configure_dbbs_defense_env
    from strategies.smrs_production import configure_smrs_defense_env
    from strategies.vamr import configure_vamr_defense_env

    configure_dinapoli_defense_env()
    configure_dbbs_defense_env()
    configure_vamr_defense_env()
    configure_smrs_defense_env()

    configure_live_runtime(enable_llm=gemini_ok)
    _llm_auditor_enabled = gemini_ok

    if gemini_ok:
        try:
            auditor = get_auditor()
            warm = auditor.warmup()
            summary["llm_auditor"] = "enabled"
            summary["llm_warmup_ms"] = str(warm.get("llm_latency_ms", 0))
            summary["llm_timeout_sec"] = str(LIVE_INFERENCE_TIMEOUT_SEC)
            summary["llm_model"] = str(warm.get("model", resolve_gemini_model()))
            print(
                f"[Prop EA] LLM auditor: enabled (Gemini, warmup {warm.get('llm_latency_ms', 0)}ms, "
                f"timeout {LIVE_INFERENCE_TIMEOUT_SEC}s)"
            )
            logger.info("LLM auditor enabled with warmup: %s", warm)
        except Exception as exc:
            configure_live_runtime(enable_llm=False)
            _llm_auditor_enabled = False
            summary["llm_auditor"] = "disabled"
            summary["llm_auditor_detail"] = str(exc)
            print(f"[Prop EA] LLM auditor: disabled ({exc})")
            logger.warning("LLM auditor warmup failed: %s", exc)
    else:
        summary["llm_auditor"] = "disabled"
        summary["llm_auditor_detail"] = "GEMINI_API_KEY not set"
        print("[Prop EA] LLM auditor: disabled (GEMINI_API_KEY missing, using simulation)")

    from live_pyramid.config import live_pyramid_env_enabled
    from live_pyramid.l6_log import live_pyramid_log_path

    registry = get_live_pyramid_registry()
    summary["live_pyramid"] = "enabled" if live_pyramid_env_enabled() else "disabled"
    summary["live_pyramid_sessions"] = str(len(registry))
    summary["live_pyramid_log"] = str(live_pyramid_log_path())

    _runtime_started = True
    return summary


def shutdown_bridge_runtime() -> None:
    """Bridge 停止時: バックグラウンドサービスを終了。"""
    global _calendar_service, _runtime_started, _live_pyramid_registry

    from feature_engineering import configure_live_runtime

    if _calendar_service is not None:
        try:
            _calendar_service.stop()
        except Exception as exc:
            logger.warning("Calendar service stop error: %s", exc)
        _calendar_service = None

    if _live_pyramid_registry is not None:
        _live_pyramid_registry.reset()
        _live_pyramid_registry = None

    configure_live_runtime(enable_llm=False)
    _runtime_started = False
    logger.info("Bridge runtime shutdown complete")


def get_runtime_status() -> dict[str, str]:
    """/health 用の統合ステータス。"""
    from calendar_service import get_calendar_status
    from feature_engineering import USE_LLM_AUDITOR
    from llm_auditor import DEFAULT_MODEL, is_gemini_configured, resolve_gemini_model

    status: dict[str, str] = {
        "gemini": "ready" if is_gemini_configured() else "unconfigured",
        "llm_model": resolve_gemini_model(),
    }
    cal = get_calendar_status()
    status["calendar"] = cal.get("calendar", "unavailable")
    status["calendar_detail"] = cal.get("detail", "")
    if cal.get("next_event_minutes") is not None:
        status["next_news_minutes"] = str(cal.get("next_event_minutes"))
    status["llm_auditor"] = "enabled" if USE_LLM_AUDITOR else "disabled"
    status["runtime"] = "started" if _runtime_started else "stopped"
    from live_pyramid.config import live_pyramid_env_enabled

    status["live_pyramid"] = "enabled" if live_pyramid_env_enabled() else "disabled"
    status["live_pyramid_sessions"] = str(len(get_live_pyramid_registry()))
    from live_pyramid.l6_log import live_pyramid_log_path

    status["live_pyramid_log"] = str(live_pyramid_log_path())
    return status

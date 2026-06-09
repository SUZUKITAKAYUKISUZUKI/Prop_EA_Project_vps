"""
calendar_service.py — 経済カレンダー連携（ForexFactory JSON）

- 外部カレンダーを取得し cache/calendar.json へ書き出す（EA とは分離）
- feature_engineering / mt5_bridge はキャッシュを読み取り専用で参照
- AI 自己学習・パラメータ自動更新は行わない

起動:
    python calendar_service.py
    python calendar_service.py --once
"""

from __future__ import annotations

import argparse
import json
import logging
import ssl
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "cache"
CALENDAR_CACHE_PATH = CACHE_DIR / "calendar.json"

FOREXFACTORY_JSON_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
# ForexFactory: 同一 URL は 5 分あたり最大 2 回まで（超過時 429）
MIN_FETCH_INTERVAL_SEC = 150
DEFAULT_REFRESH_SECONDS = 900  # 15分
DEFAULT_MIN_IMPACT = "HIGH"  # L1 へ渡す最低重要度: HIGH | MEDIUM | LOW

IMPACT_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "HOLIDAY": 0}
IMPACT_ALIASES = {
    "LOW": "LOW",
    "MEDIUM": "MEDIUM",
    "MED": "MEDIUM",
    "HIGH": "HIGH",
    "HOLIDAY": "LOW",
    "": "LOW",
}

logger = logging.getLogger("calendar_service")

SSL_VPS_FIX_HINT = (
    "Run: py -3 -m pip install certifi  then  py -3 calendar_service.py --once  "
    "(or scripts\\fix_vps_calendar_ssl.bat)"
)


def _is_ssl_cert_error(exc: BaseException) -> bool:
    if isinstance(exc, ssl.SSLCertVerificationError):
        return True
    reason = getattr(exc, "reason", None)
    if isinstance(reason, ssl.SSLCertVerificationError):
        return True
    text = str(exc).lower()
    return "certificate verify failed" in text or "certificate_verify_failed" in text


def _https_ssl_context() -> ssl.SSLContext:
    """Windows VPS 等で CA バンドル不足のとき certifi を使う。"""
    import sys

    ca_candidates: list[str] = []
    try:
        import certifi

        ca_candidates.append(certifi.where())
    except ImportError:
        pass

    bundled = Path(sys.prefix) / "Lib" / "site-packages" / "certifi" / "cacert.pem"
    if bundled.exists():
        ca_candidates.append(str(bundled))

    seen: set[str] = set()
    for cafile in ca_candidates:
        if cafile in seen:
            continue
        seen.add(cafile)
        try:
            ctx = ssl.create_default_context(cafile=cafile)
            logger.debug("SSL context using CA bundle: %s", cafile)
            return ctx
        except ssl.SSLError:
            continue

    logger.warning(
        "certifi CA bundle unavailable; default SSL context may fail on Windows VPS. %s",
        SSL_VPS_FIX_HINT,
    )
    return ssl.create_default_context()


def _format_fetch_error(exc: Exception) -> str:
    reason = getattr(exc, "reason", None) or str(exc)
    if _is_ssl_cert_error(exc):
        return f"{reason}. {SSL_VPS_FIX_HINT}"
    return str(reason)


def _append_ssl_hint_if_needed(message: str) -> str:
    if SSL_VPS_FIX_HINT in message:
        return message
    text = message.lower()
    if "certificate verify failed" in text or "certificate_verify_failed" in text:
        return f"{message}. {SSL_VPS_FIX_HINT}"
    return message


class CalendarFetchError(Exception):
    """カレンダー取得失敗（キャッシュフォールバック可能）。"""


class CalendarRateLimitError(CalendarFetchError):
    """ForexFactory レート制限 (HTTP 429)。"""


@dataclass(frozen=True)
class CalendarEvent:
    title: str
    currency: str
    impact: str
    event_time_utc: datetime

    @property
    def event_time_ms(self) -> int:
        return int(self.event_time_utc.timestamp() * 1000)


def normalize_impact(raw: str) -> str:
    token = str(raw or "").strip().upper()
    return IMPACT_ALIASES.get(token, "LOW")


def impact_meets_minimum(impact: str, minimum: str) -> bool:
    return IMPACT_RANK.get(impact, 0) >= IMPACT_RANK.get(minimum, 3)


def parse_event_datetime(raw_date: str) -> datetime | None:
    """ForexFactory JSON の date フィールドを UTC aware datetime へ変換。"""
    if not raw_date:
        return None
    text = raw_date.strip()
    try:
        if text.endswith("Z"):
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%m-%d-%y", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(text, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def fetch_forexfactory_events(timeout: float = 20.0, retries: int = 2) -> list[dict[str, Any]]:
    """ForexFactory 互換 JSON を取得（読み取り専用・外部参照）。"""
    last_error: Exception | None = None
    url = FOREXFACTORY_JSON_URL
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "PropEA-CalendarService/1.0"},
            )
            with urllib.request.urlopen(
                request,
                timeout=timeout,
                context=_https_ssl_context(),
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, list):
                raise ValueError("Unexpected calendar payload format")
            return payload
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 429:
                raise CalendarRateLimitError(
                    "ForexFactory rate limit (HTTP 429); retry after a few minutes"
                ) from exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
        except (urllib.error.URLError, TimeoutError, ValueError, ssl.SSLCertVerificationError) as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
    if last_error:
        if _is_ssl_cert_error(last_error):
            raise CalendarFetchError(_format_fetch_error(last_error)) from last_error
        raise last_error
    return []


def parse_events(raw_events: list[dict[str, Any]]) -> list[CalendarEvent]:
    parsed: list[CalendarEvent] = []
    for item in raw_events:
        event_time = parse_event_datetime(str(item.get("date", "")))
        if event_time is None:
            continue
        parsed.append(
            CalendarEvent(
                title=str(item.get("title", "Unknown")),
                currency=str(item.get("country", "")).upper(),
                impact=normalize_impact(str(item.get("impact", ""))),
                event_time_utc=event_time,
            )
        )
    parsed.sort(key=lambda e: e.event_time_utc)
    return parsed


def compute_next_event(
    events: list[CalendarEvent],
    reference: datetime | None = None,
    min_impact: str = DEFAULT_MIN_IMPACT,
) -> dict[str, Any] | None:
    """reference 以降の次イベントをミリ秒/分単位で返す。"""
    ref = reference or datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    else:
        ref = ref.astimezone(timezone.utc)

    minimum = normalize_impact(min_impact)
    for event in events:
        if event.event_time_utc <= ref:
            continue
        if not impact_meets_minimum(event.impact, minimum):
            continue
        delta = event.event_time_utc - ref
        ms = int(delta.total_seconds() * 1000)
        minutes = max(0, ms // 60_000)
        return {
            "title": event.title,
            "currency": event.currency,
            "impact": event.impact,
            "event_time_utc": event.event_time_utc.isoformat(),
            "event_time_ms": event.event_time_ms,
            "milliseconds_to_event": ms,
            "minutes_to_event": int(minutes),
        }
    return None


def build_calendar_cache(
    min_impact: str = DEFAULT_MIN_IMPACT,
    reference: datetime | None = None,
) -> dict[str, Any]:
    raw = fetch_forexfactory_events()
    events = parse_events(raw)
    ref = reference or datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    else:
        ref = ref.astimezone(timezone.utc)

    minimum = normalize_impact(min_impact)
    tracked = [
        {
            "title": e.title,
            "currency": e.currency,
            "impact": e.impact,
            "event_time_utc": e.event_time_utc.isoformat(),
            "event_time_ms": e.event_time_ms,
        }
        for e in events
        if impact_meets_minimum(e.impact, minimum)
    ]
    next_event = compute_next_event(events, reference=ref, min_impact=minimum)
    now_ms = int(ref.timestamp() * 1000)

    return {
        "updated_at_utc": ref.isoformat(),
        "updated_at_ms": now_ms,
        "source": "forexfactory_json",
        "source_url": FOREXFACTORY_JSON_URL,
        "refresh_mode": "live",
        "min_impact_filter": minimum,
        "events_tracked": tracked,
        "next_event": next_event,
        "next_high_impact": next_event if next_event and next_event.get("impact") == "HIGH" else None,
    }


def write_calendar_cache(
    cache_path: Path = CALENDAR_CACHE_PATH,
    min_impact: str = DEFAULT_MIN_IMPACT,
) -> dict[str, Any]:
    payload = build_calendar_cache(min_impact=min_impact)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote calendar cache: %s (events=%d)", cache_path, len(payload["events_tracked"]))
    return payload


def load_calendar_cache(cache_path: Path = CALENDAR_CACHE_PATH) -> dict[str, Any] | None:
    """キャッシュ読み取り（ネットワークアクセスなし）。"""
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _cache_file_age_seconds(cache_path: Path = CALENDAR_CACHE_PATH) -> float | None:
    try:
        return time.time() - cache_path.stat().st_mtime
    except OSError:
        return None


def should_fetch_calendar(cache_path: Path = CALENDAR_CACHE_PATH) -> bool:
    """レート制限回避のため、直近取得から MIN_FETCH_INTERVAL_SEC 未満ならスキップ。"""
    age = _cache_file_age_seconds(cache_path)
    if age is None:
        return True
    return age >= MIN_FETCH_INTERVAL_SEC


def rehydrate_calendar_cache(
    cached: dict[str, Any],
    min_impact: str = DEFAULT_MIN_IMPACT,
    cache_path: Path = CALENDAR_CACHE_PATH,
    *,
    refresh_mode: str = "cache",
) -> dict[str, Any]:
    """ネットワークなしで next_event / 残り時間を再計算してキャッシュを更新。"""
    ref = datetime.now(timezone.utc)
    events = _events_from_cache(cached)
    minimum = normalize_impact(min_impact)
    next_event = compute_next_event(events, reference=ref, min_impact=minimum)
    payload = dict(cached)
    payload["updated_at_utc"] = ref.isoformat()
    payload["updated_at_ms"] = int(ref.timestamp() * 1000)
    payload["refresh_mode"] = refresh_mode
    payload["next_event"] = next_event
    payload["next_high_impact"] = (
        next_event if next_event and next_event.get("impact") == "HIGH" else None
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _use_cached_calendar(
    cache_path: Path,
    min_impact: str,
    reason: str,
) -> dict[str, Any]:
    cached = load_calendar_cache(cache_path)
    if not cached:
        raise CalendarFetchError(_append_ssl_hint_if_needed(reason))
    logger.info("Calendar: using cache (%s)", reason)
    return rehydrate_calendar_cache(
        cached,
        min_impact=min_impact,
        cache_path=cache_path,
        refresh_mode="cache",
    )


def get_minutes_to_next_news(
    reference: datetime | Any | None = None,
    cache_path: Path = CALENDAR_CACHE_PATH,
    min_impact: str = DEFAULT_MIN_IMPACT,
) -> tuple[int | None, str, int | None]:
    """
    キャッシュから L1 用の分数/ミリ秒を返す（読み取り専用）。

    Returns:
        (minutes, impact_level, milliseconds) — キャッシュ未命中時は (None, "", None)
    """
    cache = load_calendar_cache(cache_path)
    if not cache:
        return None, "", None

    ref = _to_utc_datetime(reference)
    next_event = cache.get("next_event")
    if not next_event:
        events = _events_from_cache(cache)
        next_event = compute_next_event(events, reference=ref, min_impact=min_impact)
        if not next_event:
            return None, "", None

    event_time = parse_event_datetime(str(next_event.get("event_time_utc", "")))
    if event_time is None:
        minutes = next_event.get("minutes_to_event")
        ms = next_event.get("milliseconds_to_event")
        if minutes is None:
            return None, "", None
        return int(minutes), str(next_event.get("impact", "")), int(ms or int(minutes) * 60_000)

    delta_ms = int((event_time - ref).total_seconds() * 1000)
    if delta_ms < 0:
        return None, "", None
    return max(0, delta_ms // 60_000), str(next_event.get("impact", "")), delta_ms


def _to_utc_datetime(reference: datetime | Any | None) -> datetime:
    if reference is None:
        return datetime.now(timezone.utc)
    if hasattr(reference, "to_pydatetime"):
        reference = reference.to_pydatetime()
    if isinstance(reference, datetime):
        if reference.tzinfo is None:
            return reference.replace(tzinfo=timezone.utc)
        return reference.astimezone(timezone.utc)
    raise TypeError("Unsupported reference time type")


def _events_from_cache(cache: dict[str, Any]) -> list[CalendarEvent]:
    events: list[CalendarEvent] = []
    for item in cache.get("events_tracked", []):
        event_time = parse_event_datetime(str(item.get("event_time_utc", "")))
        if event_time is None:
            continue
        events.append(
            CalendarEvent(
                title=str(item.get("title", "")),
                currency=str(item.get("currency", "")),
                impact=normalize_impact(str(item.get("impact", ""))),
                event_time_utc=event_time,
            )
        )
    events.sort(key=lambda e: e.event_time_utc)
    return events


# pandas は get_minutes_to_next_news の reference 変換用
try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]


def get_calendar_status() -> dict[str, Any]:
    """キャッシュ状態（/health 用）。"""
    cache = load_calendar_cache()
    if not cache:
        return {
            "calendar": "unavailable",
            "detail": "cache/calendar.json not found",
        }
    refresh_mode = str(cache.get("refresh_mode", "live"))
    age_sec = _cache_file_age_seconds()
    age_note = f", cache age {int(age_sec)}s" if age_sec is not None else ""
    next_event = cache.get("next_event")
    updated = cache.get("updated_at_utc", "")
    if not next_event:
        events = _events_from_cache(cache)
        next_event = compute_next_event(events, min_impact=str(cache.get("min_impact_filter", DEFAULT_MIN_IMPACT)))
        if not next_event:
            return {
                "calendar": "empty",
                "detail": f"cache loaded ({updated}) but no upcoming HIGH event{age_note}",
                "refresh_mode": refresh_mode,
            }
    mode_note = " (offline cache)" if refresh_mode == "cache" else ""
    return {
        "calendar": "ready",
        "detail": (
            f"next: {next_event.get('title', 'event')} in {next_event.get('minutes_to_event')} min"
            f"{mode_note}{age_note}"
        ),
        "next_event_minutes": next_event.get("minutes_to_event"),
        "next_event_impact": next_event.get("impact"),
        "refresh_mode": refresh_mode,
    }


class CalendarBackgroundService:
    """バックグラウンドで cache/calendar.json を定期更新。"""

    def __init__(
        self,
        refresh_seconds: int = DEFAULT_REFRESH_SECONDS,
        cache_path: Path = CALENDAR_CACHE_PATH,
        min_impact: str = DEFAULT_MIN_IMPACT,
    ) -> None:
        self.refresh_seconds = refresh_seconds
        self.cache_path = cache_path
        self.min_impact = min_impact
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def refresh_once(self) -> dict[str, Any]:
        if not should_fetch_calendar(self.cache_path):
            cached = load_calendar_cache(self.cache_path)
            if cached:
                logger.info(
                    "Calendar: skip fetch (rate-limit window %ss); reusing cache",
                    MIN_FETCH_INTERVAL_SEC,
                )
                return rehydrate_calendar_cache(
                    cached,
                    min_impact=self.min_impact,
                    cache_path=self.cache_path,
                    refresh_mode="cache",
                )

        try:
            return write_calendar_cache(self.cache_path, min_impact=self.min_impact)
        except CalendarRateLimitError:
            return _use_cached_calendar(
                self.cache_path,
                self.min_impact,
                "ForexFactory rate limit (HTTP 429)",
            )
        except CalendarFetchError:
            raise
        except (urllib.error.URLError, TimeoutError, ValueError, ssl.SSLCertVerificationError) as exc:
            return _use_cached_calendar(
                self.cache_path,
                self.min_impact,
                _format_fetch_error(exc),
            )

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            if self._stop.wait(self.refresh_seconds):
                break
            try:
                self.refresh_once()
            except CalendarFetchError as exc:
                logger.warning("Background calendar refresh skipped: %s", exc)
            except Exception as exc:  # noqa: BLE001 — バックグラウンド継続優先
                logger.exception("Background refresh error: %s", exc)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="calendar-service", daemon=True)
        self._thread.start()
        logger.info("Calendar background service started (interval=%ss)", self.refresh_seconds)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Prop EA calendar cache service")
    parser.add_argument("--once", action="store_true", help="Fetch once and exit")
    parser.add_argument("--interval", type=int, default=DEFAULT_REFRESH_SECONDS, help="Refresh seconds")
    parser.add_argument("--min-impact", default=DEFAULT_MIN_IMPACT, choices=["HIGH", "MEDIUM", "LOW"])
    args = parser.parse_args()

    service = CalendarBackgroundService(
        refresh_seconds=args.interval,
        min_impact=args.min_impact,
    )
    if args.once:
        try:
            payload = service.refresh_once()
        except CalendarFetchError as exc:
            print(f"Calendar refresh failed: {exc}")
            cached = load_calendar_cache()
            if cached:
                print("Using existing cache/calendar.json")
                payload = cached
            else:
                raise SystemExit(1) from exc
        except Exception as exc:
            print(f"Calendar refresh failed: {exc}")
            cached = load_calendar_cache()
            if cached:
                print("Using existing cache/calendar.json")
                payload = cached
            else:
                raise SystemExit(1) from exc
        next_event = payload.get("next_event") or {}
        print(
            f"Cache updated: next={next_event.get('title')} "
            f"in {next_event.get('minutes_to_event')} min "
            f"({next_event.get('milliseconds_to_event')} ms)"
        )
        return

    service.refresh_once()
    service.start()
    print(f"Calendar service running. Cache: {CALENDAR_CACHE_PATH}")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        service.stop()
        print("Stopped.")


if __name__ == "__main__":
    main()

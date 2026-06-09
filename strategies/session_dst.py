"""
strategies/session_dst.py — GMT 固定データ向け米国 DST セッションシフト共通ユーティリティ

PROP_DATA_DST_TYPE（`PROP_DATA_DST_TYPE=GMT_FIXED`）が設定され、
対象日が米国サマータイム期間の場合、セッション境界を 1 時間前倒し（-1h）する。
"""

from __future__ import annotations

import os
from datetime import date, timedelta

DATA_DST_TYPE = os.getenv("PROP_DATA_DST_TYPE", "NONE").strip().upper()


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """month 内の n 番目の weekday（0=Mon … 6=Sun）の日付。"""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def is_us_dst_date(session_date: date) -> bool:
    """
    米国サマータイム期間の簡易判定（3 月第 2 日曜 〜 11 月第 1 日曜、終了日未満）。

    GMT 固定データ向けのセッション 1 時間前倒しに使用する。
    """
    dst_start = _nth_weekday_of_month(session_date.year, 3, 6, 2)
    dst_end = _nth_weekday_of_month(session_date.year, 11, 6, 1)
    return dst_start <= session_date < dst_end


def hour_range(start: int, end: int) -> range:
    """inclusive start/end → Python range(end+1)。"""
    if start > end:
        return range(0)
    return range(start, end + 1)


def shift_hour(session_date: date, hour: int, dst_type: str = DATA_DST_TYPE) -> int:
    """単一時刻を DST 設定に応じて解決（負値は 0 にクランプ）。"""
    if dst_type == "GMT_FIXED" and is_us_dst_date(session_date):
        return max(0, hour - 1)
    return hour


def shift_hour_range(
    session_date: date,
    start: int,
    end_inclusive: int,
    dst_type: str = DATA_DST_TYPE,
) -> range:
    """inclusive 時刻帯を DST 設定に応じて解決。"""
    if dst_type == "GMT_FIXED" and is_us_dst_date(session_date):
        start -= 1
        end_inclusive -= 1
    return hour_range(max(0, start), max(0, end_inclusive))

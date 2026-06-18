"""Market regime detection for CACE v1.5."""
from __future__ import annotations

import math
from typing import Any

from src.database.db_manager import DatabaseManager
from src.repositories.base import create_default_db_manager


class RegimeDetector:
    DEFAULT_SYMBOL = "EURUSD"
    DEFAULT_TIMEFRAME = "H1"
    LOOKBACK = 120

    def __init__(self, db: DatabaseManager | None = None) -> None:
        self._db = db or create_default_db_manager()

    def detect(self, *, symbol: str | None = None, timeframe: str | None = None) -> dict[str, Any]:
        sym = symbol or self.DEFAULT_SYMBOL
        tf = timeframe or self.DEFAULT_TIMEFRAME
        candles = self._load_candles(sym, tf)
        if len(candles) < 20:
            return self._fallback_metrics()

        closes = [float(c["close"]) for c in candles if c.get("close") is not None]
        highs = [float(c["high"]) for c in candles if c.get("high") is not None]
        lows = [float(c["low"]) for c in candles if c.get("low") is not None]
        if len(closes) < 20:
            return self._fallback_metrics()

        returns = [
            (closes[i] - closes[i - 1]) / closes[i - 1] if closes[i - 1] else 0.0
            for i in range(1, len(closes))
        ]
        volatility = self._std(returns[-30:]) * math.sqrt(24 * 252) * 100
        atr = self._atr(highs, lows, closes, period=14)
        atr_pct = (atr / closes[-1] * 100.0) if closes[-1] else 0.0
        atr_percentile = min(100.0, max(0.0, atr_pct * 25.0))
        trend_strength = abs(closes[-1] - closes[0]) / max(closes[0], 1e-9) * 100.0
        range_score = self._range_score(highs[-30:], lows[-30:])

        return {
            "symbol": sym,
            "timeframe": tf,
            "trend_strength": round(trend_strength, 2),
            "volatility": round(volatility, 2),
            "atr_percentile": round(atr_percentile, 1),
            "range_score": round(range_score, 1),
            "sample_size": len(closes),
        }

    def _load_candles(self, symbol: str, timeframe: str) -> list[dict[str, Any]]:
        try:
            if self._db.market is None:
                return []
            rows = self._db.query(
                """
                SELECT open, high, low, close, volume, dt
                FROM candles
                WHERE symbol=? AND timeframe=?
                ORDER BY dt DESC
                LIMIT ?
                """,
                (symbol, timeframe, self.LOOKBACK),
                market=True,
            )
            return [dict(r) for r in reversed(rows)]
        except (RuntimeError, Exception):
            return []

    def _atr(self, highs: list[float], lows: list[float], closes: list[float], *, period: int) -> float:
        trs: list[float] = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs.append(tr)
        window = trs[-period:] if len(trs) >= period else trs
        return sum(window) / max(1, len(window))

    def _range_score(self, highs: list[float], lows: list[float]) -> float:
        if not highs or not lows:
            return 50.0
        span = max(highs) - min(lows)
        mid = (max(highs) + min(lows)) / 2.0
        if mid <= 0:
            return 50.0
        return min(100.0, span / mid * 100.0)

    def _std(self, values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        return math.sqrt(var)

    def _fallback_metrics(self) -> dict[str, Any]:
        return {
            "symbol": self.DEFAULT_SYMBOL,
            "timeframe": self.DEFAULT_TIMEFRAME,
            "trend_strength": 1.5,
            "volatility": 12.0,
            "atr_percentile": 50.0,
            "range_score": 45.0,
            "sample_size": 0,
            "fallback": True,
        }

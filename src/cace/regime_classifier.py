"""Regime classification for CACE v1.5."""
from __future__ import annotations

from typing import Any

from src.cace.confidence_v15_config import REGIME_MODIFIERS, REGIME_TYPES


class RegimeClassifier:
    def classify(self, metrics: dict[str, Any]) -> dict[str, Any]:
        trend = float(metrics.get("trend_strength") or 0)
        volatility = float(metrics.get("volatility") or 0)
        atr_pct = float(metrics.get("atr_percentile") or 50)
        range_score = float(metrics.get("range_score") or 50)

        regime = self._pick_regime(trend, volatility, atr_pct, range_score)
        modifier = float(REGIME_MODIFIERS.get(regime, 0.0))
        return {
            "regime": regime,
            "confidence_modifier": modifier,
            "metrics": {
                "trend_strength": trend,
                "volatility": volatility,
                "atr_percentile": atr_pct,
                "range_score": range_score,
            },
            "rationale": self._rationale(regime, metrics),
        }

    def _pick_regime(
        self,
        trend: float,
        volatility: float,
        atr_pct: float,
        range_score: float,
    ) -> str:
        if volatility >= 22 or atr_pct >= 75:
            return "HIGH_VOLATILITY"
        if volatility <= 8 and atr_pct <= 30:
            return "LOW_VOLATILITY"
        if trend >= 2.5 and range_score >= 40:
            return "TRENDING"
        if trend < 1.0 and range_score <= 35:
            return "RANGING"
        if 1.0 <= trend < 2.5 and 30 < range_score < 50:
            return "TRANSITIONAL"
        if trend >= 1.5:
            return "TRENDING"
        return "RANGING"

    def _rationale(self, regime: str, metrics: dict[str, Any]) -> str:
        if metrics.get("fallback"):
            return f"Regime classified as {regime} using fallback metrics (limited market data)."
        return (
            f"Regime {regime} from trend={metrics.get('trend_strength')}, "
            f"volatility={metrics.get('volatility')}, atr_pct={metrics.get('atr_percentile')}."
        )

    def all_regimes(self) -> tuple[str, ...]:
        return REGIME_TYPES

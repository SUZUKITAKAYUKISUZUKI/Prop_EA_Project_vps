"""Explainability for CACE confidence scores."""
from __future__ import annotations

from typing import Any

from src.cace.confidence_factors import ConfidenceFactors


class ConfidenceExplainability:
    def allocation_reasons(
        self,
        factors: ConfidenceFactors,
        *,
        paae: dict[str, Any],
        slm: dict[str, Any],
        pdts: dict[str, Any],
    ) -> list[str]:
        reasons: list[str] = []
        if factors.historical_reliability >= 70:
            reasons.append("Strong historical allocation reliability")
        if factors.monte_carlo_stability >= 70:
            reasons.append("Low Monte Carlo variance")
        if factors.forecast_stability >= 70:
            reasons.append("Strong forecast horizon agreement")
        if factors.portfolio_health >= 70:
            reasons.append("Healthy portfolio condition")
        if factors.lifecycle_quality >= 70:
            reasons.append("Core strategies dominate allocation")

        quality = paae.get("quality_scores") or {}
        if quality and min(quality.values()) >= 60:
            reasons.append("High PAAE quality scores across strategies")

        ranking = pdts.get("recommendation_ranking") or []
        if len(ranking) >= 2:
            top = float(ranking[0].get("score") or 0)
            second = float(ranking[1].get("score") or 0)
            if top - second >= 5:
                reasons.append("Clear PDTS scenario ranking lead")

        core_weight = self._core_allocation_weight(paae, slm)
        if core_weight >= 0.5:
            reasons.append("Core strategies dominate allocation")

        if factors.historical_reliability < 45:
            reasons.append("Limited allocation history for reliability estimate")
        if factors.monte_carlo_stability < 45:
            reasons.append("Elevated Monte Carlo outcome variance")
        if factors.forecast_stability < 45:
            reasons.append("Forecast horizons disagree materially")

        return reasons[:6] or ["Baseline confidence from available governance signals"]

    def strategy_reasons(
        self,
        *,
        strategy: str,
        confidence: float,
        portfolio_fit: float,
        lifecycle_stage: str,
    ) -> list[str]:
        reasons: list[str] = []
        if lifecycle_stage.upper() == "CORE":
            reasons.append("CORE lifecycle stage")
        if portfolio_fit >= 80:
            reasons.append("Strong portfolio fit")
        elif portfolio_fit < 50:
            reasons.append("Weak portfolio fit reduces confidence")
        if confidence >= 85:
            reasons.append("High composite strategy confidence")
        elif confidence < 55:
            reasons.append("Mixed governance signals for strategy")
        return reasons[:4] or [f"Strategy {strategy} confidence from lifecycle and risk signals"]

    def portfolio_reasons(
        self,
        factors: ConfidenceFactors,
        *,
        allocation_confidence: float,
        strategy_confidences: list[float],
    ) -> list[str]:
        reasons: list[str] = []
        if allocation_confidence >= 80:
            reasons.append("High allocation recommendation confidence")
        if factors.portfolio_health >= 70:
            reasons.append("Portfolio health supports allocation trust")
        if strategy_confidences:
            avg = sum(strategy_confidences) / len(strategy_confidences)
            if avg >= 75:
                reasons.append("Strategy-level confidence is consistently high")
            elif avg < 55:
                reasons.append("Strategy-level confidence is mixed")
        if factors.forecast_stability >= 75:
            reasons.append("AGE v4 futures agree across horizons")
        return reasons[:5] or ["Portfolio confidence aggregated from CACE components"]

    def top_risks(self, factors: ConfidenceFactors, *, age_v4: dict[str, Any]) -> list[str]:
        risks: list[str] = []
        if factors.monte_carlo_stability < 55:
            risks.append("Simulation outcome variance is elevated")
        if factors.forecast_stability < 55:
            risks.append("Multi-horizon forecasts diverge")
        if factors.portfolio_health < 55:
            risks.append("Portfolio health is below trust threshold")
        if float(age_v4.get("expected_risk") or 0) >= 15:
            risks.append("AGE v4 projects elevated governance risk")
        return risks[:4]

    def top_opportunities(self, factors: ConfidenceFactors, *, paae: dict[str, Any]) -> list[str]:
        opps: list[str] = []
        if factors.lifecycle_quality >= 75:
            opps.append("Strong SLM lifecycle quality supports allocation")
        underweight = paae.get("drift_alerts") or []
        increases = [a for a in underweight if a.get("direction") == "increase"]
        if increases:
            opps.append(f"Underweight opportunity: {increases[0].get('strategy')}")
        if factors.historical_reliability >= 80:
            opps.append("Historical allocation changes have been reliable")
        return opps[:4]

    def _core_allocation_weight(self, paae: dict[str, Any], slm: dict[str, Any]) -> float:
        weights = paae.get("recommended_weights") or {}
        if not weights:
            return 0.0
        core_codes = {
            str(row.get("strategy"))
            for row in slm.get("strategies") or []
            if str(row.get("stage") or "").upper() == "CORE"
        }
        return sum(float(weights.get(code, 0)) for code in core_codes)

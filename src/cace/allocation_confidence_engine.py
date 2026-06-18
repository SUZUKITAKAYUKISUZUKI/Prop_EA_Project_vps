"""Per-allocation confidence evaluation for CACE."""
from __future__ import annotations

from typing import Any

from src.cace.confidence_calculator import ConfidenceCalculator
from src.cace.confidence_explainability import ConfidenceExplainability
from src.cace.confidence_factors import ConfidenceFactors
from src.cace.forecast_confidence_engine import ForecastConfidenceEngine
from src.cace.historical_reliability_engine import HistoricalReliabilityEngine
from src.cace.monte_carlo_confidence_engine import MonteCarloConfidenceEngine


class AllocationConfidenceEngine:
    def __init__(self) -> None:
        self._historical = HistoricalReliabilityEngine()
        self._monte_carlo = MonteCarloConfidenceEngine()
        self._forecast = ForecastConfidenceEngine()
        self._calculator = ConfidenceCalculator()
        self._explain = ConfidenceExplainability()

    def evaluate(
        self,
        *,
        allocation_id: str,
        profile_id: str,
        weights: dict[str, float],
        paae: dict[str, Any],
        pdts: dict[str, Any],
        prae_v2: dict[str, Any],
        state_analytics: dict[str, Any],
        slm: dict[str, Any],
        age_v4: dict[str, Any],
        age_v3: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        recommended = paae.get("recommended_weights") or weights
        factors = ConfidenceFactors(
            historical_reliability=self._historical.score(
                profile_id=profile_id,
                paae=paae,
                recommended_weights=recommended,
            ),
            monte_carlo_stability=self._monte_carlo.score(pdts=pdts),
            forecast_stability=self._forecast.score(age_v4=age_v4, age_v3=age_v3),
            portfolio_health=self._portfolio_health(state_analytics, prae_v2),
            lifecycle_quality=self._lifecycle_quality(slm, recommended),
        )
        confidence = self._calculator.compute(factors)
        category = self._calculator.category(confidence)
        metrics = self._expected_metrics(pdts, age_v4)

        return {
            "allocation_id": allocation_id,
            "profile_id": profile_id,
            "allocation_json": {k: round(float(v) * 100.0, 1) for k, v in recommended.items()},
            "confidence": confidence,
            "category": category,
            "expected_r": metrics["expected_r"],
            "expected_pf": metrics["expected_pf"],
            "expected_dd": metrics["expected_dd"],
            "factors": factors.to_dict(),
            "reason": self._explain.allocation_reasons(factors, paae=paae, slm=slm, pdts=pdts),
        }

    def _portfolio_health(self, state_analytics: dict[str, Any], prae_v2: dict[str, Any]) -> float:
        health = float(
            state_analytics.get("health_score")
            or (prae_v2.get("health_report") or {}).get("health_score")
            or 50.0
        )
        stability = float(state_analytics.get("funded_stability_score") or health)
        recovery_penalty = int(state_analytics.get("recovery_events") or 0) * 5
        return max(0.0, min(100.0, (health * 0.6 + stability * 0.4) - recovery_penalty))

    def _lifecycle_quality(self, slm: dict[str, Any], weights: dict[str, float]) -> float:
        strategies = slm.get("strategies") or []
        if not strategies:
            return 50.0
        weighted = 0.0
        total_w = 0.0
        for row in strategies:
            code = str(row.get("strategy") or "")
            w = float(weights.get(code, 0))
            score = float(row.get("score") or 50)
            fit = float(row.get("portfolio_fit_score") or 50)
            stage = str(row.get("stage") or "").upper()
            stage_bonus = {"CORE": 15, "PRODUCTION": 8, "INCUBATION": 0, "RECOVERY": -5}.get(stage, 0)
            weighted += w * (score * 0.5 + fit * 0.5 + stage_bonus)
            total_w += w
        if total_w <= 0:
            scores = [float(r.get("score") or 50) for r in strategies]
            return sum(scores) / len(scores)
        return max(0.0, min(100.0, weighted / total_w))

    def _expected_metrics(self, pdts: dict[str, Any], age_v4: dict[str, Any]) -> dict[str, float]:
        cmp = pdts.get("scenario_comparison") or {}
        recommended = cmp.get("recommended") or {}
        best = age_v4.get("best_future_metrics") or {}
        return {
            "expected_r": round(
                float(best.get("expected_r") or recommended.get("expected_r") or 150.0),
                1,
            ),
            "expected_pf": round(
                float(best.get("expected_pf") or recommended.get("expected_pf") or 2.5),
                2,
            ),
            "expected_dd": round(
                float(best.get("expected_dd") or recommended.get("expected_dd") or 6.0),
                1,
            ),
        }

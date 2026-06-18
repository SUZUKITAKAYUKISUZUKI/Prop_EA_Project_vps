"""Per-strategy confidence evaluation for CACE."""
from __future__ import annotations

from typing import Any

from src.cace.confidence_calculator import ConfidenceCalculator
from src.cace.confidence_explainability import ConfidenceExplainability
from src.cace.confidence_factors import ConfidenceFactors
from src.cace.confidence_normalizer import ConfidenceNormalizer


class StrategyConfidenceEngine:
    def __init__(self) -> None:
        self._calculator = ConfidenceCalculator()
        self._explain = ConfidenceExplainability()
        self._normalizer = ConfidenceNormalizer()

    def evaluate_all(
        self,
        *,
        paae: dict[str, Any],
        prae_v2: dict[str, Any],
        slm: dict[str, Any],
        state_analytics: dict[str, Any],
        portfolio_factors: ConfidenceFactors,
    ) -> list[dict[str, Any]]:
        strategies = slm.get("strategies") or []
        quality = paae.get("quality_scores") or {}
        risk_rows = {str(r.get("strategy")): r for r in prae_v2.get("strategy_risk") or []}
        results: list[dict[str, Any]] = []

        for row in strategies:
            code = str(row.get("strategy") or "")
            if not code:
                continue
            results.append(
                self.evaluate_one(
                    strategy=code,
                    row=row,
                    quality_score=float(quality.get(code, 50.0)),
                    risk_row=risk_rows.get(code) or {},
                    state_analytics=state_analytics,
                    portfolio_factors=portfolio_factors,
                )
            )
        results.sort(key=lambda r: float(r.get("confidence") or 0), reverse=True)
        return results

    def evaluate_one(
        self,
        *,
        strategy: str,
        row: dict[str, Any],
        quality_score: float,
        risk_row: dict[str, Any],
        state_analytics: dict[str, Any],
        portfolio_factors: ConfidenceFactors,
    ) -> dict[str, Any]:
        portfolio_fit = float(row.get("portfolio_fit_score") or 50.0)
        lifecycle_stage = str(row.get("stage") or "INCUBATION")
        slm_score = float(row.get("score") or 50.0)
        risk_score = float(risk_row.get("risk_score") or 50.0)

        factors = ConfidenceFactors(
            historical_reliability=portfolio_factors.historical_reliability,
            monte_carlo_stability=portfolio_factors.monte_carlo_stability,
            forecast_stability=portfolio_factors.forecast_stability,
            portfolio_health=float(state_analytics.get("health_score") or portfolio_factors.portfolio_health),
            lifecycle_quality=self._normalizer.clamp(slm_score * 0.5 + portfolio_fit * 0.5),
        )
        strategy_adjust = self._strategy_adjustment(lifecycle_stage, quality_score, risk_score)
        confidence = self._normalizer.clamp(self._calculator.compute(factors) * 0.85 + strategy_adjust * 0.15)
        category = self._calculator.category(confidence)

        return {
            "strategy": strategy,
            "confidence": confidence,
            "category": category,
            "portfolio_fit": round(portfolio_fit, 1),
            "lifecycle_stage": lifecycle_stage,
            "slm_score": round(slm_score, 1),
            "quality_score": round(quality_score, 1),
            "reason": self._explain.strategy_reasons(
                strategy=strategy,
                confidence=confidence,
                portfolio_fit=portfolio_fit,
                lifecycle_stage=lifecycle_stage,
            ),
        }

    def _strategy_adjustment(self, stage: str, quality: float, risk: float) -> float:
        stage_bonus = {"CORE": 12, "PRODUCTION": 6, "INCUBATION": 0, "RECOVERY": -8, "RETIRED": -20}.get(
            stage.upper(), 0
        )
        return max(0.0, min(100.0, quality * 0.5 + (100.0 - risk) * 0.3 + stage_bonus + 10))

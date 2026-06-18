"""Executive learning from decision outcomes."""
from __future__ import annotations

from typing import Any


class ExecutiveLearningEngine:
    def evaluate(
        self,
        *,
        outcomes: list[dict[str, Any]],
        effectiveness: dict[str, Any],
        apm_v1_report: dict[str, Any],
    ) -> dict[str, Any]:
        lessons = self._build_lessons(outcomes, effectiveness)
        learning_quality = self._learning_quality(outcomes, lessons)
        portfolio_improvement = self._portfolio_improvement(outcomes, apm_v1_report)

        return {
            "learning_quality": learning_quality,
            "portfolio_improvement": portfolio_improvement,
            "lessons": lessons,
            "lesson_count": len(lessons),
            "improvement_opportunities": self._improvement_opportunities(effectiveness),
        }

    def _build_lessons(self, outcomes: list[dict[str, Any]], effectiveness: dict[str, Any]) -> list[dict[str, Any]]:
        lessons: list[dict[str, Any]] = []
        failures = [o for o in outcomes if o.get("outcome_class") == "FAILURE"]
        successes = [o for o in outcomes if o.get("outcome_class") == "SUCCESS"]

        for outcome in failures:
            if "PROMOTE" in str(outcome.get("decision_type", "")).upper():
                lessons.append(
                    {
                        "source_module": "SLM",
                        "lesson_type": "promotion_caution",
                        "description": (
                            f"Promoting {outcome.get('strategy') or 'strategy'} below expected benefit "
                            f"increased recovery probability."
                        ),
                        "impact_score": round(100.0 - float(outcome.get("success_score") or 0), 2),
                        "confidence": 88.0,
                    }
                )

        promo_acc = float(effectiveness.get("effectiveness_by_category", {}).get("promotion_accuracy") or 0)
        if promo_acc and promo_acc < 65:
            lessons.append(
                {
                    "source_module": "SLM",
                    "lesson_type": "portfolio_fit_threshold",
                    "description": "Promoting strategies below PortfolioFit 70 increased recovery probability.",
                    "impact_score": 85.0,
                    "confidence": 93.0,
                }
            )

        if successes:
            best = max(successes, key=lambda o: float(o.get("success_score") or 0))
            lessons.append(
                {
                    "source_module": "APM",
                    "lesson_type": "best_practice",
                    "description": (
                        f"{best.get('decision_type')} decisions with confidence "
                        f"{best.get('confidence')} produced reliable outcomes."
                    ),
                    "impact_score": float(best.get("success_score") or 80),
                    "confidence": 80.0,
                }
            )

        if not lessons:
            lessons.append(
                {
                    "source_module": "PORTFOLIO_OS",
                    "lesson_type": "baseline",
                    "description": "Insufficient evaluated decisions — continue logging executive outcomes.",
                    "impact_score": 50.0,
                    "confidence": 60.0,
                }
            )
        return lessons

    def _learning_quality(self, outcomes: list[dict[str, Any]], lessons: list[dict[str, Any]]) -> float:
        if not outcomes:
            return 55.0
        avg_success = sum(float(o.get("success_score") or 0) for o in outcomes) / len(outcomes)
        lesson_bonus = min(20.0, len(lessons) * 4.0)
        return round(min(100.0, avg_success * 0.7 + lesson_bonus + 10.0), 2)

    def _portfolio_improvement(self, outcomes: list[dict[str, Any]], apm_v1_report: dict[str, Any]) -> float:
        base = float(apm_v1_report.get("executive_health") or 70)
        if not outcomes:
            return base
        delta = sum(float(o.get("actual_benefit") or 0) - float(o.get("predicted_benefit") or 0) for o in outcomes)
        return round(max(0.0, min(100.0, base + delta * 2.0)), 2)

    def _improvement_opportunities(self, effectiveness: dict[str, Any]) -> list[str]:
        ops: list[str] = []
        for key, value in (effectiveness.get("effectiveness_by_category") or {}).items():
            if value and value < 65:
                ops.append(f"Improve {key.replace('_', ' ')} — currently {value:.1f}")
        return ops or ["Maintain current executive governance discipline."]

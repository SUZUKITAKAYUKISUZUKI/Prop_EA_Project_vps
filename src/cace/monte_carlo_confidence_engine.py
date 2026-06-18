"""Monte Carlo stability confidence for CACE."""
from __future__ import annotations

from typing import Any

from src.cace.confidence_normalizer import ConfidenceNormalizer


class MonteCarloConfidenceEngine:
    def __init__(self, normalizer: ConfidenceNormalizer | None = None) -> None:
        self._normalizer = normalizer or ConfidenceNormalizer()

    def score(self, *, pdts: dict[str, Any]) -> float:
        simulation = pdts.get("simulation") or {}
        scenarios = pdts.get("scenario_results") or pdts.get("recommendation_ranking") or []

        mc_scores: list[float] = []
        for key in ("expected_r", "expected_pf", "pass_rate", "score"):
            value = simulation.get(key)
            if value is not None:
                mc_scores.append(float(value))

        for row in scenarios[:8]:
            for key in ("expected_r", "expected_pf", "pass_rate", "score"):
                if row.get(key) is not None:
                    mc_scores.append(float(row[key]))

        if len(mc_scores) < 2 and scenarios:
            mc_scores = [float(s.get("score") or 0) for s in scenarios if s.get("score") is not None]

        if len(mc_scores) < 2:
            cmp = pdts.get("scenario_comparison") or {}
            current = float((cmp.get("current") or {}).get("score") or 70)
            recommended = float((cmp.get("recommended") or {}).get("score") or 75)
            spread = abs(recommended - current)
            return self._normalizer.clamp(85.0 - spread)

        return self._normalizer.normalize_variance(mc_scores)

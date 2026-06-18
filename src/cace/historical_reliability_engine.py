"""Historical allocation reliability for CACE."""
from __future__ import annotations

from typing import Any

from src.adaptive_allocation.allocation_history import AllocationHistoryRepository
from src.cace.confidence_normalizer import ConfidenceNormalizer


class HistoricalReliabilityEngine:
    def __init__(self, normalizer: ConfidenceNormalizer | None = None) -> None:
        self._normalizer = normalizer or ConfidenceNormalizer()

    def score(
        self,
        *,
        profile_id: str,
        paae: dict[str, Any],
        recommended_weights: dict[str, float],
    ) -> float:
        repo = AllocationHistoryRepository(owns_connection=False)
        try:
            history = repo.recent_changes(profile_id=profile_id, limit=40)
        finally:
            repo.close()

        if not history:
            quality = paae.get("quality_scores") or {}
            if quality:
                return self._normalizer.clamp(sum(quality.values()) / len(quality))
            return 55.0

        positive = 0
        evaluated = 0
        for row in history:
            old_w = float(row.get("old_weight") or 0)
            new_w = float(row.get("new_weight") or 0)
            health = row.get("health_score")
            if health is None:
                continue
            evaluated += 1
            if float(health) >= 60 and abs(new_w - old_w) <= 0.15:
                positive += 1

        if evaluated == 0:
            rec_keys = set(recommended_weights)
            hist_keys = {str(r.get("strategy")) for r in history}
            overlap = len(rec_keys & hist_keys) / max(1, len(rec_keys))
            return self._normalizer.clamp(50.0 + overlap * 35.0)

        reliability = (positive / evaluated) * 100.0
        recency_bonus = min(10.0, len(history) / 4.0)
        return self._normalizer.clamp(reliability + recency_bonus)

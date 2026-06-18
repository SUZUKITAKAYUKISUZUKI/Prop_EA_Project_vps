"""Regime-aware confidence adjustment for CACE v1.5."""
from __future__ import annotations

from typing import Any

from src.cace.confidence_normalizer import ConfidenceNormalizer
from src.cace.regime_classifier import RegimeClassifier
from src.cace.regime_detector import RegimeDetector
from src.cace.regime_repository import RegimeRepository


class ConfidenceRegimeEngine:
    def __init__(
        self,
        *,
        detector: RegimeDetector | None = None,
        classifier: RegimeClassifier | None = None,
        repo: RegimeRepository | None = None,
        normalizer: ConfidenceNormalizer | None = None,
    ) -> None:
        self._detector = detector or RegimeDetector()
        self._classifier = classifier or RegimeClassifier()
        self._repo = repo if repo is not None else RegimeRepository()
        self._normalizer = normalizer or ConfidenceNormalizer()

    def evaluate(
        self,
        *,
        profile_id: str,
        raw_confidence: float,
        persist: bool = False,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        if use_cache and not persist:
            cached = self._repo.get_cached_regime(profile_id) if self._repo else None
            if cached and "raw_confidence" in cached and cached.get("raw_confidence") == raw_confidence:
                return cached

        metrics = self._detector.detect()
        classified = self._classifier.classify(metrics)
        modifier = float(classified.get("confidence_modifier") or 0)
        adjusted = self._normalizer.clamp(raw_confidence + modifier)

        payload = {
            "regime": classified.get("regime"),
            "confidence_modifier": modifier,
            "raw_confidence": round(raw_confidence, 1),
            "adjusted_confidence": adjusted,
            "metrics": classified.get("metrics"),
            "rationale": classified.get("rationale"),
            "regime_appropriate": self._regime_appropriate(raw_confidence, classified.get("regime"), modifier),
        }
        if persist and self._repo is not None:
            self._repo.save_regime_snapshot(profile_id=profile_id, regime_payload=payload)
        elif use_cache and self._repo is not None:
            self._repo._cache.set(f"cace:v15:regime:{profile_id}", payload)
        return payload

    def _regime_appropriate(self, raw: float, regime: str | None, modifier: float) -> bool:
        if raw >= 80 and regime == "HIGH_VOLATILITY" and modifier < 0:
            return False
        if raw < 55 and regime == "LOW_VOLATILITY" and modifier > 0:
            return True
        return True

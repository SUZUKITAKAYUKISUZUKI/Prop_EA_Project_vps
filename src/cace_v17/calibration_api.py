"""CACE v1.7 calibration orchestration — evaluation only."""
from __future__ import annotations

from typing import Any

from src.cace.confidence_cache import ConfidenceCache
from src.cace_v17.calibration_config import (
    CACHE_V17_ACCURACY,
    CACHE_V17_CALIBRATION,
    CACHE_V17_INTELLIGENCE,
    CACHE_V17_LEARNING,
    CACHE_V17_RELIABILITY,
    MODULE_KEYS,
)
from src.cace_v17.calibration_report import CalibrationReport
from src.cace_v17.calibration_repository import CalibrationRepository
from src.cace_v17.confidence_backtest_engine import ConfidenceBacktestEngine
from src.cace_v17.confidence_calibration_engine import ConfidenceCalibrationEngine
from src.cace_v17.confidence_error_engine import ConfidenceErrorEngine
from src.cace_v17.confidence_learning_engine import ConfidenceLearningEngine
from src.cace_v17.confidence_reliability_engine import ConfidenceReliabilityEngine
from src.cace_v17.decision_accuracy_engine import DecisionAccuracyEngine
from src.cace_v17.recommendation_accuracy_engine import RecommendationAccuracyEngine


class CalibrationEngine:
    """Aggregates calibration, accuracy, reliability, and learning intelligence."""

    def __init__(
        self,
        *,
        backtest_engine: ConfidenceBacktestEngine | None = None,
        error_engine: ConfidenceErrorEngine | None = None,
        calibration_engine: ConfidenceCalibrationEngine | None = None,
        decision_accuracy_engine: DecisionAccuracyEngine | None = None,
        recommendation_accuracy_engine: RecommendationAccuracyEngine | None = None,
        reliability_engine: ConfidenceReliabilityEngine | None = None,
        learning_engine: ConfidenceLearningEngine | None = None,
        repo: CalibrationRepository | None = None,
        reporter: CalibrationReport | None = None,
        cache: ConfidenceCache | None = None,
        owns_connections: bool = False,
    ) -> None:
        self._owns = owns_connections
        self._backtest = backtest_engine or ConfidenceBacktestEngine()
        self._error = error_engine or ConfidenceErrorEngine()
        self._calibration = calibration_engine or ConfidenceCalibrationEngine()
        self._decision_accuracy = decision_accuracy_engine or DecisionAccuracyEngine()
        self._recommendation_accuracy = recommendation_accuracy_engine or RecommendationAccuracyEngine()
        self._reliability = reliability_engine or ConfidenceReliabilityEngine()
        self._learning = learning_engine or ConfidenceLearningEngine()
        self._repo = repo or CalibrationRepository(owns_connection=self._owns)
        self._reporter = reporter or CalibrationReport()
        self._cache = cache or ConfidenceCache()
        self._last_report: dict[str, Any] | None = None

    def close(self) -> None:
        if self._owns:
            self._repo.close()

    def run(
        self,
        *,
        profile_id: str,
        cace_v16_report: dict[str, Any],
        upstream: dict[str, Any] | None = None,
        historical_decisions: list[dict[str, Any]] | None = None,
        outcome_overrides: dict[str, dict[str, Any]] | None = None,
        persist: bool = True,
        use_cache: bool = True,
        capture_new_decision: bool = True,
    ) -> dict[str, Any]:
        cache_key = CACHE_V17_INTELLIGENCE.format(profile_id=profile_id)
        if use_cache and not persist:
            cached = self._cache.get_if_fresh(cache_key)
            if cached:
                self._last_report = cached
                return cached

        upstream = upstream or {}
        records = list(historical_decisions or self._repo.load_decisions(profile_id=profile_id))
        records = self._backtest.apply_outcomes(records, outcome_overrides=outcome_overrides)

        if capture_new_decision and persist:
            new_decision = self._backtest.capture_current_decision(
                profile_id=profile_id,
                cace_v16_report=cace_v16_report,
                upstream=upstream,
            )
            records.insert(0, new_decision)
            self._repo.save_decision(new_decision)
            for module_decision in self._capture_module_decisions(profile_id, upstream, cace_v16_report):
                records.insert(0, module_decision)
                self._repo.save_decision(module_decision)

        evaluated: list[dict[str, Any]] = []
        pending: list[dict[str, Any]] = []
        for record in records:
            item = dict(record)
            if item.get("evaluated") and item.get("actual_benefit") is not None:
                enriched = self._error.evaluate_record(item)
                enriched["accuracy_score"] = enriched.get("confidence_accuracy")
                if persist and item.get("decision_id"):
                    self._repo.update_decision(enriched)
                evaluated.append(enriched)
            else:
                pending.append(item)

        calibration = self._calibration.evaluate(evaluated)
        decision_accuracy = self._decision_accuracy.evaluate(evaluated)
        recommendation_accuracy = self._recommendation_accuracy.evaluate(evaluated)
        reliability = self._reliability.evaluate(evaluated)
        learning = self._learning.evaluate(
            calibration=calibration,
            decision_accuracy=decision_accuracy,
            recommendation_accuracy=recommendation_accuracy,
            reliability=reliability,
        )

        report = self._reporter.build(
            profile_id=profile_id,
            cace_v16_report=cace_v16_report,
            calibration=calibration,
            decision_accuracy=decision_accuracy,
            recommendation_accuracy=recommendation_accuracy,
            reliability=reliability,
            learning=learning,
            evaluated_records=evaluated,
            pending_decisions=pending,
        )

        if persist:
            self._repo.save_calibration_snapshot(
                profile_id=profile_id,
                calibration_score=float(calibration.get("calibration_score") or 0),
                calibration_category=str(calibration.get("calibration_category") or "POOR"),
                decision_accuracy_score=float(decision_accuracy.get("decision_accuracy_score") or 0),
                reliability_trend=str(reliability.get("reliability_trend") or "STABLE"),
                payload=report,
            )
            self._repo.save_learning_notes(
                profile_id=profile_id,
                notes=list(learning.get("confidence_learning_notes") or []),
            )

        if use_cache or persist:
            self._cache.set(cache_key, report)
            self._cache.set(CACHE_V17_CALIBRATION.format(profile_id=profile_id), calibration)
            self._cache.set(CACHE_V17_ACCURACY.format(profile_id=profile_id), decision_accuracy)
            self._cache.set(CACHE_V17_RELIABILITY.format(profile_id=profile_id), reliability)
            self._cache.set(CACHE_V17_LEARNING.format(profile_id=profile_id), learning)

        self._last_report = report
        return report

    def _capture_module_decisions(
        self,
        profile_id: str,
        upstream: dict[str, Any],
        cace_v16_report: dict[str, Any],
    ) -> list[dict[str, Any]]:
        decisions: list[dict[str, Any]] = []
        base = self._backtest.capture_current_decision(
            profile_id=profile_id,
            cace_v16_report=cace_v16_report,
            upstream=upstream,
        )
        module_confidence = {
            "PAAE": 82.0,
            "PDTS": 88.0,
            "SLM": 76.0,
            "AGE": float((upstream.get("age_v4") or {}).get("strategic_confidence") or 78),
            "CACE": float(cace_v16_report.get("confidence") or 90),
        }
        for module in MODULE_KEYS:
            item = dict(base)
            item["decision_id"] = f"{base['decision_id']}-{module}"
            item["module"] = module
            item["confidence"] = module_confidence.get(module, 70.0)
            decisions.append(item)
        return decisions

    def get_last_report(self) -> dict[str, Any]:
        return self._last_report or {}


def extract_calibration(report: dict[str, Any]) -> dict[str, Any]:
    return dict(report.get("calibration") or {})


def extract_decision_accuracy(report: dict[str, Any]) -> dict[str, Any]:
    return dict(report.get("decision_accuracy") or {})


def extract_reliability(report: dict[str, Any]) -> dict[str, Any]:
    return dict(report.get("reliability") or {})


def extract_learning(report: dict[str, Any]) -> dict[str, Any]:
    return dict(report.get("learning") or {})


def build_engine(*, owns_connections: bool = False) -> CalibrationEngine:
    return CalibrationEngine(owns_connections=owns_connections)

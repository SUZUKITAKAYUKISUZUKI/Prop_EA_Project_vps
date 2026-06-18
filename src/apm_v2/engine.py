"""APM v2 executive learning orchestration — governance outcomes only."""
from __future__ import annotations

from typing import Any

from src.apm_v2.board_report import BoardReport
from src.apm_v2.config import (
    CACHE_V2_BOARD,
    CACHE_V2_INTELLIGENCE,
    CACHE_V2_LESSONS,
    CACHE_V2_MEMORY,
    CACHE_V2_SCORE,
)
from src.apm_v2.decision_effectiveness_engine import DecisionEffectivenessEngine
from src.apm_v2.decision_outcome_engine import DecisionOutcomeEngine
from src.apm_v2.executive_board_engine import ExecutiveBoardEngine
from src.apm_v2.executive_learning_engine import ExecutiveLearningEngine
from src.apm_v2.executive_memory_engine import ExecutiveMemoryEngine
from src.apm_v2.executive_score_engine import ExecutiveScoreEngine
from src.apm_v2.memory_snapshot_engine import MemorySnapshotEngine
from src.apm_v2.portfolio_memory_repository import PortfolioMemoryRepository
from src.apm_v2.recommendation_feedback_engine import RecommendationFeedbackEngine
from src.apm_v2.roadmap_learning_engine import RoadmapLearningEngine
from src.cace.confidence_cache import ConfidenceCache


class ApmV2Engine:
    """Portfolio memory and executive board — never trades."""

    def __init__(
        self,
        *,
        outcome_engine: DecisionOutcomeEngine | None = None,
        effectiveness_engine: DecisionEffectivenessEngine | None = None,
        board_engine: ExecutiveBoardEngine | None = None,
        learning_engine: ExecutiveLearningEngine | None = None,
        memory_engine: ExecutiveMemoryEngine | None = None,
        score_engine: ExecutiveScoreEngine | None = None,
        snapshot_engine: MemorySnapshotEngine | None = None,
        feedback_engine: RecommendationFeedbackEngine | None = None,
        roadmap_engine: RoadmapLearningEngine | None = None,
        repo: PortfolioMemoryRepository | None = None,
        reporter: BoardReport | None = None,
        cache: ConfidenceCache | None = None,
        owns_connections: bool = False,
    ) -> None:
        self._owns = owns_connections
        self._outcomes = outcome_engine or DecisionOutcomeEngine()
        self._effectiveness = effectiveness_engine or DecisionEffectivenessEngine()
        self._board = board_engine or ExecutiveBoardEngine()
        self._learning = learning_engine or ExecutiveLearningEngine()
        self._memory = memory_engine or ExecutiveMemoryEngine()
        self._score = score_engine or ExecutiveScoreEngine()
        self._snapshot = snapshot_engine or MemorySnapshotEngine()
        self._feedback = feedback_engine or RecommendationFeedbackEngine()
        self._roadmap = roadmap_engine or RoadmapLearningEngine()
        self._repo = repo or PortfolioMemoryRepository(owns_connection=self._owns)
        self._reporter = reporter or BoardReport()
        self._cache = cache or ConfidenceCache()
        self._last_report: dict[str, Any] | None = None

    def close(self) -> None:
        if self._owns:
            self._repo.close()

    def run(
        self,
        *,
        profile_id: str,
        apm_v1_report: dict[str, Any],
        cace_v17_report: dict[str, Any],
        mie_report: dict[str, Any],
        upstream: dict[str, Any],
        executed_decisions: list[dict[str, Any]] | None = None,
        outcome_overrides: dict[str, dict[str, Any]] | None = None,
        persist: bool = True,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        cache_key = CACHE_V2_INTELLIGENCE.format(profile_id=profile_id)
        if use_cache and not persist:
            cached = self._cache.get_if_fresh(cache_key)
            if cached:
                self._last_report = cached
                return cached

        decisions = executed_decisions or self._repo.load_executed_decisions(profile_id=profile_id)
        outcomes = self._outcomes.evaluate_executed_decisions(
            decisions,
            outcome_overrides=outcome_overrides,
        )
        if not outcomes:
            outcomes = self._repo.load_outcomes(profile_id=profile_id)

        effectiveness = self._effectiveness.evaluate(outcomes)
        learning = self._learning.evaluate(
            outcomes=outcomes,
            effectiveness=effectiveness,
            apm_v1_report=apm_v1_report,
        )
        lessons = learning.get("lessons") or []
        memories = self._memory.evaluate(
            outcomes=outcomes,
            lessons=lessons,
            effectiveness=effectiveness,
        )
        board = self._board.evaluate(
            apm_v1_report=apm_v1_report,
            mie_report=mie_report,
            cace_v17_report=cace_v17_report,
            upstream=upstream,
        )
        executive_score = self._score.evaluate(
            effectiveness=effectiveness,
            learning=learning,
            apm_v1_report=apm_v1_report,
            cace_v17_report=cace_v17_report,
            board=board,
        )
        snapshot = self._snapshot.build(memories=memories, lessons=lessons, outcomes=outcomes)
        feedback = self._feedback.evaluate(board=board, apm_v1_report=apm_v1_report, learning=learning)
        roadmap_learning = self._roadmap.evaluate(
            apm_v1_report=apm_v1_report,
            outcomes=outcomes,
            learning=learning,
        )

        report = self._reporter.build(
            profile_id=profile_id,
            executive_score=executive_score,
            board=board,
            effectiveness=effectiveness,
            learning=learning,
            memory_snapshot=snapshot,
            feedback=feedback,
            roadmap_learning=roadmap_learning,
            outcomes=outcomes,
            memories=memories,
            lessons=lessons,
        )

        if persist and outcomes:
            self._repo.save_outcomes(profile_id=profile_id, outcomes=outcomes)
        if persist and memories:
            self._repo.save_memories(profile_id=profile_id, memories=memories)
        if persist and lessons:
            self._repo.save_lessons(profile_id=profile_id, lessons=lessons)

        if use_cache or persist:
            self._cache.set(cache_key, report)
            self._cache.set(CACHE_V2_MEMORY.format(profile_id=profile_id), memories)
            self._cache.set(CACHE_V2_LESSONS.format(profile_id=profile_id), lessons)
            self._cache.set(CACHE_V2_BOARD.format(profile_id=profile_id), board)
            self._cache.set(CACHE_V2_SCORE.format(profile_id=profile_id), executive_score)

        self._last_report = report
        return report

    def get_last_report(self) -> dict[str, Any]:
        return self._last_report or {}

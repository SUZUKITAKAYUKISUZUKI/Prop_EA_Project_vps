"""RC2 Live Operations Layer orchestrator."""
from __future__ import annotations

from datetime import date
from typing import Any

from src.cace.confidence_cache import ConfidenceCache
from src.live_operations.anomaly_detection_engine import AnomalyDetectionEngine
from src.live_operations.config import CACHE_RC2_BRIEFING, CACHE_RC2_DIGEST, CACHE_RC2_OPERATIONS
from src.live_operations.daily_briefing_engine import DailyBriefingEngine
from src.live_operations.daily_operations_engine import DailyOperationsEngine
from src.live_operations.executive_digest_engine import ExecutiveDigestEngine
from src.live_operations.issue_tracker import IssueTracker
from src.live_operations.live_readiness_engine import LiveReadinessEngine
from src.live_operations.live_report import LiveReport
from src.live_operations.live_repository import LiveRepository
from src.live_operations.morning_report_engine import MorningReportEngine
from src.live_operations.notification_engine import NotificationEngine
from src.live_operations.operational_metrics_engine import OperationalMetricsEngine
from src.live_operations.portfolio_watchdog import PortfolioWatchdog


class LiveOperationsEngine:
    """RC2 — transforms Portfolio OS into a daily operations system."""

    def __init__(
        self,
        *,
        daily_ops: DailyOperationsEngine | None = None,
        briefing_engine: DailyBriefingEngine | None = None,
        digest_engine: ExecutiveDigestEngine | None = None,
        morning_engine: MorningReportEngine | None = None,
        watchdog: PortfolioWatchdog | None = None,
        anomaly_engine: AnomalyDetectionEngine | None = None,
        issue_tracker: IssueTracker | None = None,
        notification_engine: NotificationEngine | None = None,
        metrics_engine: OperationalMetricsEngine | None = None,
        readiness_engine: LiveReadinessEngine | None = None,
        repo: LiveRepository | None = None,
        reporter: LiveReport | None = None,
        cache: ConfidenceCache | None = None,
        owns_connections: bool = False,
    ) -> None:
        self._owns = owns_connections
        self._daily_ops = daily_ops or DailyOperationsEngine()
        self._briefing = briefing_engine or DailyBriefingEngine()
        self._digest = digest_engine or ExecutiveDigestEngine()
        self._morning = morning_engine or MorningReportEngine()
        self._watchdog = watchdog or PortfolioWatchdog()
        self._anomaly = anomaly_engine or AnomalyDetectionEngine()
        self._issues = issue_tracker or IssueTracker()
        self._notifications = notification_engine or NotificationEngine()
        self._metrics = metrics_engine or OperationalMetricsEngine()
        self._readiness = readiness_engine or LiveReadinessEngine()
        self._repo = repo or LiveRepository(owns_connection=self._owns)
        self._reporter = reporter or LiveReport()
        self._cache = cache or ConfidenceCache()
        self._last_report: dict[str, Any] | None = None

    def close(self) -> None:
        if self._owns:
            self._repo.close()

    def run(
        self,
        *,
        profile_id: str,
        ops_context: dict[str, Any],
        persist: bool = True,
        use_cache: bool = True,
        briefing_date: str | None = None,
    ) -> dict[str, Any]:
        bdate = briefing_date or date.today().isoformat()
        cache_key = CACHE_RC2_OPERATIONS.format(profile_id=profile_id)
        if use_cache and not persist:
            cached = self._cache.get_if_fresh(cache_key)
            if cached:
                self._last_report = cached
                return cached

        ops_context = {**ops_context, "profile_id": profile_id}
        history = self._repo.load_briefing_history(profile_id=profile_id)
        historical_reliability = self._repo.historical_reliability_score(profile_id=profile_id)

        watchdog = self._watchdog.evaluate(ops_context=ops_context)
        anomalies = self._anomaly.evaluate(ops_context=ops_context, history=history)
        issues = self._issues.evaluate(
            watchdog=watchdog,
            anomalies=anomalies,
            production=ops_context.get("production_report"),
        )
        notifications = self._notifications.evaluate(issues=issues, anomalies=anomalies)

        ai_cio = ops_context.get("ai_cio_report") or {}
        user_action_load, required_actions = self._metrics.count_required_actions(ai_cio=ai_cio)
        required_actions = self._daily_ops.build_required_actions(ai_cio=ai_cio, issues=issues)

        metrics = self._metrics.evaluate(
            ops_context=ops_context,
            user_action_load=user_action_load,
            historical_reliability=historical_reliability,
        )

        orl = ops_context.get("orl_report") or {}
        production = ops_context.get("production_report") or {}
        readiness = self._readiness.evaluate(
            operational_score=float(metrics.get("operational_score") or 0),
            live_readiness_components={
                "orl_readiness": orl.get("readiness_score"),
                "production_readiness": production.get("production_readiness_score"),
                "user_action_load": user_action_load,
            },
            issues=issues,
        )

        briefing = self._briefing.evaluate(
            ops_context=ops_context,
            operational_score=float(metrics.get("operational_score") or 0),
            live_readiness=float(readiness.get("live_readiness") or 0),
            user_action_load=user_action_load,
            required_actions=required_actions,
            briefing_date=bdate,
        )
        digest = self._digest.evaluate(briefing=briefing, required_actions=required_actions)
        morning = self._morning.evaluate(briefing=briefing, notifications=notifications)

        report = self._reporter.build(
            profile_id=profile_id,
            briefing=briefing,
            digest=digest,
            metrics=metrics,
            readiness=readiness,
            watchdog=watchdog,
            anomalies=anomalies,
            issues=issues,
            notifications=notifications,
            morning=morning,
        )

        if persist:
            self._repo.save_briefing(profile_id=profile_id, briefing=briefing)
            self._repo.save_digest(profile_id=profile_id, digest=digest)
            for alert in notifications.get("operational_alerts") or []:
                self._repo.save_alert(
                    profile_id=profile_id,
                    level=str(alert.get("level") or "INFO"),
                    message=str(alert.get("message") or ""),
                    payload=alert,
                )
            for anomaly in anomalies.get("anomalies") or []:
                self._repo.save_anomaly(profile_id=profile_id, anomaly=anomaly)
            self._repo.save_operations_history(
                profile_id=profile_id,
                operational_score=float(metrics.get("operational_score") or 0),
                live_readiness=float(readiness.get("live_readiness") or 0),
                user_action_load=user_action_load,
                report=report,
            )

        if use_cache or persist:
            self._cache.set(cache_key, report)
            self._cache.set(CACHE_RC2_BRIEFING.format(profile_id=profile_id, date=bdate), briefing)
            self._cache.set(CACHE_RC2_DIGEST.format(profile_id=profile_id, date=bdate), digest)

        self._last_report = report
        return report

    def get_last_report(self) -> dict[str, Any]:
        return self._last_report or {}

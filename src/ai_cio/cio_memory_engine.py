"""CIO memory synthesis from APM v2 institutional memory."""
from __future__ import annotations

from typing import Any

from src.ai_cio.config import PRIORITY_CAPITAL_PRESERVATION, PRIORITY_GROWTH, PRIORITY_SURVIVAL


class CioMemoryEngine:
    def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
        memories = context.get("executive_memory") or []
        lessons = context.get("executive_lessons") or []
        outcomes = context.get("decision_outcomes") or []
        snapshot = context.get("memory_snapshot") or {}

        past_successes = self._past_successes(memories, outcomes, snapshot)
        past_mistakes = self._past_mistakes(memories, outcomes, snapshot)
        recurring = self._recurring_problems(lessons, outcomes)

        recommended_profile = self._recommended_profile(context)
        top_priority = self._top_priority(context, recurring, past_mistakes)

        return {
            "past_successes": past_successes,
            "past_mistakes": past_mistakes,
            "recurring_problems": recurring,
            "recommended_profile": recommended_profile,
            "top_priority": top_priority,
            "executive_lessons": lessons[:10],
        }

    def _past_successes(
        self,
        memories: list[dict[str, Any]],
        outcomes: list[dict[str, Any]],
        snapshot: dict[str, Any],
    ) -> list[str]:
        items: list[str] = []
        for row in snapshot.get("best_decisions") or []:
            title = row.get("title") or row.get("decision_type")
            if title:
                items.append(str(title))
        for outcome in outcomes:
            if str(outcome.get("outcome_class") or "").upper() == "SUCCESS":
                items.append(
                    f"{outcome.get('decision_type')} on {outcome.get('strategy') or 'portfolio'} succeeded"
                )
        for mem in memories:
            if float(mem.get("success_rate") or 0) >= 75:
                items.append(str(mem.get("title") or mem.get("summary") or "Institutional success pattern"))
        return list(dict.fromkeys(items))[:5]

    def _past_mistakes(
        self,
        memories: list[dict[str, Any]],
        outcomes: list[dict[str, Any]],
        snapshot: dict[str, Any],
    ) -> list[str]:
        items: list[str] = []
        for row in snapshot.get("worst_decisions") or []:
            title = row.get("title") or row.get("decision_type")
            if title:
                items.append(str(title))
        for outcome in outcomes:
            if str(outcome.get("outcome_class") or "").upper() == "FAILURE":
                items.append(
                    f"{outcome.get('decision_type')} on {outcome.get('strategy') or 'portfolio'} underperformed"
                )
        for mem in memories:
            if float(mem.get("success_rate") or 100) < 50:
                items.append(str(mem.get("title") or mem.get("summary") or "Institutional failure pattern"))
        return list(dict.fromkeys(items))[:5]

    def _recurring_problems(
        self,
        lessons: list[dict[str, Any]],
        outcomes: list[dict[str, Any]],
    ) -> list[str]:
        items: list[str] = []
        failure_types: dict[str, int] = {}
        for outcome in outcomes:
            if str(outcome.get("outcome_class") or "").upper() == "FAILURE":
                key = str(outcome.get("decision_type") or "unknown")
                failure_types[key] = failure_types.get(key, 0) + 1
        for key, count in sorted(failure_types.items(), key=lambda x: x[1], reverse=True):
            if count >= 2:
                items.append(f"Recurring {key} failures ({count}x)")
        for lesson in lessons:
            if str(lesson.get("lesson_type") or "").endswith("caution"):
                items.append(str(lesson.get("description") or lesson.get("lesson_type")))
        return list(dict.fromkeys(items))[:5]

    def _recommended_profile(self, context: dict[str, Any]) -> str:
        states = set(context.get("investment_state") or [])
        current = str(context.get("profile_id") or "FundedBalanced")
        if "RECOVERY_MODE" in states or "HIGH_RISK" in states:
            return "FundedConservative" if current != "FundedConservative" else current
        if "HIGH_GROWTH" in states and float(context.get("risk_score") or 0) >= 70:
            return "FundedAggressive" if current == "FundedBalanced" else current
        return "FundedBalanced"

    def _top_priority(
        self,
        context: dict[str, Any],
        recurring: list[str],
        mistakes: list[str],
    ) -> str:
        states = set(context.get("investment_state") or [])
        if "RECOVERY_MODE" in states:
            return "Enter recovery protocol and reduce risk exposure"
        if recurring:
            return recurring[0]
        if mistakes:
            return f"Avoid repeating: {mistakes[0]}"
        if "STRATEGY_CONCENTRATED" in states:
            return "Reduce concentration risk"
        if context.get("top_risk"):
            return str(context.get("top_risk"))
        return "Maintain current executive posture"

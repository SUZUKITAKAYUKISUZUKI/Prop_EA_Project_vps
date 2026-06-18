"""Recommendation consistency validation across intelligence layers."""
from __future__ import annotations

from typing import Any


class RecommendationValidator:
    def evaluate(
        self,
        *,
        ai_cio_report: dict[str, Any] | None,
        apm_report: dict[str, Any] | None,
        age_report: dict[str, Any] | None,
        cace_report: dict[str, Any] | None,
    ) -> dict[str, Any]:
        issues: list[str] = []
        conflicts: list[str] = []
        missing: list[str] = []
        stale: list[str] = []

        cio_actions = self._extract_actions(ai_cio_report, source="AI_CIO")
        apm_actions = self._extract_apm_actions(apm_report)
        age_action = self._normalize(str((age_report or {}).get("recommended_action") or ""))
        cace_action = self._normalize(
            str((cace_report or {}).get("recommended_action") or (cace_report or {}).get("recommended_action_type") or "")
        )

        if not cio_actions:
            missing.append("AI CIO recommendations missing")
        if not apm_actions and not (apm_report or {}).get("recommendations"):
            missing.append("APM recommendations missing")
        if not age_action or age_action == "do_nothing":
            if cio_actions and any("promote" in a or "increase" in a for a in cio_actions):
                conflicts.append("AGE recommends DO_NOTHING but AI CIO suggests growth action")
        if not cace_action:
            missing.append("CACE recommended action missing")

        growth_cio = any("promote" in a or "accumulate" in a or "increase" in a for a in cio_actions)
        defensive_apm = any("reduce" in a or "recovery" in a for a in apm_actions)
        if growth_cio and defensive_apm:
            conflicts.append("AI CIO growth posture conflicts with APM defensive actions")

        if age_action and cace_action and age_action != cace_action and age_action != "do_nothing":
            if not self._compatible(age_action, cace_action):
                conflicts.append(f"AGE ({age_action}) conflicts with CACE ({cace_action})")

        if (ai_cio_report or {}).get("cio_opinion") == "NO_ACTION" and cio_actions:
            stale.append("AI CIO opinion is NO_ACTION but recommendations exist")

        issues.extend(conflicts)
        issues.extend(missing)
        issues.extend(stale)

        penalty = len(conflicts) * 15 + len(missing) * 10 + len(stale) * 5
        score = max(0.0, 100.0 - penalty)

        return {
            "recommendation_consistency": round(score, 2),
            "conflicts": conflicts,
            "missing": missing,
            "stale": stale,
            "issues": issues,
            "healthy": score >= 85 and not conflicts,
            "chain": {
                "ai_cio": cio_actions,
                "apm": apm_actions,
                "age": age_action,
                "cace": cace_action,
            },
        }

    def _extract_actions(self, report: dict[str, Any] | None, *, source: str) -> list[str]:
        if not report:
            return []
        actions: list[str] = []
        for rec in report.get("recommendations") or []:
            action = self._normalize(str(rec.get("action") or rec.get("description") or ""))
            if action:
                actions.append(action)
        opinion = self._normalize(str(report.get("cio_opinion") or ""))
        if opinion:
            actions.append(opinion)
        return actions

    def _extract_apm_actions(self, report: dict[str, Any] | None) -> list[str]:
        if not report:
            return []
        actions: list[str] = []
        rec = (report.get("recommendations") or {}).get("recommended_action")
        if rec:
            actions.append(self._normalize(str(rec)))
        for item in report.get("execution_queue") or report.get("approval_queue") or []:
            action = self._normalize(str(item.get("action_type") or ""))
            if action:
                actions.append(action)
        for item in report.get("opportunities") or []:
            actions.append("promote_strategy")
        return list(dict.fromkeys(actions))

    def _normalize(self, value: str) -> str:
        return value.strip().lower().replace("-", "_").replace(" ", "_")

    def _compatible(self, a: str, b: str) -> bool:
        if a == b:
            return True
        compatible_pairs = {
            ("do_nothing", "maintain"),
            ("promote_strategy", "accumulate"),
            ("reduce_risk", "defensive"),
        }
        return (a, b) in compatible_pairs or (b, a) in compatible_pairs

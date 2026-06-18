"""Recommendation chain audit for RC1."""
from __future__ import annotations

from typing import Any


class RecommendationChainValidator:
    def evaluate(self, *, chain_context: dict[str, Any]) -> dict[str, Any]:
        issues: list[str] = []
        duplicates: list[str] = []
        contradictions: list[str] = []
        stale: list[str] = []
        loops: list[str] = []

        layers = {
            "ai_cio": self._actions(chain_context.get("ai_cio_report")),
            "cil": self._cil_actions(chain_context.get("cil_report")),
            "apm": self._apm_actions(chain_context.get("apm_report")),
            "age": self._single(chain_context.get("age_report"), "recommended_action"),
            "paae": self._paae_actions(chain_context.get("paae_report")),
        }

        all_actions: list[str] = []
        for layer, actions in layers.items():
            if not actions and layer in ("ai_cio", "apm"):
                issues.append(f"Missing recommendations at {layer}")
            for action in actions:
                if action in all_actions:
                    duplicates.append(f"{action} duplicated across chain")
                all_actions.append(action)

        growth = any("promote" in a or "increase" in a or "accumulate" in a for a in layers["ai_cio"])
        defensive = any("reduce" in a or "recovery" in a or "defensive" in a for a in layers["apm"])
        if growth and defensive:
            contradictions.append("Growth posture in AI CIO conflicts with defensive APM actions")

        age_action = layers["age"][0] if layers["age"] else ""
        if age_action in ("do_nothing", "maintain") and growth:
            contradictions.append("AGE passive while AI CIO suggests growth")

        if layers["ai_cio"] and layers["ai_cio"] == layers["apm"]:
            loops.append("AI CIO and APM emit identical primary action — possible recommendation loop")

        if (chain_context.get("ai_cio_report") or {}).get("cio_opinion") == "NO_ACTION" and layers["ai_cio"]:
            stale.append("AI CIO opinion NO_ACTION but recommendations exist")

        issues.extend(contradictions)
        issues.extend(duplicates)
        issues.extend(stale)
        issues.extend(loops)

        penalty = len(contradictions) * 20 + len(duplicates) * 10 + len(stale) * 8 + len(loops) * 15
        score = max(0.0, 100.0 - penalty)

        return {
            "recommendation_consistency": round(score, 2),
            "recommendation_chain_health": round(score, 2),
            "contradictions": contradictions,
            "duplicates": duplicates,
            "stale": stale,
            "loops": loops,
            "issues": issues,
            "chain": layers,
            "healthy": score >= 85 and not contradictions,
        }

    def _actions(self, report: dict[str, Any] | None) -> list[str]:
        if not report:
            return []
        out: list[str] = []
        for rec in report.get("recommendations") or []:
            out.append(self._norm(str(rec.get("action") or rec.get("description") or "")))
        opinion = self._norm(str(report.get("cio_opinion") or ""))
        if opinion:
            out.append(opinion)
        return [a for a in out if a]

    def _cil_actions(self, report: dict[str, Any] | None) -> list[str]:
        if not report:
            return []
        out = [self._norm(str(report.get("top_opportunity") or ""))]
        return [a for a in out if a]

    def _apm_actions(self, report: dict[str, Any] | None) -> list[str]:
        if not report:
            return []
        out: list[str] = []
        rec = (report.get("recommendations") or {}).get("recommended_action")
        if rec:
            out.append(self._norm(str(rec)))
        for item in report.get("execution_queue") or []:
            out.append(self._norm(str(item.get("action_type") or "")))
        return [a for a in out if a]

    def _paae_actions(self, report: dict[str, Any] | None) -> list[str]:
        if not report:
            return []
        current = report.get("current_weights") or {}
        recommended = report.get("recommended_weights") or {}
        if not recommended:
            return []
        drift = sum(abs(float(recommended.get(k, 0)) - float(current.get(k, 0))) for k in set(current) | set(recommended))
        if drift > 0.05:
            return ["allocation_rebalance"]
        return ["maintain_allocation"]

    def _single(self, report: dict[str, Any] | None, key: str) -> list[str]:
        if not report:
            return []
        val = self._norm(str(report.get(key) or ""))
        return [val] if val else []

    def _norm(self, value: str) -> str:
        return value.strip().lower().replace("-", "_").replace(" ", "_")

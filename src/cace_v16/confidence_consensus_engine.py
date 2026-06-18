"""Cross-module confidence consensus for CACE v1.6."""
from __future__ import annotations

from collections import Counter
from typing import Any

from src.cace_v16.confidence_v16_models import consensus_category


class ConfidenceConsensusEngine:
    """Measure agreement across Portfolio OS intelligence modules."""

    MODULE_KEYS = ("PAAE", "PDTS", "SLM", "AGE", "PRAE", "STATE_ANALYTICS")

    def evaluate(
        self,
        *,
        paae: dict[str, Any] | None = None,
        pdts: dict[str, Any] | None = None,
        slm: dict[str, Any] | None = None,
        age_v4: dict[str, Any] | None = None,
        prae_v2: dict[str, Any] | None = None,
        state_analytics: dict[str, Any] | None = None,
        explicit_recommendations: dict[str, str | dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        modules: dict[str, dict[str, Any]] = {}

        if explicit_recommendations:
            for key, value in explicit_recommendations.items():
                modules[key.upper()] = self._normalize_entry(value)

        sources = {
            "PAAE": paae,
            "PDTS": pdts,
            "SLM": slm,
            "AGE": age_v4,
            "PRAE": prae_v2,
            "STATE_ANALYTICS": state_analytics,
        }
        extractors = {
            "PAAE": self._paae_recommendation,
            "PDTS": self._pdts_recommendation,
            "SLM": self._slm_recommendation,
            "AGE": self._age_recommendation,
            "PRAE": self._prae_recommendation,
            "STATE_ANALYTICS": self._state_recommendation,
        }
        for key, payload in sources.items():
            if payload is not None and key not in modules:
                modules[key] = extractors[key](payload)

        actions = [m["action"] for m in modules.values() if m.get("action")]
        if not actions:
            return self._empty_result()

        counts = Counter(actions)
        recommended_action, agree_count = counts.most_common(1)[0]
        total_modules = len(actions)
        agreement_ratio = round(agree_count / total_modules, 3)
        consensus_score = round(25.0 + agreement_ratio * 75.0, 1)

        return {
            "recommended_action": recommended_action,
            "consensus_score": consensus_score,
            "consensus_category": consensus_category(consensus_score),
            "agreement_ratio": agreement_ratio,
            "agree_count": agree_count,
            "total_modules": total_modules,
            "module_agreements": modules,
            "participant_table": [
                {"participant": name, "action": data.get("action"), "confidence": data.get("confidence")}
                for name, data in sorted(modules.items())
            ],
        }

    def _empty_result(self) -> dict[str, Any]:
        return {
            "recommended_action": "NO_ACTION",
            "consensus_score": 25.0,
            "consensus_category": "VERY_LOW",
            "agreement_ratio": 0.0,
            "agree_count": 0,
            "total_modules": 0,
            "module_agreements": {},
            "participant_table": [],
        }

    def _normalize_entry(self, value: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(value, dict):
            return {
                "action": str(value.get("action") or "NO_ACTION").upper().replace(" ", "_"),
                "confidence": round(float(value.get("confidence") or 50.0), 1),
            }
        return {"action": str(value).upper().replace(" ", "_"), "confidence": 50.0}

    def _paae_recommendation(self, paae: dict[str, Any]) -> dict[str, Any]:
        current = paae.get("current_weights") or {}
        recommended = paae.get("recommended_weights") or {}
        quality = paae.get("quality_scores") or {}
        best_code = ""
        best_gap = 0.0
        for code in set(current) | set(recommended):
            cur = self._weight(current.get(code, 0))
            rec = self._weight(recommended.get(code, 0))
            gap = rec - cur
            if abs(gap) > abs(best_gap):
                best_gap = gap
                best_code = code
        if not best_code:
            action = "NO_ACTION"
        elif best_gap > 0.02:
            action = f"INCREASE_{best_code}"
        elif best_gap < -0.02:
            action = f"REDUCE_{best_code}"
        else:
            action = "NO_ACTION"
        conf = float(quality.get(best_code, 70)) if best_code else 60.0
        return {"action": action, "confidence": round(conf, 1)}

    def _pdts_recommendation(self, pdts: dict[str, Any]) -> dict[str, Any]:
        ranking = pdts.get("recommendation_ranking") or []
        cmp = pdts.get("scenario_comparison") or {}
        recommended = cmp.get("recommended") or {}
        if ranking:
            top = ranking[0]
            action = str(top.get("scenario") or top.get("label") or "REBALANCE").upper().replace(" ", "_")
            conf = float(top.get("score") or top.get("pass_rate") or recommended.get("score") or 70)
        else:
            action = "REBALANCE_TO_RECOMMENDED"
            conf = float(recommended.get("score") or 70)
        for row in ranking:
            scenario = str(row.get("scenario") or "").upper()
            if "REDUCE" in scenario:
                action = scenario.replace(" ", "_")
                break
        return {"action": action, "confidence": round(conf, 1)}

    def _slm_recommendation(self, slm: dict[str, Any]) -> dict[str, Any]:
        for row in slm.get("promotion_candidates") or []:
            code = str(row.get("strategy") or "")
            if code:
                return {
                    "action": f"PROMOTE_{code}",
                    "confidence": round(float(row.get("score") or 75), 1),
                }
        for row in slm.get("retirement_candidates") or []:
            code = str(row.get("strategy") or "")
            if code:
                return {
                    "action": f"RETIRE_{code}",
                    "confidence": round(float(row.get("score") or 55), 1),
                }
        for row in slm.get("strategies") or []:
            code = str(row.get("strategy") or "")
            stage = str(row.get("stage") or "").upper()
            if stage == "RECOVERY" and code:
                return {
                    "action": f"REDUCE_{code}",
                    "confidence": round(float(row.get("portfolio_fit_score") or 50), 1),
                }
        return {"action": "NO_ACTION", "confidence": 60.0}

    def _age_recommendation(self, age_v4: dict[str, Any]) -> dict[str, Any]:
        action = str(age_v4.get("recommended_action") or "DO_NOTHING").upper().replace(" ", "_")
        conf = float(
            age_v4.get("strategic_confidence")
            or age_v4.get("confidence")
            or age_v4.get("strategic_score")
            or 70
        )
        return {"action": action, "confidence": round(conf, 1)}

    def _prae_recommendation(self, prae_v2: dict[str, Any]) -> dict[str, Any]:
        risks = prae_v2.get("strategy_risk") or []
        if risks:
            top = max(risks, key=lambda r: float(r.get("risk_score") or 0))
            code = str(top.get("strategy") or "")
            if code:
                return {
                    "action": f"REDUCE_{code}",
                    "confidence": round(100.0 - float(top.get("risk_score") or 50), 1),
                }
        health = (prae_v2.get("health_report") or {}).get("health_score")
        return {"action": "MONITOR_RISK", "confidence": round(float(health or 60), 1)}

    def _state_recommendation(self, state_analytics: dict[str, Any]) -> dict[str, Any]:
        state = str(state_analytics.get("current_state") or "").lower()
        conf = float(state_analytics.get("health_score") or state_analytics.get("funded_stability_score") or 60)
        if state == "recovery":
            return {"action": "PROFILE_SWITCH", "confidence": round(conf, 1)}
        if state == "challenge":
            return {"action": "AGGRESSIVE_GROWTH", "confidence": round(conf, 1)}
        return {"action": "NO_ACTION", "confidence": round(conf, 1)}

    def _weight(self, value: Any) -> float:
        w = float(value or 0)
        return w / 100.0 if w > 1.0 else w

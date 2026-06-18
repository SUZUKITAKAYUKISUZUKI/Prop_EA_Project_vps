"""Executive daily digest for RC2 — one glance portfolio operations."""
from __future__ import annotations

from typing import Any


class ExecutiveDigestEngine:
    def evaluate(self, *, briefing: dict[str, Any], required_actions: list[str]) -> dict[str, Any]:
        actions = required_actions or briefing.get("required_actions") or []
        action_lines = actions if actions else ["なし"]

        digest_text = "\n".join(
            [
                "■ 今日のPortfolio OS",
                "",
                f"CIO Opinion:",
                str(briefing.get("cio_opinion") or "—"),
                "",
                "Opportunity:",
                str(briefing.get("top_opportunity") or "—"),
                "",
                "Risk:",
                str(briefing.get("top_risk") or "—"),
                "",
                f"Health:",
                str(briefing.get("system_health") or "—"),
                "",
                f"Readiness:",
                str(briefing.get("readiness") or "—"),
                "",
                "推奨アクション:",
                *action_lines,
            ]
        )

        return {
            "daily_digest": digest_text,
            "digest_summary": {
                "cio_opinion": briefing.get("cio_opinion"),
                "top_opportunity": briefing.get("top_opportunity"),
                "top_risk": briefing.get("top_risk"),
                "system_health": briefing.get("system_health"),
                "readiness": briefing.get("readiness"),
                "required_actions": actions,
            },
        }

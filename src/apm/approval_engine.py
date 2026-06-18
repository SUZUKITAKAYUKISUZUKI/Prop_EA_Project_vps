"""Human approval workflow for APM v1."""
from __future__ import annotations

from typing import Any

from src.apm.config import APPROVAL_STATUSES


class ApprovalEngine:
    def queue_for_approval(self, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        queued = []
        for action in actions:
            item = dict(action)
            if item.get("action_type") == "NO_ACTION":
                item["status"] = "EXECUTED"
            else:
                item["status"] = "PENDING_APPROVAL"
            queued.append(item)
        return queued

    def approve(self, action: dict[str, Any]) -> dict[str, Any]:
        updated = dict(action)
        if updated.get("status") == "PENDING_APPROVAL":
            updated["status"] = "APPROVED"
        return updated

    def mark_executed(self, action: dict[str, Any]) -> dict[str, Any]:
        updated = dict(action)
        if updated.get("status") == "APPROVED":
            updated["status"] = "EXECUTED"
        return updated

    def reject(self, action: dict[str, Any], *, reason: str = "") -> dict[str, Any]:
        updated = dict(action)
        updated["status"] = "REJECTED"
        updated["rejection_reason"] = reason
        return updated

    def validate_status(self, status: str) -> bool:
        return status in APPROVAL_STATUSES

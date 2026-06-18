"""Execution queue management for APM v1 — governance only, no broker execution."""
from __future__ import annotations

from typing import Any

from src.apm.approval_engine import ApprovalEngine


class ExecutionQueue:
    def __init__(self, *, approval_engine: ApprovalEngine | None = None) -> None:
        self._approval = approval_engine or ApprovalEngine()

    def enqueue(self, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._approval.queue_for_approval(actions)

    def approve_action(self, action: dict[str, Any]) -> dict[str, Any]:
        approved = self._approval.approve(action)
        return self._approval.mark_executed(approved)

    def reject_action(self, action: dict[str, Any], *, reason: str = "") -> dict[str, Any]:
        return self._approval.reject(action, reason=reason)

    def pending(self, queue: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [item for item in queue if item.get("status") == "PENDING_APPROVAL"]

    def approved(self, queue: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [item for item in queue if item.get("status") in {"APPROVED", "EXECUTED"}]

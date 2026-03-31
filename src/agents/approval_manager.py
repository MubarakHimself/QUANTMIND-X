"""
Human-in-the-Loop Approval Manager

Central approval queue that integrates with:
- Workflow gates (SIT_GATE, HUMAN_APPROVAL_GATE)
- Tool-level approvals (requires_approval=True on ToolDefinition)
- Agent-level approvals (PRE_TOOL_USE hook in BaseAgent / DepartmentHead)
- SSE notifications to the Svelte UI
- Department mail system for persistent cross-department notifications
- Database persistence + resume logic so pending approvals survive restarts
- Graph memory for audit trail of human decisions

Approval flow:
1. Agent/Workflow creates an ApprovalRequest
2. ApprovalManager persists to DB, notifies UI via SSE, sends dept mail
3. Human approves/rejects via the API
4. The awaiting asyncio.Event is set, unblocking the agent/workflow
5. Resolution written to graph memory + DB for audit trail

Resume flow (on server restart):
1. resume_pending() loads PENDING rows from DB
2. Re-creates in-memory ApprovalRequest objects (with new asyncio.Events)
3. Sends SSE + mail reminders so UI/departments know they still need action
4. Workflow coordinator calls resume_waiting_workflows() to re-attach

Architecture reference: PRD §11.8, architecture §5.1 human-in-the-loop gates
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Types
# =============================================================================

class ApprovalType(str, Enum):
    """Types of approval requests."""
    WORKFLOW_GATE = "workflow_gate"         # SIT_GATE, HUMAN_APPROVAL_GATE
    TOOL_EXECUTION = "tool_execution"      # Tool requires_approval=True
    AGENT_ACTION = "agent_action"          # General agent action needing approval
    EA_PROMOTION = "ea_promotion"          # EA lifecycle: paper_validated → active
    TRADE_EXECUTION = "trade_execution"    # Live trade execution
    DEPLOYMENT = "deployment"              # EA deployment to live


class ApprovalStatus(str, Enum):
    """Status of an approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ApprovalUrgency(str, Enum):
    """Urgency level — determines UI presentation."""
    LOW = "low"           # Info banner, no sound
    MEDIUM = "medium"     # Highlighted card
    HIGH = "high"         # Alert with sound
    CRITICAL = "critical" # Modal blocking, flash


@dataclass
class ApprovalRequest:
    """A request for human approval."""
    id: str
    approval_type: ApprovalType
    urgency: ApprovalUrgency
    title: str
    description: str
    department: str
    agent_id: str
    context: Dict[str, Any] = field(default_factory=dict)
    # Workflow-specific
    workflow_id: Optional[str] = None
    workflow_stage: Optional[str] = None
    strategy_id: Optional[str] = None
    # Tool-specific
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    # Resume context — everything needed to restart the paused entity
    # For workflows: workflow dict snapshot, payload, task history
    # For agents: session_id, checkpoint_id, conversation tail
    resume_context: Optional[Dict[str, Any]] = None
    # State
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: str = ""
    resolved_at: Optional[str] = None
    resolved_by: Optional[str] = None
    rejection_reason: Optional[str] = None
    # Internal: asyncio event for blocking callers
    _event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API / SSE (excludes _event)."""
        return {
            "id": self.id,
            "approval_type": self.approval_type.value,
            "urgency": self.urgency.value,
            "title": self.title,
            "description": self.description,
            "department": self.department,
            "agent_id": self.agent_id,
            "context": self.context,
            "workflow_id": self.workflow_id,
            "workflow_stage": self.workflow_stage,
            "strategy_id": self.strategy_id,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "resume_context": self.resume_context,
            "status": self.status.value,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "resolved_by": self.resolved_by,
            "rejection_reason": self.rejection_reason,
        }


# =============================================================================
# Approval Manager Singleton
# =============================================================================

class ApprovalManager:
    """
    Central approval queue with DB persistence + resume support.

    Agents call `request_approval()` which returns an ApprovalRequest.
    They can then `await wait_for_approval(request_id)` to block until
    the human responds.

    On server restart, call `resume_pending()` to reload unresolved
    approval requests from the DB and re-notify the UI.
    """

    def __init__(self):
        self._requests: Dict[str, ApprovalRequest] = {}
        self._history: List[Dict[str, Any]] = []  # resolved requests
        self._on_request_callbacks: List[Callable] = []
        self._max_history = 200
        self._db_initialized = False

    # ── DB Persistence ──────────────────────────────────────────────────

    def _ensure_table(self) -> None:
        """Ensure the approval_requests DB table exists."""
        if self._db_initialized:
            return
        try:
            from src.database.engine import engine
            from src.database.models.approval_request import init_approval_requests_table
            init_approval_requests_table(engine)
            self._db_initialized = True
        except Exception as e:
            logger.debug(f"Approval table init skipped: {e}")

    def _persist(self, req: ApprovalRequest) -> None:
        """Persist an approval request to the database."""
        self._ensure_table()
        try:
            from src.database.models.approval_request import ApprovalRequestModel
            from src.database.models import get_db_session
            db = get_db_session()
            try:
                row = ApprovalRequestModel(
                    request_id=req.id,
                    approval_type=req.approval_type.value,
                    urgency=req.urgency.value,
                    status=req.status.value,
                    title=req.title,
                    description=req.description,
                    department=req.department,
                    agent_id=req.agent_id,
                    workflow_id=req.workflow_id,
                    workflow_stage=req.workflow_stage,
                    strategy_id=req.strategy_id,
                    tool_name=req.tool_name,
                    tool_input_json=json.dumps(req.tool_input) if req.tool_input else None,
                    context_json=json.dumps(req.context) if req.context else None,
                    resume_context_json=json.dumps(req.resume_context) if req.resume_context else None,
                )
                db.add(row)
                db.commit()
                logger.debug(f"Approval persisted to DB: {req.id}")
            finally:
                db.close()
        except Exception as e:
            logger.debug(f"Approval DB persist skipped: {e}")

    def _update_db_status(self, req: ApprovalRequest) -> None:
        """Update an approval request's status in the database."""
        try:
            from src.database.models.approval_request import ApprovalRequestModel
            from src.database.models import get_db_session
            db = get_db_session()
            try:
                row = db.query(ApprovalRequestModel).filter(
                    ApprovalRequestModel.request_id == req.id
                ).first()
                if row:
                    row.status = req.status.value
                    row.resolved_by = req.resolved_by
                    row.resolved_at = (
                        datetime.fromisoformat(req.resolved_at)
                        if req.resolved_at else None
                    )
                    row.rejection_reason = req.rejection_reason
                    db.commit()
            finally:
                db.close()
        except Exception as e:
            logger.debug(f"Approval DB status update skipped: {e}")

    def _load_pending_from_db(self) -> List[Dict[str, Any]]:
        """Load all pending approval requests from the database."""
        self._ensure_table()
        try:
            from src.database.models.approval_request import ApprovalRequestModel
            from src.database.models import get_db_session
            db = get_db_session()
            try:
                rows = db.query(ApprovalRequestModel).filter(
                    ApprovalRequestModel.status == "pending"
                ).order_by(ApprovalRequestModel.created_at.asc()).all()

                results = []
                for row in rows:
                    results.append({
                        "id": row.request_id,
                        "approval_type": row.approval_type,
                        "urgency": row.urgency,
                        "status": row.status,
                        "title": row.title,
                        "description": row.description or "",
                        "department": row.department or "",
                        "agent_id": row.agent_id or "",
                        "workflow_id": row.workflow_id,
                        "workflow_stage": row.workflow_stage,
                        "strategy_id": row.strategy_id,
                        "tool_name": row.tool_name,
                        "tool_input": json.loads(row.tool_input_json) if row.tool_input_json else None,
                        "context": json.loads(row.context_json) if row.context_json else {},
                        "resume_context": json.loads(row.resume_context_json) if row.resume_context_json else None,
                        "created_at": row.created_at.isoformat() if row.created_at else "",
                    })
                return results
            finally:
                db.close()
        except Exception as e:
            logger.debug(f"Approval DB load skipped: {e}")
            return []

    # ── Resume Logic ────────────────────────────────────────────────────

    def resume_pending(self) -> int:
        """
        Resume pending approvals from DB after server restart.

        Re-creates in-memory ApprovalRequest objects with new asyncio.Events
        and re-sends SSE + department mail reminders.

        Returns the count of resumed approvals.
        """
        rows = self._load_pending_from_db()
        resumed = 0

        for row in rows:
            if row["id"] in self._requests:
                continue  # Already in memory

            req = ApprovalRequest(
                id=row["id"],
                approval_type=ApprovalType(row["approval_type"]),
                urgency=ApprovalUrgency(row.get("urgency", "medium")),
                title=row["title"],
                description=row["description"],
                department=row["department"],
                agent_id=row["agent_id"],
                context=row.get("context", {}),
                workflow_id=row.get("workflow_id"),
                workflow_stage=row.get("workflow_stage"),
                strategy_id=row.get("strategy_id"),
                tool_name=row.get("tool_name"),
                tool_input=row.get("tool_input"),
                resume_context=row.get("resume_context"),
                created_at=row.get("created_at", ""),
            )

            self._requests[req.id] = req

            # Re-send notifications so UI knows about pending approvals
            self._notify_ui(req, event_type="approval_resumed")

            resumed += 1

        if resumed > 0:
            logger.info(f"Resumed {resumed} pending approvals from DB")

        return resumed

    # ── Graph Memory — Audit Trail ──────────────────────────────────────

    def _write_memory(self, req: ApprovalRequest, event: str) -> None:
        """Write approval event to graph memory for long-term audit trail."""
        try:
            from src.memory.graph.facade import GraphMemoryFacade
            facade = GraphMemoryFacade()
            facade.retain(
                node_type="DECISION",
                content=(
                    f"HITL {event}: [{req.approval_type.value}] {req.title}. "
                    f"Status: {req.status.value}. "
                    f"Resolved by: {req.resolved_by or 'pending'}. "
                    f"Department: {req.department}. "
                    f"Workflow: {req.workflow_id or 'N/A'}."
                ),
                category="hitl_audit",
                tags=[
                    "hitl", event, req.approval_type.value,
                    req.department or "unknown",
                    req.workflow_id or "no_workflow",
                ],
                metadata={
                    "approval_id": req.id,
                    "approval_type": req.approval_type.value,
                    "status": req.status.value,
                    "resolved_by": req.resolved_by,
                    "rejection_reason": req.rejection_reason,
                    "workflow_id": req.workflow_id,
                    "strategy_id": req.strategy_id,
                    "created_at": req.created_at,
                    "resolved_at": req.resolved_at,
                },
            )
        except Exception as e:
            logger.debug(f"Approval memory write skipped: {e}")

    # ── Create approval requests ─────────────────────────────────────────

    def request_approval(
        self,
        approval_type: ApprovalType,
        title: str,
        description: str,
        department: str,
        agent_id: str = "",
        urgency: ApprovalUrgency = ApprovalUrgency.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None,
        workflow_stage: Optional[str] = None,
        strategy_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        tool_input: Optional[Dict[str, Any]] = None,
        resume_context: Optional[Dict[str, Any]] = None,
    ) -> ApprovalRequest:
        """
        Create a new approval request, persist to DB, and notify the UI.

        Args:
            resume_context: Dict with everything needed to resume the paused
                            entity. For workflows: workflow snapshot. For agents:
                            session_id, checkpoint_id, conversation tail, etc.

        Returns the ApprovalRequest — callers can await wait_for_approval()
        to block until resolved.
        """
        request_id = f"apr_{uuid.uuid4().hex[:12]}"

        req = ApprovalRequest(
            id=request_id,
            approval_type=approval_type,
            urgency=urgency,
            title=title,
            description=description,
            department=department,
            agent_id=agent_id,
            context=context or {},
            workflow_id=workflow_id,
            workflow_stage=workflow_stage,
            strategy_id=strategy_id,
            tool_name=tool_name,
            tool_input=tool_input,
            resume_context=resume_context,
        )

        self._requests[request_id] = req

        # Persist to DB for durability
        self._persist(req)

        # Notify UI via SSE thought stream + department mail
        self._notify_ui(req, event_type="approval_requested")

        # Write to graph memory for audit trail
        self._write_memory(req, event="requested")

        # Fire any registered callbacks
        for cb in self._on_request_callbacks:
            try:
                cb(req)
            except Exception as e:
                logger.debug(f"Approval callback error: {e}")

        logger.info(
            f"Approval requested: {request_id} [{approval_type.value}] "
            f"{title} (dept={department})"
        )

        return req

    # ── Resolve approval requests ────────────────────────────────────────

    def approve(
        self,
        request_id: str,
        approved_by: str = "human",
    ) -> bool:
        """Approve a pending request. Unblocks any awaiting agents."""
        req = self._requests.get(request_id)
        if not req or req.status != ApprovalStatus.PENDING:
            return False

        req.status = ApprovalStatus.APPROVED
        req.resolved_at = datetime.now(timezone.utc).isoformat()
        req.resolved_by = approved_by
        req._event.set()

        self._archive(request_id)
        self._update_db_status(req)
        self._notify_ui(req, event_type="approval_resolved")
        self._write_memory(req, event="approved")

        logger.info(f"Approval granted: {request_id} by {approved_by}")
        return True

    def reject(
        self,
        request_id: str,
        reason: str = "",
        rejected_by: str = "human",
    ) -> bool:
        """Reject a pending request. Unblocks any awaiting agents."""
        req = self._requests.get(request_id)
        if not req or req.status != ApprovalStatus.PENDING:
            return False

        req.status = ApprovalStatus.REJECTED
        req.resolved_at = datetime.now(timezone.utc).isoformat()
        req.resolved_by = rejected_by
        req.rejection_reason = reason
        req._event.set()

        self._archive(request_id)
        self._update_db_status(req)
        self._notify_ui(req, event_type="approval_resolved")
        self._write_memory(req, event="rejected")

        logger.info(f"Approval rejected: {request_id} reason={reason}")
        return True

    def cancel(self, request_id: str) -> bool:
        """Cancel a pending request (system-initiated)."""
        req = self._requests.get(request_id)
        if not req or req.status != ApprovalStatus.PENDING:
            return False

        req.status = ApprovalStatus.CANCELLED
        req.resolved_at = datetime.now(timezone.utc).isoformat()
        req._event.set()

        self._archive(request_id)
        self._update_db_status(req)
        return True

    # ── Wait for resolution ──────────────────────────────────────────────

    async def wait_for_approval(
        self,
        request_id: str,
        timeout: float = 3600,  # 1 hour default
    ) -> ApprovalRequest:
        """
        Block until the approval is resolved (approved/rejected/cancelled).

        Args:
            request_id: The approval request ID
            timeout: Max seconds to wait (default 1 hour)

        Returns:
            The resolved ApprovalRequest

        Raises:
            TimeoutError: If timeout expires
            KeyError: If request_id not found
        """
        req = self._requests.get(request_id)
        if not req:
            # Check history
            for h in reversed(self._history):
                if h.get("id") == request_id:
                    # Already resolved — return a synthetic request
                    resolved = ApprovalRequest(
                        id=request_id,
                        approval_type=ApprovalType(h.get("approval_type", "agent_action")),
                        urgency=ApprovalUrgency.MEDIUM,
                        title=h.get("title", ""),
                        description=h.get("description", ""),
                        department=h.get("department", ""),
                        agent_id=h.get("agent_id", ""),
                        status=ApprovalStatus(h.get("status", "cancelled")),
                    )
                    resolved._event.set()
                    return resolved
            raise KeyError(f"Approval request {request_id} not found")

        if req.status != ApprovalStatus.PENDING:
            return req  # Already resolved

        try:
            await asyncio.wait_for(req._event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            req.status = ApprovalStatus.EXPIRED
            req.resolved_at = datetime.now(timezone.utc).isoformat()
            self._archive(request_id)
            self._update_db_status(req)
            self._notify_ui(req, event_type="approval_expired")
            self._write_memory(req, event="expired")
            logger.warning(f"Approval expired: {request_id} (timeout={timeout}s)")

        return req

    # ── Query ────────────────────────────────────────────────────────────

    def get_pending(self) -> List[Dict[str, Any]]:
        """Get all pending approval requests."""
        return [
            r.to_dict()
            for r in self._requests.values()
            if r.status == ApprovalStatus.PENDING
        ]

    def get_request(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific approval request."""
        req = self._requests.get(request_id)
        if req:
            return req.to_dict()
        for h in reversed(self._history):
            if h.get("id") == request_id:
                return h
        return None

    def get_resume_context(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get the resume context for a specific approval request."""
        req = self._requests.get(request_id)
        if req:
            return req.resume_context
        # Check DB
        rows = self._load_pending_from_db()
        for row in rows:
            if row["id"] == request_id:
                return row.get("resume_context")
        return None

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get resolved approval history."""
        return list(reversed(self._history[-limit:]))

    def get_pending_count(self) -> int:
        """Get count of pending approvals."""
        return sum(
            1 for r in self._requests.values()
            if r.status == ApprovalStatus.PENDING
        )

    # ── Callbacks ────────────────────────────────────────────────────────

    def on_request(self, callback: Callable) -> None:
        """Register a callback fired when a new approval is requested."""
        self._on_request_callbacks.append(callback)

    # ── Internal ─────────────────────────────────────────────────────────

    def _archive(self, request_id: str) -> None:
        """Move resolved request from active to history."""
        req = self._requests.pop(request_id, None)
        if req:
            self._history.append(req.to_dict())
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

    def _notify_ui(self, req: ApprovalRequest, event_type: str) -> None:
        """Publish approval event to SSE thought stream + department mail."""
        # 1. SSE thought stream (real-time UI)
        try:
            from src.api.agent_thought_stream_endpoints import get_thought_publisher
            payload = req.to_dict()
            payload["event_type"] = event_type
            get_thought_publisher().publish(
                department=req.department,
                thought=json.dumps(payload),
                thought_type="observation",
            )
        except Exception as e:
            logger.debug(f"Approval SSE notification skipped: {e}")

        # 2. Department mail (persistent notification + cross-department routing)
        self._send_department_mail(req, event_type)

    def _send_department_mail(self, req: ApprovalRequest, event_type: str) -> None:
        """Route approval events through the department mail system."""
        try:
            from src.agents.departments.department_mail import (
                get_mail_service, Priority as MailPriority, MessageType,
            )
            mail = get_mail_service()

            urgency_map = {
                ApprovalUrgency.LOW: MailPriority.LOW,
                ApprovalUrgency.MEDIUM: MailPriority.NORMAL,
                ApprovalUrgency.HIGH: MailPriority.HIGH,
                ApprovalUrgency.CRITICAL: MailPriority.URGENT,
            }
            priority = urgency_map.get(req.urgency, MailPriority.NORMAL)

            if event_type in ("approval_requested", "approval_resumed"):
                msg_type = MessageType.APPROVAL_REQUEST
                prefix = "Approval Required" if event_type == "approval_requested" else "Pending Approval (Resumed)"
                subject = f"{prefix}: {req.title}"
                body = (
                    f"{req.description}\n\n"
                    f"Type: {req.approval_type.value}\n"
                    f"Agent: {req.agent_id or 'system'}\n"
                    f"Approval ID: {req.id}"
                )
                if event_type == "approval_resumed":
                    body += "\n\n[This approval was pending before server restart and still needs attention.]"
            elif event_type == "approval_resolved":
                action = req.status.value.upper()
                msg_type = (
                    MessageType.APPROVAL_APPROVED
                    if req.status == ApprovalStatus.APPROVED
                    else MessageType.APPROVAL_REJECTED
                )
                subject = f"Approval {action}: {req.title}"
                body = (
                    f"Resolved by: {req.resolved_by or 'unknown'}\n"
                    f"Status: {req.status.value}"
                )
                if req.rejection_reason:
                    body += f"\nReason: {req.rejection_reason}"
            elif event_type == "approval_expired":
                msg_type = MessageType.APPROVAL_REJECTED
                subject = f"Approval Expired: {req.title}"
                body = f"Timed out after waiting for human response.\nApproval ID: {req.id}"
            else:
                return

            mail.send(
                from_dept="floor_manager",
                to_dept=req.department or "development",
                type=msg_type,
                subject=subject,
                body=body,
                priority=priority,
                gate_id=req.id,
                workflow_id=req.workflow_id or "",
            )
            logger.debug(f"Approval mail sent: {req.id} ({event_type})")
        except Exception as e:
            logger.debug(f"Approval department mail skipped: {e}")


# =============================================================================
# Singleton
# =============================================================================

_manager: Optional[ApprovalManager] = None


def get_approval_manager() -> ApprovalManager:
    """Return the singleton ApprovalManager instance."""
    global _manager
    if _manager is None:
        _manager = ApprovalManager()
    return _manager

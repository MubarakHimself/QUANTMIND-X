"""
Persistent Approval Request Model

Database-backed storage for HITL approval requests so they survive
server restarts and can be resumed. Also stores workflow checkpoint
context needed to resume paused workflows/agents after approval.

Architecture reference: PRD §11.8, HITL gates + memory-resume
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, DateTime, Text, Float, Index,
)
from .base import Base


class ApprovalRequestModel(Base):
    """
    Persistent approval request — mirrors the in-memory ApprovalRequest
    but survives server restarts.

    The resume_context JSON blob stores everything needed to restart
    the paused workflow or agent after human resolution.
    """
    __tablename__ = "approval_requests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(36), unique=True, nullable=False, index=True)
    approval_type = Column(String(30), nullable=False)
    urgency = Column(String(20), nullable=False, default="medium")
    status = Column(String(20), nullable=False, default="pending", index=True)

    # Human-readable
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)

    # Context references
    department = Column(String(50), nullable=True, index=True)
    agent_id = Column(String(100), nullable=True)
    workflow_id = Column(String(100), nullable=True, index=True)
    workflow_stage = Column(String(100), nullable=True)
    strategy_id = Column(String(100), nullable=True)
    tool_name = Column(String(100), nullable=True)

    # Tool input / extra context (JSON)
    tool_input_json = Column(Text, nullable=True)
    context_json = Column(Text, nullable=True)

    # Resume context — everything needed to restart the paused entity
    # For workflows: workflow_id, stage, payload, task history
    # For agents: agent_id, session_id, checkpoint_id, conversation tail
    resume_context_json = Column(Text, nullable=True)

    # Resolution
    resolved_by = Column(String(100), nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    rejection_reason = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(
        DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    expires_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_approval_status_dept", "status", "department"),
        Index("ix_approval_workflow", "workflow_id", "status"),
    )


def init_approval_requests_table(engine):
    """Create the approval_requests table if it doesn't exist."""
    try:
        ApprovalRequestModel.__table__.create(engine, checkfirst=True)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            f"Could not create approval_requests table: {e}"
        )

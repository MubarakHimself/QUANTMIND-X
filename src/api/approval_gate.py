"""
Approval Gate API Endpoints

Phase 7: Semi-Automatic Workflow Integration

Provides REST API endpoints for workflow stage approval gates.
- Accepts workflow stage transitions
- Requires user approval before moving to next stage
- Stores approval status in database
- Sends department mail notifications on creation, approval, and rejection

Endpoints:
- POST /api/approval-gates - Create an approval request
- GET /api/approval-gates/{gate_id} - Get approval status
- POST /api/approval-gates/{gate_id}/approve - Approve a gate
- POST /api/approval-gates/{gate_id}/reject - Reject a gate
- GET /api/approval-gates/workflow/{workflow_id} - Get all gates for a workflow
- GET /api/approval-gates/pending - List pending approvals
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from enum import Enum
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.database.models import Base
from src.database.engine import engine, get_session

logger = logging.getLogger(__name__)

# Import WebSocket broadcast function (optional - gracefully fails if not available)
try:
    from src.api.websocket_endpoints import broadcast_approval_gate
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False
    logger.warning("WebSocket broadcast not available for approval gates")

# Import department mail service for notifications
try:
    from src.agents.departments.department_mail import get_mail_service, Priority as MailPriority
    MAIL_AVAILABLE = True
except ImportError:
    MAIL_AVAILABLE = False
    logger.warning("Department mail not available for approval gates")

router = APIRouter(prefix="/api/approval-gates", tags=["approval-gates"])


# =============================================================================
# Enums
# =============================================================================

class ApprovalStatus(str, Enum):
    PENDING = "pending"
    PENDING_REVIEW = "pending_review"  # Alpha Forge: awaiting human review
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED_REVIEW = "expired_review"  # Alpha Forge: timeout after 7 days


class GateType(str, Enum):
    STAGE_TRANSITION = "stage_transition"
    DEPLOYMENT = "deployment"
    RISK_CHECK = "risk_check"
    MANUAL_REVIEW = "manual_review"
    # Alpha Forge specific gate types
    ALPHA_FORGE_BACKTEST = "alpha_forge_backtest"  # Post-backtest approval
    ALPHA_FORGE_DEPLOYMENT = "alpha_forge_deployment"  # Pre-live deployment


# =============================================================================
# Database Models
# =============================================================================

from sqlalchemy import Column, Integer, String, DateTime, Enum as SQLEnum, Text


class ApprovalGateModel(Base):
    """SQLAlchemy model for approval gates."""
    __tablename__ = "approval_gates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    gate_id = Column(String(36), unique=True, nullable=False, index=True)
    workflow_id = Column(String(36), nullable=False, index=True)
    workflow_type = Column(String(50), nullable=True)
    from_stage = Column(String(100), nullable=False)
    to_stage = Column(String(100), nullable=False)
    gate_type = Column(SQLEnum(GateType), nullable=False, default=GateType.STAGE_TRANSITION)
    status = Column(SQLEnum(ApprovalStatus), nullable=False, default=ApprovalStatus.PENDING)
    requester = Column(String(100), nullable=True)
    assigned_to = Column(String(100), nullable=True, index=True)
    approver = Column(String(100), nullable=True)
    reason = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    extra_data = Column(Text, nullable=True)  # JSON string for additional data
    # Alpha Forge specific fields
    strategy_id = Column(String(36), nullable=True, index=True)  # Strategy being approved
    metrics_snapshot = Column(Text, nullable=True)  # JSON string for backtest metrics
    revision_feedback = Column(Text, nullable=True)  # Feedback from revision request
    expires_at = Column(DateTime, nullable=True)  # When PENDING_REVIEW expires (15 min)
    hard_expires_at = Column(DateTime, nullable=True)  # Hard timeout (7 days)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    approved_at = Column(DateTime, nullable=True)
    rejected_at = Column(DateTime, nullable=True)


# Create table if not exists
def init_approval_gates_table():
    """Initialize the approval_gates table if it doesn't exist."""
    try:
        # Create table if not exists
        ApprovalGateModel.__table__.create(engine, checkfirst=True)

        # Add new columns if they don't exist (for existing databases)
        from sqlalchemy import text
        session = get_session()
        try:
            # Check and add each new column if it doesn't exist
            columns_to_add = [
                ("strategy_id", "ALTER TABLE approval_gates ADD COLUMN strategy_id VARCHAR(36)"),
                ("metrics_snapshot", "ALTER TABLE approval_gates ADD COLUMN metrics_snapshot TEXT"),
                ("revision_feedback", "ALTER TABLE approval_gates ADD COLUMN revision_feedback TEXT"),
                ("expires_at", "ALTER TABLE approval_gates ADD COLUMN expires_at TIMESTAMP"),
                ("hard_expires_at", "ALTER TABLE approval_gates ADD COLUMN hard_expires_at TIMESTAMP"),
            ]

            for col_name, alter_stmt in columns_to_add:
                try:
                    session.execute(text(f"SELECT {col_name} FROM approval_gates LIMIT 1"))
                except Exception:
                    try:
                        session.execute(text(alter_stmt))
                        logger.info(f"Added column {col_name} to approval_gates table")
                    except Exception:
                        pass  # Column might already exist or table doesn't exist yet
        finally:
            session.close()

        logger.info("approval_gates table initialized")
    except Exception as e:
        logger.warning(f"Could not create approval_gates table: {e}")


# =============================================================================
# Pydantic Models
# =============================================================================

class ApprovalGateCreate(BaseModel):
    """Request model for creating an approval gate."""
    workflow_id: str = Field(..., description="ID of the workflow requesting approval")
    workflow_type: Optional[str] = Field(None, description="Type of workflow (e.g., video_ingest_to_ea, trd_to_ea)")
    from_stage: str = Field(..., description="Current workflow stage")
    to_stage: str = Field(..., description="Next workflow stage to transition to")
    gate_type: GateType = Field(default=GateType.STAGE_TRANSITION, description="Type of approval gate")
    requester: Optional[str] = Field(None, description="User requesting approval")
    assigned_to: Optional[str] = Field(None, description="User or role assigned to approve this gate")
    reason: Optional[str] = Field(None, description="Reason for the transition")
    extra_data: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    # Alpha Forge specific fields
    strategy_id: Optional[str] = Field(None, description="Strategy ID for Alpha Forge approval")
    metrics_snapshot: Optional[Dict[str, Any]] = Field(default=None, description="Backtest metrics snapshot")


class ApprovalGateUpdate(BaseModel):
    """Request model for approving or rejecting a gate."""
    approver: str = Field(..., description="User approving or rejecting")
    notes: Optional[str] = Field(None, description="Notes for the approval/rejection")


class ApprovalGateResponse(BaseModel):
    """Response model for approval gate."""
    gate_id: str
    workflow_id: str
    workflow_type: Optional[str]
    from_stage: str
    to_stage: str
    gate_type: GateType
    status: ApprovalStatus
    requester: Optional[str]
    assigned_to: Optional[str]
    approver: Optional[str]
    reason: Optional[str]
    notes: Optional[str]
    extra_data: Optional[Dict[str, Any]]
    # Alpha Forge specific fields
    strategy_id: Optional[str]
    metrics_snapshot: Optional[Dict[str, Any]]
    revision_feedback: Optional[str]
    expires_at: Optional[str]
    hard_expires_at: Optional[str]
    created_at: str
    updated_at: str
    approved_at: Optional[str]
    rejected_at: Optional[str]


class ApprovalGateRevisionRequest(BaseModel):
    """Request model for requesting revision of a gate."""
    approver: str = Field(..., description="User requesting revision")
    feedback: str = Field(..., description="Feedback for the revision")
    create_new_gate: bool = Field(default=True, description="Whether to create a new gate after revision")


class ApprovalGateListResponse(BaseModel):
    """Response model for listing approval gates."""
    gates: List[ApprovalGateResponse]
    total: int


class ApprovalActionResponse(BaseModel):
    """Response model for approval actions."""
    success: bool
    gate_id: str
    status: ApprovalStatus
    message: str
    approver: str
    timestamp: str


# =============================================================================
# Database Operations
# =============================================================================

def _get_db():
    """Get database session."""
    session = get_session()
    try:
        yield session
    finally:
        session.close()


def _model_to_response(model: ApprovalGateModel) -> ApprovalGateResponse:
    """Convert SQLAlchemy model to Pydantic response."""
    import json
    extra_data = None
    metrics_snapshot = None

    if model.extra_data:
        try:
            extra_data = json.loads(model.extra_data)
        except json.JSONDecodeError:
            pass

    if model.metrics_snapshot:
        try:
            metrics_snapshot = json.loads(model.metrics_snapshot)
        except json.JSONDecodeError:
            pass

    return ApprovalGateResponse(
        gate_id=model.gate_id,
        workflow_id=model.workflow_id,
        workflow_type=model.workflow_type,
        from_stage=model.from_stage,
        to_stage=model.to_stage,
        gate_type=model.gate_type,
        status=model.status,
        requester=model.requester,
        assigned_to=model.assigned_to,
        approver=model.approver,
        reason=model.reason,
        notes=model.notes,
        extra_data=extra_data,
        # Alpha Forge specific fields
        strategy_id=model.strategy_id,
        metrics_snapshot=metrics_snapshot,
        revision_feedback=model.revision_feedback,
        expires_at=model.expires_at.isoformat() if model.expires_at else None,
        hard_expires_at=model.hard_expires_at.isoformat() if model.hard_expires_at else None,
        created_at=model.created_at.isoformat() if model.created_at else None,
        updated_at=model.updated_at.isoformat() if model.updated_at else None,
        approved_at=model.approved_at.isoformat() if model.approved_at else None,
        rejected_at=model.rejected_at.isoformat() if model.rejected_at else None,
    )


def _send_approval_mail(
    gate: ApprovalGateModel,
    action: str,
    approver_name: Optional[str] = None,
) -> None:
    """
    Send department mail notification for approval gate events.

    Args:
        gate: The approval gate model
        action: The action (created, approved, rejected)
        approver_name: Name of the approver (for approved/rejected actions)
    """
    if not MAIL_AVAILABLE:
        return

    try:
        mail_service = get_mail_service()

        # Determine target department based on assignment or workflow type
        to_dept = gate.assigned_to if gate.assigned_to else "development"

        # Map workflow types to departments
        workflow_dept_map = {
            "video_ingest_to_ea": "research",
            "trd_to_ea": "development",
            "backtest": "research",
        }
        if gate.workflow_type and gate.workflow_type in workflow_dept_map:
            to_dept = workflow_dept_map[gate.workflow_type]

        # Determine the appropriate action and body based on action type
        if action == "created":
            subject = f"Approval Required: {gate.from_stage} -> {gate.to_stage}"
            body = f"""An approval is required to transition from '{gate.from_stage}' to '{gate.to_stage}'.

Workflow ID: {gate.workflow_id}
Workflow Type: {gate.workflow_type or 'N/A'}
Gate Type: {gate.gate_type.value}
Requester: {gate.requester or 'System'}

"""
            if gate.reason:
                body += f"Reason: {gate.reason}\n\n"
            body += f"Gate ID: {gate.gate_id}\n\n"
            body += "Please review and approve or reject this transition."

            priority = MailPriority.HIGH
        elif action == "approved":
            subject = f"Approved: {gate.from_stage} -> {gate.to_stage}"
            body = f"""The transition from '{gate.from_stage}' to '{gate.to_stage}' has been approved.

Workflow ID: {gate.workflow_id}
Approved by: {approver_name or gate.approver or 'Unknown'}

"""
            if gate.notes:
                body += f"Notes: {gate.notes}\n\n"
            body += f"Gate ID: {gate.gate_id}"

            priority = MailPriority.NORMAL
        else:  # rejected
            subject = f"Rejected: {gate.from_stage} -> {gate.to_stage}"
            body = f"""The transition from '{gate.from_stage}' to '{gate.to_stage}' has been rejected.

Workflow ID: {gate.workflow_id}
Rejected by: {approver_name or gate.approver or 'Unknown'}

"""
            if gate.notes:
                body += f"Reason: {gate.notes}\n\n"
            body += f"Gate ID: {gate.gate_id}"

            priority = MailPriority.HIGH

        # Send the notification
        from src.agents.departments.department_mail import MessageType
        mail_service.send(
            from_dept="floor_manager",
            to_dept=to_dept,
            type=MessageType.APPROVAL_REQUEST if action == "created" else
                 MessageType.APPROVAL_APPROVED if action == "approved" else
                 MessageType.APPROVAL_REJECTED,
            subject=subject,
            body=body,
            priority=priority,
            gate_id=gate.gate_id,
            workflow_id=gate.workflow_id,
            from_stage=gate.from_stage,
            to_stage=gate.to_stage,
        )

        logger.info(f"Sent approval mail notification for gate {gate.gate_id} ({action})")

    except Exception as e:
        logger.warning(f"Failed to send approval mail notification: {e}")


# =============================================================================
# API Endpoints
# =============================================================================

# NOTE: Routes with static path segments MUST come before dynamic /{gate_id}
# to avoid route conflicts where /pending or /workflow/xxx is matched as a gate_id


@router.get("/pending", response_model=ApprovalGateListResponse)
async def list_pending_gates(
    workflow_id: Optional[str] = Query(None, description="Filter by workflow ID"),
    assigned_to: Optional[str] = Query(None, description="Filter by assigned user/role"),
    limit: int = Query(50, description="Maximum number of gates to return"),
    db: Session = Depends(_get_db)
):
    """
    List all pending approval gates.

    Optionally filter by workflow_id or assigned_to.
    """
    query = db.query(ApprovalGateModel).filter(ApprovalGateModel.status == ApprovalStatus.PENDING)

    if workflow_id:
        query = query.filter(ApprovalGateModel.workflow_id == workflow_id)

    if assigned_to:
        query = query.filter(ApprovalGateModel.assigned_to == assigned_to)

    gates = query.order_by(ApprovalGateModel.created_at.asc()).limit(limit).all()

    return ApprovalGateListResponse(
        gates=[_model_to_response(gate) for gate in gates],
        total=len(gates)
    )


@router.get("/workflow/{workflow_id}", response_model=ApprovalGateListResponse)
async def get_workflow_gates(
    workflow_id: str,
    db: Session = Depends(_get_db)
):
    """
    Get all approval gates for a specific workflow.
    """
    gates = db.query(ApprovalGateModel).filter(
        ApprovalGateModel.workflow_id == workflow_id
    ).order_by(ApprovalGateModel.created_at.asc()).all()

    return ApprovalGateListResponse(
        gates=[_model_to_response(gate) for gate in gates],
        total=len(gates)
    )


@router.post("", response_model=ApprovalGateResponse, status_code=201)
async def create_approval_gate(
    request: ApprovalGateCreate,
    db: Session = Depends(_get_db)
):
    """
    Create a new approval gate request.

    This endpoint is called when a workflow wants to transition to a new stage
    but requires user approval first.
    """
    import json
    from datetime import timedelta

    # Generate unique gate ID
    gate_id = str(uuid.uuid4())

    # Determine initial status - Alpha Forge gates use PENDING_REVIEW
    initial_status = ApprovalStatus.PENDING
    if request.gate_type in [GateType.ALPHA_FORGE_BACKTEST, GateType.ALPHA_FORGE_DEPLOYMENT]:
        initial_status = ApprovalStatus.PENDING_REVIEW

    # Calculate expiration times for Alpha Forge gates
    now = datetime.now(timezone.utc)
    expires_at = None
    hard_expires_at = None

    if request.gate_type in [GateType.ALPHA_FORGE_BACKTEST, GateType.ALPHA_FORGE_DEPLOYMENT]:
        # 15-minute soft timeout
        expires_at = now + timedelta(minutes=15)
        # 7-day hard timeout
        hard_expires_at = now + timedelta(days=7)

    # Create the gate record
    gate = ApprovalGateModel(
        gate_id=gate_id,
        workflow_id=request.workflow_id,
        workflow_type=request.workflow_type,
        from_stage=request.from_stage,
        to_stage=request.to_stage,
        gate_type=request.gate_type,
        status=initial_status,
        requester=request.requester,
        assigned_to=request.assigned_to,
        reason=request.reason,
        extra_data=json.dumps(request.extra_data) if request.extra_data else None,
        # Alpha Forge specific fields
        strategy_id=request.strategy_id,
        metrics_snapshot=json.dumps(request.metrics_snapshot) if request.metrics_snapshot else None,
        expires_at=expires_at,
        hard_expires_at=hard_expires_at,
        created_at=now,
        updated_at=now,
    )

    db.add(gate)
    db.commit()
    db.refresh(gate)

    logger.info(f"Created approval gate {gate_id} for workflow {request.workflow_id}")

    # Broadcast approval gate creation via WebSocket
    if WS_AVAILABLE:
        try:
            await broadcast_approval_gate(
                gate_id=gate_id,
                workflow_id=request.workflow_id,
                action="created",
                from_stage=request.from_stage,
                to_stage=request.to_stage,
                gate_type=request.gate_type.value
            )
        except Exception as e:
            logger.warning(f"Failed to broadcast approval gate creation: {e}")

    # Send department mail notification
    _send_approval_mail(gate, "created")

    return _model_to_response(gate)


@router.get("/alpha-forge/pending", response_model=ApprovalGateListResponse)
async def list_alpha_forge_pending(
    assigned_to: Optional[str] = Query(None, description="Filter by assigned user"),
    limit: int = Query(50, description="Maximum number of gates to return"),
    db: Session = Depends(_get_db)
):
    """
    List all pending Alpha Forge approval gates.

    Returns gates with ALPHA_FORGE_BACKTEST or ALPHA_FORGE_DEPLOYMENT gate types
    that are in PENDING_REVIEW status.
    """
    query = db.query(ApprovalGateModel).filter(
        ApprovalGateModel.status == ApprovalStatus.PENDING_REVIEW,
        ApprovalGateModel.gate_type.in_([
            GateType.ALPHA_FORGE_BACKTEST,
            GateType.ALPHA_FORGE_DEPLOYMENT
        ])
    )

    if assigned_to:
        query = query.filter(ApprovalGateModel.assigned_to == assigned_to)

    gates = query.order_by(ApprovalGateModel.created_at.asc()).limit(limit).all()

    return ApprovalGateListResponse(
        gates=[_model_to_response(gate) for gate in gates],
        total=len(gates)
    )


@router.get("/{gate_id}", response_model=ApprovalGateResponse)
async def get_approval_gate(
    gate_id: str,
    db: Session = Depends(_get_db)
):
    """
    Get approval gate by ID.
    """
    gate = db.query(ApprovalGateModel).filter(ApprovalGateModel.gate_id == gate_id).first()

    if not gate:
        raise HTTPException(status_code=404, detail=f"Approval gate {gate_id} not found")

    return _model_to_response(gate)


@router.post("/{gate_id}/approve", response_model=ApprovalActionResponse)
async def approve_gate(
    gate_id: str,
    request: ApprovalGateUpdate,
    db: Session = Depends(_get_db)
):
    """
    Approve an approval gate.

    This allows the workflow to proceed to the next stage.
    For Alpha Forge gates (PENDING_REVIEW), stores immutable audit record.
    """
    gate = db.query(ApprovalGateModel).filter(ApprovalGateModel.gate_id == gate_id).first()

    if not gate:
        raise HTTPException(status_code=404, detail=f"Approval gate {gate_id} not found")

    # Allow approval from PENDING or PENDING_REVIEW (Alpha Forge) status
    if gate.status not in [ApprovalStatus.PENDING, ApprovalStatus.PENDING_REVIEW]:
        raise HTTPException(
            status_code=400,
            detail=f"Gate is already {gate.status.value}. Cannot approve."
        )

    # Create immutable audit record for Alpha Forge gates
    audit_record = None
    if gate.gate_type in [GateType.ALPHA_FORGE_BACKTEST, GateType.ALPHA_FORGE_DEPLOYMENT]:
        import json
        now = datetime.now(timezone.utc)
        audit_record = {
            "strategy_id": gate.strategy_id,
            "approver": request.approver,
            "approved_at_utc": now.isoformat(),
            "gate_type": gate.gate_type.value,
            "metrics_snapshot": json.loads(gate.metrics_snapshot) if gate.metrics_snapshot else None,
            "workflow_id": gate.workflow_id,
            "gate_id": gate.gate_id
        }
        logger.info(f"Created immutable audit record for approval gate {gate_id}: {audit_record}")

    # Update the gate
    gate.status = ApprovalStatus.APPROVED
    gate.approver = request.approver
    gate.notes = request.notes
    gate.approved_at = datetime.now(timezone.utc)
    gate.updated_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(gate)

    logger.info(f"Approved gate {gate_id} by {request.approver}")

    # Broadcast approval via WebSocket
    if WS_AVAILABLE:
        try:
            await broadcast_approval_gate(
                gate_id=gate_id,
                workflow_id=gate.workflow_id,
                action="approved",
                from_stage=gate.from_stage,
                to_stage=gate.to_stage,
                gate_type=gate.gate_type.value,
                approver=request.approver,
                notes=request.notes
            )
        except Exception as e:
            logger.warning(f"Failed to broadcast approval: {e}")

    # Send department mail notification
    _send_approval_mail(gate, "approved", request.approver)

    return ApprovalActionResponse(
        success=True,
        gate_id=gate_id,
        status=ApprovalStatus.APPROVED,
        message=f"Workflow can now transition from {gate.from_stage} to {gate.to_stage}",
        approver=request.approver,
        timestamp=gate.approved_at.isoformat()
    )


@router.post("/{gate_id}/reject", response_model=ApprovalActionResponse)
async def reject_gate(
    gate_id: str,
    request: ApprovalGateUpdate,
    db: Session = Depends(_get_db)
):
    """
    Reject an approval gate.

    This blocks the workflow from proceeding to the next stage.
    """
    gate = db.query(ApprovalGateModel).filter(ApprovalGateModel.gate_id == gate_id).first()

    if not gate:
        raise HTTPException(status_code=404, detail=f"Approval gate {gate_id} not found")

    # Allow rejection from PENDING or PENDING_REVIEW (Alpha Forge) status
    if gate.status not in [ApprovalStatus.PENDING, ApprovalStatus.PENDING_REVIEW]:
        raise HTTPException(
            status_code=400,
            detail=f"Gate is already {gate.status.value}. Cannot reject."
        )

    # Update the gate
    gate.status = ApprovalStatus.REJECTED
    gate.approver = request.approver
    gate.notes = request.notes
    gate.rejected_at = datetime.now(timezone.utc)
    gate.updated_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(gate)

    logger.info(f"Rejected gate {gate_id} by {request.approver}")

    # Broadcast rejection via WebSocket
    if WS_AVAILABLE:
        try:
            await broadcast_approval_gate(
                gate_id=gate_id,
                workflow_id=gate.workflow_id,
                action="rejected",
                from_stage=gate.from_stage,
                to_stage=gate.to_stage,
                gate_type=gate.gate_type.value,
                approver=request.approver,
                notes=request.notes
            )
        except Exception as e:
            logger.warning(f"Failed to broadcast rejection: {e}")

    # Send department mail notification
    _send_approval_mail(gate, "rejected", request.approver)

    return ApprovalActionResponse(
        success=True,
        gate_id=gate_id,
        status=ApprovalStatus.REJECTED,
        message=f"Workflow transition from {gate.from_stage} to {gate.to_stage} has been rejected",
        approver=request.approver,
        timestamp=gate.rejected_at.isoformat()
    )


@router.post("/{gate_id}/request-revision", response_model=ApprovalActionResponse)
async def request_revision(
    gate_id: str,
    request: ApprovalGateRevisionRequest,
    db: Session = Depends(_get_db)
):
    """
    Request revision for an approval gate.

    This allows the user to provide feedback and request the agent to re-execute
    with the feedback as context. A new approval gate can be created after revision.
    """
    gate = db.query(ApprovalGateModel).filter(ApprovalGateModel.gate_id == gate_id).first()

    if not gate:
        raise HTTPException(status_code=404, detail=f"Approval gate {gate_id} not found")

    if gate.status not in [ApprovalStatus.PENDING, ApprovalStatus.PENDING_REVIEW]:
        raise HTTPException(
            status_code=400,
            detail=f"Gate is {gate.status.value}. Cannot request revision."
        )

    # Store the revision feedback
    gate.revision_feedback = request.feedback
    gate.updated_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(gate)

    logger.info(f"Revision requested for gate {gate_id} by {request.approver}")

    # If create_new_gate is True, we'll return info for the workflow to create a new gate
    # The actual new gate creation is handled by the workflow orchestrator

    return ApprovalActionResponse(
        success=True,
        gate_id=gate_id,
        status=gate.status,
        message=f"Revision requested. Feedback stored: {request.feedback[:100]}...",
        approver=request.approver,
        timestamp=gate.updated_at.isoformat()
    )


@router.post("/{gate_id}/check-timeout", response_model=ApprovalGateResponse)
async def check_gate_timeout(
    gate_id: str,
    db: Session = Depends(_get_db)
):
    """
    Check and update gate timeout status.

    For PENDING_REVIEW gates:
    - If 15 minutes have passed: soft timeout (keep as PENDING_REVIEW)
    - If 7 days have passed: hard timeout (set to EXPIRED_REVIEW)

    Returns the updated gate status.
    """
    gate = db.query(ApprovalGateModel).filter(ApprovalGateModel.gate_id == gate_id).first()

    if not gate:
        raise HTTPException(status_code=404, detail=f"Approval gate {gate_id} not found")

    now = datetime.now(timezone.utc)
    updated = False

    # Check for hard timeout (7 days) - move to EXPIRED_REVIEW
    if gate.status == ApprovalStatus.PENDING_REVIEW and gate.hard_expires_at:
        # Ensure both datetimes are timezone-aware for comparison
        hard_exp = gate.hard_expires_at
        if hard_exp.tzinfo is None:
            hard_exp = hard_exp.replace(tzinfo=timezone.utc)
        if now >= hard_exp:
            gate.status = ApprovalStatus.EXPIRED_REVIEW
            gate.updated_at = now
            updated = True
            logger.info(f"Gate {gate_id} expired (hard timeout reached)")

    # Check for soft timeout (15 min) - just log, still PENDING_REVIEW
    elif gate.status == ApprovalStatus.PENDING_REVIEW and gate.expires_at:
        exp = gate.expires_at
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        if now >= exp:
            logger.info(f"Gate {gate_id} soft timeout reached (15 min), still pending review")

    if updated:
        db.commit()
        db.refresh(gate)

    return _model_to_response(gate)


# Initialize table on module load
init_approval_gates_table()

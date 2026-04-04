"""
Department Mail API Endpoints

REST API for cross-department agent communication.

Endpoints:
- POST /api/departments/mail/send - Send a message
- GET /api/departments/mail/inbox/{department} - Get inbox for department
- GET /api/departments/mail/{message_id} - Read specific message
- PATCH /api/departments/mail/{message_id}/read - Mark as read
- GET /api/departments/mail/stats - Get mail statistics
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.agents.departments.department_mail import (
    DepartmentMailService,
    DepartmentMessage,
    MessageType,
    Priority,
)
from src.agents.departments.types import Department
from src.api.pagination import PaginatedResponse, DEFAULT_LIMIT, DEFAULT_OFFSET, MAX_LIMIT

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/departments/mail", tags=["department-mail"])


# ============== Request/Response Models ==============

class SendMessageRequest(BaseModel):
    """Request to send a message."""
    from_dept: str = Field(..., description="Sending department")
    to_dept: str = Field(..., description="Receiving department")
    type: str = Field(
        default="status",
        description="Message type: status, question, result, error, dispatch"
    )
    subject: str = Field(..., description="Message subject")
    body: str = Field(..., description="Message content")
    priority: str = Field(
        default="normal",
        description="Priority: low, normal, high, urgent"
    )


class MessageResponse(BaseModel):
    """Response model for a single message."""
    id: str
    from_dept: str
    to_dept: str
    type: str
    subject: str
    body: str
    priority: str
    timestamp: datetime
    read: bool
    # Approval-related fields
    gate_id: Optional[str] = None
    workflow_id: Optional[str] = None
    from_stage: Optional[str] = None
    to_stage: Optional[str] = None


class InboxResponse(BaseModel):
    """Response model for department inbox."""
    department: str
    unread_count: int
    total_count: int
    messages: List[MessageResponse]


class StatsResponse(BaseModel):
    """Response model for mail statistics."""
    total_messages: int
    unread_messages: int
    by_department: Dict[str, int]
    by_type: Dict[str, int]
    by_priority: Dict[str, int]
    unread_by_department: Dict[str, int]
    recent_24h: int
    urgent_unread: int


class ListMessagesResponse(BaseModel):
    """Response model for message listing."""
    total: int
    messages: List[MessageResponse]


# ============== Helper Functions ==============

def _message_to_response(msg: DepartmentMessage) -> MessageResponse:
    """Convert DepartmentMessage to response model."""
    return MessageResponse(
        id=msg.id,
        from_dept=msg.from_dept,
        to_dept=msg.to_dept,
        type=msg.type.value,
        subject=msg.subject,
        body=msg.body,
        priority=msg.priority.value,
        timestamp=msg.timestamp,
        read=msg.read,
        gate_id=msg.gate_id,
        workflow_id=msg.workflow_id,
        from_stage=msg.from_stage,
        to_stage=msg.to_stage,
    )


def _validate_department(dept: str) -> str:
    """Validate department name."""
    valid_depts = [d.value for d in Department]
    # Allow floor_manager as special sender
    if dept == "floor_manager":
        return dept
    if dept not in valid_depts:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid department: {dept}. Valid: {valid_depts}"
        )
    return dept


def _validate_message_type(msg_type: str) -> MessageType:
    """Validate and convert message type."""
    try:
        return MessageType(msg_type.lower())
    except ValueError:
        valid_types = [t.value for t in MessageType]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid message type: {msg_type}. Valid: {valid_types}"
        )


def _validate_priority(priority: str) -> Priority:
    """Validate and convert priority."""
    try:
        return Priority(priority.lower())
    except ValueError:
        valid_priorities = [p.value for p in Priority]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid priority: {priority}. Valid: {valid_priorities}"
        )


# ============== Endpoints ==============

def _get_mail_svc():
    """Lazy shared mail service selection with Redis fallback."""
    from src.agents.departments.department_mail import get_mail_service

    return get_mail_service()

@router.post("/send", response_model=MessageResponse)
async def send_message(request: SendMessageRequest):
    """
    Send a message to a department inbox.

    Message types:
    - status: General status updates
    - question: Questions requiring response
    - result: Results from completed tasks
    - error: Error notifications
    - dispatch: Cross-department task dispatch

    Priority levels:
    - low: Non-urgent information
    - normal: Standard priority
    - high: Important but not critical
    - urgent: Requires immediate attention
    """
    mail_service = _get_mail_svc()

    # Validate inputs
    from_dept = _validate_department(request.from_dept)
    to_dept = _validate_department(request.to_dept)
    msg_type = _validate_message_type(request.type)
    priority = _validate_priority(request.priority)

    # Send message
    message = mail_service.send(
        from_dept=from_dept,
        to_dept=to_dept,
        type=msg_type,
        subject=request.subject,
        body=request.body,
        priority=priority,
    )

    logger.info(f"Message sent: {message.id} from {from_dept} to {to_dept}")

    return _message_to_response(message)


@router.get("/inbox/{department}", response_model=InboxResponse)
async def get_inbox(
    department: str,
    unread_only: bool = Query(False, description="Only return unread messages"),
    limit: int = Query(100, ge=1, le=500, description="Maximum messages to return"),
):
    """
    Get inbox for a specific department.

    Returns messages addressed to the department, sorted by timestamp (newest first).
    """
    mail_service = _get_mail_svc()

    # Validate department
    dept = _validate_department(department)

    # Get messages
    messages = mail_service.check_inbox(
        dept=dept,
        unread_only=False,
        limit=1000,  # Get all for counting
    )

    # Filter and count
    unread_count = sum(1 for m in messages if not m.read)

    if unread_only:
        messages = [m for m in messages if not m.read]

    # Apply limit
    total_count = len(messages)
    messages = messages[:limit]

    return InboxResponse(
        department=dept,
        unread_count=unread_count,
        total_count=total_count,
        messages=[_message_to_response(m) for m in messages],
    )


@router.get("/inbox", response_model=ListMessagesResponse)
async def get_all_inboxes(
    unread_only: bool = Query(False, description="Only return unread messages"),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT, description="Maximum messages to return"),
    offset: int = Query(DEFAULT_OFFSET, ge=0, description="Number of messages to skip"),
):
    """
    Get all inbox messages across all departments with pagination.

    Returns all messages addressed to any department, sorted by timestamp (newest first).
    """
    mail_service = _get_mail_svc()

    # Get all messages
    all_messages = mail_service.list_messages(limit=10000)

    # Filter to only inbox messages (not from floor_manager if it's a dispatch)
    messages = [m for m in all_messages if m.to_dept != "floor_manager"]

    if unread_only:
        messages = [m for m in messages if not m.read]

    total = len(messages)
    paginated_messages = messages[offset:offset + limit]

    return ListMessagesResponse(
        total=total,
        messages=[_message_to_response(m) for m in paginated_messages],
    )


@router.get("/sent", response_model=ListMessagesResponse)
async def get_sent_messages(
    from_dept: Optional[str] = Query(None, description="Filter by sender department"),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT, description="Maximum messages to return"),
    offset: int = Query(DEFAULT_OFFSET, ge=0, description="Number of messages to skip"),
):
    """
    Get sent messages with pagination.

    Returns messages sent by departments, sorted by timestamp (newest first).
    """
    mail_service = _get_mail_svc()

    # Get all messages
    all_messages = mail_service.list_messages(limit=10000)

    # Filter to sent messages
    if from_dept:
        _validate_department(from_dept)
        messages = [m for m in all_messages if m.from_dept == from_dept]
    else:
        messages = all_messages

    total = len(messages)
    paginated_messages = messages[offset:offset + limit]

    return ListMessagesResponse(
        total=total,
        messages=[_message_to_response(m) for m in paginated_messages],
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats_alias():
    """
    Get mail statistics (alias for /stats/overview).

    Returns counts by department, type, priority, and recent activity.
    """
    return await get_stats()


@router.patch("/read-all", response_model=Dict[str, Any])
async def mark_all_read(
    department: Optional[str] = Query(None, description="Department to mark all as read for"),
):
    """
    Mark all messages as read for a department (or all if no department specified).
    """
    mail_service = _get_mail_svc()

    if department:
        _validate_department(department)
        messages = mail_service.check_inbox(dept=department, unread_only=True, limit=1000)
    else:
        messages = mail_service.list_messages(unread_only=True, limit=1000)

    count = 0
    for msg in messages:
        if mail_service.mark_read(msg.id):
            count += 1

    return {
        "status": "success",
        "marked_read": count,
        "department": department or "all",
    }


@router.get("/{message_id}", response_model=MessageResponse)
async def get_message(message_id: str):
    """
    Read a specific message by ID.

    Returns the full message details including body.
    """
    mail_service = _get_mail_svc()

    message = mail_service.get_message(message_id)
    if message is None:
        raise HTTPException(
            status_code=404,
            detail=f"Message not found: {message_id}"
        )

    return _message_to_response(message)


@router.patch("/{message_id}/read", response_model=Dict[str, Any])
async def mark_message_read(message_id: str):
    """
    Mark a message as read.

    Updates the message's read status to true.
    """
    mail_service = _get_mail_svc()

    # Check message exists
    message = mail_service.get_message(message_id)
    if message is None:
        raise HTTPException(
            status_code=404,
            detail=f"Message not found: {message_id}"
        )

    # Mark as read
    success = mail_service.mark_read(message_id)

    return {
        "status": "success" if success else "already_read",
        "message_id": message_id,
        "read": True,
    }


@router.get("/stats/overview", response_model=StatsResponse)
async def get_stats():
    """
    Get mail statistics.

    Returns counts by department, type, priority, and recent activity.
    """
    mail_service = _get_mail_svc()
    stats = mail_service.get_stats()

    return StatsResponse(**stats)


@router.get("/list/messages", response_model=ListMessagesResponse)
async def list_messages(
    from_dept: Optional[str] = Query(None, description="Filter by sender department"),
    to_dept: Optional[str] = Query(None, description="Filter by recipient department"),
    unread_only: bool = Query(False, description="Only return unread messages"),
    type: Optional[str] = Query(None, description="Filter by message type"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    limit: int = Query(100, ge=1, le=500, description="Maximum messages to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
):
    """
    List messages with flexible filtering.

    Supports filtering by department, type, priority, and read status.
    Results are paginated with limit/offset.
    """
    mail_service = _get_mail_svc()

    # Validate filters
    msg_type = _validate_message_type(type) if type else None
    msg_priority = _validate_priority(priority) if priority else None

    if from_dept:
        _validate_department(from_dept)
    if to_dept:
        _validate_department(to_dept)

    # Get messages
    messages = mail_service.list_messages(
        from_dept=from_dept,
        to_dept=to_dept,
        unread_only=unread_only,
        message_type=msg_type,
        priority=msg_priority,
        limit=limit,
        offset=offset,
    )

    # Get total count (approximate)
    total = len(messages)

    return ListMessagesResponse(
        total=total,
        messages=[_message_to_response(m) for m in messages],
    )


@router.post("/{message_id}/reply", response_model=MessageResponse)
async def reply_to_message(
    message_id: str,
    from_dept: str = Query(..., description="Department sending the reply"),
    body: str = Query(..., description="Reply content"),
    priority: Optional[str] = Query(None, description="Priority (defaults to original)"),
):
    """
    Reply to a message.

    Creates a new message with reversed from/to departments.
    """
    mail_service = _get_mail_svc()

    # Validate
    _validate_department(from_dept)
    msg_priority = _validate_priority(priority) if priority else None

    # Send reply
    reply = mail_service.reply(
        original_message_id=message_id,
        from_dept=from_dept,
        body=body,
        priority=msg_priority,
    )

    if reply is None:
        raise HTTPException(
            status_code=404,
            detail=f"Original message not found: {message_id}"
        )

    return _message_to_response(reply)


@router.delete("/purge", response_model=Dict[str, Any])
async def purge_old_messages(
    days: int = Query(30, ge=1, le=365, description="Keep messages from last N days"),
):
    """
    Delete messages older than specified days.

    Returns count of deleted messages.
    """
    mail_service = _get_mail_svc()

    deleted_count = mail_service.purge_old_messages(days=days)

    logger.info(f"Purged {deleted_count} messages older than {days} days")

    return {
        "status": "success",
        "deleted_count": deleted_count,
        "days_kept": days,
    }


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    mail_service = _get_mail_svc()

    # Quick database check
    stats = mail_service.get_stats()

    return {
        "status": "healthy",
        "service": "department-mail",
        "total_messages": stats["total_messages"],
        "unread_messages": stats["unread_messages"],
    }
